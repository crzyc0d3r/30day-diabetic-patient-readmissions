#!/usr/bin/env python3
"""Execute every notebook under "pipeline/" in filename order with progress output.

Invoked from the capstone root, e.g. `python pipeline/run_pipeline.py ...`,
since the script itself lives in pipeline/ alongside the 01..08 notebooks.

Examples
--------
    python run_pipeline.py
    python run_pipeline.py --from 04_feature_engineering.ipynb
    python run_pipeline.py --from 01_overview.ipynb --to 05_split_encode_scale_select.ipynb
    python run_pipeline.py --to 05_split_encode_scale_select.ipynb
    python run_pipeline.py 02_data_cleaning.ipynb 03_exploratory_data_analysis.ipynb
    python run_pipeline.py --timeout 7200 --keep-going
    python run_pipeline.py --reset # wipe data/ + clear outputs, then run
    python run_pipeline.py --reset --list # only wipe and clear, don't run
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import warnings
from pathlib import Path

import nbformat
from nbconvert.preprocessors import CellExecutionError, ClearOutputPreprocessor, ExecutePreprocessor

warnings.filterwarnings("ignore", category=nbformat.validator.MissingIDFieldWarning)

# This file lives at capstone/pipeline/run_pipeline.py, so .parent.parent
# is the capstone root. Keeping ROOT at the capstone root preserves the
# semantics that NOTEBOOKS_DIR -> pipeline/, DATA_DIR -> data/, and the
# .relative_to(ROOT) prints below stay 'pipeline' and 'data'.
ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = ROOT / "pipeline"
DATA_DIR = ROOT / "data"

def discover_notebooks() -> list[Path]:
    return sorted(p for p in NOTEBOOKS_DIR.glob("*.ipynb") if not p.name.startswith("."))

def wipe_data_dir(data_dir: Path) -> int:
    """Remove every entry under `data_dir` (the directory itself stays).

    The `data/` directory is the durable artefact store for every pipeline
    rerun: `cleaned.csv`, `features.csv`, `train_test.npz`,
    `final_model.joblib`, `final_model_threshold.joblib`, and the
    Notebook-08 evaluation drops live here. `--reset` invokes this helper
    so a fresh run starts from raw inputs only. The parent directory is
    preserved (and re-created if missing) so the relative paths the
    notebooks compute against `Path('../data')` still resolve.

    Symlinks and plain files are unlinked. Subdirectories are removed
    recursively via `shutil.rmtree`. Returns the count of top-level
    entries removed for the run-summary print at the call site.
    """
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        return 0
    removed = 0
    for child in data_dir.iterdir():
        if child.is_symlink() or child.is_file():
            child.unlink()
        else:
            shutil.rmtree(child)
        removed += 1
    return removed

def clear_notebook_outputs(path: Path) -> None:
    nb = nbformat.read(path, as_version=4)
    _, nb = nbformat.validator.normalize(nb)
    ClearOutputPreprocessor().preprocess(nb, {})
    nbformat.write(nb, path)

def fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s"

def progress_bar(done: int, total: int, width: int = 30) -> str:
    filled = int(width * done / total) if total else width
    return "[" + "#" * filled + "-" * (width - filled) + f"] {done}/{total}"

# Notebook stages whose orchestration lives in the library
# (helpers/*_pipeline.py). For these we call the library function directly
# instead of paying for a Jupyter kernel + cell-by-cell execution + the
# write-back of a re-rendered notebook. The notebooks are thin
# demonstrators for human inspection. The library is the production
# execution surface, which makes `--from 06` (the common dev-loop case) ~10x
# faster.
_LIBRARY_STAGES: dict[str, str] = {
    # filename prefix : "<helpers module>:<function name>"
    "06_": "helpers.hpo_pipeline:run_hpo",
    "07_": "helpers.training_pipeline:train_baselines_and_refits",
    "08_": "helpers.conclusion_pipeline:run_conclusion_and_register",
}


def _run_library_stage(path: Path) -> tuple[bool, str]:
    """Call the matching helpers.*_pipeline entry point for an NB06/07/08 stage.

    Returns `(ok, error_or_empty)` matching :func:`run_notebook`'s signature
    so the main driver loop can dispatch uniformly.
    """
    import importlib

    # `mediwatch train` runs this script from pipeline/ (or anywhere else),
    # but the `helpers` package lives at the project root. Each notebook
    # puts the root on sys.path with a `sys.path.insert(0, PROJECT_ROOT)` cell.
    # The library-stage path does the same here before importing.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    prefix = path.name[:3]
    spec = _LIBRARY_STAGES[prefix]
    module_name, fn_name = spec.split(":", 1)
    try:
        module = importlib.import_module(module_name)
        fn = getattr(module, fn_name)
        # All three library entry points accept these defaults relative to
        # the project root (ROOT). They're explicit here so the call site
        # documents the contract.
        if fn_name == "run_hpo":
            fn(train_test_path=str(DATA_DIR / "train_test.npz"), out_dir=str(DATA_DIR))
        elif fn_name == "train_baselines_and_refits":
            fn(
                train_test_path=str(DATA_DIR / "train_test.npz"),
                tuned_results_path=str(DATA_DIR / "tuned_results.joblib"),
                out_dir=str(DATA_DIR),
            )
        elif fn_name == "run_conclusion_and_register":
            fn(
                train_test_path=str(DATA_DIR / "train_test.npz"),
                in_dir=str(DATA_DIR),
                out_dir=str(DATA_DIR),
                register=True,
            )
        else:
            return False, f"unknown library entry point: {spec}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    return True, ""


def run_notebook(path: Path, timeout: int | None, kernel: str) -> tuple[bool, str]:
    # NB06/07/08 are migrated stages: bypass nbconvert entirely and call the
    # library function. Saves Jupyter-kernel startup + cell-by-cell execution
    # overhead on what is now the production retrain path.
    if path.name[:3] in _LIBRARY_STAGES:
        return _run_library_stage(path)

    nb = nbformat.read(path, as_version=4)
    _, nb = nbformat.validator.normalize(nb)
    ep = ExecutePreprocessor(timeout=timeout, kernel_name=kernel)
    try:
        ep.preprocess(nb, {"metadata": {"path": str(path.parent)}})
    except CellExecutionError as e:
        nbformat.write(nb, path)
        return False, str(e).splitlines()[-1] if str(e) else "CellExecutionError"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    nbformat.write(nb, path)
    return True, ""


def select_notebooks(args: argparse.Namespace) -> list[Path]:
    all_nbs = discover_notebooks()
    if not all_nbs:
        sys.exit(f"No pipeline found in {NOTEBOOKS_DIR}")

    # A positional list and --from/--to are different selection modes. Passing
    # both is almost always a misread of the CLI. `--from 01 05` looks
    # like "01 through 05" but argparse binds 05 to the positional list, which
    # then silently overrides --from. Fail loudly and show the two right ways.
    if args.notebooks and (args.from_nb or args.to_nb):
        sys.exit(
            "Pass EITHER an explicit notebook list OR --from/--to, not both.\n"
            "  range:    run_pipeline.py --from 01_overview.ipynb --to 05_split_encode_scale_select.ipynb\n"
            "  explicit: run_pipeline.py 01_overview.ipynb 05_split_encode_scale_select.ipynb"
        )

    if args.notebooks:
        by_name = {p.name: p for p in all_nbs}
        chosen = []
        for name in args.notebooks:
            if name not in by_name:
                sys.exit(f"Unknown notebook: {name}\nAvailable: {', '.join(by_name)}")
            chosen.append(by_name[name])
        return chosen

    if args.from_nb or args.to_nb:
        names = [p.name for p in all_nbs]
        if args.from_nb and args.from_nb not in names:
            sys.exit(f"--from notebook not found: {args.from_nb}")
        if args.to_nb and args.to_nb not in names:
            sys.exit(f"--to notebook not found: {args.to_nb}")
        # --from defaults to the first notebook, --to to the last. The slice is
        # inclusive of both endpoints, so we add 1 to the --to index.
        start = names.index(args.from_nb) if args.from_nb else 0
        end = names.index(args.to_nb) + 1 if args.to_nb else len(all_nbs)
        if end <= start:
            sys.exit(
                f"--to ({args.to_nb}) precedes --from ({args.from_nb}) "
                "in pipeline order, so nothing to run."
            )
        return all_nbs[start:end]

    return all_nbs


def parse_args() -> argparse.Namespace:
    """Build the CLI argument parser used by :func:`main`.

    The parser is split out so the unit tests can call it directly with a
    crafted `argv` list without running the executor. The module's
    docstring is used as the parser description so `--help` prints the
    same "Examples" block the file header carries.

    Flag semantics
    --------------
    * `notebooks` (positional, optional). Run exactly this list, in
      the given order. Mutually exclusive in effect with `--from` (if
      both are passed, positional names win).
    * `--from <name>`. Discover every notebook, then start the run at
      the named one and proceed in filename order. Runs to the end unless
      `--to` caps it.
    * `--to <name>`. Stop the run at the named notebook (inclusive).
      Pairs with `--from` to run a contiguous range (`--from X --to Y`).
      Used alone it runs from the first notebook through `Y`. Combining
      either flag with a positional list is rejected.
    * `--timeout <seconds>`. Per-cell timeout passed to
      `nbconvert.ExecutePreprocessor`. Defaults to `None` (no cap).
      CI passes 10800 (3 h) for the full HPO pass.
    * `--kernel <name>`. The Jupyter kernel, defaulting to `python3`
      so the active environment's kernel is picked up.
    * `--keep-going`. Continue to the next notebook after a failure
      instead of stopping at the first one.
    * `--list`. Print the discovered notebook order and exit.
    * `--reset`. Wipe `data/` and clear notebook outputs *before*
      running. Combine with `--list` to wipe-and-list without
      re-running.
    """
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("notebooks", nargs="*", help="Specific notebook filenames to run (default: all)")
    p.add_argument("--from", dest="from_nb", metavar="NB", help="Start from this notebook (inclusive)")
    p.add_argument("--to", dest="to_nb", metavar="NB", help="Stop at this notebook (inclusive); pairs with --from for a range")
    p.add_argument("--timeout", type=int, default=None, help="Per-cell timeout in seconds (default: none)")
    p.add_argument("--kernel", default="python3", help="Jupyter kernel name (default: python3)")
    p.add_argument("--keep-going", action="store_true", help="Continue after a notebook fails")
    p.add_argument("--list", action="store_true", help="List discovered pipeline and exit")
    p.add_argument("--reset", action="store_true",
                   help="Before running, wipe data/ and clear outputs from selected pipeline")
    return p.parse_args()


def main() -> int:
    """Drive the discover → reset → execute → summarise loop for the pipeline.

    Returns
    -------
    int
        `0` on full success (every notebook `OK`), `1` when at least
        one notebook failed (under `--keep-going` this is reached after
        all notebooks attempted, otherwise after the first failure).

    Behaviour
    ---------
    1. Parse args via :func:`parse_args`.
    2. If `--list` and not `--reset`, print discovered notebooks and
       return 0.
    3. Select notebooks (`select_notebooks`): positional list, then
       `--from`, otherwise everything in filename order.
    4. If `--reset`, wipe `data/` and clear the selected notebooks'
       outputs. With `--list` after that, print and exit.
    5. Execute each notebook in order, printing a progress bar +
       per-notebook duration. On failure record the last error line and
       stop unless `--keep-going` is set.
    6. Print a summary (passed / failed / skipped + total wall-time).
    """
    args = parse_args()

    if args.list and not args.reset:
        for p in discover_notebooks():
            print(p.name)
        return 0

    notebooks = select_notebooks(args)

    if args.reset:
        removed = wipe_data_dir(DATA_DIR)
        suffix = "y" if removed == 1 else "ies"
        print(f"[reset] removed {removed} entr{suffix} from {DATA_DIR.relative_to(ROOT)}/")
        for nb in notebooks:
            clear_notebook_outputs(nb)
        print(f"[reset] cleared outputs from {len(notebooks)} notebook(s)")
        if args.list:
            for p in notebooks:
                print(p.name)
            return 0

    total = len(notebooks)
    name_w = max(len(p.name) for p in notebooks)

    print(f"Running {total} notebook(s) from {NOTEBOOKS_DIR.relative_to(ROOT)} (kernel={args.kernel})")
    print("=" * (name_w + 50))

    results: list[tuple[str, bool, float, str]] = []
    overall_start = time.time()

    for i, path in enumerate(notebooks, 1):
        prefix = f"{progress_bar(i - 1, total)}  ({i}/{total}) {path.name:<{name_w}}"
        print(f"{prefix}  ... ", end="", flush=True)

        t0 = time.time()
        ok, err = run_notebook(path, timeout=args.timeout, kernel=args.kernel)
        dur = time.time() - t0
        status = "OK  " if ok else "FAIL"
        print(f"{status}  {fmt_duration(dur):>10}")
        if not ok:
            print(f"          -> {err}")
        results.append((path.name, ok, dur, err))

        if not ok and not args.keep_going:
            break

    overall = time.time() - overall_start
    passed = sum(1 for _, ok, *_ in results if ok)
    failed = len(results) - passed
    skipped = total - len(results)

    print("=" * (name_w + 50))
    print(f"{progress_bar(len(results), total)}  total {fmt_duration(overall)}")
    print(f"  passed:  {passed}")
    print(f"  failed:  {failed}")
    if skipped:
        print(f"  skipped: {skipped}  (stopped on first failure; use --keep-going to continue)")

    if failed:
        print("\nFailed pipeline:")
        for name, ok, dur, err in results:
            if not ok:
                print(f"  - {name}  ({fmt_duration(dur)})  {err}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
