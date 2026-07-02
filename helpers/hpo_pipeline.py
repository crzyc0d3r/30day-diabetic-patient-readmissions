"""Hyperparameter-tuning orchestration for the retrain pipeline.

`run_hpo` is the single entry point. It loads the train / val arrays
from `data/train_test.npz`, initialises MLflow and Ray, dispatches a
randomised search over the five model families (XGBoost, CatBoost,
Logistic Regression, Random Forest, MLP) through one of three execution
paths, persists `data/tuned_results.joblib` and
`data/tuned_models.joblib` for the downstream training and conclusion
stages, and emits an ASHA-pruning diagnostic to
`data/hpo_diagnostics.parquet` after every Tuner run.

The three execution paths share a single per-fold scorer (`helpers/hpo.py
::_cv_fit_and_score`) so the metric definition cannot drift between them:

1. Ray Tune with `ASHAScheduler` (default when `ray` is available and
   `MEDIWATCH_USE_TUNER=1`). Per-fold reporting through `tune.report`
   lets the scheduler halve trials between rungs.
2. Bare `@ray.remote` actors (default when `MEDIWATCH_USE_TUNER=0`).
   Every trial runs to completion across all three folds with no halving.
3. Deterministic single-process `StratifiedGroupKFold` (used when `ray`
   is not installed). Three trials per model on a two-fold split, sized to
   finish under two minutes on CPU.

Both Ray-backed paths cross-validate with `StratifiedGroupKFold(n_splits=3,
groups=patient_nbr)` so multi-encounter patients land in exactly one fold,
preserving the patient-level partition contract NB05 establishes.
"""
from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd

from helpers.constants import (
    ASHA_GRACE_PERIOD,
    ASHA_MAX_T,
    ASHA_REDUCTION_FACTOR,
    NUM_SAMPLES,
    N_SPLITS_HPO,
    SEED,
)
from helpers.hpo import (
    _cv_fit_and_score,
    asha_pruning_report,
    deterministic_grid_fallback,
    make_tune_trial,
    refit_and_log_tuner_result,
    sequential_cv_fallback,
    to_tune_space,
)
from helpers.mlops_helpers import cuda_device_name, has_cuda, init_mlflow, init_ray

# Per-model discrete search grids. Sampled by every execution path (Tuner,
# bare-@ray.remote, deterministic fallback) and forwarded into the training
# stage via the best_params field in tuned_results.
SEARCH_SPACES: dict[str, dict[str, list[Any]]] = {
    "XGBoost": {
        "n_estimators": [100, 200, 300, 500, 800],
        "max_depth": [3, 4, 5, 6, 8, 10, 12],
        "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1, 0.15],
        "min_child_weight": [1, 2, 3, 5, 7],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "gamma": [0, 0.05, 0.1, 0.2, 0.5],
        "reg_alpha": [0, 0.1, 0.5, 1.0],
        "reg_lambda": [1, 2, 5],
    },
    "CatBoost": {
        "iterations": [100, 200, 300, 500, 800],
        "depth": [4, 5, 6, 7, 8, 9, 10],
        "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1, 0.15],
        "l2_leaf_reg": [1, 3, 5, 7, 9, 12],
        "bagging_temperature": [0, 0.3, 0.5, 0.8, 1.0, 1.5],
        "random_strength": [0, 0.5, 1, 2],
    },
    "MLP": {
        "lr": [5e-5, 1e-4, 3e-4, 5e-4, 1e-3, 3e-3],
        "weight_decay": [1e-6, 1e-5, 3e-5, 1e-4, 3e-4],
        "dropout": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "batch_size": [256, 512, 1024],
        "epochs": [15, 25, 35],
    },
    "Logistic Regression": {
        "C": [0.0005, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30],
        "penalty": ["l2"],
    },
    "Random Forest": {
        "n_estimators": [100, 200, 300, 500, 800],
        "max_depth": [6, 8, 10, 12, 15, 18, 22, None],
        "min_samples_split": [2, 4, 6, 8, 12],
        "min_samples_leaf": [1, 2, 3, 4, 6, 8],
        "max_features": ["sqrt", 0.5, 0.6, 0.7, 0.8],
        "class_weight": [None, "balanced", "balanced_subsample"],
    },
}


def _resolve_use_tuner(arg: bool | None) -> bool:
    """Resolve the Tuner/bare flag from explicit arg, env var, or default."""
    if arg is not None:
        return arg
    return os.environ.get("MEDIWATCH_USE_TUNER", "1") == "1"


def _resolve_ray_address(arg: str | None) -> str:
    if arg is not None:
        return arg
    return os.environ.get("RAY_ADDRESS", "ray://localhost:20001")


def _resolve_max_concurrent_trials() -> int:
    """Cap simultaneous Tune trials to the real physical GPU count (default 1).

    Each trial requests a full GPU (`tune.with_resources(..., {"gpu": 1})`).
    The compose cluster runs ray-head + 2 ray-workers, and on a single-GPU
    host every node still advertises 1 GPU, so Ray's scheduler believes it
    has 3 GPUs and lands 3 trials on the *one* physical GB10 at once. Those
    actors then contend for the GB10's unified 121 GB pool and die as
    "CatBoost GPU OOM / CUDA stream death" -- the multi-actor Ray-with-GPU
    failure mode documented in helpers/hpo.py. Every trial erroring forces
    the in-process `sequential_cv_fallback` onto the driver (50 trials x 3
    folds x 5 models), which, with Ray's memory monitor disabled, exhausts
    the pool until the kernel OOM-killer SIGKILLs the driver.

    Serialising trials to the true device count keeps each trial alone on the
    GB10 so it reports its metric instead of erroring, so the fallback never
    fires. Override with MEDIWATCH_MAX_CONCURRENT_TRIALS on a host that
    genuinely has more than one GPU.
    """
    env = os.environ.get("MEDIWATCH_MAX_CONCURRENT_TRIALS")
    if env:
        return max(1, int(env))
    return 1


def _bare_ray_hpo(
    top_models: list[str],
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_patient_ids: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    pos_weight: float,
    _HAS_CUDA: bool,
    ray_address: str,
    num_samples: int,
) -> dict[str, dict[str, Any]]:
    """Bare `@ray.remote` HPO path. Activated when `MEDIWATCH_USE_TUNER=0`.

    Every trial runs to completion across all three CV folds. No ASHA
    halving. Identical per-trial scorer to the Tuner path because both
    call `_cv_fit_and_score`.
    """
    import ray  # type: ignore[import-not-found]
    from sklearn.model_selection import StratifiedGroupKFold

    init_ray(default_address=ray_address)
    ctx = ray.cluster_resources()
    mode = "cluster" if ray_address.startswith("ray://") and ctx.get("CPU", 0) > 1 else "in-process-or-fallback"
    print(f"[run_hpo bare] RAY_ADDRESS={ray_address}, mode={mode}")

    train_ref = ray.put((X_train, y_train, train_patient_ids))

    @ray.remote(num_gpus=1)
    def _fit_and_score(name: str, config: dict, train_ref_inner, pos_weight_inner: float, has_cuda: bool):
        """One trial: 3-fold StratifiedGroupKFold CV, returns mean F1 (test + train)."""
        # _cv_fit_and_score is the single source of truth for "fit one config
        # across every fold", the same scorer the Tuner path uses.
        X_tr, y_tr, pids = train_ref_inner
        cv = StratifiedGroupKFold(n_splits=N_SPLITS_HPO, shuffle=True, random_state=SEED)
        mean_test, mean_train = _cv_fit_and_score(
            name, config, cv.split(X_tr, y_tr, groups=pids),
            X_tr, y_tr, pos_weight_inner, has_cuda,
        )
        return {
            "name": name,
            "config": config,
            "mean_test_score": mean_test,
            "mean_train_score": mean_train,
        }

    rng = random.Random(SEED)
    results: dict[str, dict[str, Any]] = {}

    for name in top_models:
        if name not in SEARCH_SPACES:
            print(f"Skipping {name}: no search space declared")
            continue

        with mlflow.start_run(
            run_name=f"hpo_{name.lower()}",
            tags={
                "phase": "hpo",
                "model": name,
                "hpo_backend": "ray_remote_bare",
                "ray_address": ray_address,
            },
        ) as parent_run:
            configs = [
                {k: rng.choice(list(v)) for k, v in SEARCH_SPACES[name].items()}
                for _ in range(num_samples)
            ]
            print(f"\n{'=' * 60}\n  HPO {name}: {num_samples} trials, 3-fold CV, "
                  f"parent_run_id={parent_run.info.run_id[:8]}\n{'=' * 60}")
            futures = [
                _fit_and_score.remote(name, cfg, train_ref, pos_weight, _HAS_CUDA)
                for cfg in configs
            ]
            trials = ray.get(futures)
            print(f"  All {len(trials)} trials complete.")

            for i, t in enumerate(trials):
                with mlflow.start_run(
                    run_name=f"hpo_{name.lower()}_trial_{i:02d}",
                    tags={"phase": "hpo_trial", "model": name},
                    nested=True,
                ):
                    mlflow.log_params({k: str(v) for k, v in t["config"].items()})
                    mlflow.log_metrics({
                        "cv_f1_mean": float(t["mean_test_score"]),
                        "cv_train_f1_mean": float(t["mean_train_score"]),
                    })

            best = max(trials, key=lambda t: t["mean_test_score"])
            df_results = pd.DataFrame(trials)
            results[name] = refit_and_log_tuner_result(
                name, best["config"], df_results,
                X_train, y_train, X_val, y_val,
                pos_weight, _HAS_CUDA, num_samples,
            )
            print(f"[{name}] Best params: {best['config']}")
            print(f"[{name}] Best CV F1:  {results[name]['best_cv_f1']:.4f}")
            print(f"[{name}] Val F1:      {results[name]['val_f1']:.4f}")

    ray.shutdown()
    return results


def _tuner_asha_hpo(
    top_models: list[str],
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_patient_ids: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    pos_weight: float,
    _HAS_CUDA: bool,
    ray_address: str,
    num_samples: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, pd.DataFrame]]:
    """Ray Tune + `ASHAScheduler` HPO path.

    Returns `(results, per_model_trial_frames)`. The trial frames feed the
    ASHA-pruning diagnostic, one row per trial with the `folds_completed`
    column that records how far each trial got before ASHA halved it. When a
    model's Tuner trials silently fail to report `mean_test_score` (the
    multi-actor Ray-with-GPU failure mode observed on some clusters), this
    falls back to a synchronous in-process three-fold sweep via
    `sequential_cv_fallback` so downstream callers still receive a populated
    record.
    """
    import ray  # type: ignore[import-not-found]
    from ray import tune  # type: ignore[import-not-found]
    from ray.tune.schedulers import ASHAScheduler  # type: ignore[import-not-found]

    init_ray(default_address=ray_address)
    ctx = ray.cluster_resources()
    mode = "cluster" if ray_address.startswith("ray://") and ctx.get("CPU", 0) > 1 else "in-process-or-fallback"
    print(f"[run_hpo tuner] RAY_ADDRESS={ray_address}, mode={mode}")

    train_ref = ray.put((X_train, y_train, train_patient_ids))
    _tune_trial = make_tune_trial(train_ref, pos_weight, _HAS_CUDA)

    results: dict[str, dict[str, Any]] = {}
    trial_frames: dict[str, pd.DataFrame] = {}

    for name in top_models:
        if name not in SEARCH_SPACES:
            continue
        with mlflow.start_run(
            run_name=f"hpo_tuner_{name.lower()}",
            tags={
                "phase": "hpo",
                "model": name,
                "hpo_backend": "ray_tune",
                "ray_address": ray_address,
                "ray_mode": mode,
            },
        ):
            tuner = tune.Tuner(
                tune.with_resources(_tune_trial, resources={"gpu": 1}),
                param_space=to_tune_space(name, SEARCH_SPACES[name]),
                tune_config=tune.TuneConfig(
                    num_samples=num_samples,
                    # Serialise trials to the physical GPU count so multiple
                    # actors never contend for the single GB10 (the multi-actor
                    # GPU OOM -> in-process fallback -> driver SIGKILL cascade).
                    max_concurrent_trials=_resolve_max_concurrent_trials(),
                    # ASHA knobs all live in helpers/constants.py so a future
                    # change to the CV fold count automatically propagates
                    # into max_t. The asha_pruning_report at the end of
                    # run_hpo quantifies per-model survival to the final
                    # rung so cross-model fairness of the halving is
                    # auditable on every run.
                    scheduler=ASHAScheduler(
                        max_t=ASHA_MAX_T,
                        grace_period=ASHA_GRACE_PERIOD,
                        reduction_factor=ASHA_REDUCTION_FACTOR,
                    ),
                    metric="mean_test_score",
                    mode="max",
                ),
            )
            tune_results = tuner.fit()
            df_results_raw = tune_results.get_dataframe()

            metric_missing = (
                "mean_test_score" not in df_results_raw.columns
                or df_results_raw["mean_test_score"].isna().all()
            )
            if metric_missing:
                first_err = next((r.error for r in tune_results if r.error is not None), None)
                print(f"[Tuner/{name}] trials reported no `mean_test_score` "
                      f"(num_errors={tune_results.num_errors}). first error: {first_err!r}")
                print(f"[Tuner/{name}] falling back to sequential CV sweep "
                      "(in-process, no Ray actor isolation).")
                df_results, best_cfg = sequential_cv_fallback(
                    name, num_samples, SEARCH_SPACES,
                    X_train, y_train, train_patient_ids,
                    pos_weight, _HAS_CUDA,
                )
                df_results = df_results.sort_values("mean_test_score", ascending=False)
            else:
                df_results = df_results_raw.sort_values("mean_test_score", ascending=False)
                best_cfg = {
                    k: v for k, v in
                    tune_results.get_best_result(metric="mean_test_score", mode="max").config.items()
                    if k != "__model__"
                }

            trial_frames[name] = df_results_raw
            results[name] = refit_and_log_tuner_result(
                name, best_cfg, df_results,
                X_train, y_train, X_val, y_val,
                pos_weight, _HAS_CUDA, num_samples,
            )

    return results, trial_frames


def run_hpo(
    train_test_path: str | Path = "data/train_test.npz",
    out_dir: str | Path = "data",
    *,
    use_tuner: bool | None = None,
    num_samples: int = NUM_SAMPLES,
    ray_address: str | None = None,
    mlflow_experiment: str = "medi-watch-readmission",
    diagnostics_path: str | Path | None = None,
    top_models: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Run hyperparameter tuning across all five model families.

    Loads the preprocessed train and validation arrays from
    `data/train_test.npz`, initialises MLflow and Ray, dispatches the HPO
    sweep through one of three execution paths (Ray Tune with ASHA halving,
    bare `@ray.remote` actors, or a deterministic single-process fallback),
    persists `tuned_results.joblib` and `tuned_models.joblib` for the
    downstream training and conclusion stages, and writes
    `data/hpo_diagnostics.parquet` with the per-model trial-survival audit.

    Parameters
    ----------
    train_test_path
        Path to the preprocessed train / val / test arrays, default
        `"data/train_test.npz"` (relative to the current working directory).
        Airflow workers pass an absolute `/workspace/data` path. Notebooks
        run from `pipeline/` pass `../data/...`.
    out_dir
        Where `tuned_results.joblib` and `tuned_models.joblib` get written.
    use_tuner
        `True` to run the Tuner+ASHA path, `False` for bare-@ray.remote,
        `None` (default) to read `MEDIWATCH_USE_TUNER` env var (default 1).
    num_samples
        Trials per model. Default `helpers.constants.NUM_SAMPLES` (50).
    ray_address
        Override the Ray client address. `None` reads `RAY_ADDRESS` env or
        `ray://localhost:20001` default.
    mlflow_experiment
        MLflow experiment name. Server reachability is enforced by
        `init_mlflow` (raises `RuntimeError` if unreachable).
    diagnostics_path
        Where to persist the ASHA-pruning DataFrame (parquet). Defaults to
        `<out_dir>/hpo_diagnostics.parquet`. `None` disables persistence
        (the stdout table still prints when the Tuner path runs).
    top_models
        Restrict to a subset of model families. Default = all five.

    Returns
    -------
    tuned_results : dict
        Per-model record:
        `{name: {best_params, best_cv_f1, val_f1, val_auc_roc, val_auc_pr,
        val_recall, val_precision, y_pred, y_prob, model, cv_results}}`.
        Downstream callers (the training and conclusion stages, and the
        inspection cells in NB06) read this directly.
    """
    train_test_path = Path(train_test_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if diagnostics_path is None:
        diagnostics_path = out_dir / "hpo_diagnostics.parquet"

    use_tuner_resolved = _resolve_use_tuner(use_tuner)
    ray_address_resolved = _resolve_ray_address(ray_address)
    models = top_models if top_models is not None else list(SEARCH_SPACES.keys())

    data = np.load(train_test_path)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    train_patient_ids = data["train_patient_ids"]

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    pos_weight = float(n_neg / max(n_pos, 1))
    print(f"[run_hpo] X_train={X_train.shape} X_val={X_val.shape} "
          f"pos_weight={pos_weight:.2f}")

    _HAS_CUDA = has_cuda()
    if _HAS_CUDA:
        print(f"[run_hpo] CUDA detected ({cuda_device_name() or 'unknown'})")

    init_mlflow(experiment=mlflow_experiment)

    try:
        import ray  # noqa: F401  # type: ignore[import-not-found]
        ray_available = True
    except ImportError:
        ray_available = False

    tuned_results: dict[str, dict[str, Any]] = {}
    trial_frames: dict[str, pd.DataFrame] = {}
    path_taken: str

    if not ray_available:
        print("[run_hpo] ray not installed -- using deterministic sklearn fallback "
              "(15 trials/model, 2-fold CV) so downstream stays runnable.")
        t0 = time.perf_counter()
        tuned_results = deterministic_grid_fallback(
            models, SEARCH_SPACES,
            X_train, y_train, train_patient_ids,
            X_val, y_val, pos_weight, _HAS_CUDA,
            n_samples=15, seed=SEED,   # bumped from 3 for stronger search on new 85-feature data
        )
        print(f"[run_hpo] fallback wrote {len(tuned_results)} model(s) "
              f"in {time.perf_counter() - t0:.1f}s.")
        path_taken = "deterministic_fallback"
    elif use_tuner_resolved:
        tuned_results, trial_frames = _tuner_asha_hpo(
            models, X_train, y_train, train_patient_ids,
            X_val, y_val, pos_weight, _HAS_CUDA,
            ray_address_resolved, num_samples,
        )
        path_taken = "tuner_asha"
    else:
        tuned_results = _bare_ray_hpo(
            models, X_train, y_train, train_patient_ids,
            X_val, y_val, pos_weight, _HAS_CUDA,
            ray_address_resolved, num_samples,
        )
        path_taken = "bare_ray_remote"

    # Strip the heavy per-prediction arrays out of tuned_results before joblib
    # so downstream callers can reload the metrics dict without paying for the
    # fitted-model deserialisation cost. The fitted models go into
    # tuned_models.joblib so the two artefacts can be loaded independently.
    tuned_summary = {
        name: {k: v for k, v in record.items()
               if k not in ("y_pred", "y_prob", "model", "cv_results")}
        for name, record in tuned_results.items()
    }
    joblib.dump(tuned_summary, out_dir / "tuned_results.joblib")
    print(f"[run_hpo] saved {out_dir / 'tuned_results.joblib'}")
    tuned_models = {name: record["model"] for name, record in tuned_results.items()}
    joblib.dump(tuned_models, out_dir / "tuned_models.joblib")
    print(f"[run_hpo] saved {out_dir / 'tuned_models.joblib'} "
          f"({len(tuned_models)} models)")

    # ASHA diagnostic: only meaningful for the Tuner+ASHA path. The bare and
    # deterministic paths run every trial to completion so folds_completed is
    # always equal to n_splits and the diagnostic has nothing to surface.
    if path_taken == "tuner_asha" and trial_frames:
        asha_pruning_report(
            trial_frames,
            out_path=str(diagnostics_path) if diagnostics_path else None,
            print_tables=True,
        )
    else:
        print(f"[run_hpo] skipping ASHA diagnostic ({path_taken} path runs every "
              "trial to completion; nothing to surface).")

    return tuned_results


def _main(argv: list[str] | None = None) -> int:
    """Command-line entry point for the retrain DAG's HPO step.

    The Airflow `run_06_hyperparameter_tuning` task invokes this module as
    `python -m helpers.hpo_pipeline` under a dedicated CPython 3.13.2 venv so
    the Ray client interpreter matches the cluster. `run_hpo` communicates
    only through files (`train_test.npz` in, joblib artefacts out), so the
    subprocess needs nothing back beyond a process exit code. The Ray address,
    MLflow URI, and Tuner flag still resolve from the inherited environment
    (`RAY_ADDRESS`, `MLFLOW_TRACKING_URI`, `MEDIWATCH_USE_TUNER`).
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m helpers.hpo_pipeline",
        description="Run the retrain HPO sweep and persist the tuned artefacts.",
    )
    parser.add_argument(
        "--train-test-path",
        default="data/train_test.npz",
        help="Path to the preprocessed train/val/test arrays (.npz).",
    )
    parser.add_argument(
        "--out-dir",
        default="data",
        help="Directory for tuned_results.joblib and tuned_models.joblib.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=NUM_SAMPLES,
        help="Trials per model. Defaults to helpers.constants.NUM_SAMPLES.",
    )
    args = parser.parse_args(argv)

    run_hpo(
        train_test_path=args.train_test_path,
        out_dir=args.out_dir,
        num_samples=args.num_samples,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
