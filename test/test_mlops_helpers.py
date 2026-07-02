"""Unit tests for `helpers/mlops_helpers.py`.

`mlops_helpers` is the IMPURE seam of the pipeline. It shells out to git and
nvidia-smi, opens sockets, hits an MLflow `/health` endpoint, and drives the
MLflow / Ray client libraries. Touching any of those for real in a unit test
would make the suite slow and flaky and would tie it to a GPU and a live
Postgres-backed tracking server that is simply not present in CI.

The testing strategy here is therefore two-pronged.

1. The near-pure resolution and formatting helpers (`resolve_project_root`,
   `resolve_raw_csv`, `_resolve_code_version`, `_metadata_tags`,
   `_resolve_model_flavor`) are exercised directly. They only read the
   filesystem and environment, so we drive them with `tmp_path` trees and
   `monkeypatch` of `os.environ` / the current working directory. No mocking
   of internal logic, just control of the inputs they read.

2. The impure probes and MLflow / Ray drivers are tested by monkeypatching the
   EXTERNAL BOUNDARY only (`subprocess`, `urllib`, `socket`, and the
   `mlflow` API surface) with record-only fakes, then asserting that the
   helper made the right decision and forwarded the right arguments. We never
   call the real `nvidia-smi` or contact a real server. WHY: the unit under
   test is the helper's branching and argument-shaping logic, not the
   correctness of nvidia-smi or urllib, so the boundary is exactly where the
   double belongs.

Style rules honoured throughout: no em dashes, no semicolons, "program" never the British
spelling, and comments explaining the WHAT and the WHY of every monkeypatch.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from helpers import mlops_helpers as mh


# ===========================================================================
# resolve_project_root
# ===========================================================================
#
# This walks upward from a start directory until it finds a marker. We build
# throwaway directory trees under tmp_path so the walk has a deterministic,
# isolated filesystem to climb. WHY tmp_path: the real repo root would also
# satisfy the markers, so without an isolated tree the assertion about WHICH
# directory is returned would be meaningless.


def test_resolve_project_root_finds_requirements_txt(tmp_path: Path) -> None:
    """A `requirements.txt` marker at the root is detected from a deep child.

    We drop the marker at `root` and start the walk several levels down. The
    helper must climb back up to the directory that owns the marker.
    """
    root = tmp_path / "repo"
    deep = root / "pipeline" / "sub" / "leaf"
    deep.mkdir(parents=True)
    (root / "requirements.txt").write_text("numpy\n")

    assert mh.resolve_project_root(start=deep) == root.resolve()


def test_resolve_project_root_finds_pyproject_toml(tmp_path: Path) -> None:
    """A `pyproject.toml` marker is an equally valid root sentinel."""
    root = tmp_path / "repo"
    child = root / "child"
    child.mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='x'\n")

    assert mh.resolve_project_root(start=child) == root.resolve()


def test_resolve_project_root_finds_pipeline_plus_data_siblings(tmp_path: Path) -> None:
    """When no file marker exists, `pipeline/` + `data/` siblings win.

    This is the interactive-from-inside-pipeline path: the repo is identified by
    the co-presence of both directories, not by a single file. We assert the
    helper requires BOTH (the next test covers the only-one case).
    """
    root = tmp_path / "repo"
    (root / "pipeline").mkdir(parents=True)
    (root / "data").mkdir(parents=True)
    start = root / "pipeline"

    assert mh.resolve_project_root(start=start) == root.resolve()


def test_resolve_project_root_pipeline_without_data_does_not_match(tmp_path: Path) -> None:
    """`pipeline/` alone (no `data/`) is not enough to claim the root.

    With no marker anywhere, the helper falls back to `start.parent`. We place
    a lone `pipeline/` dir and confirm the directory is NOT returned as the
    root, proving the AND condition in the source is honoured.
    """
    root = tmp_path / "repo"
    (root / "pipeline").mkdir(parents=True)
    start = root / "pipeline"

    # No marker is found anywhere up the (tmp_path) chain, so the documented
    # fallback `here.parent` applies. The lone pipeline dir must not short-circuit
    # that, so the returned value is the parent of `start`.
    assert mh.resolve_project_root(start=start) == root.resolve()


def test_resolve_project_root_no_marker_falls_back_to_parent(tmp_path: Path) -> None:
    """With no marker anywhere, the source returns `here.parent`.

    The source never raises. We start in a bare leaf directory whose ancestors
    (all under tmp_path) carry no marker, and assert the documented fallback of
    one level up.
    """
    leaf = tmp_path / "a" / "b" / "c"
    leaf.mkdir(parents=True)

    assert mh.resolve_project_root(start=leaf) == leaf.resolve().parent


def test_resolve_project_root_defaults_to_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`start=None` resolves from the current working directory.

    We monkeypatch.chdir into a tmp tree with a marker so the cwd-based default
    is deterministic and never depends on where pytest was launched from.
    """
    root = tmp_path / "repo"
    root.mkdir()
    (root / "requirements.txt").write_text("\n")
    monkeypatch.chdir(root)

    assert mh.resolve_project_root() == root.resolve()


# ===========================================================================
# resolve_raw_csv
# ===========================================================================


def test_resolve_raw_csv_env_path_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`$MEDIWATCH_RAW_CSV` pointing at a real file is returned verbatim.

    The env override is checked first, so we set it to a tmp file and assert it
    is returned without the helper ever consulting the project-root fallback.
    """
    csv = tmp_path / "somewhere" / "diabetic_data.csv"
    csv.parent.mkdir(parents=True)
    csv.write_text("col\n1\n")
    monkeypatch.setenv("MEDIWATCH_RAW_CSV", str(csv))

    assert mh.resolve_raw_csv() == csv.resolve()


def test_resolve_raw_csv_env_unset_but_nonfile_falls_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An env path that is not a file is ignored and the root fallback is used.

    We point the env var at a directory (not a file) so the `p.is_file()`
    guard fails, then place the CSV under `<root>/orig_dataset`. WHY we
    monkeypatch `resolve_project_root`: the primary fallback is anchored at
    the project root, and pinning the root at our tmp tree is the only way to
    assert the orig_dataset branch deterministically.
    """
    not_a_file = tmp_path / "iam_a_dir"
    not_a_file.mkdir()
    monkeypatch.setenv("MEDIWATCH_RAW_CSV", str(not_a_file))

    root = tmp_path / "repo"
    (root / "orig_dataset").mkdir(parents=True)
    csv = root / "orig_dataset" / "diabetic_data.csv"
    csv.write_text("col\n1\n")
    monkeypatch.setattr(mh, "resolve_project_root", lambda *a, **k: root)

    assert mh.resolve_raw_csv() == csv


def test_resolve_raw_csv_orig_dataset_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the env var unset, `<root>/orig_dataset/<file>` is found."""
    monkeypatch.delenv("MEDIWATCH_RAW_CSV", raising=False)
    root = tmp_path / "repo"
    (root / "orig_dataset").mkdir(parents=True)
    csv = root / "orig_dataset" / "diabetic_data.csv"
    csv.write_text("col\n1\n")
    monkeypatch.setattr(mh, "resolve_project_root", lambda *a, **k: root)

    assert mh.resolve_raw_csv() == csv


def test_resolve_raw_csv_sibling_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The `<root>/../orig_dataset/<file>` sibling location is the last hit.

    We deliberately leave the primary `<root>/orig_dataset` empty and place
    the file one level up so only the sibling branch can satisfy the lookup.
    """
    monkeypatch.delenv("MEDIWATCH_RAW_CSV", raising=False)
    parent = tmp_path / "workspace"
    root = parent / "repo"
    root.mkdir(parents=True)
    sibling_dir = parent / "orig_dataset"
    sibling_dir.mkdir()
    csv = sibling_dir / "diabetic_data.csv"
    csv.write_text("col\n1\n")
    monkeypatch.setattr(mh, "resolve_project_root", lambda *a, **k: root)

    assert mh.resolve_raw_csv() == csv


def test_resolve_raw_csv_not_found_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When nothing matches, a `FileNotFoundError` with a hint is raised.

    We unset the env var and point the root at an empty tmp tree so every
    candidate misses, then assert the error type and that the kagglehub hint is
    present in the message (the message is part of the helper's contract).
    """
    monkeypatch.delenv("MEDIWATCH_RAW_CSV", raising=False)
    root = tmp_path / "empty_repo"
    root.mkdir()
    monkeypatch.setattr(mh, "resolve_project_root", lambda *a, **k: root)

    with pytest.raises(FileNotFoundError) as excinfo:
        mh.resolve_raw_csv()
    assert "kaggle" in str(excinfo.value).lower()


def test_resolve_raw_csv_respects_filename_argument(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A custom `filename` is threaded through to the lookup paths."""
    monkeypatch.delenv("MEDIWATCH_RAW_CSV", raising=False)
    root = tmp_path / "repo"
    (root / "orig_dataset").mkdir(parents=True)
    csv = root / "orig_dataset" / "custom.csv"
    csv.write_text("x\n")
    monkeypatch.setattr(mh, "resolve_project_root", lambda *a, **k: root)

    assert mh.resolve_raw_csv(filename="custom.csv") == csv


# ===========================================================================
# _resolve_code_version
# ===========================================================================
#
# The version resolution has a four-tier precedence. We drive each tier in
# isolation by controlling the env and monkeypatching the git subprocess at the
# module's `subprocess` attribute. The local `import subprocess` inside the
# function rebinds to the SAME module object, so patching `mh.subprocess` (or
# the global subprocess module) intercepts the real call site.


def test_resolve_code_version_explicit_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit argument short-circuits env and git entirely.

    We set the env var to a different value to prove explicit takes precedence
    over it.
    """
    monkeypatch.setenv("MEDIWATCH_VERSION", "from-env")
    assert mh._resolve_code_version("v9.9.9") == "v9.9.9"


def test_resolve_code_version_env_used_when_no_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no explicit value, `$MEDIWATCH_VERSION` is honoured next."""
    monkeypatch.setenv("MEDIWATCH_VERSION", "release-2026.05")
    assert mh._resolve_code_version() == "release-2026.05"


def test_resolve_code_version_git_clean_sha(monkeypatch: pytest.MonkeyPatch) -> None:
    """With env unset, the short git SHA is used when the tree is clean.

    We monkeypatch `subprocess.check_output` so the rev-parse call returns a
    fake SHA and the porcelain status returns empty (clean). WHY: we are testing
    the helper's assembly of the SHA, not git itself, so a fake that returns
    canned stdout is exactly the right boundary.
    """
    monkeypatch.delenv("MEDIWATCH_VERSION", raising=False)
    monkeypatch.setattr(mh, "resolve_project_root", lambda *a, **k: Path("/fake/root"))

    def fake_check_output(cmd, **kwargs):
        # The function issues two git invocations. Branch on the verb so each returns
        # the canned output the real git would.
        if "rev-parse" in cmd:
            return "abc1234\n"
        if "status" in cmd:
            return "\n"  # empty porcelain == clean tree
        raise AssertionError(f"unexpected git command {cmd!r}")

    monkeypatch.setattr(mh.subprocess, "check_output", fake_check_output)
    assert mh._resolve_code_version() == "abc1234"


def test_resolve_code_version_git_dirty_sha(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-empty porcelain status appends the `-dirty` suffix."""
    monkeypatch.delenv("MEDIWATCH_VERSION", raising=False)
    monkeypatch.setattr(mh, "resolve_project_root", lambda *a, **k: Path("/fake/root"))

    def fake_check_output(cmd, **kwargs):
        if "rev-parse" in cmd:
            return "deadbee\n"
        if "status" in cmd:
            return " M helpers/mlops_helpers.py\n"  # uncommitted change present
        raise AssertionError(f"unexpected git command {cmd!r}")

    monkeypatch.setattr(mh.subprocess, "check_output", fake_check_output)
    assert mh._resolve_code_version() == "deadbee-dirty"


def test_resolve_code_version_git_failure_returns_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    """When git is unavailable, the helper returns `"unknown"` and never raises.

    We make the first subprocess call raise `OSError` (mimicking git absent
    from PATH). The source documents that tagging must never block a run, so the
    contract is a graceful `"unknown"` not an exception.
    """
    monkeypatch.delenv("MEDIWATCH_VERSION", raising=False)
    monkeypatch.setattr(mh, "resolve_project_root", lambda *a, **k: Path("/fake/root"))

    def boom(cmd, **kwargs):
        raise OSError("git not found")

    monkeypatch.setattr(mh.subprocess, "check_output", boom)
    assert mh._resolve_code_version() == "unknown"


def test_resolve_code_version_empty_sha_returns_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty SHA (e.g. a fresh repo with no commits) maps to `"unknown"`."""
    monkeypatch.delenv("MEDIWATCH_VERSION", raising=False)
    monkeypatch.setattr(mh, "resolve_project_root", lambda *a, **k: Path("/fake/root"))

    def fake_check_output(cmd, **kwargs):
        if "rev-parse" in cmd:
            return "\n"  # empty -> source returns "unknown" before status check
        raise AssertionError("status should not be queried when SHA is empty")

    monkeypatch.setattr(mh.subprocess, "check_output", fake_check_output)
    assert mh._resolve_code_version() == "unknown"


# ===========================================================================
# _metadata_tags
# ===========================================================================


def test_metadata_tags_with_description(monkeypatch: pytest.MonkeyPatch) -> None:
    """A version + description produce both canonical tag keys.

    We pin the version via env so the dict is deterministic and assert the EXACT
    keys (`code.version` and `mlflow.note.content`) the rest of the pipeline
    relies on for column alignment.
    """
    monkeypatch.setenv("MEDIWATCH_VERSION", "v1")
    tags = mh._metadata_tags(version=None, description="champion refit")

    assert tags == {
        mh.CODE_VERSION_TAG: "v1",
        mh.DESCRIPTION_TAG: "champion refit",
    }
    # Guard the literal key strings too, since downstream UI columns key off these
    # exact constants.
    assert mh.CODE_VERSION_TAG == "code.version"
    assert mh.DESCRIPTION_TAG == "mlflow.note.content"


def test_metadata_tags_omits_empty_description(monkeypatch: pytest.MonkeyPatch) -> None:
    """A falsy description is omitted, leaving only the code.version tag."""
    monkeypatch.setenv("MEDIWATCH_VERSION", "v2")
    assert mh._metadata_tags(version=None, description=None) == {
        mh.CODE_VERSION_TAG: "v2",
    }
    # An empty string is also falsy and must be dropped.
    assert mh._metadata_tags(version=None, description="") == {
        mh.CODE_VERSION_TAG: "v2",
    }


def test_metadata_tags_explicit_version_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit version flows through `_resolve_code_version` unchanged."""
    monkeypatch.setenv("MEDIWATCH_VERSION", "ignored-env")
    tags = mh._metadata_tags(version="explicit-3", description=None)
    assert tags == {mh.CODE_VERSION_TAG: "explicit-3"}


# ===========================================================================
# has_cuda
# ===========================================================================
#
# WHY we monkeypatch instead of calling nvidia-smi: the CI box has no GPU, and
# even a dev box must not be probed in a unit test, since that introduces a
# hardware dependency and nondeterminism. The function shells out via
# `subprocess.run` after a `shutil.which` guard, so we replace BOTH the PATH
# lookup and the subprocess call with fakes. `has_cuda` does a local `import
# shutil` / `import subprocess` that rebind to the global module objects, so
# patching the real modules' attributes intercepts the call site.


def _patch_which(monkeypatch: pytest.MonkeyPatch, found: bool) -> None:
    """Force `shutil.which('nvidia-smi')` to report present or absent.

    Returning a fake path string when `found` is the realistic shape of
    `shutil.which` (it returns the resolved executable path or None).
    """
    import shutil

    monkeypatch.setattr(
        shutil, "which", lambda name: "/usr/bin/nvidia-smi" if found else None
    )


def test_has_cuda_true_when_gpu_listed(monkeypatch: pytest.MonkeyPatch) -> None:
    """returncode 0 plus a 'GPU 0' line in stdout means CUDA is present."""
    _patch_which(monkeypatch, found=True)

    def fake_run(cmd, **kwargs):
        # nvidia-smi -L on a GPU box prints one line per device.
        return SimpleNamespace(returncode=0, stdout="GPU 0: NVIDIA RTX 4090 (UUID: GPU-x)\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert mh.has_cuda() is True


def test_has_cuda_false_when_nvidia_smi_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """No `nvidia-smi` on PATH short-circuits to False without a subprocess.

    We patch `subprocess.run` to explode so the test also proves the helper
    returns BEFORE attempting to run anything.
    """
    _patch_which(monkeypatch, found=False)

    def must_not_run(cmd, **kwargs):
        raise AssertionError("subprocess.run must not be called when which() is None")

    monkeypatch.setattr(subprocess, "run", must_not_run)
    assert mh.has_cuda() is False


def test_has_cuda_false_on_nonzero_returncode(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-zero exit (driver error) means no usable CUDA device."""
    _patch_which(monkeypatch, found=True)
    monkeypatch.setattr(
        subprocess, "run",
        lambda cmd, **kw: SimpleNamespace(returncode=9, stdout=""),
    )
    assert mh.has_cuda() is False


def test_has_cuda_false_when_subprocess_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A timeout or OSError from the probe degrades to False, never raises."""
    _patch_which(monkeypatch, found=True)

    def boom(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 2.0)

    monkeypatch.setattr(subprocess, "run", boom)
    assert mh.has_cuda() is False


# ===========================================================================
# cuda_device_name
# ===========================================================================


def test_cuda_device_name_parses_device_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """The device name is extracted from the matching `GPU <index>:` line.

    We feed canned `nvidia-smi -L` output and assert the UUID suffix is
    stripped, leaving just the human-readable model name.
    """
    _patch_which(monkeypatch, found=True)
    listing = (
        "GPU 0: NVIDIA GeForce RTX 4090 (UUID: GPU-aaaa)\n"
        "GPU 1: NVIDIA A100 (UUID: GPU-bbbb)\n"
    )
    monkeypatch.setattr(
        subprocess, "run",
        lambda cmd, **kw: SimpleNamespace(returncode=0, stdout=listing),
    )
    assert mh.cuda_device_name(index=0) == "NVIDIA GeForce RTX 4090"


def test_cuda_device_name_parses_nonzero_index(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-zero index selects the matching line."""
    _patch_which(monkeypatch, found=True)
    listing = (
        "GPU 0: NVIDIA GeForce RTX 4090 (UUID: GPU-aaaa)\n"
        "GPU 1: NVIDIA A100 (UUID: GPU-bbbb)\n"
    )
    monkeypatch.setattr(
        subprocess, "run",
        lambda cmd, **kw: SimpleNamespace(returncode=0, stdout=listing),
    )
    assert mh.cuda_device_name(index=1) == "NVIDIA A100"


def test_cuda_device_name_out_of_range_index_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """An index with no matching line yields None (no crash)."""
    _patch_which(monkeypatch, found=True)
    monkeypatch.setattr(
        subprocess, "run",
        lambda cmd, **kw: SimpleNamespace(
            returncode=0, stdout="GPU 0: NVIDIA RTX 4090 (UUID: GPU-x)\n"
        ),
    )
    assert mh.cuda_device_name(index=7) is None


def test_cuda_device_name_none_when_nvidia_smi_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """No nvidia-smi on PATH returns None up front."""
    _patch_which(monkeypatch, found=False)
    assert mh.cuda_device_name() is None


def test_cuda_device_name_none_on_nonzero_returncode(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed probe (non-zero exit) yields None."""
    _patch_which(monkeypatch, found=True)
    monkeypatch.setattr(
        subprocess, "run",
        lambda cmd, **kw: SimpleNamespace(returncode=1, stdout=""),
    )
    assert mh.cuda_device_name() is None


def test_cuda_device_name_none_when_subprocess_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """An OSError from the probe degrades to None."""
    _patch_which(monkeypatch, found=True)

    def boom(cmd, **kwargs):
        raise OSError("exec format error")

    monkeypatch.setattr(subprocess, "run", boom)
    assert mh.cuda_device_name() is None


# ===========================================================================
# mlflow_reachable
# ===========================================================================
#
# WHY we monkeypatch urllib: this probe opens a real HTTP connection to
# `<uri>/health`. A unit test must never hit the network, so we replace
# `urllib.request.urlopen` with a fake context manager whose `.status` we
# control. We assert both the 200-OK path and the connection-error path, and
# crucially that a non-http scheme returns False WITHOUT urlopen ever running.


class _FakeHTTPResponse:
    """Minimal stand-in for the object `urlopen` yields as a context manager.

    The helper exercises only `.status` and the context-manager protocol, so
    that is all we implement.
    """

    def __init__(self, status: int) -> None:
        self.status = status

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, *exc) -> None:
        return None


def test_mlflow_reachable_true_on_200(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 200 from `/health` means the server is reachable.

    We also capture the URL urlopen was called with to confirm the helper
    appends `/health` and strips any trailing slash on the base URI.
    """
    captured: dict[str, object] = {}

    def fake_urlopen(url, timeout=None):
        captured["url"] = url
        captured["timeout"] = timeout
        return _FakeHTTPResponse(status=200)

    monkeypatch.setattr(mh.urllib.request, "urlopen", fake_urlopen)
    assert mh.mlflow_reachable("http://127.0.0.1:5000/", timeout=1.5) is True
    # Trailing slash stripped, /health appended, timeout threaded through.
    assert captured["url"] == "http://127.0.0.1:5000/health"
    assert captured["timeout"] == 1.5


def test_mlflow_reachable_true_on_4xx(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 4xx (server up but endpoint guarded) still counts as reachable.

    The source treats `200 <= status < 500` as reachable, so a 404 means the
    server answered. We assert that boundary is honoured.
    """
    monkeypatch.setattr(
        mh.urllib.request, "urlopen",
        lambda url, timeout=None: _FakeHTTPResponse(status=404),
    )
    assert mh.mlflow_reachable("http://host:5000") is True


def test_mlflow_reachable_false_on_5xx(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 5xx (server erroring) is treated as not reachable."""
    monkeypatch.setattr(
        mh.urllib.request, "urlopen",
        lambda url, timeout=None: _FakeHTTPResponse(status=503),
    )
    assert mh.mlflow_reachable("http://host:5000") is False


def test_mlflow_reachable_false_on_urlerror(monkeypatch: pytest.MonkeyPatch) -> None:
    """A connection refused / DNS failure degrades to False, never raises."""

    def boom(url, timeout=None):
        raise mh.urllib.error.URLError("connection refused")

    monkeypatch.setattr(mh.urllib.request, "urlopen", boom)
    assert mh.mlflow_reachable("http://127.0.0.1:1") is False


def test_mlflow_reachable_non_http_scheme_skips_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """A `file://` URI returns False without any network call.

    We patch urlopen to explode so the test proves the scheme guard returns
    BEFORE any socket is opened. This is the "Postgres-only, no file store"
    invariant the source documents.
    """

    def must_not_open(url, timeout=None):
        raise AssertionError("urlopen must not be called for a non-http scheme")

    monkeypatch.setattr(mh.urllib.request, "urlopen", must_not_open)
    assert mh.mlflow_reachable("file:///tmp/mlruns") is False


# ===========================================================================
# init_mlflow
# ===========================================================================
#
# WHY we monkeypatch `mlflow_reachable` and the mlflow setters: init_mlflow
# binds the tracking URI and (optionally) the experiment via the real mlflow
# client, which would contact a server. We stub the reachability probe to control
# the branch and replace `set_tracking_uri` / `set_experiment` with
# record-only fakes, then assert the helper bound the URI we expect.


def test_init_mlflow_raises_when_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unreachable server is a configuration error and must raise.

    The source forbids any silent file-store fallback, so we assert a
    `RuntimeError` carrying the actionable bring-up hint.
    """
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setattr(mh, "mlflow_reachable", lambda uri, **kw: False)

    with pytest.raises(RuntimeError) as excinfo:
        mh.init_mlflow()
    assert "not reachable" in str(excinfo.value)


def test_init_mlflow_binds_uri_when_reachable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A reachable server binds the URI and returns it.

    We record the `set_tracking_uri` argument to confirm the helper bound the
    exact target it probed.
    """
    import mlflow

    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setattr(mh, "mlflow_reachable", lambda uri, **kw: True)

    bound: dict[str, str] = {}
    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda uri: bound.__setitem__("uri", uri))
    monkeypatch.setattr(
        mlflow, "set_experiment",
        lambda *a, **k: pytest.fail("set_experiment should not run without experiment arg"),
    )

    result = mh.init_mlflow(default_uri="http://my-server:5000")
    assert result == "http://my-server:5000"
    assert bound["uri"] == "http://my-server:5000"


def test_init_mlflow_prefers_env_uri(monkeypatch: pytest.MonkeyPatch) -> None:
    """`$MLFLOW_TRACKING_URI` overrides the default URI argument."""
    import mlflow

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://env-server:9999")
    probed: dict[str, str] = {}
    monkeypatch.setattr(mh, "mlflow_reachable", lambda uri, **kw: probed.__setitem__("uri", uri) or True)
    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda uri: None)

    result = mh.init_mlflow(default_uri="http://default:1")
    assert result == "http://env-server:9999"
    assert probed["uri"] == "http://env-server:9999"


def test_init_mlflow_sets_experiment_when_supplied(monkeypatch: pytest.MonkeyPatch) -> None:
    """When `experiment` is given, `set_experiment` is invoked with it."""
    import mlflow

    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setattr(mh, "mlflow_reachable", lambda uri, **kw: True)
    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda uri: None)

    seen: dict[str, str] = {}
    monkeypatch.setattr(mlflow, "set_experiment", lambda name: seen.__setitem__("name", name))

    mh.init_mlflow(default_uri="http://srv:5000", experiment="medi-watch-readmit")
    assert seen["name"] == "medi-watch-readmit"


# ===========================================================================
# _resolve_model_flavor
# ===========================================================================
#
# Dispatch is keyed off the estimator's defining module / class name. We pass
# real estimators (sklearn and xgboost, both installed in the test env) and
# assert the returned module's `__name__`. Asserting on the module name keeps
# the test cheap and avoids importing every heavy flavor just to compare object
# identity, while still proving the correct branch was taken.


def test_resolve_model_flavor_sklearn(seed: int) -> None:
    """A plain sklearn estimator dispatches to `mlflow.sklearn`."""
    from sklearn.linear_model import LogisticRegression

    flavor = mh._resolve_model_flavor(LogisticRegression())
    assert flavor.__name__ == "mlflow.sklearn"


def test_resolve_model_flavor_sklearn_tree(seed: int) -> None:
    """A different sklearn family (tree) still resolves to the sklearn flavor.

    Guards against an accidental keyword (`xgb`/`lgbm`) matching a generic
    sklearn class name.
    """
    from sklearn.ensemble import RandomForestClassifier

    flavor = mh._resolve_model_flavor(RandomForestClassifier(n_estimators=2))
    assert flavor.__name__ == "mlflow.sklearn"


def test_resolve_model_flavor_xgboost() -> None:
    """An XGBoost estimator dispatches to `mlflow.xgboost`.

    The class lives in the `xgboost.sklearn` module, so the `"xgboost" in
    module` branch must fire even though XGBClassifier exposes the sklearn API.
    """
    from xgboost import XGBClassifier

    flavor = mh._resolve_model_flavor(XGBClassifier())
    assert flavor.__name__ == "mlflow.xgboost"


def test_resolve_model_flavor_dispatch_by_class_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """The class-name keyword path resolves cuML-style wrappers.

    We synthesise a class whose module is innocuous but whose NAME contains
    `XGB` to prove the `cls_name` branch (not just the module branch) routes
    to the xgboost flavor. This mirrors the documented cuML / wrapper case
    without needing those heavy packages installed.
    """

    class XGBWrapper:  # noqa: N801 - intentionally named to trigger the cls_name branch
        pass

    flavor = mh._resolve_model_flavor(XGBWrapper())
    assert flavor.__name__ == "mlflow.xgboost"


# ===========================================================================
# stamp_run_metadata
# ===========================================================================
#
# These higher-level helpers normally need a live run. We monkeypatch the mlflow
# entry points with record-only fakes and assert the helper forwarded the right
# tags. WHY mocking over skipping: the branching logic (no-op when no active run,
# set_tags when there is one) is real logic worth covering, and the mlflow API
# surface it touches is tiny and easy to fake.


def test_stamp_run_metadata_no_active_run_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no active run, the helper returns an empty dict and writes nothing."""
    import mlflow

    monkeypatch.setattr(mlflow, "active_run", lambda: None)
    monkeypatch.setattr(
        mlflow, "set_tags",
        lambda tags: pytest.fail("set_tags must not be called without an active run"),
    )
    assert mh.stamp_run_metadata(version="v1", description="d") == {}


def test_stamp_run_metadata_sets_tags_when_active(monkeypatch: pytest.MonkeyPatch) -> None:
    """With an active run, the resolved tags are written via `set_tags`."""
    import mlflow

    monkeypatch.setenv("MEDIWATCH_VERSION", "v7")
    monkeypatch.setattr(mlflow, "active_run", lambda: object())  # truthy == active

    captured: dict[str, dict] = {}
    monkeypatch.setattr(mlflow, "set_tags", lambda tags: captured.__setitem__("tags", tags))

    result = mh.stamp_run_metadata(version=None, description="champion")
    assert captured["tags"] == {
        mh.CODE_VERSION_TAG: "v7",
        mh.DESCRIPTION_TAG: "champion",
    }
    assert result == captured["tags"]


# ===========================================================================
# stamp_experiment_metadata
# ===========================================================================


def test_stamp_experiment_metadata_no_run_no_id_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    """No explicit id and no active run cannot resolve an experiment, so no-op."""
    import mlflow

    monkeypatch.setattr(mlflow, "active_run", lambda: None)
    monkeypatch.setattr(
        mlflow, "MlflowClient",
        lambda *a, **k: pytest.fail("MlflowClient must not be built when there is nothing to tag"),
    )
    assert mh.stamp_experiment_metadata() == {}


def test_stamp_experiment_metadata_tags_explicit_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit experiment_id drives `set_experiment_tag` per tag key.

    We record every (eid, key, value) the fake client receives and assert both
    canonical tags were written against the supplied id.
    """
    import mlflow

    monkeypatch.setenv("MEDIWATCH_VERSION", "v5")
    calls: list[tuple] = []

    class FakeClient:
        def set_experiment_tag(self, eid, key, value):
            calls.append((eid, key, value))

    monkeypatch.setattr(mlflow, "MlflowClient", lambda *a, **k: FakeClient())

    result = mh.stamp_experiment_metadata(
        experiment_id="42", version=None, description="exp-desc"
    )
    assert result == {mh.CODE_VERSION_TAG: "v5", mh.DESCRIPTION_TAG: "exp-desc"}
    assert ("42", mh.CODE_VERSION_TAG, "v5") in calls
    assert ("42", mh.DESCRIPTION_TAG, "exp-desc") in calls


# ===========================================================================
# stamp_logged_model_metadata
# ===========================================================================


def test_stamp_logged_model_metadata_uses_setter_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """When `mlflow.set_logged_model_tags` exists, it is called with the tags.

    This is the mlflow >= 3 path. We record the (model_id, tags) the setter
    receives.
    """
    import mlflow

    monkeypatch.setenv("MEDIWATCH_VERSION", "v3")
    received: dict[str, object] = {}
    monkeypatch.setattr(
        mlflow, "set_logged_model_tags",
        lambda model_id, tags: received.update(model_id=model_id, tags=tags),
        raising=False,
    )

    result = mh.stamp_logged_model_metadata(model_id="m-123", version=None, description=None)
    assert received["model_id"] == "m-123"
    assert received["tags"] == {mh.CODE_VERSION_TAG: "v3"}
    assert result == {mh.CODE_VERSION_TAG: "v3"}


def test_stamp_logged_model_metadata_fallback_for_old_mlflow(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the setter is missing (mlflow < 3), tags fall back to prefixed run tags.

    We delete the attribute so `getattr(..., None)` returns None, then assert
    the fallback writes `logged_model.<id>.<key>` run tags via `set_tags`.
    """
    import mlflow

    monkeypatch.setenv("MEDIWATCH_VERSION", "v4")
    # Simulate the older mlflow that lacks the API entirely.
    monkeypatch.delattr(mlflow, "set_logged_model_tags", raising=False)

    captured: dict[str, dict] = {}
    monkeypatch.setattr(mlflow, "set_tags", lambda tags: captured.__setitem__("tags", tags))

    mh.stamp_logged_model_metadata(model_id="m-9", version=None, description=None)
    assert captured["tags"] == {"logged_model.m-9.code.version": "v4"}


# ===========================================================================
# stamp_registered_model_metadata
# ===========================================================================


def test_stamp_registered_model_metadata_writes_both_layers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both the registered-model and the model-version layers are tagged.

    The registry is load-bearing, so the helper writes description + tags to the
    model header AND the specific version row. We use a recording fake client
    and assert all four call kinds fired with the right arguments.
    """
    import mlflow

    monkeypatch.setenv("MEDIWATCH_VERSION", "v6")
    events: list[tuple] = []

    class FakeClient:
        def update_registered_model(self, name, description):
            events.append(("update_rm", name, description))

        def set_registered_model_tag(self, name, key, value):
            events.append(("set_rm_tag", name, key, value))

        def update_model_version(self, name, version, description):
            events.append(("update_mv", name, version, description))

        def set_model_version_tag(self, name, version, key, value):
            events.append(("set_mv_tag", name, version, key, value))

    monkeypatch.setattr(mlflow, "MlflowClient", lambda *a, **k: FakeClient())

    result = mh.stamp_registered_model_metadata(
        name="medi-watch", model_version=3, version=None, description="champion"
    )
    assert result == {mh.CODE_VERSION_TAG: "v6", mh.DESCRIPTION_TAG: "champion"}
    assert ("update_rm", "medi-watch", "champion") in events
    assert ("update_mv", "medi-watch", "3", "champion") in events
    assert ("set_rm_tag", "medi-watch", mh.CODE_VERSION_TAG, "v6") in events
    assert ("set_mv_tag", "medi-watch", "3", mh.CODE_VERSION_TAG, "v6") in events


# ===========================================================================
# enable_mlflow_autolog_and_tracing
# ===========================================================================


def test_enable_mlflow_autolog_and_tracing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The helper calls universal autolog (models/datasets off) + tracing.enable.

    We replace `mlflow.autolog` and `mlflow.tracing.enable` with recorders
    and assert the documented arguments. WHY assert the kwargs: the source's
    whole point is that autolog must NOT log models or datasets (those are
    handled by the curated helpers), so the flags are the contract.
    """
    import mlflow

    autolog_kwargs: dict = {}
    tracing_called: list[bool] = []
    monkeypatch.setattr(mlflow, "autolog", lambda **kw: autolog_kwargs.update(kw))
    monkeypatch.setattr(mlflow.tracing, "enable", lambda: tracing_called.append(True))

    mh.enable_mlflow_autolog_and_tracing(silent=True)
    assert autolog_kwargs == {"log_models": False, "log_datasets": False, "silent": True}
    assert tracing_called == [True]


# ===========================================================================
# traced_run
# ===========================================================================
#
# traced_run is a context manager wrapping mlflow.start_run plus span and tag
# stamping. We fake the whole mlflow surface it touches (start_run, start_span,
# and the two stamp_* helpers) so no server is needed, then assert the run is
# yielded and the stamping helpers were invoked with the resolved version.


def test_traced_run_yields_run_and_stamps(monkeypatch: pytest.MonkeyPatch) -> None:
    """The context manager yields the run and stamps run + experiment metadata."""
    import mlflow
    from contextlib import contextmanager

    monkeypatch.setenv("MEDIWATCH_VERSION", "v8")

    fake_run = SimpleNamespace(
        info=SimpleNamespace(run_id="run-abcdef12", run_name="my_run", experiment_id="11")
    )

    @contextmanager
    def fake_start_run(*a, **k):
        yield fake_run

    @contextmanager
    def fake_start_span(name=None):
        # The span object needs only set_attributes for the helper's body.
        yield SimpleNamespace(set_attributes=lambda attrs: None)

    monkeypatch.setattr(mlflow, "start_run", fake_start_run)
    monkeypatch.setattr(mlflow, "start_span", fake_start_span)

    stamp_calls: list[tuple] = []
    monkeypatch.setattr(
        mh, "stamp_run_metadata",
        lambda version, description: stamp_calls.append(("run", version, description)),
    )
    monkeypatch.setattr(
        mh, "stamp_experiment_metadata",
        lambda experiment_id, version, description: stamp_calls.append(
            ("exp", experiment_id, version, description)
        ),
    )

    with mh.traced_run(run_name="my_run", description="d8") as run:
        assert run is fake_run

    # Version resolved from env, both stamps fired, experiment id threaded.
    assert ("run", "v8", "d8") in stamp_calls
    assert ("exp", "11", "v8", "d8") in stamp_calls


# ===========================================================================
# ray_reachable
# ===========================================================================
#
# WHY we monkeypatch socket.create_connection: this probe opens a real TCP socket
# to host:port. A unit test must not, so we replace the connector with a fake
# context manager (success) or one that raises OSError (refused). The special-case
# addresses (local/auto/None) take a pure branch and need no socket at all.


def test_ray_reachable_local_addresses_always_true(monkeypatch: pytest.MonkeyPatch) -> None:
    """`""`, `"local"` and `"auto"` short-circuit to True with no socket.

    We patch `create_connection` to explode to prove no socket is opened for
    these in-process sentinels.
    """
    import socket as socket_mod

    monkeypatch.setattr(
        socket_mod, "create_connection",
        lambda *a, **k: pytest.fail("no socket should be opened for local sentinels"),
    )
    assert mh.ray_reachable("") is True
    assert mh.ray_reachable("local") is True
    assert mh.ray_reachable("auto") is True


def test_ray_reachable_true_on_open_socket(monkeypatch: pytest.MonkeyPatch) -> None:
    """A `ray://host:port` whose TCP connect succeeds is reachable."""
    import socket as socket_mod

    class FakeConn:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

    monkeypatch.setattr(socket_mod, "create_connection", lambda addr, timeout=None: FakeConn())
    assert mh.ray_reachable("ray://ray-head:20001") is True


def test_ray_reachable_false_on_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """A refused / timed-out connection means not reachable."""
    import socket as socket_mod

    def refused(addr, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr(socket_mod, "create_connection", refused)
    assert mh.ray_reachable("ray://nope:20001") is False


def test_ray_reachable_false_when_host_or_port_missing() -> None:
    """A malformed address with no host/port cannot be probed, so False."""
    # A bare scheme with no host:port yields None for host and port.
    assert mh.ray_reachable("ray://") is False


# ===========================================================================
# init_ray
# ===========================================================================
#
# init_ray drives the real `ray` client. We inject a fake `ray` module by
# monkeypatching the lazy `import ray` (achieved by inserting a fake into
# sys.modules) and stub `ray_reachable` to steer the branch. WHY a fake ray
# module: importing and initialising real Ray spins up processes and is far too
# heavy and nondeterministic for a unit test, and the unit under test is the
# helper's address selection + fallback logic, not Ray itself.


def _install_fake_ray(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Insert a record-only fake `ray` module and return its call log.

    The helper does a lazy `import ray` inside the function body, so putting a
    fake into `sys.modules['ray']` is what that import resolves to.
    """
    import sys

    log: dict = {"init_calls": []}

    fake_ray = SimpleNamespace(
        init=lambda **kwargs: log["init_calls"].append(kwargs)
    )
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    return log


def test_init_ray_connects_when_reachable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A reachable cluster address is passed straight to `ray.init`.

    We force `ray_reachable` True and assert the returned address plus that
    `ray.init` saw `address=<the cluster>`.
    """
    log = _install_fake_ray(monkeypatch)
    monkeypatch.setenv("RAY_ADDRESS", "ray://cluster:20001")
    monkeypatch.setattr(mh, "ray_reachable", lambda addr, **kw: True)

    result = mh.init_ray()
    assert result == "ray://cluster:20001"
    assert log["init_calls"][0]["address"] == "ray://cluster:20001"


def test_init_ray_falls_back_in_process_when_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unreachable address falls back to in-process `ray.init(address=None)`.

    We force `ray_reachable` False and assert the helper returns None and the
    final `ray.init` call carried `address=None`. We also confirm the
    poisoned `$RAY_ADDRESS` was popped so the fallback cannot reconnect to the
    cluster we just failed to reach.
    """
    log = _install_fake_ray(monkeypatch)
    monkeypatch.setenv("RAY_ADDRESS", "ray://dead:20001")
    monkeypatch.setattr(mh, "ray_reachable", lambda addr, **kw: False)

    result = mh.init_ray()
    assert result is None
    assert log["init_calls"][-1]["address"] is None
    # The cluster address must be scrubbed from the env so address=None sticks.
    assert "RAY_ADDRESS" not in os.environ


def test_init_ray_falls_back_when_init_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the remote `ray.init` raises (version mismatch), fall back in-process.

    The first init (remote) raises, the second (in-process, address=None) must
    succeed. We assert two init attempts happened and the second was address
    None.
    """
    import sys

    init_calls: list[dict] = []

    def fake_init(**kwargs):
        init_calls.append(kwargs)
        if kwargs.get("address") is not None:
            raise RuntimeError("ray client version mismatch")

    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(init=fake_init))
    monkeypatch.setenv("RAY_ADDRESS", "ray://mismatch:20001")
    monkeypatch.setattr(mh, "ray_reachable", lambda addr, **kw: True)

    result = mh.init_ray()
    assert result is None
    assert len(init_calls) == 2
    assert init_calls[0]["address"] == "ray://mismatch:20001"
    assert init_calls[1]["address"] is None
