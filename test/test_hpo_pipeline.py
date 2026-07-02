"""Unit tests for `helpers.hpo_pipeline`.

This module is the heaviest of the three orchestrators. Its production
paths spin up Ray (a Tune plus ASHA sweep or bare `@ray.remote` actors) and
log every trial to MLflow. Neither belongs in a unit test. The third
execution path, however, is a deterministic single-process sklearn fallback
(taken whenever `ray` is not importable) which IS safe to drive end-to-end
once MLflow is neutered.

Strategy:

1. Test the PURE resolver helpers `_resolve_use_tuner` and
   `_resolve_ray_address` directly, covering the arg, env, and default
   precedence the source documents.
2. Assert the `SEARCH_SPACES` constant exposes the five documented model
   families, each with a non-empty parameter grid.
3. Drive `run_hpo` down the DETERMINISTIC sklearn fallback path. We force
   that path by monkeypatching the module's `import ray` attempt to fail
   (so `ray_available` becomes False), restrict the sweep to the two cheap
   sklearn families, neuter every MLflow boundary, and assert the returned
   dict is keyed by model name with the documented per-model fields.

We never initialise a real Ray cluster and never contact a real MLflow
server. All artefact IO is confined to `tmp_path`.

Pickle and joblib note: the joblib artefacts loaded below are ones the test
itself wrote into `tmp_path` moments earlier and are trusted by
construction, matching the module's documented pickle-safety convention.
"""

from __future__ import annotations

import builtins

import numpy as np
import pytest

from helpers import hpo_pipeline as hp


# ===========================================================================
# _resolve_use_tuner
# ===========================================================================
def test_resolve_use_tuner_arg_wins_over_env(monkeypatch):
    """An explicit `arg` always overrides the environment variable."""
    monkeypatch.setenv("MEDIWATCH_USE_TUNER", "0")
    # arg=True must win even though the env var says 0.
    assert hp._resolve_use_tuner(True) is True
    assert hp._resolve_use_tuner(False) is False


def test_resolve_use_tuner_reads_env_when_arg_none(monkeypatch):
    """With `arg=None` the env var decides: '1' maps to True, anything else False."""
    monkeypatch.setenv("MEDIWATCH_USE_TUNER", "1")
    assert hp._resolve_use_tuner(None) is True
    monkeypatch.setenv("MEDIWATCH_USE_TUNER", "0")
    assert hp._resolve_use_tuner(None) is False


def test_resolve_use_tuner_default_is_true(monkeypatch):
    """With no arg and no env var the documented default is True (Tuner path)."""
    monkeypatch.delenv("MEDIWATCH_USE_TUNER", raising=False)
    assert hp._resolve_use_tuner(None) is True


# ===========================================================================
# _resolve_ray_address
# ===========================================================================
def test_resolve_ray_address_arg_wins(monkeypatch):
    """An explicit address argument overrides the environment variable."""
    monkeypatch.setenv("RAY_ADDRESS", "ray://from-env:20001")
    assert hp._resolve_ray_address("ray://explicit:9999") == "ray://explicit:9999"


def test_resolve_ray_address_reads_env_when_arg_none(monkeypatch):
    """With `arg=None` the `RAY_ADDRESS` env var is used."""
    monkeypatch.setenv("RAY_ADDRESS", "ray://from-env:20001")
    assert hp._resolve_ray_address(None) == "ray://from-env:20001"


def test_resolve_ray_address_default_when_unset(monkeypatch):
    """With no arg and no env var the documented localhost default is used."""
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    assert hp._resolve_ray_address(None) == "ray://localhost:20001"


# ===========================================================================
# SEARCH_SPACES
# ===========================================================================
def test_search_spaces_has_five_documented_families_with_nonempty_grids():
    """The constant exposes the five model families, each with a real grid."""
    expected = {
        "XGBoost", "CatBoost", "MLP",
        "Logistic Regression", "Random Forest",
    }
    assert set(hp.SEARCH_SPACES.keys()) == expected

    for name, grid in hp.SEARCH_SPACES.items():
        # Each family maps to a non-empty dict of hyperparameter to choices.
        assert isinstance(grid, dict)
        assert len(grid) > 0, name
        for param, choices in grid.items():
            # Every parameter offers at least one candidate value to sample.
            assert len(list(choices)) > 0, (name, param)


# ===========================================================================
# run_hpo  (deterministic sklearn fallback, Ray disabled, MLflow neutered)
# ===========================================================================
def _force_ray_unavailable(monkeypatch):
    """Make `import ray` raise ImportError inside `run_hpo`.

    `run_hpo` wraps `import ray` in a try/except to set `ray_available`.
    By patching the builtin import to reject 'ray' we deterministically steer
    the function into its single-process sklearn fallback path, which needs no
    cluster and finishes fast.
    """
    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "ray" or name.startswith("ray."):
            raise ImportError("ray disabled for this unit test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)


def _neuter_mlflow_for_hpo(monkeypatch):
    """No-op every MLflow boundary the fallback path touches.

    The deterministic fallback lives in `helpers.hpo.deterministic_grid_fallback`
    which opens an MLflow run plus span and calls several logging and metadata
    stamping helpers. We patch all of them, plus `hpo_pipeline.init_mlflow`
    (which raises when the server is unreachable), so the fallback runs purely
    in-process.
    """
    import helpers.hpo as hpo_mod

    class _NullCtx:
        """Context manager mimicking `mlflow.start_run` and `start_span`.

        `deterministic_grid_fallback` reads `run.info.run_id` and
        `run.info.experiment_id` off the start_run handle, so the fake run
        object exposes a minimal `.info` namespace.
        """

        class _Info:
            run_id = "test-run-id"
            experiment_id = "test-exp-id"

        def __init__(self):
            self.info = _NullCtx._Info()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def set_attributes(self, *a, **k):
            return None

    # hpo_pipeline-level boundary.
    monkeypatch.setattr(hp, "init_mlflow", lambda *a, **k: None)

    # Boundaries used inside helpers.hpo.deterministic_grid_fallback.
    monkeypatch.setattr(hpo_mod.mlflow, "start_run", lambda *a, **k: _NullCtx())
    monkeypatch.setattr(hpo_mod.mlflow, "start_span", lambda *a, **k: _NullCtx())
    monkeypatch.setattr(hpo_mod.mlflow, "log_params", lambda *a, **k: None)
    monkeypatch.setattr(hpo_mod.mlflow, "log_metrics", lambda *a, **k: None)
    monkeypatch.setattr(hpo_mod, "stamp_run_metadata", lambda *a, **k: None)
    monkeypatch.setattr(hpo_mod, "stamp_experiment_metadata", lambda *a, **k: None)
    monkeypatch.setattr(hpo_mod, "log_training_dataset", lambda *a, **k: object())
    monkeypatch.setattr(hpo_mod, "log_estimator_to_mlflow", lambda *a, **k: object())


def _write_hpo_npz(path, rng):
    """Write a `train_test.npz` with the keys `run_hpo` reads.

    `run_hpo` reads `X_train`, `y_train`, `X_val`, `y_val` and
    `train_patient_ids`. The patient ids must repeat across rows so the
    StratifiedGroupKFold splitter in the fallback has multi-row groups to
    keep together, and there must be enough groups per class for a 2-fold
    grouped split to be feasible.
    """
    n_train, n_val = 120, 40

    def _make(m):
        y = np.concatenate([np.zeros(m // 2), np.ones(m // 2)]).astype(int)
        X = np.column_stack([
            np.where(y == 1, 2.0, -2.0) + rng.normal(0, 0.6, size=m),
            rng.normal(0, 1, size=m),
            rng.normal(0, 1, size=m),
        ]).astype(np.float64)
        return X, y

    X_train, y_train = _make(n_train)
    X_val, y_val = _make(n_val)

    # Give each class many distinct patient groups so StratifiedGroupKFold can
    # place whole patients into folds without starving a fold of a class.
    # Each patient owns two consecutive rows within a class block.
    half = n_train // 2
    groups_per_class = half // 2
    base = np.repeat(np.arange(groups_per_class), 2)
    train_patient_ids = np.concatenate([base, base + groups_per_class]).astype(int)
    assert len(train_patient_ids) == n_train

    np.savez(
        path,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        train_patient_ids=train_patient_ids,
    )


def test_run_hpo_deterministic_fallback(tmp_path, monkeypatch, rng):
    """Drive `run_hpo` down the sklearn fallback and assert the record shape.

    Ray is forced unavailable so the deterministic single-process path runs.
    We restrict the sweep to the two cheap sklearn families (LR and RF) to keep
    it fast, neuter MLflow, and confirm the returned dict is keyed by model
    name with the documented per-model fields, plus that the two summary
    joblib artefacts were written into `out_dir`.
    """
    import joblib

    _force_ray_unavailable(monkeypatch)
    _neuter_mlflow_for_hpo(monkeypatch)
    # has_cuda is imported into the hpo_pipeline namespace, so force CPU.
    monkeypatch.setattr(hp, "has_cuda", lambda: False)

    npz_path = tmp_path / "train_test.npz"
    _write_hpo_npz(npz_path, rng)

    out_dir = tmp_path / "out"
    cheap_models = ["Logistic Regression", "Random Forest"]
    results = hp.run_hpo(
        train_test_path=str(npz_path),
        out_dir=str(out_dir),
        top_models=cheap_models,
        diagnostics_path=None,
    )

    # Returned dict keyed by model name.
    assert set(results.keys()) == set(cheap_models)

    # Each per-model record carries the documented fields.
    for name, record in results.items():
        for field in ("best_params", "best_cv_f1", "val_f1", "val_auc_roc",
                      "val_auc_pr", "val_recall", "val_precision",
                      "y_pred", "y_prob", "model", "cv_results"):
            assert field in record, (name, field)
        # Sanity on the scalar metrics.
        assert 0.0 <= float(record["val_f1"]) <= 1.0
        assert 0.0 <= float(record["best_cv_f1"]) <= 1.0
        assert isinstance(record["best_params"], dict)

    # Summary artefacts written to out_dir.
    tuned_results_path = out_dir / "tuned_results.joblib"
    tuned_models_path = out_dir / "tuned_models.joblib"
    assert tuned_results_path.exists()
    assert tuned_models_path.exists()

    # The summary joblib strips the heavy per-prediction arrays and the fitted
    # model, keeping only the lightweight metric record.
    summary = joblib.load(tuned_results_path)
    assert set(summary.keys()) == set(cheap_models)
    for record in summary.values():
        assert "y_pred" not in record
        assert "y_prob" not in record
        assert "model" not in record
        assert "cv_results" not in record
        assert "val_f1" in record

    # The models joblib carries the fitted estimators, one per family.
    models = joblib.load(tuned_models_path)
    assert set(models.keys()) == set(cheap_models)
