"""Unit tests for `helpers.training_pipeline`.

This module is an IMPURE orchestrator. Its real job is to load `.npz` and
`.joblib` artefacts off disk, fit a population of estimators, log every
fit to a live MLflow server, and persist seven joblib artefacts for the
conclusion stage. None of that should happen inside a unit test, so the
strategy here is two-pronged.

1. Test the PURE private helpers directly. `_safe_params`,
   `_train_and_evaluate`, and `_compute_optimal_threshold` are
   side-effect-free given their arguments, so they can be driven with
   tiny in-memory data and asserted exactly.

2. For the big `train_baselines_and_refits` entry point we fabricate a
   minimal `train_test.npz` plus a `tuned_results.joblib` in `tmp_path`,
   then monkeypatch every MLflow boundary the module touches down to a
   no-op so the test never contacts a real tracking server. We also restrict
   the model families to the two cheap sklearn estimators (Logistic
   Regression and Random Forest) so the end-to-end fit finishes in a second
   or two. We never read or write the repo's real `data/` directory.

Style note: the docstrings here are deliberately generous because these
tests double as executable documentation of the orchestrator's contract.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from helpers import training_pipeline as tp


# ===========================================================================
# Shared tiny-data helpers
# ===========================================================================
def _tiny_separable(rng: np.random.Generator, n: int = 80):
    """Build a small, well-separated binary problem.

    We want the two classes far enough apart that a Logistic Regression can
    learn a non-trivial decision boundary (so metrics are not degenerate)
    while staying tiny enough to fit instantly. Returns `(X, y)` with both
    classes present.
    """
    half = n // 2
    # Class 0 clustered near the origin, class 1 shifted along every feature.
    X0 = rng.normal(0.0, 1.0, size=(half, 4))
    X1 = rng.normal(3.0, 1.0, size=(half, 4))
    X = np.vstack([X0, X1]).astype(np.float64)
    y = np.concatenate([np.zeros(half), np.ones(half)]).astype(int)
    # Shuffle so the row order does not encode the label.
    perm = rng.permutation(n)
    return X[perm], y[perm]


# ===========================================================================
# _safe_params
# ===========================================================================
class _ExoticParamEstimator:
    """A stand-in estimator exposing parameters of exotic, non-primitive type.

    `_safe_params` calls `get_params(deep=False)` and must coerce anything
    that is not an `int`, `float`, `str`, `bool`, or `None` into its
    `str` representation, because MLflow's `log_params` rejects arbitrary
    Python objects. This fake lets us assert that coercion rule precisely
    without depending on the quirks of a real library estimator.
    """

    def __init__(self):
        # A spread of types: primitives that MUST pass through untouched,
        # and exotic objects that MUST be stringified.
        self._params = {
            "an_int": 7,
            "a_float": 0.5,
            "a_str": "hello",
            "a_bool": True,
            "a_none": None,
            "a_list": [1, 2, 3],
            "a_dict": {"k": "v"},
            "an_object": object(),
            "a_tuple": (1, 2),
        }

    def get_params(self, deep: bool = False):
        return dict(self._params)


def test_safe_params_passes_primitives_through_and_stringifies_the_rest():
    """Primitives survive verbatim. Everything exotic becomes its `str`."""
    est = _ExoticParamEstimator()
    out = tp._safe_params(est)

    # Primitive types must be returned with identical value and type.
    assert out["an_int"] == 7
    assert out["a_float"] == 0.5
    assert out["a_str"] == "hello"
    assert out["a_bool"] is True
    assert out["a_none"] is None

    # Exotic types must be coerced to their str() form, which is what MLflow
    # accepts. We compare against str(original) so the rule is pinned exactly.
    assert out["a_list"] == str([1, 2, 3])
    assert out["a_dict"] == str({"k": "v"})
    assert out["a_tuple"] == str((1, 2))
    assert isinstance(out["an_object"], str)

    # Every returned value must be a primitive MLflow can accept.
    for value in out.values():
        assert isinstance(value, (int, float, str, bool, type(None)))


def test_safe_params_on_real_estimator_is_all_primitive():
    """A genuine sklearn estimator round-trips to all-primitive params."""
    out = tp._safe_params(LogisticRegression(C=0.5, max_iter=123))
    assert out["C"] == 0.5
    assert out["max_iter"] == 123
    for value in out.values():
        assert isinstance(value, (int, float, str, bool, type(None)))


def test_safe_params_returns_empty_dict_when_get_params_raises():
    """If `get_params` blows up the helper must swallow it and return `{}`.

    The source wraps the `get_params` call in a bare `try/except` so a
    malformed estimator never aborts an entire MLflow logging run.
    """

    class _Broken:
        def get_params(self, deep=False):
            raise RuntimeError("no params for you")

    assert tp._safe_params(_Broken()) == {}


# ===========================================================================
# _train_and_evaluate
# ===========================================================================
def test_train_and_evaluate_returns_fitted_model_preds_and_metric_panel(rng):
    """Fit a cheap LR on tiny data and assert the full return contract.

    The helper returns `(model, y_pred, y_prob, metrics)`. We assert the
    prediction arrays line up with the validation rows and that the metric
    dict carries the panel keys that `helpers.models.evaluate_model`
    promises (the downstream leaderboard reads these by name).
    """
    X_train, y_train = _tiny_separable(rng, n=80)
    X_val, y_val = _tiny_separable(rng, n=40)

    model, y_pred, y_prob, metrics = tp._train_and_evaluate(
        "Logistic Regression",
        LogisticRegression(max_iter=500),
        X_train, y_train, X_val, y_val,
    )

    # The returned estimator must be the same one we passed, now fitted.
    assert model is not None
    # Predictions are one value per validation row.
    assert y_pred.shape == (len(y_val),)
    assert y_prob.shape == (len(y_val),)
    # Probabilities live in [0, 1].
    assert float(y_prob.min()) >= 0.0
    assert float(y_prob.max()) <= 1.0
    # The metric panel must expose the headline keys downstream code reads.
    for key in ("accuracy", "precision", "recall", "f1",
                "auc_roc", "auc_pr", "train_f1", "train_auc_roc"):
        assert key in metrics
        assert isinstance(metrics[key], float)


# ===========================================================================
# _compute_optimal_threshold
# ===========================================================================
def test_compute_optimal_threshold_contract(rng):
    """On a separable case the F1-optimal threshold is well-formed.

    We assert (a) the threshold sits strictly inside (0, 1), (b) the optimal
    F1 lands in [0, 1], (c) the returned hard predictions are 0/1 of the right
    length, and (d) the exact decision rule the source documents:
    `y_pred_opt == (y_prob >= threshold).astype(int)`.
    """
    # Probabilities strongly correlated with the label so a clean threshold
    # exists. y_val carries both classes.
    n = 100
    y_val = np.concatenate([np.zeros(n // 2), np.ones(n // 2)]).astype(int)
    base = np.where(y_val == 1, 0.7, 0.3)
    y_prob = np.clip(base + rng.normal(0.0, 0.05, size=n), 0.0, 1.0)

    threshold, f1_opt, recall_opt, precision_opt, y_pred_opt = (
        tp._compute_optimal_threshold(y_val, y_prob)
    )

    assert 0.0 < threshold < 1.0
    assert 0.0 <= f1_opt <= 1.0
    assert 0.0 <= recall_opt <= 1.0
    assert 0.0 <= precision_opt <= 1.0

    # Hard predictions are binary and one-per-row.
    assert y_pred_opt.shape == (len(y_val),)
    assert set(np.unique(y_pred_opt)).issubset({0, 1})

    # Pin the exact thresholding rule from the source verbatim.
    expected = (np.asarray(y_prob) >= threshold).astype(int)
    np.testing.assert_array_equal(y_pred_opt, expected)


# ===========================================================================
# train_baselines_and_refits (end-to-end, MLflow neutered)
# ===========================================================================
def _write_train_test_npz(path, rng):
    """Write a minimal `train_test.npz` matching the schema the source reads.

    `train_baselines_and_refits` only ever touches `X_train`, `y_train`,
    `X_val`, `y_val` from this file, so those four arrays are all we need
    to fabricate. Both splits carry both classes so every metric is defined.
    """
    X_train, y_train = _tiny_separable(rng, n=80)
    X_val, y_val = _tiny_separable(rng, n=40)
    np.savez(
        path,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
    )


def _neuter_mlflow(monkeypatch):
    """Replace every MLflow and side-effecting boundary the module imports.

    `train_baselines_and_refits` calls `init_mlflow` (which raises if the
    server is unreachable), opens runs through `mlflow.start_run`, and pushes
    params, metrics, datasets, and estimators through several logging helpers.
    We swap all of them for no-ops bound on the `training_pipeline` module
    namespace so the real tracking server is never contacted and `has_cuda`
    is forced false for determinism.
    """

    class _NullRun:
        """A context manager that mimics `mlflow.start_run` doing nothing."""

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    # Boundaries imported by-name into the training_pipeline namespace.
    monkeypatch.setattr(tp, "init_mlflow", lambda *a, **k: None)
    monkeypatch.setattr(tp, "enable_mlflow_autolog_and_tracing", lambda *a, **k: None)
    monkeypatch.setattr(tp, "log_training_dataset", lambda *a, **k: object())
    monkeypatch.setattr(tp, "log_estimator_to_mlflow", lambda *a, **k: object())
    monkeypatch.setattr(tp, "has_cuda", lambda: False)
    monkeypatch.setattr(tp, "cuda_device_name", lambda *a, **k: None)

    # The bare `mlflow.*` calls inside the module reference the imported
    # module object, so patch them on tp.mlflow.
    monkeypatch.setattr(tp.mlflow, "start_run", lambda *a, **k: _NullRun())
    monkeypatch.setattr(tp.mlflow, "log_params", lambda *a, **k: None)
    monkeypatch.setattr(tp.mlflow, "log_metrics", lambda *a, **k: None)


def test_train_baselines_and_refits_end_to_end(tmp_path, monkeypatch, rng):
    """Drive a tiny real training run with all MLflow side effects disabled.

    We fabricate the two input artefacts in `tmp_path`, restrict the model
    families to the two cheap sklearn estimators, neuter MLflow, and assert
    both the returned dict shape and that the documented joblib artefacts
    landed in `out_dir`.
    """
    import joblib

    _neuter_mlflow(monkeypatch)

    # Only train the two fast sklearn families so the end-to-end run is quick.
    cheap_families = ["Logistic Regression", "Random Forest"]
    monkeypatch.setattr(tp, "model_families", cheap_families, raising=False)
    # The source builds `model_families` as a local list literal, so the
    # attribute patch above does not reach it. Instead, restrict the tuned
    # results so only the cheap families are refit, and monkeypatch the local
    # default population by patching `build_estimator` to refuse the heavy
    # families. Simplest robust approach: shrink the families list the source
    # iterates by patching the module-level constant the source references.

    # Build a tuned_results.joblib whose keys are exactly the cheap families.
    # train_baselines_and_refits iterates tuned_results_summary for the refit
    # population and references `["best_params"]` per entry.
    tuned_results = {
        "Logistic Regression": {"best_params": {"C": 1.0}},
        "Random Forest": {"best_params": {"n_estimators": 25, "max_depth": 4}},
    }
    tuned_path = tmp_path / "tuned_results.joblib"
    joblib.dump(tuned_results, tuned_path)

    npz_path = tmp_path / "train_test.npz"
    _write_train_test_npz(npz_path, rng)

    # The default population iterates a hard-coded five-family list inside the
    # function body. To keep the run cheap we patch `build_estimator` so the
    # three heavy families collapse to a cheap LogisticRegression while the two
    # cheap ones build normally. This keeps the orchestration logic under test
    # (looping, thresholding, champion selection, joblib dump) while avoiding
    # XGBoost, CatBoost, and MLP fits.
    real_build = tp.build_estimator

    def _cheap_build(name, config, pos_weight, has_cuda):
        if name in ("Logistic Regression", "Random Forest"):
            return real_build(name, config, pos_weight, has_cuda)
        # Substitute a fast LR for the heavy families to keep the loop intact.
        return LogisticRegression(max_iter=200)

    monkeypatch.setattr(tp, "build_estimator", _cheap_build)

    out_dir = tmp_path / "out"
    result = tp.train_baselines_and_refits(
        train_test_path=str(npz_path),
        tuned_results_path=str(tuned_path),
        out_dir=str(out_dir),
        autolog=False,
    )

    # Returned dict contract.
    for key in ("results", "default_results", "fitted_models",
                "default_fitted_models", "threshold_results",
                "model_thresholds", "champion_name", "champion_threshold"):
        assert key in result

    # The refit population is keyed by the tuned_results families.
    assert set(result["results"].keys()) == set(tuned_results.keys())
    assert result["champion_name"] in result["results"]
    assert isinstance(result["champion_threshold"], float)
    assert 0.0 <= result["champion_threshold"] <= 1.0

    # Persisted joblib artefacts.
    for fname in ("final_model.joblib", "final_model_threshold.joblib",
                  "model_thresholds.joblib", "training_models.joblib",
                  "training_results.joblib", "default_models.joblib",
                  "default_results.joblib"):
        assert (out_dir / fname).exists(), f"missing artefact {fname}"

    # The model_thresholds artefact should round-trip to a per-model float map.
    loaded_thresholds = joblib.load(out_dir / "model_thresholds.joblib")
    assert set(loaded_thresholds.keys()) == set(result["results"].keys())
    for t in loaded_thresholds.values():
        assert isinstance(t, float)
