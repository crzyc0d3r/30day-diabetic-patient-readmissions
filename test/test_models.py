"""Unit tests for `helpers.models`.

This module exercises the four public building blocks the readmission
pipeline shares across notebooks NB06 (HPO), NB07 (training), and NB08
(stacking + final evaluation):

* `evaluate_model`: the shared val-and-train metric panel.
* `build_estimator`: the model-name to estimator dispatch.
* `MLPWrapper`: the sklearn-compatible bridge around the torch MLP.
* `ReadmissionMLP`: the raw `nn.Module` itself.

WHY these matter: every notebook leans on this single source of truth, so a
silent change to a metric key, an estimator's contract, or the MLP's output
shape would ripple through the whole project. The tests pin the documented
behaviour so that drift is caught here instead of mid-pipeline.

All torch work is forced onto the CPU and held to a handful of epochs so the
suite runs in seconds. We never assume a CUDA device is present.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from helpers.models import (
    MLPWrapper,
    ReadmissionMLP,
    build_estimator,
    evaluate_model,
)


# Local fixtures.
#
# conftest.py supplies (y_true, y_proba) for metric tests, but estimator
# training needs a numeric feature matrix X plus a label vector y with BOTH
# classes present. We build a tiny, deterministic one here rather than reuse
# the metric fixture, because the two needs genuinely differ: metrics want
# probabilities, training wants features.

# Force every torch path in this file onto the CPU. The test venv ships a CUDA
# build of torch, but we run CPU-only on purpose so results do not depend on a
# GPU being available.
CPU = torch.device("cpu")


@pytest.fixture
def tiny_training_data(rng):
    """A small numeric (X, y) problem with both classes present.

    Forty rows by six features is ample for every estimator to fit in well
    under a second while still giving sklearn / xgboost / catboost a real
    (non-degenerate) target to learn. We guarantee both classes appear by
    constructing y from a thresholded linear signal rather than a coin flip,
    so no estimator ever sees a single-class fold.
    """
    n_rows, n_features = 40, 6
    X = rng.normal(0.0, 1.0, size=(n_rows, n_features)).astype(np.float32)
    # A linear signal plus noise, thresholded at its median, gives a balanced
    # two-class label that is learnable but not trivially perfect.
    signal = X @ rng.normal(0.0, 1.0, size=n_features) + rng.normal(0.0, 0.5, size=n_rows)
    y = (signal > np.median(signal)).astype(int)
    # Belt and braces: assert both classes are present so a flaky RNG seed can
    # never silently produce a single-class target that masks a real failure.
    assert set(np.unique(y)) == {0, 1}
    return X, y


# evaluate_model
def test_evaluate_model_returns_documented_keys_in_valid_ranges():
    """`evaluate_model` must return exactly the documented metric keys.

    WHAT: we hand the function a tiny perfectly-separable case and check the
    returned dict against the eight keys promised in the docstring
    (accuracy, precision, recall, f1, auc_roc, auc_pr, train_f1,
    train_auc_roc). WHY exact key spelling matters: NB07 and NB08 index this
    dict by name to build their leaderboards, so a renamed key would surface
    as a KeyError deep in a notebook rather than here.

    Every metric is a probability-like score, so each value must land in the
    closed [0, 1] interval. On this clean case the headline scores should be
    perfect, which also confirms we wired the arguments in the right order.
    """
    # A hand-built, perfectly-separated validation set: the first half are
    # negatives, the second half positives, and the predicted labels match.
    y_test = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 1])
    # Probabilities are monotonic with the label so AUC-ROC / AUC-PR are 1.0.
    y_prob = np.array([0.05, 0.10, 0.20, 0.80, 0.90, 0.95])

    # The train arrays mirror the same clean structure so the train-side
    # metrics are equally well-defined.
    y_train = np.array([0, 0, 1, 1])
    y_train_pred = np.array([0, 0, 1, 1])
    y_train_prob = np.array([0.10, 0.20, 0.80, 0.90])

    metrics = evaluate_model(
        y_test, y_pred, y_prob, y_train, y_train_pred, y_train_prob
    )

    expected_keys = {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc_roc",
        "auc_pr",
        "train_f1",
        "train_auc_roc",
    }
    assert set(metrics.keys()) == expected_keys

    # Every metric is a [0, 1] score. This guards against argument-order bugs
    # that could push a value outside the valid range.
    for key, value in metrics.items():
        assert 0.0 <= value <= 1.0, f"{key}={value} outside [0, 1]"

    # On a perfectly separable case the headline scores are exactly 1.0, which
    # confirms y_test / y_pred / y_prob were threaded through in the right slots.
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["auc_roc"] == pytest.approx(1.0)
    assert metrics["auc_pr"] == pytest.approx(1.0)


def test_evaluate_model_imperfect_case_lands_in_range(binary_classification_data):
    """On a realistic noisy case every metric stays strictly inside [0, 1].

    Reusing the shared `binary_classification_data` fixture gives us a
    non-degenerate problem where scores land in a believable mid-range rather
    than a suspicious perfect 1.0. WHY: a clean case alone could hide a metric
    that silently clamps or errors on overlapping classes.
    """
    y_true, y_proba = binary_classification_data
    # Threshold the probabilities at 0.5 to get hard predictions, exactly the
    # way the pipeline derives labels from scores.
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = evaluate_model(y_true, y_pred, y_proba, y_true, y_pred, y_proba)
    for key, value in metrics.items():
        assert 0.0 <= value <= 1.0, f"{key}={value} outside [0, 1]"


# build_estimator
#
# The five model names the dispatch understands. We parametrize so each model
# is its own test case, which makes a single broken branch obvious in the
# pytest report instead of hiding behind a loop.
MODEL_NAMES = [
    "XGBoost",
    "CatBoost",
    "Logistic Regression",
    "Random Forest",
    "MLP",
]


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_build_estimator_returns_fittable_estimator(name, tiny_training_data):
    """Each model name yields a sklearn-style estimator that fits and predicts.

    WHAT: we ask `build_estimator` for each of the five supported models,
    then confirm the returned object honours the sklearn contract enough for
    the pipeline's needs: it exposes `fit` / `predict`, and after fitting a
    tiny (X, y) it predicts a label per row.

    WHY `has_cuda=False`: the test host runs CPU-only, so we must not ask
    XGBoost for `device=cuda` or CatBoost for `task_type=GPU`. Passing
    `has_cuda=False` selects the CPU code paths the dispatch documents.

    The MLP is kept tiny (a few epochs) so its fit stays fast.
    """
    X, y = tiny_training_data

    # Keep the MLP's epoch budget small so the torch fit runs in a blink. The
    # other estimators ignore the 'epochs' key, so we inject it only for MLP.
    config = {"epochs": 2} if name == "MLP" else {}

    estimator = build_estimator(name, config=config, pos_weight=1.0, has_cuda=False)

    # Every supported model must expose the core sklearn classifier verbs.
    assert hasattr(estimator, "fit")
    assert hasattr(estimator, "predict")

    estimator.fit(X, y)
    preds = estimator.predict(X)

    # Predict must return one label per input row.
    assert len(preds) == len(X)

    # Every model in this pipeline is a probabilistic classifier, so
    # predict_proba must exist and return a row-aligned, two-column matrix.
    assert hasattr(estimator, "predict_proba")
    proba = estimator.predict_proba(X)
    assert proba.shape[0] == len(X)
    assert proba.shape[1] == 2


def test_build_estimator_mlp_is_mlpwrapper(tiny_training_data):
    """The "MLP" branch specifically returns an `MLPWrapper` instance.

    WHY this is its own test: NB08 relies on the MLP path producing the
    sklearn-compatible wrapper (not a bare `nn.Module`) so it can sit beside
    the tree models in the same leaderboard. We pin the concrete type so a
    refactor that returns a raw module is caught immediately.
    """
    estimator = build_estimator("MLP", config={"epochs": 1}, pos_weight=1.0, has_cuda=False)
    assert isinstance(estimator, MLPWrapper)


def test_build_estimator_unknown_name_raises():
    """An unrecognised model name raises `ValueError` rather than failing late.

    The dispatch is typed with a `Literal` for static checking, but at
    runtime an unknown string must fail loudly and early.
    """
    with pytest.raises(ValueError):
        build_estimator("NotAModel", config={}, pos_weight=1.0, has_cuda=False)


# MLPWrapper
def test_mlpwrapper_fit_returns_self(tiny_training_data):
    """`fit` must return `self` to satisfy the sklearn estimator contract.

    WHY: sklearn utilities (Pipeline, GridSearchCV) chain `estimator.fit(...)`
    and reuse the return value. A wrapper that returned `None` would break
    every such chain, so the contract is non-negotiable.
    """
    X, y = tiny_training_data
    wrapper = MLPWrapper(epochs=2)
    returned = wrapper.fit(X, y)
    assert returned is wrapper


def test_mlpwrapper_predict_shape_and_proba_normalisation(tiny_training_data):
    """`predict` gives 1-D labels and `predict_proba` gives normalised rows.

    WHAT we check:
      * `predict` returns a 1-D array of length len(X) (one label per row).
      * `predict_proba` returns a two-column ndarray whose rows each sum to
        ~1.0, matching every other sklearn binary classifier.

    WHY the proba-sum check matters: the wrapper builds the negative-class
    column as `1 - positive`, so a broken implementation that forgot the
    complement would fail this row-sum assertion.
    """
    X, y = tiny_training_data
    wrapper = MLPWrapper(epochs=2)
    wrapper.fit(X, y)

    preds = wrapper.predict(X)
    # 1-D labels, one per input row.
    assert preds.ndim == 1
    assert len(preds) == len(X)

    proba = wrapper.predict_proba(X)
    # Two columns: P(class=0), P(class=1).
    assert proba.shape == (len(X), 2)
    # Each row is a proper probability distribution summing to ~1.
    row_sums = proba.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5)


def test_mlpwrapper_inference_does_not_crash_in_eval_mode(tiny_training_data):
    """Inference on a fitted wrapper runs cleanly with the module in eval mode.

    The wrapper flips the underlying module into inference mode before scoring.
    We confirm a second, independent prediction call still succeeds and returns
    finite probabilities: eval-mode inference is stable and side-effect free.
    """
    X, y = tiny_training_data
    wrapper = MLPWrapper(epochs=2)
    wrapper.fit(X, y)

    # Calling predict_proba twice exercises the eval-mode path repeatedly and
    # confirms it neither crashes nor produces NaNs or infs.
    first = wrapper.predict_proba(X)
    second = wrapper.predict_proba(X)
    assert np.all(np.isfinite(first))
    assert np.all(np.isfinite(second))


# ReadmissionMLP
def test_readmission_mlp_forward_output_shape():
    """A forward pass returns `(batch, 1)` logits for a small float tensor.

    WHAT: we build the module with a small `n_features`, feed a small float
    batch through it, and assert the output is shaped `(batch, 1)`, a single
    raw logit per row. WHY the explicit shape: NB07's loss
    (`BCEWithLogitsLoss`) and the §7.8 threshold sweep both assume one logit
    column, so a change to the head's output width would break training.

    We use a batch size > 1 because the module contains `BatchNorm1d`, which
    needs at least two rows to compute batch statistics in training mode.
    """
    n_features = 6
    batch = 8
    model = ReadmissionMLP(n_features).to(CPU)

    x = torch.randn(batch, n_features, device=CPU)
    out = model(x)

    # One raw logit per row: shape must be exactly (batch, 1).
    assert tuple(out.shape) == (batch, 1)
    # Logits should be finite real numbers, not NaN or inf.
    assert torch.all(torch.isfinite(out))
