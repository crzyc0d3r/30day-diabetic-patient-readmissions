"""Unit tests for `helpers.evaluation`.

This module pins the behaviour of the pure evaluation primitives shared by the
retrain orchestration DAG and the NB07/NB08 notebooks. These algorithms were
centralised so the automated retrain gate and the human-readable report agree
on the same numbers, and these tests guard the exact return shapes and the
numeric invariants every caller depends on.

Style note for maintainers: the assertions below favour hand-computed
expectations over re-deriving the answer with the same library the production
code uses. A test that mirrors the implementation cannot catch a regression in
that implementation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from helpers.evaluation import (
    best_f1_threshold,
    bootstrap_lift_ci,
    metric_panel,
    per_subgroup_metrics,
    per_subgroup_recall,
)


# ===========================================================================
# bootstrap_lift_ci
# ===========================================================================


def test_bootstrap_lift_ci_returns_three_floats(binary_classification_data):
    """The CI helper must return exactly three Python floats.

    The retrain gate unpacks `lo, hi, point` directly. A wrong arity or a
    numpy scalar leaking through (which serialises differently in the gate's
    JSON audit trail) would break the downstream contract.
    """
    y_true, y_proba = binary_classification_data
    # Build two distinct prediction vectors from the shared probabilities so
    # the candidate and prior genuinely differ.
    cand = (y_proba >= 0.45).astype(int)
    prior = (y_proba >= 0.55).astype(int)

    result = bootstrap_lift_ci(cand, prior, y_true, n_resamples=200, seed=17)

    assert isinstance(result, tuple)
    assert len(result) == 3
    for value in result:
        # Plain float, never a numpy type, so JSON and equality behave.
        assert type(value) is float


def test_bootstrap_lift_ci_ordering(binary_classification_data):
    """The CI must bracket the point estimate: lo <= point <= hi.

    This is the load-bearing invariant for the lift gate. If the point lift
    fell outside its own confidence interval, the gate's "is the lower bound
    above zero" logic would be meaningless.
    """
    y_true, y_proba = binary_classification_data
    cand = (y_proba >= 0.40).astype(int)
    prior = (y_proba >= 0.60).astype(int)

    lo, hi, point = bootstrap_lift_ci(cand, prior, y_true, n_resamples=300, seed=17)

    assert lo <= hi
    assert lo <= point <= hi


def test_bootstrap_lift_ci_is_deterministic(binary_classification_data):
    """The same seed and the same inputs must reproduce the same CI bit for bit.

    Reproducibility is non-negotiable for an automated gate. A retrain decision
    must be auditable after the fact, so two runs with the pinned seed must
    agree exactly.
    """
    y_true, y_proba = binary_classification_data
    cand = (y_proba >= 0.45).astype(int)
    prior = (y_proba >= 0.55).astype(int)

    first = bootstrap_lift_ci(cand, prior, y_true, n_resamples=250, seed=17)
    second = bootstrap_lift_ci(cand, prior, y_true, n_resamples=250, seed=17)

    assert first == second


def test_bootstrap_lift_ci_different_seed_changes_resamples(binary_classification_data):
    """A different seed should generally move the bootstrap CI bounds.

    This confirms the seed drives the resampling rather than being ignored. The
    point lift is computed on the full set and stays fixed, but the resampled
    lo and hi are expected to differ.
    """
    y_true, y_proba = binary_classification_data
    cand = (y_proba >= 0.45).astype(int)
    prior = (y_proba >= 0.55).astype(int)

    lo_a, hi_a, point_a = bootstrap_lift_ci(cand, prior, y_true, n_resamples=250, seed=1)
    lo_b, hi_b, point_b = bootstrap_lift_ci(cand, prior, y_true, n_resamples=250, seed=2)

    # The full-set point lift does not depend on the resampling seed.
    assert point_a == point_b
    # At least one CI bound should shift when the resampling seed changes.
    assert (lo_a, hi_a) != (lo_b, hi_b)


def test_bootstrap_lift_ci_identical_predictions_zero_lift():
    """When candidate and prior are identical the lift is exactly zero.

    If two models make the same predictions there is no improvement to detect,
    so the point lift and the entire bootstrap distribution must collapse to
    zero. This guards against an off-by-one or mislabelled-vector bug that would
    manufacture phantom lift between identical models.
    """
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=120)
    preds = (rng.random(120) >= 0.5).astype(int)

    lo, hi, point = bootstrap_lift_ci(preds, preds, y_true, n_resamples=200, seed=17)

    # Every resampled F1 difference is candidate_f1 minus prior_f1 on the SAME
    # vector, which is identically zero, so the whole CI must be zero.
    assert point == 0.0
    assert lo == 0.0
    assert hi == 0.0


# ===========================================================================
# per_subgroup_recall
# ===========================================================================


def test_per_subgroup_recall_keys_and_values():
    """The recall convenience wrapper must return one float recall per bucket.

    The equity check in the retrain gate consumes this flat `{label: recall}`
    mapping, so the keys must be every non-empty subgroup and the values must
    match the recall axis of the full metrics.
    """
    # Two subgroups, hand-constructed so recall is obvious.
    y_true = np.array([1, 1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 0, 1, 1, 0])
    groups = np.array(["a", "a", "a", "b", "b", "b"])

    recalls = per_subgroup_recall(y_true, y_pred, groups)

    assert set(recalls.keys()) == {"a", "b"}
    # Group a: positives at idx0,1, pred 1,0, so tp=1, fn=1, recall 0.5.
    assert recalls["a"] == pytest.approx(0.5)
    # Group b: positives at idx3,4, pred 1,1, so tp=2, fn=0, recall 1.0.
    assert recalls["b"] == pytest.approx(1.0)


def test_per_subgroup_recall_matches_full_metrics():
    """The wrapper's recall must equal per_subgroup_metrics' recall exactly.

    per_subgroup_recall is documented as a thin view over per_subgroup_metrics.
    If the two ever diverge, the gate's recall check and the report's fairness
    table would silently disagree, which is the precise failure this module
    exists to prevent.
    """
    y_true = np.array([1, 0, 1, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0, 1, 0])
    groups = np.array(["x", "x", "y", "y", "y", "x"])

    flat = per_subgroup_recall(y_true, y_pred, groups)
    full = per_subgroup_metrics(y_true, y_pred, groups)

    for label in full:
        assert flat[label] == full[label]["recall"] or (
            math.isnan(flat[label]) and math.isnan(full[label]["recall"])
        )


# ===========================================================================
# per_subgroup_metrics
# ===========================================================================


_EXPECTED_METRIC_KEYS = {
    "n",
    "positive_rate",
    "predicted_positive_rate",
    "recall",
    "precision",
    "fpr",
    "f1",
}


def test_per_subgroup_metrics_key_shape():
    """Every subgroup row must carry exactly the seven documented keys.

    NB08's fairness audit and the retrain gate both index this dict by name. A
    missing or extra key would raise a KeyError in one caller while the other
    stayed green, so the shape itself is part of the contract.
    """
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 1])
    groups = np.array(["g", "g", "g", "g"])

    out = per_subgroup_metrics(y_true, y_pred, groups)

    assert set(out.keys()) == {"g"}
    assert set(out["g"].keys()) == _EXPECTED_METRIC_KEYS


def test_per_subgroup_metrics_hand_computed_case():
    """A fully hand-worked confusion matrix pins every metric value.

    This is the anchor test for the subgroup math. A single group with a known
    confusion matrix lets us assert recall, precision, fpr, f1 and the two base
    rates against arithmetic done by hand rather than by sklearn.

    Group layout (y_true, y_pred):
        (1, 1) -> tp
        (1, 0) -> fn
        (0, 1) -> fp
        (0, 0) -> tn
    So tp=1, fn=1, fp=1, tn=1.
    """
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0])
    groups = np.array(["only", "only", "only", "only"])

    row = per_subgroup_metrics(y_true, y_pred, groups)["only"]

    assert row["n"] == 4.0
    # Two of four rows are truly positive, so the base positive rate is 0.5.
    assert row["positive_rate"] == pytest.approx(0.5)
    # Two of four rows are predicted positive, so the predicted positive rate is 0.5.
    assert row["predicted_positive_rate"] == pytest.approx(0.5)
    # recall = tp / (tp + fn) = 1 / 2.
    assert row["recall"] == pytest.approx(0.5)
    # precision = tp / (tp + fp) = 1 / 2.
    assert row["precision"] == pytest.approx(0.5)
    # fpr = fp / (fp + tn) = 1 / 2.
    assert row["fpr"] == pytest.approx(0.5)
    # f1 = 2 * 0.5 * 0.5 / (0.5 + 0.5) = 0.5.
    assert row["f1"] == pytest.approx(0.5)


def test_per_subgroup_metrics_degenerate_buckets_yield_nan():
    """No-positive and no-prediction buckets must surface NaN, not raise.

    The module computes the confusion matrix by hand so empty denominators
    become NaN rather than triggering sklearn warnings. A bucket with zero true
    positives has an undefined recall, and the gate relies on filtering those
    NaNs instead of catching exceptions.
    """
    # Group "neg" has no positives at all, so the recall denominator is zero.
    # No predicted positives, so the precision denominator is zero.
    y_true = np.array([0, 0, 0])
    y_pred = np.array([0, 0, 0])
    groups = np.array(["neg", "neg", "neg"])

    row = per_subgroup_metrics(y_true, y_pred, groups)["neg"]

    assert math.isnan(row["recall"])
    assert math.isnan(row["precision"])
    assert math.isnan(row["f1"])
    # fpr is well defined here: fp=0, tn=3, so 0 / 3 = 0.0.
    assert row["fpr"] == pytest.approx(0.0)


def test_per_subgroup_metrics_skips_only_present_levels():
    """Only labels present in the data become keys.

    np.unique drives the bucketing, so the output must contain exactly the
    distinct labels observed and nothing more. This stops a stale or expected
    subgroup name from sneaking into the audit when no rows carry it.
    """
    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([1, 0, 1, 0])
    groups = np.array(["p", "p", "q", "q"])

    out = per_subgroup_metrics(y_true, y_pred, groups)

    assert set(out.keys()) == {"p", "q"}


# ===========================================================================
# best_f1_threshold
# ===========================================================================


def test_best_f1_threshold_return_shape(binary_classification_data):
    """The sweep must return a `(threshold, f1)` pair of plain floats.

    NB07 writes this threshold into the registered model's tags. A numpy scalar
    leaking out would change how the value serialises and could drift from what
    the notebook plots.
    """
    y_true, y_proba = binary_classification_data

    threshold, f1 = best_f1_threshold(y_true, y_proba)

    assert type(threshold) is float
    assert type(f1) is float
    # F1 is bounded to [0, 1] by definition.
    assert 0.0 <= f1 <= 1.0


def test_best_f1_threshold_separable_case():
    """On a cleanly separable problem the sweep must find a perfect F1.

    When a threshold exists that perfectly splits the classes, the optimiser
    must discover it and report F1 == 1.0. The chosen threshold must also sit in
    the gap between the two probability clusters so it would reproduce the
    perfect split at scoring time.
    """
    # Negatives clustered near 0.1, positives near 0.9, leaving a clean gap.
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_proba = np.array([0.05, 0.10, 0.12, 0.15, 0.85, 0.88, 0.90, 0.95])

    threshold, f1 = best_f1_threshold(y_true, y_proba)

    assert f1 == pytest.approx(1.0)
    # Any threshold strictly above the top negative and at or below the bottom
    # positive yields the perfect split.
    assert 0.15 < threshold <= 0.85


def test_best_f1_threshold_respects_custom_grid():
    """A custom grid must constrain the returned threshold to that grid.

    Callers pass a bespoke grid when they want the threshold reported on a
    coarser or shifted scale. The function must only ever return a value drawn
    from the supplied grid, never from the default 91-point linspace.
    """
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.2, 0.3, 0.7, 0.8])
    custom_grid = np.array([0.25, 0.50, 0.75])

    threshold, _ = best_f1_threshold(y_true, y_proba, grid=custom_grid)

    # The chosen threshold must be one of the three grid points we passed.
    assert threshold in set(float(g) for g in custom_grid)


@pytest.mark.parametrize(
    "grid",
    [
        np.array([0.5]),
        np.linspace(0.1, 0.9, 9),
        np.array([0.3, 0.6, 0.9]),
    ],
)
def test_best_f1_threshold_threshold_drawn_from_grid(binary_classification_data, grid):
    """Across several grids, the returned threshold is always a grid member.

    Parametrising over grid shapes confirms the invariant holds regardless of
    grid size: the optimiser argmaxes over the grid and returns `g[idx]`, so
    the output can never be an interpolated off-grid value.
    """
    y_true, y_proba = binary_classification_data

    threshold, f1 = best_f1_threshold(y_true, y_proba, grid=grid)

    assert any(threshold == pytest.approx(float(g)) for g in grid)
    assert 0.0 <= f1 <= 1.0


# ===========================================================================
# metric_panel
# ===========================================================================


_EXPECTED_PANEL_KEYS = {
    "precision_pos",
    "recall_pos",
    "f1_pos",
    "balanced_accuracy",
    "matthews_cc",
    "cohens_kappa",
    "auc_roc",
    "auc_pr",
    "brier",
    "log_loss",
}


def test_metric_panel_has_all_ten_keys(binary_classification_data):
    """The panel must expose exactly the ten documented metric keys.

    NB08 renders this panel as a fixed table and the retrain audit logs each
    metric by name. A renamed or dropped key would break the report layout and
    the logged comparison at once.
    """
    y_true, y_proba = binary_classification_data
    y_pred = (y_proba >= 0.5).astype(int)

    panel = metric_panel(y_true, y_pred, y_proba)

    assert set(panel.keys()) == _EXPECTED_PANEL_KEYS
    assert len(panel) == 10


def test_metric_panel_values_are_floats_and_finite(binary_classification_data):
    """On a healthy two-class input every panel metric is a finite float.

    With a non-degenerate dataset none of the metrics should fall back to NaN.
    This is the happy-path counterpart to the degenerate test below and proves
    the safe wrapper does not over-eagerly swallow valid results.
    """
    y_true, y_proba = binary_classification_data
    y_pred = (y_proba >= 0.5).astype(int)

    panel = metric_panel(y_true, y_pred, y_proba)

    for name, value in panel.items():
        assert type(value) is float, name
        assert math.isfinite(value), name


def test_metric_panel_single_class_nan_fallback():
    """Single-class input must yield NaN for the metrics that cannot exist.

    AUC-ROC is undefined when only one class is present, and MCC and kappa
    collapse as well. The panel is computed inside bootstrap loops that
    routinely hit single-class folds, so each undefined metric must degrade to
    NaN rather than raise and abort the loop.
    """
    # Every label is the positive class, so ROC AUC and friends are undefined.
    y_true = np.ones(20, dtype=int)
    y_pred = np.ones(20, dtype=int)
    y_proba = np.full(20, 0.8)

    panel = metric_panel(y_true, y_pred, y_proba)

    # The panel survives (all ten keys still present) instead of raising.
    assert set(panel.keys()) == _EXPECTED_PANEL_KEYS
    # AUC-ROC is mathematically undefined on one class, so it must be NaN.
    assert math.isnan(panel["auc_roc"])
