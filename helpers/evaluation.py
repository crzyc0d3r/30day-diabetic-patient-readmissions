"""Evaluation utilities shared by the pipeline notebooks and the
retrain orchestration DAG.

These functions centralise the gate math that the notebooks and the
retrain DAG both depend on. 'pipeline/07_model_training.ipynb' and
'pipeline/08_conclusion.ipynb' run the threshold-sweep and subgroup-metric
helpers, and 'infra/airflow/dags/retrain_on_drift_dag.py' runs the
bootstrap-lift CI and per-subgroup recall. Holding the algorithms in one
place means a change shows up in every caller at once and the same numbers
underpin both the human-readable NB08 report and the automated retrain gate.

Each function is a pure-Python computation. No notebook hooks, no
IPython, no matplotlib, no plotting. Safe to import from any kernel:
the production inference path, an Airflow task, or a
notebook cell.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Default 91-point threshold grid (0.05 → 0.95 in 0.01 steps). Re-exported
# from helpers.constants but pinned here as the function-level default so
# best_f1_threshold remains usable without an explicit grid argument.
_DEFAULT_THRESHOLD_GRID = np.linspace(0.05, 0.95, 91)


def bootstrap_lift_ci(
    cand_preds: np.ndarray,
    prior_preds: np.ndarray,
    y_true: np.ndarray,
    *,
    n_resamples: int = 1000,
    alpha: float = 0.95,
    seed: int = 17,
) -> tuple[float, float, float]:
    """Bootstrap a (lo, hi, point) CI on candidate_f1 - prior_f1.

    Resamples row indices with replacement, then scores both models on
    the *same* resampled indices each iteration so the resulting lift
    distribution accounts for prediction correlation. This is the
    PAIRED bootstrap. The naive INDEPENDENT-resample variant draws
    separate indices for candidate and prior and inflates the variance
    of the lift, producing a wider CI that can hide a true regression.
    Reviewers comparing this CI to a hand-rolled one elsewhere should
    expect the paired CI to be strictly narrower at the same n_resamples.

    Returns the point lift on the full set plus the (1 - alpha) two-sided
    CI. Used by retrain_on_drift_dag's lift-CI gate and by NB08's
    candidate-vs-prior comparison, so the same numbers must show up in
    both places.
    """
    rng = np.random.default_rng(seed)
    y_true_arr = np.asarray(y_true)
    cand_arr = np.asarray(cand_preds)
    prior_arr = np.asarray(prior_preds)
    n = len(y_true_arr)

    point_lift = float(
        f1_score(y_true_arr, cand_arr) - f1_score(y_true_arr, prior_arr)
    )

    lifts = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        lifts[i] = (
            f1_score(y_true_arr[idx], cand_arr[idx])
            - f1_score(y_true_arr[idx], prior_arr[idx])
        )
    lo = float(np.quantile(lifts, (1 - alpha) / 2))
    hi = float(np.quantile(lifts, 1 - (1 - alpha) / 2))
    return lo, hi, point_lift


def per_subgroup_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subgroup_labels: np.ndarray,
) -> dict[str, float]:
    """Return '{subgroup_label: recall}' for every non-empty bucket.

    Thin convenience over :func:`per_subgroup_metrics` for callers that
    only need the recall axis (the retrain gate's equity check).
    """
    full = per_subgroup_metrics(y_true, y_pred, subgroup_labels)
    return {label: row["recall"] for label, row in full.items()}


def per_subgroup_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subgroup_labels: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Return '{subgroup_label: {n, positive_rate, predicted_positive_rate,
    recall, precision, fpr, f1}}' for every non-empty bucket.

    Computes the per-subgroup confusion matrix directly so we don't pay
    repeated sklearn dispatch overhead and so empty/edge cases (no
    positives, no negatives) surface as 'nan' rather than warnings.

    NB08's fairness audit and the retrain gate both rely on this exact
    shape. If you add a metric here, make sure both callers are happy.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    labels_arr = np.asarray(subgroup_labels)

    out: dict[str, dict[str, float]] = {}
    for level in np.unique(labels_arr):
        mask = labels_arr == level
        n = int(mask.sum())
        if n == 0:
            continue
        yt = y_true_arr[mask]
        yp = y_pred_arr[mask]
        tp = int(((yt == 1) & (yp == 1)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        tn = int(((yt == 0) & (yp == 0)).sum())
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        fpr = fp / (fp + tn) if (fp + tn) else float("nan")
        f1 = (
            (2 * precision * recall / (precision + recall))
            if (precision + recall)
            else float("nan")
        )
        out[str(level)] = {
            "n": float(n),
            "positive_rate": float(yt.mean()),
            "predicted_positive_rate": float(yp.mean()),
            "recall": recall,
            "precision": precision,
            "fpr": fpr,
            "f1": f1,
        }
    return out


def best_f1_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    grid: np.ndarray | None = None,
) -> tuple[float, float]:
    """Sweep 'grid' (defaults to `np.linspace(0.05, 0.95, 91)`),
    threshold `y_proba` at each value, and return the
    `(threshold, f1)` pair that maximises F1 on `y_true`.

    NB07 selects a per-model operating threshold and NB08 selects one
    for the final champion. Both need to see the same grid so the
    threshold reported in the registered model's tags lines up with
    what the notebooks plot.
    """
    g = _DEFAULT_THRESHOLD_GRID if grid is None else np.asarray(grid)
    y_true_arr = np.asarray(y_true)
    y_proba_arr = np.asarray(y_proba)
    scores = np.array(
        [f1_score(y_true_arr, (y_proba_arr >= t).astype(int)) for t in g]
    )
    idx = int(np.argmax(scores))
    return float(g[idx]), float(scores[idx])


def metric_panel(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, float]:
    """Compute the full NB08 metric panel: precision, recall, F1,
    balanced accuracy, MCC, kappa, AUC-ROC, AUC-PR, Brier, log loss.

    Returns NaN for any individual metric that raises (e.g. AUC on a
    single-class fold) so the panel as a whole survives degenerate
    inputs. Bootstrap loops over this routinely.
    """

    def _safe(fn, *args: Any) -> float:
        try:
            return float(fn(*args))
        except Exception:
            return float("nan")

    return {
        "precision_pos":     _safe(precision_score, y_true, y_pred),
        "recall_pos":        _safe(recall_score, y_true, y_pred),
        "f1_pos":            _safe(f1_score, y_true, y_pred),
        "balanced_accuracy": _safe(balanced_accuracy_score, y_true, y_pred),
        "matthews_cc":       _safe(matthews_corrcoef, y_true, y_pred),
        "cohens_kappa":      _safe(cohen_kappa_score, y_true, y_pred),
        "auc_roc":           _safe(roc_auc_score, y_true, y_proba),
        "auc_pr":            _safe(average_precision_score, y_true, y_proba),
        "brier":             _safe(brier_score_loss, y_true, y_proba),
        "log_loss":          _safe(log_loss, y_true, y_proba),
    }
