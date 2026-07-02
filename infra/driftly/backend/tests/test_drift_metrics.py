"""Unit tests for drift_metrics.compute_drift — the per-feature orchestration."""
from __future__ import annotations

import numpy as np
import pandas as pd

from drift_metrics import compute_drift, worst
from helpers.drift_sim import MONITORED_CATEGORICAL, MONITORED_CONTINUOUS


def _frame(n, rng, *, shift=0.0):
    data = {}
    for col in MONITORED_CONTINUOUS:
        data[col] = rng.normal(5.0 + shift, 2.0, n).clip(min=0)
    cats = ["a", "b", "c", "d"]
    w = np.array([4.0, 3.0, 2.0, 1.0]) + np.array([0, 0, shift, shift * 2])
    for col in MONITORED_CATEGORICAL:
        data[col] = rng.choice(cats, n, p=w / w.sum())
    return pd.DataFrame(data)


def test_worst_picks_most_severe():
    assert worst([]) == "OK"
    assert worst(["OK", "WARN", "OK"]) == "WARN"
    assert worst(["WARN", "ALERT", "OK"]) == "ALERT"


def test_identical_frames_verdict_ok():
    rng = np.random.default_rng(1)
    ref = _frame(2000, rng)
    result = compute_drift(ref, ref.copy(), "self")
    assert result["verdict"] == "OK"
    assert result["summary"]["alert"] == 0
    assert result["reference_rows"] == 2000


def test_strong_shift_escalates_verdict():
    rng = np.random.default_rng(2)
    ref = _frame(2000, rng)
    cur = _frame(2000, rng, shift=6.0)
    result = compute_drift(ref, cur, "spike")
    assert result["verdict"] in ("WARN", "ALERT")
    assert result["summary"]["warn"] + result["summary"]["alert"] > 0


def test_feature_schema_and_metric_presence():
    rng = np.random.default_rng(3)
    ref = _frame(1500, rng)
    result = compute_drift(ref, _frame(1500, rng), "x")
    by_name = {f["name"]: f for f in result["features"]}

    num = by_name[MONITORED_CONTINUOUS[0]]
    assert num["type"] == "numeric"
    assert num["wasserstein"] is not None and num["psi"] is not None and num["kl"] is not None
    assert "bins" in num["histogram"]

    cat = by_name[MONITORED_CATEGORICAL[0]]
    assert cat["type"] == "categorical"
    assert cat["wasserstein"] is None          # no ground distance for nominal vars
    assert "categories" in cat["histogram"]


def test_intersection_only_when_columns_missing():
    rng = np.random.default_rng(4)
    ref = _frame(800, rng)
    cur = _frame(800, rng).drop(columns=[MONITORED_CONTINUOUS[0]])
    result = compute_drift(ref, cur, "partial")
    names = {f["name"] for f in result["features"]}
    assert MONITORED_CONTINUOUS[0] not in names      # dropped column is skipped
    assert MONITORED_CONTINUOUS[1] in names          # the rest still scored


def test_numeric_histogram_fractions_sum_to_one():
    rng = np.random.default_rng(5)
    ref = _frame(1000, rng)
    result = compute_drift(ref, _frame(1000, rng), "x")
    num = next(f for f in result["features"] if f["type"] == "numeric")
    assert sum(num["histogram"]["reference"]) == \
        __import__("pytest").approx(1.0, abs=1e-3)
