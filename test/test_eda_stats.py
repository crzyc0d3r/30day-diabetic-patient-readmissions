"""Unit tests for helpers/eda_stats.py — the EDA effect-size / CI / adjusted-OR
helpers used to quantify the readmission analysis in NB03.

The suite checks the statistics against textbook values (a 2x2 with a known odds
ratio, a Wilson interval at p=0.5) and against planted effects (a logistic model
with a known coefficient), so a regression in the math is caught immediately. It
is hermetic: every frame is constructed in-test, nothing reads real data.

Per project conventions this file avoids em dashes and semicolons and uses the
spelling "program".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from helpers.eda_stats import (
    adjusted_logit,
    odds_ratio_ci,
    rate_with_ci,
    univariate_or,
    wilson_ci,
)


# --------------------------------------------------------------------------- #
# wilson_ci
# --------------------------------------------------------------------------- #
def test_wilson_ci_half_known_value():
    """Wilson 95% CI for 50/100 is approximately (0.404, 0.596)."""
    lo, hi = wilson_ci(50, 100)
    assert lo == pytest.approx(0.404, abs=0.003)
    assert hi == pytest.approx(0.596, abs=0.003)


def test_wilson_ci_stays_in_unit_interval_at_extremes():
    lo, hi = wilson_ci(0, 20)
    assert lo == pytest.approx(0.0, abs=1e-9) and 0.0 < hi < 1.0
    lo, hi = wilson_ci(20, 20)
    assert 0.0 < lo < 1.0 and hi == pytest.approx(1.0, abs=1e-9)


def test_wilson_ci_brackets_point_and_narrows_with_n():
    p = 0.3
    narrow = wilson_ci(300, 1000)
    wide = wilson_ci(3, 10)
    assert narrow[0] < p < narrow[1] and wide[0] < p < wide[1]
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_wilson_ci_zero_n_is_nan():
    lo, hi = wilson_ci(0, 0)
    assert np.isnan(lo) and np.isnan(hi)


# --------------------------------------------------------------------------- #
# odds_ratio_ci
# --------------------------------------------------------------------------- #
def test_odds_ratio_known_value():
    """a=20,b=10,c=10,d=20 gives OR = (20*20)/(10*10) = 4.0, CI excludes 1."""
    or_, lo, hi = odds_ratio_ci(20, 10, 10, 20)
    assert or_ == pytest.approx(4.0, rel=1e-9)
    assert 1.0 < lo < hi


def test_odds_ratio_one_for_no_association():
    or_, lo, hi = odds_ratio_ci(25, 25, 25, 25)
    assert or_ == pytest.approx(1.0, rel=1e-9)
    assert lo < 1.0 < hi


def test_odds_ratio_zero_cell_stays_finite():
    or_, lo, hi = odds_ratio_ci(0, 12, 8, 30)  # zero exposed-events
    assert np.isfinite(or_) and np.isfinite(lo) and np.isfinite(hi)
    assert or_ < 1.0  # exposure protective when exposed-events are ~0


# --------------------------------------------------------------------------- #
# univariate_or
# --------------------------------------------------------------------------- #
def test_univariate_or_recovers_strong_association():
    # indicator perfectly raises event odds
    x = np.array([1] * 100 + [0] * 100)
    y = np.array([1] * 80 + [0] * 20 + [1] * 20 + [0] * 80)
    df = pd.DataFrame({"x": x, "y": y})
    res = univariate_or(df, "y", "x")
    assert res["or"] > 8.0
    assert res["ci_low"] > 1.0
    assert res["n_exposed"] == 100 and res["n_unexposed"] == 100


# --------------------------------------------------------------------------- #
# rate_with_ci
# --------------------------------------------------------------------------- #
def test_rate_with_ci_counts_and_rates():
    df = pd.DataFrame({
        "grp": ["a"] * 100 + ["b"] * 50,
        "y": [1] * 30 + [0] * 70 + [1] * 25 + [0] * 25,
    })
    res = rate_with_ci(df, "grp", "y")
    assert res.loc["a", "n"] == 100 and res.loc["a", "events"] == 30
    assert res.loc["a", "rate"] == pytest.approx(0.30)
    assert res.loc["b", "rate"] == pytest.approx(0.50)
    for g in ("a", "b"):
        assert res.loc[g, "ci_low"] < res.loc[g, "rate"] < res.loc[g, "ci_high"]


def test_rate_with_ci_accepts_aligned_series():
    df = pd.DataFrame({"grp": ["a", "a", "b", "b"], "readmitted": ["<30", "NO", "<30", "<30"]})
    target = (df["readmitted"] == "<30").astype(int)
    res = rate_with_ci(df, "grp", target)
    assert res.loc["b", "rate"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# adjusted_logit (needs statsmodels)
# --------------------------------------------------------------------------- #
def test_adjusted_logit_recovers_planted_effect():
    pytest.importorskip("statsmodels")
    rng = np.random.default_rng(7)
    n = 4000
    x1 = rng.normal(size=n)          # true predictor, coef +1.0 -> OR ~ e^1 = 2.72
    x2 = rng.normal(size=n)          # confounder, correlated with x1
    x1 = x1 + 0.5 * x2
    logits = -0.5 + 1.0 * x1 + 0.3 * x2
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logits))).astype(int)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})

    table = adjusted_logit(df, "y", ["x1", "x2"])
    row = table.set_index("term").loc["x1"]
    assert row["OR"] == pytest.approx(np.exp(1.0), rel=0.20)  # ~2.7
    assert row["ci_low"] > 1.0                                # effect is significant
    assert (table["term"] == "Intercept").sum() == 0          # intercept excluded
