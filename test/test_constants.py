"""Guard tests for helpers/constants.py.

WHAT this module guards.
`helpers.constants` centralises every tuned knob the medi-watch pipeline
relies on: the overfit gates, the canonical RNG seed, the HPO/ASHA splitter
sizes, the retrain/promotion gate thresholds, the threshold-sweep grid, and
the column groups the cleaning and feature-engineering layers must agree on.

WHY these tests.
These are not behavioural tests. They are INVARIANT guards. Each constant
carries a docstring in the source explaining WHY it holds its value (a
calibration, an SLA, a Postgres TOAST limit, a reproducibility contract). An
accidental edit to any of those numbers would silently change a gate, a
split, or a sweep across notebooks, DAGs, and the inference service. These
assertions pin the documented value, or the documented relationship between
two values, so such an edit fails CI loudly instead of shipping. Where a
relationship is what matters (TIGHT < OVERFIT, WARN < ALERT, ASHA_MAX_T ==
N_SPLITS_HPO) we assert the relationship rather than the literal, so the
intent survives a deliberate retune of the underlying number.
"""

from __future__ import annotations

import numpy as np

from helpers import constants


def test_seed_is_canonical_42() -> None:
    """SEED pins the pipeline's randomness for bit-for-bit reruns.

    Every split, shuffle, and bootstrap derives from this value, so it must
    stay 42, the documented canonical seed, or reproducibility breaks.
    """
    assert constants.SEED == 42


def test_overfit_thresholds_and_ordering() -> None:
    """The two overfit gates hold their calibrated values and stay ordered.

    0.15 is the §6-baseline-calibrated overfit gap. 0.10 is the tighter gate
    for the noisier §7.5 CV-fold reads. The tight gate must be strictly
    smaller, otherwise "tight" would not be tighter.
    """
    assert constants.OVERFIT_THRESHOLD == 0.15
    assert constants.TIGHT_OVERFIT_THRESHOLD == 0.10
    assert constants.TIGHT_OVERFIT_THRESHOLD < constants.OVERFIT_THRESHOLD


def test_unknown_categorical_sentinel() -> None:
    """The missing-category sentinel must match the OHE fit-time level.

    nb02 cleaning and both inference services fill "?" with this exact
    string. If it drifts, handle_unknown="ignore" silently zero-vectors the
    column at inference, so the literal value is pinned here.
    """
    assert constants.UNKNOWN_CATEGORICAL == "Unknown"


def test_asha_max_t_tracks_n_splits_hpo() -> None:
    """ASHA's max_t is pinned to the CV fold count (one rung per fold).

    The scheduler runs one rung per CV fold, so ASHA_MAX_T must equal
    N_SPLITS_HPO. Asserting the relationship rather than the literal 3 keeps
    the invariant true even when the fold count is deliberately retuned.
    """
    assert constants.ASHA_MAX_T == constants.N_SPLITS_HPO


def test_bootstrap_ci_alpha_is_a_valid_coverage() -> None:
    """The bootstrap CI coverage is the documented 95% and a valid fraction.

    0.95 makes the gate's 'lower bound > LIFT_FLOOR' a 95% confidence claim.
    It must also be a proper probability strictly inside (0, 1).
    """
    assert constants.BOOTSTRAP_CI_ALPHA == 0.95
    assert 0 < constants.BOOTSTRAP_CI_ALPHA < 1


def test_drift_psi_warn_below_alert() -> None:
    """PSI WARN must sit below ALERT so escalation is monotonic.

    0.1 warns. 0.2 alerts and raises the gating MLflow tag. WARN below ALERT
    is what makes the two-tier routing meaningful.
    """
    assert constants.DRIFT_PSI_WARN < constants.DRIFT_PSI_ALERT


def test_threshold_sweep_grid_shape_and_ordering() -> None:
    """The best-F1 sweep grid is a sorted 91-point linspace over [0.05, 0.95].

    NB07 and NB08 sweep this identical grid so the registered operating
    threshold matches the conclusion-notebook plot. It must be a numpy array
    of length 91, span roughly 0.05 to 0.95, and ascend so the sweep is well
    defined.
    """
    grid = constants.THRESHOLD_SWEEP_GRID
    assert isinstance(grid, np.ndarray)
    assert grid.shape == (91,)
    assert grid[0] == np.float64(0.05) or abs(float(grid[0]) - 0.05) < 1e-9
    assert abs(float(grid[-1]) - 0.95) < 1e-9
    # Strictly ascending: every step is positive, so the sweep is sorted.
    assert np.all(np.diff(grid) > 0)


def test_categorical_missing_cols_superset_of_mnar_flags() -> None:
    """The "?"-coded columns must contain the MNAR-flag subset.

    CATEGORICAL_MISSING_COLS are every column the UCI source codes as "?".
    MNAR_FLAG_COLS is the subset that gets per-column missingness flags, so
    every MNAR-flag column must also appear in the broader "?"-coded tuple.
    """
    assert isinstance(constants.CATEGORICAL_MISSING_COLS, tuple)
    assert isinstance(constants.MNAR_FLAG_COLS, tuple)
    assert len(constants.MNAR_FLAG_COLS) == 3
    assert set(constants.MNAR_FLAG_COLS).issubset(set(constants.CATEGORICAL_MISSING_COLS))


def test_positive_int_capacities() -> None:
    """Bootstrap resample count must be a positive integer.

    BOOTSTRAP_RESAMPLES is the resample count for the CI. It is a count, so
    a non-positive or non-integer value would be nonsensical.
    """
    assert isinstance(constants.BOOTSTRAP_RESAMPLES, int)
    assert constants.BOOTSTRAP_RESAMPLES > 0


def test_retrain_gate_documented_values() -> None:
    """The retrain/promotion gate knobs hold their documented SLA values.

    COOLDOWN_DAYS (7) is the minimum gap between champion swaps.
    EQUITY_RECALL_TOL (0.05) is the maximum tolerated per-subgroup recall
    drop. LIFT_FLOOR (0.005) is the smallest F1 lift the lower CI bound must
    clear to promote. These are the operational gate constants, pinned
    exactly.
    """
    assert constants.COOLDOWN_DAYS == 7
    assert constants.EQUITY_RECALL_TOL == 0.05
    assert constants.LIFT_FLOOR == 0.005
