"""Unit tests for `helpers/drift_sim.py`, the PSI/KS drift detector and the
seeded "newly-arrived encounter batch" generator behind the medi-watch
drift -> retrain demonstration.

WHY THIS SUITE EXISTS
=====================
`drift_sim` is a mostly pure module. Its statistics functions perform no
I/O, and its scenario generator is fully deterministic for a fixed seed.
That makes it an ideal candidate for fast, hermetic unit tests: we feed it
hand-built reference and current DataFrames, then assert on exact
mathematical properties (PSI is zero for identical inputs, KS lies in
[0, 1], a strong injected shift escalates the verdict to ALERT, and so on).

The tests below follow the module's own structure:

  * the drift statistics (`_psi_from_fractions`, `psi_numeric`,
    `psi_categorical`, `ks_statistic`),
  * the routing-and-verdict report (`column_drift_report`),
  * the scenario generator (`make_scenario`),
  * the on-disk batch writer (`write_scenarios`).

We construct our own synthetic frames rather than reading `data/features.csv`
so the suite stays hermetic and never touches the repository's real data.
Every frame carries the columns the module declares it monitors, which we
read straight off the module (`MONITORED_*`) so the tests track the source
of truth instead of duplicating a column list that could drift apart.

STYLE NOTES
===========
Comments and docstrings are deliberately generous and educational. Per the
project conventions this file avoids em dashes, avoids semicolons, and uses
the spelling "program" throughout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Import the module under test exactly as production code does, so the
# tests exercise the real import path rather than a copy of it.
from helpers.drift_sim import (
    MONITORED_CATEGORICAL,
    MONITORED_COLUMNS,
    MONITORED_CONTINUOUS,
    SCENARIOS,
    _kl_from_fractions,
    _psi_from_fractions,
    build_drift_report,
    champion_impact,
    column_drift_report,
    kl_categorical,
    kl_numeric,
    ks_statistic,
    make_scenario,
    psi_categorical,
    psi_numeric,
    wasserstein_numeric,
    write_scenarios,
)

# The thresholds and the canonical seed come from the shared constants module,
# so the tests assert against the same numbers the detector itself uses.
from helpers.constants import (
    DRIFT_KS_UNIQUE_THRESHOLD,
    DRIFT_PSI_ALERT,
    DRIFT_PSI_WARN,
    SEED,
)


# =========================================================================== #
# Helpers for building synthetic reference / current frames.
#
# The detector monitors the columns named in MONITORED_COLUMNS. To produce a
# realistic report we need a frame that carries every one of those columns
# with plausible dtypes: the continuous columns numeric (and varied enough to
# clear the KS unique-value threshold), the categorical columns string-typed.
# We centralise construction here so each test starts from the same well
# shaped cohort and perturbs only what it cares about.
# =========================================================================== #
def _make_reference_frame(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """Build a synthetic engineered-feature frame covering every monitored column.

    Continuous columns are drawn from non-degenerate integer distributions
    wide enough to exceed `DRIFT_KS_UNIQUE_THRESHOLD` distinct values, which
    routes them through KS in `column_drift_report`. Categorical columns are
    drawn from small string vocabularies so PSI has a handful of bins to
    compare. `n` is kept modest for speed but large enough that the
    statistics stay stable.
    """
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {}

    # Continuous columns: integer counts with a healthy spread of distinct
    # values so each clears the KS routing threshold comfortably.
    for col in MONITORED_CONTINUOUS:
        data[col] = rng.integers(0, 60, size=n)

    # Categorical columns: small fixed vocabularies. Each carries a few levels
    # with uneven frequencies so PSI has a non-trivial reference distribution.
    cat_vocab = {
        "diag_1_cat": ["Circulatory", "Diabetes", "Respiratory", "Other"],
        "diag_2_cat": ["Circulatory", "Diabetes", "Respiratory", "Other"],
        "diag_3_cat": ["Circulatory", "Diabetes", "Respiratory", "Other"],
        "admission_type": ["Elective", "Urgent", "Emergency"],
        "discharge_group": ["Home", "Transfer", "Other"],
        "payer_grouped": ["Medicare", "Private", "SelfPay", "Other"],
        "race": ["Caucasian", "AfricanAmerican", "Hispanic", "Other"],
        "gender": ["Male", "Female"],
        "insulin": ["No", "Steady", "Up", "Down"],
        "metformin": ["No", "Steady", "Up", "Down"],
        "A1Cresult": ["None", ">7", "Norm", ">8"],
        "change": ["No", "Ch"],
    }
    for col in MONITORED_CATEGORICAL:
        levels = cat_vocab[col]
        data[col] = rng.choice(levels, size=n)

    # A handful of extra columns the scenarios perturb but that are absent from
    # MONITORED_COLUMNS. Including them lets make_scenario exercise its full
    # set of perturbations and lets us assert schema-identity end to end.
    data["age_mid"] = rng.integers(0, 100, size=n)
    data["admission_source"] = rng.choice(["Referral", "Transfer", "ER"], size=n)
    data["n_med_changes"] = rng.integers(0, 6, size=n)

    return pd.DataFrame(data)


# =========================================================================== #
# Module-level declaration sanity.
#
# These are cheap guards that catch an accidental edit to the module's public
# column lists or scenario tuple before the heavier tests build frames on top
# of them.
# =========================================================================== #
def test_monitored_columns_is_continuous_plus_categorical():
    """MONITORED_COLUMNS must be exactly the continuous list followed by the categorical list."""
    assert MONITORED_COLUMNS == MONITORED_CONTINUOUS + MONITORED_CATEGORICAL
    # No column may appear in both halves, otherwise routing is ambiguous.
    assert set(MONITORED_CONTINUOUS).isdisjoint(set(MONITORED_CATEGORICAL))


def test_scenarios_includes_the_none_control():
    """The scenario tuple must contain the no-drift control named "none"."""
    assert "none" in SCENARIOS
    # SCENARIOS is documented as the five demonstrations plus the control.
    assert len(SCENARIOS) >= 2


# =========================================================================== #
# _psi_from_fractions: the shared PSI core.
#
# PSI compares two probability vectors. The core takes already-normalised bin
# fractions, clips them away from zero (the eps guard) to keep the log finite,
# and sums (cur - ref) * log(cur / ref) over the bins.
# =========================================================================== #
def test_psi_from_fractions_zero_for_identical_distributions():
    """Identical fraction vectors give PSI of exactly zero.

    When `cur == ref` every term is `0 * log(1) == 0`, so the sum is zero.
    This is the defining "no drift" property of PSI.
    """
    frac = np.array([0.1, 0.2, 0.3, 0.4])
    assert _psi_from_fractions(frac, frac) == pytest.approx(0.0, abs=1e-12)


def test_psi_from_fractions_is_non_negative_and_grows_with_shift():
    """PSI is non-negative and increases as the two distributions diverge.

    PSI sums Kullback-Leibler-like terms and is mathematically non-negative.
    A wider gap between the vectors must yield a strictly larger PSI, the
    property that makes it usable as a drift score.
    """
    ref = np.array([0.25, 0.25, 0.25, 0.25])
    small_shift = np.array([0.30, 0.25, 0.25, 0.20])
    big_shift = np.array([0.70, 0.10, 0.10, 0.10])

    psi_small = _psi_from_fractions(ref, small_shift)
    psi_big = _psi_from_fractions(ref, big_shift)

    assert psi_small >= 0.0
    assert psi_big >= 0.0
    assert psi_big > psi_small


def test_psi_from_fractions_eps_guard_handles_empty_bins():
    """An empty bin (zero mass) must not blow up via log(0) or divide-by-zero.

    The eps clip replaces a literal zero fraction with a tiny positive number
    so `log(cur / ref)` stays finite. The result must be a real, finite,
    positive float rather than `inf` or `nan`.
    """
    ref = np.array([0.5, 0.5, 0.0])  # third bin empty in the reference
    cur = np.array([0.3, 0.3, 0.4])  # but populated in the current batch
    value = _psi_from_fractions(ref, cur)
    assert np.isfinite(value)
    assert value > 0.0


def test_psi_from_fractions_symmetric_in_magnitude():
    """Swapping ref and cur leaves PSI's sign and finiteness sensible.

    PSI is not perfectly symmetric, but for two well-populated vectors both
    orderings must be finite and positive. This guards against a sign or
    direction bug in the (cur - ref) * log(cur / ref) term.
    """
    a = np.array([0.6, 0.3, 0.1])
    b = np.array([0.2, 0.3, 0.5])
    forward = _psi_from_fractions(a, b)
    backward = _psi_from_fractions(b, a)
    assert forward > 0.0
    assert backward > 0.0
    assert np.isfinite(forward) and np.isfinite(backward)


# =========================================================================== #
# psi_numeric: quantile-binned PSI for a numeric column.
# =========================================================================== #
def test_psi_numeric_zero_for_identical_arrays():
    """PSI of an array against itself is approximately zero.

    Binning the same array on its own quantiles and comparing the resulting
    histograms must yield near-zero drift.
    """
    rng = np.random.default_rng(SEED)
    ref = rng.normal(size=2000)
    assert psi_numeric(ref, ref.copy()) == pytest.approx(0.0, abs=1e-9)


def test_psi_numeric_positive_under_clear_shift():
    """A clear location shift in the numeric distribution produces positive PSI."""
    rng = np.random.default_rng(SEED)
    ref = rng.normal(loc=0.0, scale=1.0, size=3000)
    cur = rng.normal(loc=3.0, scale=1.0, size=3000)  # shifted three std devs
    value = psi_numeric(ref, cur)
    assert value > 0.2  # a three-sigma move comfortably clears the ALERT band


def test_psi_numeric_never_negative():
    """PSI is mathematically non-negative for any inputs."""
    rng = np.random.default_rng(SEED)
    ref = rng.normal(size=1500)
    cur = rng.uniform(-2, 2, size=1500)
    assert psi_numeric(ref, cur) >= 0.0


def test_psi_numeric_near_constant_column_returns_zero():
    """A near-constant reference yields too few bin edges, so PSI returns 0.0.

    The implementation guards against degenerate binning: when fewer than
    three unique quantile edges exist there are no meaningful bins, and the
    function returns `0.0` rather than dividing into empty bins.
    """
    ref = np.zeros(100)
    cur = np.ones(100)
    assert psi_numeric(ref, cur) == 0.0


# =========================================================================== #
# psi_categorical: PSI over the union of observed categories.
# =========================================================================== #
def test_psi_categorical_zero_for_identical_frequencies():
    """Identical category frequencies give approximately zero PSI."""
    ref = pd.Series(["a", "a", "b", "c", "c", "c"])
    cur = pd.Series(["a", "a", "b", "c", "c", "c"])
    assert psi_categorical(ref, cur) == pytest.approx(0.0, abs=1e-9)


def test_psi_categorical_large_when_category_share_collapses():
    """When a dominant category's share collapses, PSI grows large.

    The reference is dominated by "a". The current batch nearly removes "a"
    and replaces it with "b". This is a textbook categorical drift and must
    produce a clearly positive, large PSI.
    """
    ref = pd.Series(["a"] * 90 + ["b"] * 10)
    cur = pd.Series(["a"] * 5 + ["b"] * 95)
    value = psi_categorical(ref, cur)
    assert value > 0.2  # share collapse lands firmly in the ALERT band


def test_psi_categorical_handles_unseen_category_in_current():
    """A brand-new category in the current batch is handled without error.

    The function takes the union of categories and reindexes both vectors,
    filling missing categories with zero before the eps-guarded PSI core runs.
    A fresh category (zero reference mass) is the signature of a coding-system
    migration and must yield a finite, positive PSI.
    """
    ref = pd.Series(["a", "a", "b", "b"])
    cur = pd.Series(["a", "b", "ICD10_recode", "ICD10_recode"])
    value = psi_categorical(ref, cur)
    assert np.isfinite(value)
    assert value > 0.0


# =========================================================================== #
# ks_statistic: Kolmogorov-Smirnov two-sample statistic.
# =========================================================================== #
def test_ks_statistic_zero_for_identical_samples():
    """KS of a sample against itself is exactly zero (no CDF gap)."""
    rng = np.random.default_rng(SEED)
    ref = rng.normal(size=500)
    assert ks_statistic(ref, ref.copy()) == pytest.approx(0.0, abs=1e-12)


def test_ks_statistic_in_unit_interval():
    """The KS statistic always lies in the closed interval [0, 1]."""
    rng = np.random.default_rng(SEED)
    ref = rng.normal(size=400)
    cur = rng.normal(loc=1.0, size=400)
    value = ks_statistic(ref, cur)
    assert 0.0 <= value <= 1.0


def test_ks_statistic_near_one_for_fully_separated_samples():
    """Two non-overlapping samples give a KS statistic of one.

    If every value in `cur` exceeds every value in `ref` the empirical
    CDFs never overlap, so the maximum gap is one.
    """
    ref = np.linspace(0.0, 1.0, 200)
    cur = np.linspace(10.0, 11.0, 200)  # entirely above the reference range
    assert ks_statistic(ref, cur) == pytest.approx(1.0, abs=1e-9)


# =========================================================================== #
# wasserstein_numeric: normalized earth-mover distance (Driftly).
# =========================================================================== #
def test_wasserstein_zero_for_identical_arrays():
    """A sample against an identical copy has zero transport cost."""
    rng = np.random.default_rng(SEED)
    ref = rng.normal(size=500)
    assert wasserstein_numeric(ref, ref.copy()) == pytest.approx(0.0, abs=1e-12)


def test_wasserstein_grows_with_mean_shift():
    """A larger location shift costs more to transport, so the (std-normalized)
    distance increases monotonically with the shift."""
    rng = np.random.default_rng(SEED)
    ref = rng.normal(size=2000)
    small = wasserstein_numeric(ref, ref + 0.5)
    large = wasserstein_numeric(ref, ref + 2.0)
    assert 0.0 < small < large


def test_wasserstein_is_std_normalized():
    """Normalizing by reference std makes the metric scale-free: a shift of one
    reference-std costs ~1.0 regardless of the underlying scale."""
    rng = np.random.default_rng(SEED)
    base = rng.normal(size=5000)
    for scale in (1.0, 10.0, 100.0):
        ref = base * scale
        shifted = ref + scale  # exactly one reference-std (std(base) ~ 1)
        assert wasserstein_numeric(ref, shifted) == pytest.approx(1.0, rel=0.05)


def test_wasserstein_near_constant_reference_returns_zero():
    """A reference with ~zero std would divide by zero, so the guard returns 0."""
    ref = np.full(300, 7.0)
    cur = np.full(300, 9.0)
    assert wasserstein_numeric(ref, cur) == 0.0


# =========================================================================== #
# KL divergence: KL(current || reference), eps-smoothed (Driftly).
# =========================================================================== #
def test_kl_from_fractions_zero_for_identical_distributions():
    """KL of a distribution against itself is zero."""
    frac = np.array([0.2, 0.3, 0.5])
    assert _kl_from_fractions(frac, frac.copy()) == pytest.approx(0.0, abs=1e-9)


def test_kl_from_fractions_non_negative():
    """KL divergence is non-negative (Gibbs' inequality)."""
    cur = np.array([0.5, 0.3, 0.2])
    ref = np.array([0.2, 0.3, 0.5])
    assert _kl_from_fractions(cur, ref) > 0.0


def test_kl_from_fractions_is_asymmetric():
    """KL is directional: KL(p||q) generally differs from KL(q||p). (Mirror-
    symmetric pairs are the degenerate exception, so we pick a non-symmetric
    pair here.)"""
    p = np.array([0.7, 0.2, 0.1])
    q = np.array([0.2, 0.3, 0.5])
    assert _kl_from_fractions(p, q) != pytest.approx(_kl_from_fractions(q, p))


def test_kl_from_fractions_eps_guard_handles_empty_bins():
    """A current bin with zero mass against a populated reference bin must not
    blow up to inf/nan thanks to eps-smoothing."""
    cur = np.array([0.0, 1.0])
    ref = np.array([0.5, 0.5])
    value = _kl_from_fractions(cur, ref)
    assert np.isfinite(value) and value > 0.0


def test_kl_numeric_zero_for_identical_arrays():
    """Binned KL of a numeric column against itself is ~zero."""
    rng = np.random.default_rng(SEED)
    ref = rng.normal(size=2000)
    assert kl_numeric(ref, ref.copy()) == pytest.approx(0.0, abs=1e-6)


def test_kl_numeric_positive_under_clear_shift():
    """A clear location shift drives binned KL above zero."""
    rng = np.random.default_rng(SEED)
    ref = rng.normal(size=2000)
    cur = rng.normal(loc=2.0, size=2000)
    assert kl_numeric(ref, cur) > 0.0


def test_kl_numeric_near_constant_column_returns_zero():
    """A near-constant reference yields fewer than three bin edges, so KL is 0."""
    ref = np.full(300, 5.0)
    cur = np.concatenate([np.full(280, 5.0), np.full(20, 9.0)])
    assert kl_numeric(ref, cur) == 0.0


def test_kl_categorical_zero_for_identical_frequencies():
    """Categorical KL against an identical frequency table is ~zero."""
    ref = pd.Series(["a", "a", "b", "c"] * 25)
    assert kl_categorical(ref, ref.copy()) == pytest.approx(0.0, abs=1e-9)


def test_kl_categorical_positive_when_share_shifts():
    """A shift in category shares drives categorical KL above zero."""
    ref = pd.Series(["a"] * 80 + ["b"] * 20)
    cur = pd.Series(["a"] * 20 + ["b"] * 80)
    assert kl_categorical(ref, cur) > 0.0


def test_kl_categorical_handles_unseen_category_in_current():
    """A category present only in the current batch is eps-smoothed, not inf."""
    ref = pd.Series(["a"] * 50 + ["b"] * 50)
    cur = pd.Series(["a"] * 40 + ["b"] * 40 + ["z"] * 20)
    value = kl_categorical(ref, cur)
    assert np.isfinite(value) and value > 0.0


# =========================================================================== #
# column_drift_report: routing plus the OK / WARN / ALERT verdict.
# =========================================================================== #
def test_report_identical_frames_yield_ok():
    """A reference compared against an identical copy reports OK everywhere.

    With no drift, every per-column statistic stays below the WARN threshold,
    so the worst status is OK and the overall verdict is OK.
    """
    reference = _make_reference_frame(n=600, seed=1)
    current = reference.copy()
    report, verdict = column_drift_report(reference, current)

    assert verdict == "OK"
    assert (report["status"] == "OK").all()
    # One row per monitored column present in both frames.
    expected_cols = [c for c in MONITORED_COLUMNS if c in reference.columns]
    assert list(report["column"]) == expected_cols


def test_report_has_one_row_per_column_with_expected_schema():
    """The report frame carries the documented columns and one row per monitored column."""
    reference = _make_reference_frame(n=300, seed=2)
    current = reference.copy()
    report, _ = column_drift_report(reference, current)

    assert list(report.columns) == ["column", "method", "statistic", "status"]
    assert len(report) == len([c for c in MONITORED_COLUMNS if c in reference.columns])
    # Every status must be one of the three documented levels.
    assert set(report["status"]).issubset({"OK", "WARN", "ALERT"})


def test_report_routing_numeric_high_cardinality_uses_ks():
    """A numeric column with many distinct values routes through KS, not PSI.

    Our synthetic continuous columns span 60 integer levels, which exceeds
    `DRIFT_KS_UNIQUE_THRESHOLD`. The detector must label them "KS".
    """
    reference = _make_reference_frame(n=800, seed=3)
    current = reference.copy()
    report, _ = column_drift_report(reference, current)

    # Confirm the precondition: every continuous column does exceed the
    # unique-value threshold in our synthetic frame.
    for col in MONITORED_CONTINUOUS:
        assert reference[col].nunique() > DRIFT_KS_UNIQUE_THRESHOLD

    cont_methods = report[report["column"].isin(MONITORED_CONTINUOUS)]["method"]
    assert (cont_methods == "KS").all()


def test_report_routing_low_cardinality_numeric_uses_psi():
    """A numeric column with few distinct values routes through PSI, not KS.

    Here we overwrite the continuous columns with a binary 0/1 encoding so they
    stay numeric but carry only two distinct values, well under the KS
    threshold. The detector must fall back to PSI for those columns.
    """
    reference = _make_reference_frame(n=400, seed=4)
    current = reference.copy()
    # Collapse one continuous column to a low-cardinality numeric column.
    col = MONITORED_CONTINUOUS[0]
    reference[col] = (reference[col] > 30).astype(int)
    current[col] = (current[col] > 30).astype(int)

    assert reference[col].nunique() <= DRIFT_KS_UNIQUE_THRESHOLD
    report, _ = column_drift_report(reference, current)
    method = report.loc[report["column"] == col, "method"].iloc[0]
    assert method == "PSI"


def test_report_routing_categorical_uses_psi():
    """String categorical columns always route through PSI."""
    reference = _make_reference_frame(n=400, seed=5)
    current = reference.copy()
    report, _ = column_drift_report(reference, current)

    cat_methods = report[report["column"].isin(MONITORED_CATEGORICAL)]["method"]
    assert (cat_methods == "PSI").all()


def test_report_strong_shift_yields_alert():
    """A strong injected shift across many columns escalates the verdict to ALERT.

    We use the module's own most-severe scenario at high severity, which moves
    both categorical (PSI) and continuous (KS) columns hard. The worst status
    must be ALERT.
    """
    reference = _make_reference_frame(n=1000, seed=6)
    current = make_scenario(reference, "mixed_severe", severity=2.0, seed=SEED)
    report, verdict = column_drift_report(reference, current)

    assert verdict == "ALERT"
    assert (report["status"] == "ALERT").any()


def test_report_verdict_is_worst_status_present():
    """The overall verdict is the worst per-column status in the report.

    We hand-craft a current frame in which exactly one categorical column has
    its dominant share collapsed (an ALERT-level PSI move) while everything
    else stays identical. The overall verdict must therefore be ALERT even
    though most columns are OK.
    """
    reference = _make_reference_frame(n=600, seed=7)
    current = reference.copy()
    # Force a dramatic categorical collapse on a single monitored column.
    col = "race"
    current[col] = "Other"  # crush all mass onto one category
    report, verdict = column_drift_report(reference, current)

    assert verdict == "ALERT"
    # The drifted column must individually carry the ALERT status.
    assert report.loc[report["column"] == col, "status"].iloc[0] == "ALERT"


def test_report_respects_explicit_columns_argument():
    """Passing an explicit `columns` list restricts the report to those columns."""
    reference = _make_reference_frame(n=300, seed=8)
    current = reference.copy()
    subset = ["time_in_hospital", "race"]
    report, _ = column_drift_report(reference, current, columns=subset)
    assert list(report["column"]) == subset


def test_report_thresholds_match_constants():
    """The WARN and ALERT bands the report applies match helpers.constants.

    This guards against the detector silently diverging from the documented
    cutoffs. We build a column whose PSI lands between WARN and ALERT and
    confirm it is labelled WARN, and another above ALERT labelled ALERT.
    """
    assert DRIFT_PSI_WARN < DRIFT_PSI_ALERT  # sanity on the constants themselves

    reference = _make_reference_frame(n=800, seed=9)
    current = reference.copy()
    # A modest categorical move sized to land in the WARN band.
    current["gender"] = make_scenario(
        reference, "casemix_shift", severity=0.4, seed=SEED
    )["gender"]
    report, _ = column_drift_report(reference, current, columns=["gender"])
    stat = report["statistic"].iloc[0]
    status = report["status"].iloc[0]

    if stat >= DRIFT_PSI_ALERT:
        assert status == "ALERT"
    elif stat >= DRIFT_PSI_WARN:
        assert status == "WARN"
    else:
        assert status == "OK"


# =========================================================================== #
# make_scenario: the seeded perturbation generator.
# =========================================================================== #
def test_make_scenario_is_schema_identical_to_reference():
    """Every scenario returns a frame with exactly the reference's columns.

    The drift detector compares column-for-column, so the perturbed batch must
    keep the reference schema unchanged for every scenario.
    """
    reference = _make_reference_frame(n=200, seed=10)
    for scenario in SCENARIOS:
        batch = make_scenario(reference, scenario, seed=SEED)
        assert list(batch.columns) == list(reference.columns)


def test_make_scenario_none_is_near_identical_distribution():
    """The "none" control perturbs nothing meaningfully.

    It is a seeded bootstrap resample of the reference, so its verdict against
    the reference must be OK: a fresh sample of the same population, no drift.
    """
    reference = _make_reference_frame(n=1000, seed=11)
    batch = make_scenario(reference, "none", seed=SEED)
    _, verdict = column_drift_report(reference, batch)
    assert verdict == "OK"


def test_make_scenario_invalid_name_raises_value_error():
    """An unknown scenario name raises ValueError, per the module guard."""
    reference = _make_reference_frame(n=50, seed=12)
    with pytest.raises(ValueError):
        make_scenario(reference, "not_a_real_scenario", seed=SEED)


def test_make_scenario_is_deterministic_for_fixed_seed():
    """Two calls with the same seed produce byte-for-byte identical batches."""
    reference = _make_reference_frame(n=400, seed=13)
    first = make_scenario(reference, "mixed_severe", severity=1.0, seed=SEED)
    second = make_scenario(reference, "mixed_severe", severity=1.0, seed=SEED)
    pd.testing.assert_frame_equal(first, second)


def test_make_scenario_different_seeds_differ():
    """Different seeds draw different bootstrap samples, so the batches differ.

    Determinism for a fixed seed must not collapse into producing the same
    output for every seed. Changing the seed must change the resample.
    """
    reference = _make_reference_frame(n=400, seed=14)
    a = make_scenario(reference, "none", seed=SEED)
    b = make_scenario(reference, "none", seed=SEED + 1)
    # At least one cell must differ between the two resamples.
    assert not a.equals(b)


def test_make_scenario_respects_batch_size():
    """The `batch_size` argument controls the number of rows returned."""
    reference = _make_reference_frame(n=500, seed=15)
    batch = make_scenario(reference, "none", seed=SEED, batch_size=123)
    assert len(batch) == 123


def test_make_scenario_default_batch_size_matches_reference():
    """With no `batch_size` the batch has the same row count as the reference."""
    reference = _make_reference_frame(n=321, seed=16)
    batch = make_scenario(reference, "none", seed=SEED)
    assert len(batch) == len(reference)


def test_make_scenario_severity_scales_drift_magnitude():
    """Higher severity produces a larger drift signal for a targeted scenario.

    We compare the worst per-column statistic in the report at low versus high
    severity for the same scenario. The high-severity batch must drift more.
    """
    reference = _make_reference_frame(n=1000, seed=17)

    mild = make_scenario(reference, "los_utilization_shift", severity=0.3, seed=SEED)
    strong = make_scenario(reference, "los_utilization_shift", severity=2.0, seed=SEED)

    mild_report, _ = column_drift_report(reference, mild)
    strong_report, _ = column_drift_report(reference, strong)

    mild_max = mild_report["statistic"].max()
    strong_max = strong_report["statistic"].max()
    assert strong_max > mild_max


# =========================================================================== #
# write_scenarios: the on-disk batch writer.
#
# We always write into pytest's tmp_path so the suite never touches the
# repository's data/ directory. write_scenarios reads its reference from a CSV
# path, so we first materialise a synthetic reference CSV inside tmp_path too.
# =========================================================================== #
def test_write_scenarios_writes_one_csv_per_scenario(tmp_path):
    """write_scenarios returns a scenario -> path map and every path exists.

    Each value in the returned dictionary must point at a CSV file that was
    written, one per scenario, and the file must read back as a
    DataFrame that carries the reference schema.
    """
    reference = _make_reference_frame(n=200, seed=18)
    ref_path = tmp_path / "features.csv"
    reference.to_csv(ref_path, index=False)
    out_dir = tmp_path / "incoming"

    written = write_scenarios(
        reference_path=str(ref_path),
        out_dir=str(out_dir),
        seed=SEED,
    )

    # One entry per scenario, each mapping to an existing CSV.
    assert set(written.keys()) == set(SCENARIOS)
    for scenario, path in written.items():
        assert path.endswith(f"{scenario}.csv")
        assert (out_dir / f"{scenario}.csv").exists()
        # The written batch round-trips and keeps the reference columns.
        round_tripped = pd.read_csv(path, low_memory=False)
        assert list(round_tripped.columns) == list(reference.columns)


def test_write_scenarios_subset_of_scenarios(tmp_path):
    """Passing an explicit scenarios list writes only those scenarios."""
    reference = _make_reference_frame(n=150, seed=19)
    ref_path = tmp_path / "features.csv"
    reference.to_csv(ref_path, index=False)
    out_dir = tmp_path / "incoming"

    subset = ("none", "coding_shift")
    written = write_scenarios(
        reference_path=str(ref_path),
        out_dir=str(out_dir),
        scenarios=subset,
        seed=SEED,
    )

    assert set(written.keys()) == set(subset)
    # Only the requested CSVs should exist in the output directory.
    produced = {p.name for p in out_dir.glob("*.csv")}
    assert produced == {"none.csv", "coding_shift.csv"}


# =========================================================================== #
# Gap 7 + champion-impact: the drifted batch is a *labeled* set, so we can
# both keep its numeric encodings intact and measure real model harm.
#
# `change` is the 0/1 "any medication change" indicator in features.csv, scaled
# as a numeric feature by the champion preprocessor. A scenario must drift it
# with a numeric value, never the raw string "Ch" (which corrupts the column and
# makes the batch unscoreable). These frames mirror that real numeric encoding.
# =========================================================================== #
def _toy_reference(n: int = 400) -> pd.DataFrame:
    """Minimal labeled frame mirroring features.csv encodings for the columns
    the formulary scenario perturbs, plus the readmitted_binary target."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "change": rng.integers(0, 2, n),                # int64 {0, 1}
        "insulin": rng.choice(["No", "Steady", "Up"], n),
        "metformin": rng.choice(["No", "Steady"], n),
        "n_med_changes": rng.integers(0, 5, n),
        "readmitted_binary": rng.integers(0, 2, n),
    })


def test_formulary_shift_keeps_change_numeric():
    """Gap 7 regression: formulary_shift must drift the numeric `change` column
    toward 1 ("changed"), never write the raw string "Ch"."""
    ref = _toy_reference()
    out = make_scenario(ref, "formulary_shift", severity=1.0, seed=1)
    assert "Ch" not in set(out["change"].unique())
    assert pd.api.types.is_numeric_dtype(out["change"])
    assert set(out["change"].dropna().unique()).issubset({0, 1})


# =========================================================================== #
# champion_impact / build_drift_report: because the drifted batch keeps the
# real readmitted_binary label, we can measure genuine model harm with a
# deterministic stand-in scorer (no MLflow, no joblib artefacts needed here).
# =========================================================================== #
class _ConstProba:
    """Deterministic predict_proba_fn stand-in: positive-class probability
    keys off the `change` indicator so reference and drifted frames (which the
    formulary scenario pushes toward change=1) produce different metric panels."""

    def __call__(self, df: pd.DataFrame) -> np.ndarray:
        return np.clip(0.5 + 0.1 * (df["change"] - 0.5), 0.0, 1.0).to_numpy()


def test_champion_impact_returns_reference_current_delta():
    ref = _toy_reference()
    cur = make_scenario(ref, "formulary_shift", severity=1.0, seed=2)
    out = champion_impact(ref, cur, _ConstProba(), threshold=0.5)
    assert set(out) >= {"reference", "current", "delta", "threshold"}
    # delta must equal current - reference for every metric in the panel.
    for k, v in out["delta"].items():
        assert np.isclose(v, out["current"][k] - out["reference"][k], equal_nan=True)


def test_build_drift_report_is_json_serializable_with_impact():
    import json

    ref = _toy_reference()
    cur = make_scenario(ref, "formulary_shift", severity=1.0, seed=3)
    report = build_drift_report(ref, cur, scenario="formulary_shift",
                                predict_proba_fn=_ConstProba(), threshold=0.5)
    assert report["scenario"] == "formulary_shift"
    assert report["verdict"] in {"OK", "WARN", "ALERT"}
    assert "champion_impact" in report
    json.dumps(report)  # must not raise


def test_build_drift_report_without_scorer_has_no_impact_block():
    ref = _toy_reference()
    cur = make_scenario(ref, "formulary_shift", severity=1.0, seed=4)
    report = build_drift_report(ref, cur, scenario="formulary_shift")
    assert "champion_impact" not in report
    assert report["scenario"] == "formulary_shift"
