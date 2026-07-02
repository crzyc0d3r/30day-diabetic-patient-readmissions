"""Unit tests for the PURE feature-engineering helpers in helpers/feature_eng.py.

WHAT this module does
=====================
helpers/feature_eng.py is a collection of small, side-effect-light functions
that each bolt one conceptual group of derived columns onto a DataFrame:
per-day ratios, cross-product interactions, discharge/age interactions,
polynomial (squared) terms, and binary threshold flags. There is also a single
`map_icd9` scalar function that collapses an ICD-9 diagnosis string to a
chapter bucket, and an `add_all_interactions` orchestrator that runs every
group in the canonical notebook order.

WHY these tests look the way they do
====================================
The functions are PURE in the useful sense: their output is a deterministic
function of the input column values, with no randomness, no file IO, and no
hidden global state. That lets us build a TINY DataFrame with explicit, known
integer values and HAND-COMPUTE every expected output cell. We do not lean on
the shared `raw_cohort_df` fixture because feature_eng expects
already-engineered numeric columns (service_utilization, total_prior_visits,
discharge_group, age_numeric) that the raw cohort does not carry. A purpose-built
fixture with round numbers makes the expected values obvious to a reader and
keeps the arithmetic checkable by the eye.

Throughout, we also assert against the module-level constant tuples
(PER_DAY_FEATURES, CROSS_PRODUCT_FEATURES, DISCHARGE_AGE_FEATURES,
POLYNOMIAL_FEATURES, THRESHOLD_FLAG_FEATURES, ALL_INTERACTION_FEATURES) rather
than re-typing column-name string literals. If the module renames a feature, the
test follows it automatically instead of going stale.
"""

from __future__ import annotations

import pandas as pd
import pytest

from helpers.feature_eng import (
    ALL_INTERACTION_FEATURES,
    CROSS_PRODUCT_FEATURES,
    DISCHARGE_AGE_FEATURES,
    PER_DAY_FEATURES,
    POLYNOMIAL_FEATURES,
    THRESHOLD_FLAG_FEATURES,
    add_all_interactions,
    add_cross_product_interactions,
    add_discharge_and_age_interactions,
    add_per_day_ratios,
    add_polynomial_features,
    add_threshold_flags,
    map_icd9,
)


# ===========================================================================
# Local fixture
# ===========================================================================
@pytest.fixture
def engineered_df() -> pd.DataFrame:
    """A tiny three-row frame with EVERY column the helpers read, using round
    integers so expected feature values can be computed by hand.

    The three rows are deliberately chosen to exercise edge cases:

    Row 0: a "high utiliser" with time_in_hospital == 0. The zero length of stay
            is the important case: the per-day ratios clip length of stay at a
            floor of 1, so division must NOT blow up. With LOS clipped to 1, the
            ratios equal the raw numerators.
    Row 1: a moderate-utilisation "Home" discharge, young enough (age 40) and
            with enough inpatient visits (3) to trip the young_high_utilizer
            flag.
    Row 2: a low-utilisation "Transfer" discharge, old (age 70) so the
            young_high_utilizer flag must stay 0, and below every upper-tail
             threshold, so the binary flags are mostly 0.

    Every column carries a known value, so we can assert exact equality rather
    than approximate ranges.
    """
    return pd.DataFrame(
        {
            # Counts read by the per-day, cross-product, polynomial, and flag
            # helpers.
            "time_in_hospital": [0, 5, 2],
            "num_lab_procedures": [40, 20, 10],
            "num_procedures": [6, 4, 1],
            "num_medications": [20, 10, 5],
            "service_utilization": [12, 8, 3],
            "number_inpatient": [4, 3, 1],
            "number_emergency": [2, 0, 1],
            "number_diagnoses": [9, 7, 3],
            "total_prior_visits": [5, 3, 2],
            # Columns read-only by the discharge/age helper.
            "discharge_group": ["Home", "Home", "Transfer"],
            "age_numeric": [45, 40, 70],
        }
    )


# ===========================================================================
# map_icd9
# ===========================================================================
# Each tuple is (input_code, expected_bucket). We parametrize a wide spread of
# inputs to pin down every branch of the function:
#   - missing or sentinel values collapse to "Unknown"
#   - codes starting with V or v collapse to "V-codes"
#   - codes starting with E or e collapse to "E-codes"
#   - numeric strings (and floats) are zero-padded to a 3-digit chapter
#   - anything non-numeric that is not V/E/Unknown falls through to "Other"
@pytest.mark.parametrize(
    "code, expected",
    [
        # === Missing or sentinel to "Unknown" ===
        (None, "Unknown"),                # actual Python None is NaN-like
        (float("nan"), "Unknown"),        # numpy/pandas NaN float
        ("?", "Other"),                   # the "?" sentinel is NOT special-cased
        ("Unknown", "Unknown"),           # literal "Unknown" string short-circuits
        ("", "Unknown"),                  # empty string is treated as missing
        ("   ", "Unknown"),               # whitespace-only strips to empty -> Unknown
        # === V-codes ===
        ("V57", "V-codes"),               # rehabilitation V-code
        ("v45", "V-codes"),               # lowercase v still buckets as V-codes
        ("V", "V-codes"),                 # bare V prefix
        # === E-codes ===
        ("E930", "E-codes"),              # external-cause E-code
        ("e849", "E-codes"),              # lowercase e still buckets as E-codes
        # === Numeric strings to zero-padded 3-digit chapter ===
        ("250", "250"),                   # already three digits
        ("8", "008"),                     # single digit zero-padded to width 3
        ("42", "042"),                    # two digits zero-padded
        ("250.83", "250"),               # decimal portion is truncated by int(float(...))
        ("428", "428"),
        ("0", "000"),                     # zero pads to "000"
        # === Float inputs ===
        (250.83, "250"),                  # a real float, not a string
        (8.0, "008"),                     # float that is integral
        # === Other (non-numeric, non V/E) ===
        ("ABC", "Other"),                 # alphabetic rubbish falls through
        ("12X", "Other"),                 # starts numeric but is not parseable
    ],
)
def test_map_icd9_buckets(code, expected):
    """map_icd9 collapses each code to the documented chapter or sentinel.

    WHY: the mapping must be deterministic and identical between the research
    notebook and the production inference path, so we lock down every branch
    with hand-chosen representatives. The zero-padding behaviour
    (`f"{int(float(s)):03d}"`) is the easiest to get wrong, so single and
    double-digit codes are tested explicitly.
    """
    assert map_icd9(code) == expected


def test_map_icd9_returns_str():
    """Every return value is a plain str, never a number or NaN.

    WHY: downstream one-hot encoding groups on string categories, so a stray int
    chapter would create a spurious extra category.
    """
    for code in ("250", "V57", "E930", "?", None, 8.0):
        assert isinstance(map_icd9(code), str)


# ===========================================================================
# add_per_day_ratios
# ===========================================================================
def test_add_per_day_ratios_values(engineered_df):
    """Per-day ratios divide each volume count by length of stay (floored at 1).

    Hand computation, using LOS = max(time_in_hospital, 1):
      Row 0: LOS = max(0, 1) = 1  -> ratios equal the raw numerators
              labs_per_day        = 40 / 1 = 40
              procedures_per_day  =  6 / 1 =  6
              meds_per_day        = 20 / 1 = 20
              utilisation_per_day = 12 / 1 = 12
      Row 1: LOS = 5
              labs_per_day        = 20 / 5 =  4
              procedures_per_day  =  4 / 5 =  0.8
              meds_per_day        = 10 / 5 =  2
              utilisation_per_day =  8 / 5 =  1.6
      Row 2: LOS = 2
              labs_per_day        = 10 / 2 =  5
              procedures_per_day  =  1 / 2 =  0.5
              meds_per_day        =  5 / 2 =  2.5
              utilisation_per_day =  3 / 2 =  1.5
    """
    out = add_per_day_ratios(engineered_df)

    # The function returns a DataFrame carrying exactly the documented columns.
    assert isinstance(out, pd.DataFrame)
    for col in PER_DAY_FEATURES:
        assert col in out.columns

    assert list(out["labs_per_day"]) == [40.0, 4.0, 5.0]
    assert list(out["procedures_per_day"]) == [6.0, 0.8, 0.5]
    assert list(out["meds_per_day"]) == [20.0, 2.0, 2.5]
    assert list(out["utilization_per_day"]) == [12.0, 1.6, 1.5]


def test_add_per_day_ratios_zero_los_does_not_blow_up(engineered_df):
    """The time_in_hospital == 0 row must produce finite ratios, not inf or NaN.

    WHY: this is the whole reason the length of stay is clipped at a floor of 1.
    Row 0 has LOS 0, so without the clip every ratio would be a divide-by-zero.
    We assert the values are finite AND equal to the raw numerators (clip to 1).
    """
    out = add_per_day_ratios(engineered_df)
    row0 = out.iloc[0]
    for col in PER_DAY_FEATURES:
        assert pd.notna(row0[col])
        assert row0[col] != float("inf")
    # With LOS clipped to 1, the row-0 ratio equals the numerator itself.
    assert row0["labs_per_day"] == 40.0
    assert row0["utilization_per_day"] == 12.0


# ===========================================================================
# add_cross_product_interactions
# ===========================================================================
def test_add_cross_product_interactions_values(engineered_df):
    """Six pairwise products, four of them anchored on number_inpatient.

    Hand computation per row (inpatient, emergency, diagnoses, meds, time):
      Row 0: inp=4 emg=2 diag=9 meds=20 time=0
      Row 1: inp=3 emg=0 diag=7 meds=10 time=5
      Row 2: inp=1 emg=1 diag=3 meds=5 time=2

      inpatient_x_emergency = inp * emg -> [8, 0, 1]
      inpatient_x_diagnoses = inp * diag -> [36, 21, 3]
      inpatient_x_meds = inp * meds -> [80, 30, 5]
      inpatient_x_time = inp * time -> [0, 15, 2]
      emergency_x_diagnoses = emg * diag -> [18, 0, 3]
      meds_x_diagnoses = meds * diag-> [180, 70, 15]
    """
    out = add_cross_product_interactions(engineered_df)

    assert isinstance(out, pd.DataFrame)
    for col in CROSS_PRODUCT_FEATURES:
        assert col in out.columns

    assert list(out["inpatient_x_emergency"]) == [8, 0, 1]
    assert list(out["inpatient_x_diagnoses"]) == [36, 21, 3]
    assert list(out["inpatient_x_meds"]) == [80, 30, 5]
    assert list(out["inpatient_x_time"]) == [0, 15, 2]
    assert list(out["emergency_x_diagnoses"]) == [18, 0, 3]
    assert list(out["meds_x_diagnoses"]) == [180, 70, 15]


# ===========================================================================
# add_discharge_and_age_interactions
# ===========================================================================
def test_add_discharge_and_age_interactions_values(engineered_df):
    """Discharge-indicator products, an age product, and the young-utiliser flag.

    Hand computation per row (inpatient, discharge_group, age_numeric):
      Row 0: inp=4 grp="Home" age=45
      Row 1: inp=3 grp="Home" age=40
      Row 2: inp=1 grp="Transfer" age=70

      inpatient_x_discharge_home     = inp * (grp == "Home")      -> [4, 3, 0]
      inpatient_x_discharge_transfer = inp * (grp == "Transfer")  -> [0, 0, 1]
      inpatient_x_discharge_snf      = inp * (grp == "SNF/Rehab") -> [0, 0, 0]
      inpatient_x_age                = inp * age                  -> [180, 120, 70]
      young_high_utiliser = (age < 50) AND (inp >= 2):
          Row 0: 45 < 50 True,  inp 4 >= 2 True  -> 1
          Row 1: 40 < 50 True,  inp 3 >= 2 True  -> 1
          Row 2: 70 < 50 False                   -> 0
    """
    out = add_discharge_and_age_interactions(engineered_df)

    assert isinstance(out, pd.DataFrame)
    for col in DISCHARGE_AGE_FEATURES:
        assert col in out.columns

    assert list(out["inpatient_x_discharge_home"]) == [4, 3, 0]
    assert list(out["inpatient_x_discharge_transfer"]) == [0, 0, 1]
    assert list(out["inpatient_x_discharge_snf"]) == [0, 0, 0]
    assert list(out["inpatient_x_age"]) == [180, 120, 70]
    assert list(out["young_high_utilizer"]) == [1, 1, 0]


def test_discharge_snf_indicator_is_all_zero_when_level_absent(engineered_df):
    """An absent discharge level yields an all-zero indicator, not an error.

    WHY: the docstring promises the helper is safe to call when a level was
    dropped during cleaning. None of our rows are "SNF/Rehab", so that column
    must be uniformly 0 rather than raising a KeyError.
    """
    out = add_discharge_and_age_interactions(engineered_df)
    assert set(out["inpatient_x_discharge_snf"]) == {0}


# ===========================================================================
# add_polynomial_features
# ===========================================================================
def test_add_polynomial_features_values(engineered_df):
    """Three squared terms computed element-wise on the raw counts.

    Hand computation per row:
      inpatient_sq = number_inpatient ** 2 -> [16, 9, 1]
      emergency_sq = number_emergency ** 2 -> [4, 0, 1]
      total_visits_sq = total_prior_visits ** 2 -> [25, 9, 4]
    """
    out = add_polynomial_features(engineered_df)

    assert isinstance(out, pd.DataFrame)
    for col in POLYNOMIAL_FEATURES:
        assert col in out.columns

    # Verify the square relationship explicitly against the source columns.
    assert list(out["inpatient_sq"]) == list(out["number_inpatient"] ** 2)
    assert list(out["emergency_sq"]) == list(out["number_emergency"] ** 2)
    assert list(out["total_visits_sq"]) == list(out["total_prior_visits"] ** 2)

    # And against the hand-computed constants.
    assert list(out["inpatient_sq"]) == [16, 9, 1]
    assert list(out["emergency_sq"]) == [4, 0, 1]
    assert list(out["total_visits_sq"]) == [25, 9, 4]


# ===========================================================================
# add_threshold_flags
# ===========================================================================
def test_add_threshold_flags_values(engineered_df):
    """Six binary upper-tail flags, each a >= or > cut on one count column.

    Hand computation per row, using the source values:
      total_prior_visits = [5, 3, 2] number_inpatient = [4, 3, 1]
      number_emergency = [2, 0, 1] time_in_hospital = [0, 5, 2]
      num_medications = [20, 10, 5] number_diagnoses = [9, 7, 3]

      high_utilizer = total_prior_visits >= 3 -> [1, 1, 0]
      frequent_inpatient= number_inpatient >= 2 -> [1, 1, 0]
      any_emergency = number_emergency > 0 -> [1, 0, 1]
      long_stay = time_in_hospital >= 7 -> [0, 0, 0]
      many_meds = num_medications >= 15-> [1, 0, 0]
      many_diagnoses = number_diagnoses >= 7 -> [1, 1, 0]
    """
    out = add_threshold_flags(engineered_df)

    assert isinstance(out, pd.DataFrame)
    for col in THRESHOLD_FLAG_FEATURES:
        assert col in out.columns

    assert list(out["high_utilizer"]) == [1, 1, 0]
    assert list(out["frequent_inpatient"]) == [1, 1, 0]
    assert list(out["any_emergency"]) == [1, 0, 1]
    assert list(out["long_stay"]) == [0, 0, 0]
    assert list(out["many_meds"]) == [1, 0, 0]
    assert list(out["many_diagnoses"]) == [1, 1, 0]


def test_threshold_flags_are_strictly_binary(engineered_df):
    """Every flag column holds only the integers 0 and 1, nothing else.

    WHY: these are meant to be clean step signals for the linear models. A stray
    boolean or float would change how downstream scalers treat them.
    """
    out = add_threshold_flags(engineered_df)
    for col in THRESHOLD_FLAG_FEATURES:
        assert set(out[col].unique()).issubset({0, 1})


# ===========================================================================
# add_all_interactions (the orchestrator)
# ===========================================================================
def test_add_all_interactions_adds_every_feature(engineered_df):
    """The orchestrator applies all five groups and yields every documented column.

    WHY: production callers use this one function to build the full interaction
    block. We assert that EVERY name in ALL_INTERACTION_FEATURES is present in
    the result, which transitively proves all five sub-helpers ran.
    """
    out = add_all_interactions(engineered_df)

    assert isinstance(out, pd.DataFrame)
    for col in ALL_INTERACTION_FEATURES:
        assert col in out.columns


def test_add_all_interactions_matches_individual_calls(engineered_df):
    """Running the groups one by one yields the same values as the orchestrator.

    WHY: this guards the documented "canonical order" contract. Because every
    helper only ADDS columns (it never reads a column another helper writes), the
    orchestrator output must be cell-for-cell identical to applying the groups
    sequentially ourselves. We compute the reference on a separate copy so the
    two pipelines cannot share a mutation.
    """
    reference = engineered_df.copy()
    reference = add_per_day_ratios(reference)
    reference = add_cross_product_interactions(reference)
    reference = add_discharge_and_age_interactions(reference)
    reference = add_polynomial_features(reference)
    reference = add_threshold_flags(reference)

    out = add_all_interactions(engineered_df.copy())

    # Compare only the engineered columns, in a stable order, on both frames.
    cols = list(ALL_INTERACTION_FEATURES)
    pd.testing.assert_frame_equal(out[cols], reference[cols])


# ===========================================================================
# Module-level constant tuples
# ===========================================================================
def test_all_interaction_features_is_the_union_of_groups():
    """ALL_INTERACTION_FEATURES is exactly the ordered concatenation of groups.

    WHY: the module builds ALL_INTERACTION_FEATURES by adding the five group
    tuples together. We assert that union explicitly so a future edit that
    forgets to register a new group in the master tuple is caught.
    """
    expected = (
        PER_DAY_FEATURES
        + CROSS_PRODUCT_FEATURES
        + DISCHARGE_AGE_FEATURES
        + POLYNOMIAL_FEATURES
        + THRESHOLD_FLAG_FEATURES
    )
    assert ALL_INTERACTION_FEATURES == expected


def test_constant_tuples_have_expected_sizes():
    """Each group tuple carries the documented number of feature names.

    WHY: a quick structural smoke check. The docstrings promise 4 per-day, 6
    cross-product, discharge/age 5, 3 polynomials, and 6 threshold features, for
    24 total. Locking the counts catches an accidental duplicate or drop.
    """
    assert len(PER_DAY_FEATURES) == 4
    assert len(CROSS_PRODUCT_FEATURES) == 6
    assert len(DISCHARGE_AGE_FEATURES) == 5
    assert len(POLYNOMIAL_FEATURES) == 3
    assert len(THRESHOLD_FLAG_FEATURES) == 6
    assert len(ALL_INTERACTION_FEATURES) == 24


def test_constant_tuples_have_no_duplicate_names():
    """No feature name appears twice across all groups combined.

    WHY: duplicate names would mean two helpers fight over one DataFrame column,
    silently overwriting each other. The set of all names must be as large as
    the tuple itself.
    """
    assert len(set(ALL_INTERACTION_FEATURES)) == len(ALL_INTERACTION_FEATURES)
