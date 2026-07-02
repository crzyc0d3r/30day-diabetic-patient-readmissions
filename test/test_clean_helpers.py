"""Unit tests for the PURE cleaning helpers in `helpers/clean_helpers.py`.

These helpers implement NB02 section 2.8 of the medi-watch pipeline. They are
"pure" in the operational sense the module docstring promises: no file I/O and
no mutation of globals. They are NOT pure in the functional sense. Every helper
mutates the DataFrame it is handed (`df[col] = ...` or `df.drop(...)`) AND
returns that same object so callers can chain them. These tests pin down that
mutate-and-return contract, the exact new column names and dtypes each helper
produces, and the train-mask leakage guard that keeps validation and test row
frequencies from steering which columns and categories survive cleaning.

Note on the split program these tests protect: the rollup helpers accept an
optional `train_mask` so the top-K category computation only ever sees train
rows. If validation or test frequencies leaked into that decision, the model
would gain a subtle information advantage it will not have in production.
Several tests below construct data where a category is common ONLY outside the
train mask and then assert that category is correctly discarded.

Style constraints honoured throughout this file: no em dashes, no semicolons,
and the spelling "program" rather than the British variant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from helpers.clean_helpers import (
    build_train_mask_for_rollup,
    canonicalize_admin_ids,
    canonicalize_label,
    derive_lab_measured_flags,
    derive_mnar_flags,
    drop_near_constant,
    log1p_utilization,
    refill_categorical_unknowns,
    restore_unknown_nans,
    rollup_icd9,
    rollup_specialty_top10,
)
from helpers.constants import SEED, UNKNOWN_CATEGORICAL


# ===========================================================================
# Helpers local to this test module.
# ===========================================================================
def _grouped_patient_cohort(n_patients: int = 40) -> pd.DataFrame:
    """Build a patient-grouped cohort large enough for the two-stage split.

    The conftest `raw_cohort_df` carries only 10 unique patients, too few for
    `build_train_mask_for_rollup`'s nested stratified `train_test_split`
    calls (the inner split can leave a patient-level class with a single member,
    and sklearn rightly refuses to stratify it). So for the train-mask tests we
    roll a deterministic, well-populated cohort: `n_patients` patients, each
    appearing on exactly two encounter rows, with both patient-level
    any-positive classes (0 and 1) comfortably populated.

    Each patient is given a fixed readmission pattern by index parity so the
    patient-level `.max()` aggregation lands roughly half positive and half
    negative. This keeps the stratifier happy and makes the 70/10/20 split
    well-defined and reproducible.
    """
    patient_nbr: list[int] = []
    readmitted: list[str] = []
    for i in range(n_patients):
        # Two encounter rows per patient so the patient-grouped invariant
        # (all of a patient's rows land in the same split) has something to
        # constrain.
        patient_nbr.extend([2000 + i, 2000 + i])
        if i % 2 == 0:
            # At least one positive encounter, so the patient is positive after .max().
            readmitted.extend(["<30", "NO"])
        else:
            # No positive encounter, so the patient is negative after .max().
            readmitted.extend(["NO", ">30"])
    return pd.DataFrame({"patient_nbr": patient_nbr, "readmitted": readmitted})


# ===========================================================================
# restore_unknown_nans
# ===========================================================================
class TestRestoreUnknownNans:
    """`restore_unknown_nans` undoes the 'Unknown' fill back to real NaN.

    Why this matters: the MNAR missing-indicator helpers compute signal from
    `isna()`. If the categorical columns still carry the literal "Unknown"
    token from an earlier fill, `isna()` reports zero missing and the MNAR
    flag collapses to all zeros, silently destroying a real clinical signal.
    """

    def test_replaces_unknown_token_with_nan(self):
        """The literal UNKNOWN_CATEGORICAL token becomes a true NaN.

        We assert the post-call cell is detected by `isna()` because that is
        exactly the predicate the downstream MNAR helper relies on.
        """
        df = pd.DataFrame({"race": [UNKNOWN_CATEGORICAL, "Caucasian", UNKNOWN_CATEGORICAL]})
        out = restore_unknown_nans(df, cols=["race"])
        assert out["race"].isna().tolist() == [True, False, True]

    def test_mutates_in_place_and_returns_same_object(self):
        """Contract: mutate the passed frame and return that very object.

        Identity (`is`) is the assertion here, not equality, because the
        chaining pattern in NB02 depends on every helper handing back the same
        object rather than a copy.
        """
        df = pd.DataFrame({"race": [UNKNOWN_CATEGORICAL, "X"]})
        out = restore_unknown_nans(df, cols=["race"])
        assert out is df

    def test_non_unknown_values_are_left_untouched(self):
        """Only the sentinel is rewritten, every real category survives.

        This guards against an over-eager replace that nukes legitimate values.
        """
        df = pd.DataFrame({"race": ["Caucasian", "AfricanAmerican", UNKNOWN_CATEGORICAL]})
        restore_unknown_nans(df, cols=["race"])
        assert df["race"].tolist()[:2] == ["Caucasian", "AfricanAmerican"]

    def test_missing_column_is_skipped_silently(self):
        """A column in `cols` but absent from the frame is a no-op, not a crash.

        The helper guards every access with `if c in df.columns`, so passing a
        column name the frame does not have must be tolerated. This keeps the
        same default column tuple usable across frames at different pipeline
        stages where some columns have already been dropped.
        """
        df = pd.DataFrame({"gender": ["Male", "Female"]})
        out = restore_unknown_nans(df, cols=["race", "payer_code"])
        assert out is df
        assert list(out.columns) == ["gender"]


# ===========================================================================
# derive_mnar_flags
# ===========================================================================
class TestDeriveMnarFlags:
    """`derive_mnar_flags` appends `<col>_missing` Int8 indicator columns.

    The nullable `Int8` dtype is load-bearing: a plain int column would be
    promoted to float64 the moment a downstream merge reintroduces NaN, losing
    the crisp 0/1 missing semantic the mutual-information selector consumes.
    """

    def test_adds_missing_flag_column_with_int8_dtype(self):
        """A `race_missing` column appears and is nullable Int8.

        The dtype assertion is the important part. We check the literal string
        "Int8" so a regression to int8 or float64 is caught immediately.
        """
        df = pd.DataFrame({"race": [np.nan, "X", "Y", np.nan]})
        derive_mnar_flags(df, cols=["race"])
        assert "race_missing" in df.columns
        assert str(df["race_missing"].dtype) == "Int8"

    def test_flag_is_one_where_missing_zero_where_present(self):
        """The indicator is 1 exactly on the NaN rows and 0 elsewhere.

        This is the semantic heart of an MNAR flag: the missingness itself is
        the modelled signal.
        """
        df = pd.DataFrame({"race": [np.nan, "X", "Y", np.nan]})
        derive_mnar_flags(df, cols=["race"])
        assert df["race_missing"].tolist() == [1, 0, 0, 1]

    def test_mutates_in_place_and_returns_same_object(self):
        """Mutate-and-return identity contract."""
        df = pd.DataFrame({"race": [np.nan, "X"]})
        out = derive_mnar_flags(df, cols=["race"])
        assert out is df

    @pytest.mark.parametrize(
        "cells, expected_flags",
        [
            ([np.nan, np.nan, np.nan], [1, 1, 1]),  # all missing
            (["a", "b", "c"], [0, 0, 0]),           # none missing
            ([np.nan, "b", np.nan], [1, 0, 1]),     # mixed
        ],
    )
    def test_flag_pattern_tabular(self, cells, expected_flags):
        """Tabular sweep from the missing pattern to the resulting flag vector.

        Parametrizing keeps the all-missing, none-missing, and mixed cases in
        one place so the mapping from NaN positions to 0/1 is unambiguous.
        """
        df = pd.DataFrame({"payer_code": cells})
        derive_mnar_flags(df, cols=["payer_code"])
        assert df["payer_code_missing"].tolist() == expected_flags

    def test_ordering_invariant_flag_survives_a_later_unknown_fill(self):
        """MNAR flags MUST be derived BEFORE the 'Unknown' refill, not after.

        The module documents this ordering invariant explicitly. We demonstrate
        the failure mode it guards against: if we refill the NaNs to "Unknown"
        FIRST, the flag derived afterwards collapses to all zeros and the signal
        is gone. Deriving first then refilling preserves the captured signal.
        """
        # Correct order: derive the flag while NaN is still NaN.
        good = pd.DataFrame({"race": [np.nan, "X", np.nan]})
        derive_mnar_flags(good, cols=["race"])
        refill_categorical_unknowns(good, cols=["race"])
        assert good["race_missing"].tolist() == [1, 0, 1]

        # Wrong order: refill first, then derive. The flag is now all zeros
        # because every NaN has become the literal "Unknown" token.
        bad = pd.DataFrame({"race": [np.nan, "X", np.nan]})
        refill_categorical_unknowns(bad, cols=["race"])
        derive_mnar_flags(bad, cols=["race"])
        assert bad["race_missing"].tolist() == [0, 0, 0]


# ===========================================================================
# derive_lab_measured_flags
# ===========================================================================
class TestDeriveLabMeasuredFlags:
    """`derive_lab_measured_flags` records was-tested signal and fills labs.

    For the two lab columns it adds `A1C_measured` and `glu_measured` Int8
    flags (whether the lab was run at all is itself a risk signal) and then
    fills the genuinely missing lab cells with the literal "not_measured" so the
    absence becomes a real categorical level rather than an imputed value.
    """

    def test_adds_both_measured_flags_as_int8(self):
        """Both flag columns appear and are nullable Int8.

        Matching `derive_mnar_flags`, the dtype choice protects the 0/1
        semantic across downstream NaN-reintroducing merges.
        """
        df = pd.DataFrame({"A1Cresult": [np.nan, ">7"], "max_glu_serum": [">200", np.nan]})
        derive_lab_measured_flags(df)
        assert str(df["A1C_measured"].dtype) == "Int8"
        assert str(df["glu_measured"].dtype) == "Int8"

    def test_measured_flag_reflects_presence_before_fill(self):
        """The flag is 1 where a lab value was present, 0 where it was NaN.

        Crucially this is computed BEFORE the not_measured fill, so the flag
        reflects the real testing decision rather than the post-fill state.
        """
        df = pd.DataFrame({"A1Cresult": [np.nan, ">7", "Norm"], "max_glu_serum": ["None", np.nan, ">200"]})
        derive_lab_measured_flags(df)
        assert df["A1C_measured"].tolist() == [0, 1, 1]
        assert df["glu_measured"].tolist() == [1, 0, 1]

    def test_missing_labs_filled_with_not_measured_token(self):
        """NaN lab cells become the literal 'not_measured' string.

        Treating "not measured" as a category preserves the was-tested signal
        without fabricating a fictitious lab result.
        """
        df = pd.DataFrame({"A1Cresult": [np.nan, ">7"], "max_glu_serum": [np.nan, "Norm"]})
        derive_lab_measured_flags(df)
        assert df["A1Cresult"].tolist() == ["not_measured", ">7"]
        assert df["max_glu_serum"].tolist() == ["not_measured", "Norm"]

    def test_literal_none_string_counts_as_measured(self):
        """The literal string 'None' is a present value, not missing.

        Subtle but real: the helper only maps the empty string "" and actual NaN
        to not_measured. The UCI source uses the literal token "None" to mean
        "test run, result normal-ish/none", so it must register as measured (flag
        1) and must NOT be overwritten with not_measured. This test pins that
        distinction so a future refactor does not conflate the two.
        """
        df = pd.DataFrame({"A1Cresult": ["None"], "max_glu_serum": ["None"]})
        derive_lab_measured_flags(df)
        assert df["A1C_measured"].tolist() == [1]
        assert df["glu_measured"].tolist() == [1]
        assert df["A1Cresult"].tolist() == ["None"]
        assert df["max_glu_serum"].tolist() == ["None"]

    def test_empty_string_is_treated_as_not_measured(self):
        """An empty-string lab cell is normalised to NaN then filled.

        The helper replaces "" with NaN before computing the flag, so an empty
        cell reads as not-tested (flag 0) and ends up filled with not_measured.
        """
        df = pd.DataFrame({"A1Cresult": [""], "max_glu_serum": [""]})
        derive_lab_measured_flags(df)
        assert df["A1C_measured"].tolist() == [0]
        assert df["glu_measured"].tolist() == [0]
        assert df["A1Cresult"].tolist() == ["not_measured"]
        assert df["max_glu_serum"].tolist() == ["not_measured"]

    def test_mutates_in_place_and_returns_same_object(self):
        """Mutate-and-return identity contract."""
        df = pd.DataFrame({"A1Cresult": [np.nan], "max_glu_serum": [np.nan]})
        out = derive_lab_measured_flags(df)
        assert out is df

    def test_runs_on_realistic_cohort(self, raw_cohort_df):
        """End-to-end smoke on the conftest cohort produces no NaN labs.

        After the helper runs, the two lab columns are fully populated (every
        cell that was missing now reads not_measured) and both flag columns
        exist with the Int8 dtype.
        """
        derive_lab_measured_flags(raw_cohort_df)
        assert raw_cohort_df["A1Cresult"].isna().sum() == 0
        assert raw_cohort_df["max_glu_serum"].isna().sum() == 0
        assert str(raw_cohort_df["A1C_measured"].dtype) == "Int8"
        assert str(raw_cohort_df["glu_measured"].dtype) == "Int8"


# ===========================================================================
# refill_categorical_unknowns
# ===========================================================================
class TestRefillCategoricalUnknowns:
    """`refill_categorical_unknowns` fills NaN with the UNKNOWN_CATEGORICAL token.

    This is the inverse partner of `restore_unknown_nans` and runs AFTER the
    MNAR flags have been derived, so cleaned.csv is human-readable and the
    OneHotEncoder sees a stable "Unknown" level at fit time.
    """

    def test_fills_nan_with_unknown_token(self):
        """Every NaN becomes exactly UNKNOWN_CATEGORICAL, present values stay.

        We assert against the imported constant rather than a hard-coded string
        so the test tracks the single source of truth in helpers.constants.
        """
        df = pd.DataFrame({"race": [np.nan, "Caucasian", np.nan]})
        refill_categorical_unknowns(df, cols=["race"])
        assert df["race"].tolist() == [UNKNOWN_CATEGORICAL, "Caucasian", UNKNOWN_CATEGORICAL]

    def test_no_nan_remains_after_fill(self):
        """The post-condition is zero missing values in the filled column."""
        df = pd.DataFrame({"payer_code": [np.nan, "MC", np.nan, "HM"]})
        refill_categorical_unknowns(df, cols=["payer_code"])
        assert df["payer_code"].isna().sum() == 0

    def test_round_trips_with_restore_unknown_nans(self):
        """restore then refill is an identity on the set of missing positions.

        Starting from a filled column, restoring to NaN and refilling must land
        back on the same values. This proves the two helpers are true inverses
        on the sentinel, which the NB02 program relies on when it temporarily
        un-fills to compute MNAR signal and then re-fills for encoding.
        """
        original = pd.DataFrame({"race": [UNKNOWN_CATEGORICAL, "X", UNKNOWN_CATEGORICAL]})
        df = original.copy()
        restore_unknown_nans(df, cols=["race"])
        refill_categorical_unknowns(df, cols=["race"])
        assert df["race"].tolist() == original["race"].tolist()

    def test_mutates_in_place_and_returns_same_object(self):
        """Mutate-and-return identity contract."""
        df = pd.DataFrame({"race": [np.nan, "X"]})
        out = refill_categorical_unknowns(df, cols=["race"])
        assert out is df


# ===========================================================================
# canonicalize_admin_ids
# ===========================================================================
class TestCanonicalizeAdminIds:
    """`canonicalize_admin_ids` rewrites integer NULL sentinels to NaN.

    The three admin id columns encode "unknown/not available/not mapped" as
    specific integer codes. This helper turns those codes into NaN and stops,
    leaving the follow-on 'Unknown' fill to `refill_categorical_unknowns`.
    """

    @pytest.mark.parametrize(
        "column, sentinel_codes",
        [
            ("admission_type_id", [5, 6, 8]),
            ("discharge_disposition_id", [18, 25, 26]),
            ("admission_source_id", [9, 15, 17, 20, 21]),
        ],
    )
    def test_sentinel_codes_become_nan(self, column, sentinel_codes):
        """Each documented sentinel code in each admin column maps to NaN.

        Tabular over the three columns and their distinct code sets so a typo in
        any single sentinel list would surface here.
        """
        df = pd.DataFrame({column: sentinel_codes})
        canonicalize_admin_ids(df)
        assert df[column].isna().all()

    def test_non_sentinel_values_are_preserved(self):
        """Real (non-sentinel) admin codes are left exactly as they were.

        Code 1 is a legitimate admission type and must not be coerced to NaN.
        """
        df = pd.DataFrame({"admission_type_id": [1, 5, 2, 6, 3]})
        canonicalize_admin_ids(df)
        # Positions 1 and 3 hold sentinels 5 and 6, the rest survive.
        assert df["admission_type_id"].isna().tolist() == [False, True, False, True, False]
        assert df.loc[[0, 2, 4], "admission_type_id"].tolist() == [1, 2, 3]

    def test_mutates_in_place_and_returns_same_object(self):
        """Mutate-and-return identity contract."""
        df = pd.DataFrame({"admission_type_id": [5, 1]})
        out = canonicalize_admin_ids(df)
        assert out is df

    def test_missing_admin_columns_are_skipped(self):
        """Frames without the admin columns pass through untouched.

        The helper guards each column with `if col in df.columns`, so a frame
        that never carried admin ids must not raise.
        """
        df = pd.DataFrame({"gender": ["Male"]})
        out = canonicalize_admin_ids(df)
        assert out is df
        assert list(out.columns) == ["gender"]


# ===========================================================================
# canonicalize_label
# ===========================================================================
class TestCanonicalizeLabel:
    """`canonicalize_label` derives `readmitted_canonical` and `age_mid`.

    The readmission label arrives in mixed spellings ("NO", ">30", "<30") and
    must collapse to the canonical {no, gt30, lt30}. The age bucket string is
    turned into its numeric midpoint for downstream numeric modelling.
    """

    @pytest.mark.parametrize(
        "raw, canonical",
        [
            ("NO", "no"),
            ("no", "no"),
            (">30", "gt30"),
            ("gt30", "gt30"),
            ("<30", "lt30"),
            ("lt30", "lt30"),
        ],
    )
    def test_readmitted_maps_to_canonical(self, raw, canonical):
        """Every accepted spelling of the label maps to its canonical form.

        Parametrized across the full accepted vocabulary, including the already
        canonical inputs which must pass through idempotently.
        """
        df = pd.DataFrame({"readmitted": [raw]})
        canonicalize_label(df)
        assert df["readmitted_canonical"].tolist() == [canonical]

    @pytest.mark.parametrize(
        "bucket, midpoint",
        [
            ("[0-10)", 5),
            ("[40-50)", 45),
            ("[70-80)", 75),
            ("[90-100)", 95),
        ],
    )
    def test_age_bucket_maps_to_numeric_midpoint(self, bucket, midpoint):
        """Each `[a-b)` age bucket becomes the numeric midpoint a+5.

        `age_mid` is the canonical numeric age column NB04 keeps, so the exact
        midpoint arithmetic matters.
        """
        df = pd.DataFrame({"age": [bucket]})
        canonicalize_label(df)
        assert df["age_mid"].tolist() == [midpoint]

    def test_whitespace_padded_label_is_stripped_then_mapped(self):
        """A label with surrounding whitespace still canonicalizes.

        The helper strips before mapping, so a stray "  <30  " must not slip
        through as an unmapped value (which would trip the internal assertion).
        """
        df = pd.DataFrame({"readmitted": ["  <30  "]})
        canonicalize_label(df)
        assert df["readmitted_canonical"].tolist() == ["lt30"]

    def test_unmapped_label_raises_assertion(self):
        """An unrecognised non-null label trips the guard assertion.

        The helper asserts that every non-null readmitted value mapped to a
        canonical form. A junk value like "MAYBE" should therefore raise rather
        than silently producing a NaN canonical label that would poison the
        target downstream.
        """
        df = pd.DataFrame({"readmitted": ["MAYBE"]})
        with pytest.raises(AssertionError):
            canonicalize_label(df)

    def test_existing_age_mid_is_not_recomputed(self):
        """A pre-existing `age_mid` column is left as-is.

        The helper only derives `age_mid` when it is absent, so a frame that
        already carries it (for instance from a prior run) keeps its values.
        """
        df = pd.DataFrame({"age": ["[0-10)"], "age_mid": [999]})
        canonicalize_label(df)
        assert df["age_mid"].tolist() == [999]

    def test_mutates_in_place_and_returns_same_object(self, raw_cohort_df):
        """Mutate-and-return identity contract on the realistic cohort.

        The conftest cohort uses only mappable labels and standard age buckets,
        so this also doubles as an end-to-end smoke that both derived columns
        appear without tripping the internal assertion.
        """
        out = canonicalize_label(raw_cohort_df)
        assert out is raw_cohort_df
        assert "readmitted_canonical" in out.columns
        assert "age_mid" in out.columns
        assert set(out["readmitted_canonical"].unique()) <= {"no", "gt30", "lt30"}


# ===========================================================================
# drop_near_constant
# ===========================================================================
class TestDropNearConstant:
    """`drop_near_constant` removes columns dominated by a single modal value.

    A column where one value covers at least `threshold` of the rows carries
    almost no information for any model. The protected id and target
    columns (patient_nbr, encounter_id, readmitted) are never dropped.

    Mutation nuance worth pinning: unlike the other helpers, this one calls
    `df = df.drop(columns=...)` which rebinds to a NEW frame when something is
    dropped, so the returned object is NOT the input object in that case. When
    nothing crosses the threshold the original object passes straight through.
    """

    def test_drops_column_above_threshold(self):
        """A column that is 100% one value is dropped at the default 0.99 cut."""
        df = pd.DataFrame({"const": [1] * 100, "varied": list(range(100))})
        out = drop_near_constant(df)
        assert "const" not in out.columns
        assert "varied" in out.columns

    def test_keeps_column_below_threshold(self):
        """A column whose modal share is under the cutoff is retained.

        Here the modal value covers 98 percent of rows, below the 0.99 default,
        so the column survives.
        """
        col = [0] * 98 + [1, 2]
        df = pd.DataFrame({"mostly_const": col})
        out = drop_near_constant(df)
        assert "mostly_const" in out.columns

    @pytest.mark.parametrize("protected", ["patient_nbr", "encounter_id", "readmitted"])
    def test_protected_columns_are_never_dropped(self, protected):
        """The id and target columns survive even when constant.

        These columns are structurally near-constant in some cohorts (a single
        patient, a single label) but must never be removed because the rest of
        the pipeline keys on them. Parametrized across all three protected names.
        """
        df = pd.DataFrame({protected: [7] * 100, "junk": [9] * 100})
        out = drop_near_constant(df)
        assert protected in out.columns
        # The unprotected constant column is still removed.
        assert "junk" not in out.columns

    def test_returns_new_object_when_a_column_is_dropped(self):
        """When a drop happens the returned frame is a fresh object.

        This documents the subtlety that `drop_near_constant` does not follow
        the in-place-same-object contract of its siblings. Callers that captured
        the original reference would not see the dropped column reflected there.
        """
        df = pd.DataFrame({"const": [1] * 100, "varied": list(range(100))})
        out = drop_near_constant(df)
        assert out is not df

    def test_returns_same_object_when_nothing_is_dropped(self):
        """When no column crosses the threshold the input passes straight through.

        With nothing to drop the helper never rebinds, so the original object is
        returned unchanged.
        """
        df = pd.DataFrame({"a": list(range(100)), "b": list(range(100, 200))})
        out = drop_near_constant(df)
        assert out is df

    def test_threshold_is_inclusive_at_the_boundary(self):
        """A modal share exactly equal to the threshold triggers the drop.

        The comparison is `>=` so a column at precisely the cutoff is removed.
        We build a column that is 90 percent one value and pass threshold 0.90.
        """
        col = [5] * 90 + list(range(10))
        df = pd.DataFrame({"boundary": col})
        out = drop_near_constant(df, threshold=0.90)
        assert "boundary" not in out.columns

    def test_train_mask_restricts_the_modal_share_computation(self):
        """The modal share is measured on train rows only when a mask is given.

        Leakage guard demonstration. We construct a column that is constant
        within the train rows (so it looks zero-information to the model) but
        varied across the full frame. With the mask, the helper sees only the
        constant train slice and drops the column. Without the mask it would see
        the variety and keep it. Computing this decision on train rows only is
        what stops validation and test frequencies from steering which columns
        survive cleaning.
        """
        n = 100
        train_mask = pd.Series([True] * 50 + [False] * 50)
        # Constant across the train half, fully varied across the test half.
        col = [7] * 50 + list(range(50))
        df = pd.DataFrame({"leaky": col, "keepme": list(range(n))})

        # With the mask the column reads as constant-in-train and is dropped.
        with_mask = drop_near_constant(df.copy(), train_mask=train_mask)
        assert "leaky" not in with_mask.columns

        # Without the mask the same column reads as varied and is kept, proving
        # the mask is what changed the decision.
        without_mask = drop_near_constant(df.copy())
        assert "leaky" in without_mask.columns


# ===========================================================================
# rollup_icd9
# ===========================================================================
class TestRollupIcd9:
    """`rollup_icd9` buckets diagnosis codes by their top-K 3-digit prefix.

    For each diag_X column it extracts the leading numeric ICD-9 prefix, keeps
    the top-K most frequent, and collapses the long tail to "Other". The new
    column is named `diag_X_cat` and the original column is left in place.
    """

    def test_adds_cat_columns_for_each_diag(self):
        """A `diag_X_cat` column is produced for each present diag column.

        We give every code the same prefix so the result is trivially the kept
        prefix. The point here is purely that the three sidecar columns appear.
        """
        df = pd.DataFrame(
            {
                "diag_1": ["250.83", "250.1"],
                "diag_2": ["401", "402"],
                "diag_3": ["414", "414"],
            }
        )
        rollup_icd9(df, top_k=25)
        for c in ("diag_1_cat", "diag_2_cat", "diag_3_cat"):
            assert c in df.columns

    def test_extracts_three_digit_prefix(self):
        """The bucket value is the leading numeric prefix, not the full code.

        "250.83" and "250.1" both reduce to prefix "250", so the rolled-up
        column should read "250" for both rows.
        """
        df = pd.DataFrame({"diag_1": ["250.83", "250.1"]})
        rollup_icd9(df, top_k=25)
        assert df["diag_1_cat"].tolist() == ["250", "250"]

    def test_tail_prefixes_collapse_to_other_under_small_top_k(self):
        """With top_k=1 only the single most frequent prefix survives.

        Prefix "250" appears three times and "401" once. With K=1 only "250" is
        kept and the "401" row collapses to "Other", proving the long-tail
        bucketing works.
        """
        df = pd.DataFrame({"diag_1": ["250", "250", "250", "401"]})
        rollup_icd9(df, top_k=1)
        assert df["diag_1_cat"].tolist() == ["250", "250", "250", "Other"]

    def test_mutates_in_place_and_returns_same_object(self):
        """Mutate-and-return identity contract."""
        df = pd.DataFrame({"diag_1": ["250", "401"]})
        out = rollup_icd9(df, top_k=25)
        assert out is df

    def test_train_mask_restricts_top_k_selection(self):
        """The top-K is chosen from train rows only when a mask is supplied.

        Leakage guard demonstration for the ICD-9 rollup. Prefix "999" is
        frequent ONLY outside the train mask. With top_k=1 the train-only
        frequency table elects "250" as the single keeper, so every "999" row
        (which lives outside the train slice) must collapse to "Other". This
        proves validation and test frequencies cannot smuggle a category into
        the kept set.
        """
        # Train rows (mask True): "250" dominates. Test rows (mask False): "999"
        # dominates. The global frequency of "999" is high, but it must be
        # ignored because it never appears in train.
        diag = ["250"] * 6 + ["999"] * 10
        train_mask = pd.Series([True] * 6 + [False] * 10)
        df = pd.DataFrame({"diag_1": diag})
        rollup_icd9(df, top_k=1, train_mask=train_mask)

        # "999" was frequent only outside the mask, so it is NOT kept.
        kept = set(df["diag_1_cat"].unique())
        assert "250" in kept
        assert "999" not in kept
        # Every out-of-train "999" row collapsed to Other.
        assert df.loc[6:, "diag_1_cat"].eq("Other").all()


# ===========================================================================
# rollup_specialty_top10
# ===========================================================================
class TestRollupSpecialtyTop10:
    """`rollup_specialty_top10` keeps the top-10 specialties, rest to 'Other'.

    Produces a `medical_specialty_grouped` column. The original
    `medical_specialty` column is preserved.
    """

    def test_adds_grouped_column(self):
        """The `medical_specialty_grouped` sidecar column appears."""
        df = pd.DataFrame({"medical_specialty": ["Cardiology", "Surgery"]})
        rollup_specialty_top10(df)
        assert "medical_specialty_grouped" in df.columns

    def test_collapses_tail_to_other_beyond_top_10(self):
        """An 11th specialty beyond the top-10 collapses to 'Other'.

        We build ten frequent specialties plus one rare extra. The rare one is
        outside the top-10 and must map to "Other" while the frequent ten keep
        their own names.
        """
        rows = []
        for i in range(10):
            # Each of the first ten specialties appears three times.
            rows.extend([f"Spec{i}"] * 3)
        rows.append("RareSpec")  # the eleventh, appears once
        df = pd.DataFrame({"medical_specialty": rows})
        rollup_specialty_top10(df)
        grouped = df["medical_specialty_grouped"]
        assert grouped.iloc[-1] == "Other"
        # The ten frequent specialties kept their own labels (no Other among them).
        assert "Other" not in set(grouped.iloc[:-1])

    def test_mutates_in_place_and_returns_same_object(self):
        """Mutate-and-return identity contract."""
        df = pd.DataFrame({"medical_specialty": ["Cardiology"]})
        out = rollup_specialty_top10(df)
        assert out is df

    def test_train_mask_restricts_top_10_selection(self):
        """The top-10 is computed on train rows only when a mask is given.

        Leakage guard demonstration. "RareInTrain" appears frequently but ONLY
        outside the train mask. The train slice carries just two specialties, so
        the train-only top-10 cannot include "RareInTrain", and every such row
        (all outside train) must collapse to "Other". This is the specialty
        analogue of the ICD-9 leakage guard above.
        """
        specialties = ["A"] * 10 + ["B"] * 10 + ["RareInTrain"] * 10
        train_mask = pd.Series([True] * 20 + [False] * 10)
        df = pd.DataFrame({"medical_specialty": specialties})
        rollup_specialty_top10(df, train_mask=train_mask)

        kept = set(df["medical_specialty_grouped"].unique())
        # A and B were the only train-visible specialties, so they are kept.
        assert {"A", "B"} <= kept
        # RareInTrain was frequent only outside the mask and is therefore Other.
        assert "RareInTrain" not in kept
        assert df.loc[20:, "medical_specialty_grouped"].eq("Other").all()


# ===========================================================================
# log1p_utilization
# ===========================================================================
class TestLog1pUtilization:
    """`log1p_utilization` adds log1p sidecars and has-any flags for counts.

    The three utilization count columns are heavy-tailed, so the helper emits a
    `log_<col>` sidecar (numerically stable log1p) plus a `has_any_<col>`
    Int8 flag marking whether the patient had any such event.
    """

    @pytest.mark.parametrize(
        "col",
        ["number_outpatient", "number_emergency", "number_inpatient"],
    )
    def test_adds_log_and_flag_columns(self, col):
        """Each utilization column gains a log sidecar and a has-any flag.

        Parametrized across all three count columns so a missing branch for any
        single column is caught.
        """
        df = pd.DataFrame({col: [0, 1, 5]})
        log1p_utilization(df)
        assert f"log_{col}" in df.columns
        assert f"has_any_{col}" in df.columns

    def test_log_sidecar_is_log1p_of_the_count(self):
        """The sidecar equals numpy's log1p of the original count.

        log1p(0) is 0 and log1p(x) handles small counts without the log(0)
        blow-up, which is exactly why it is used over a raw log here.
        """
        df = pd.DataFrame({"number_inpatient": [0, 3]})
        log1p_utilization(df)
        expected = np.log1p([0, 3])
        assert np.allclose(df["log_number_inpatient"].to_numpy(), expected)

    def test_has_any_flag_is_int8_and_correct(self):
        """The has-any flag is Int8 and is 1 exactly when the count exceeds zero.

        A zero count yields flag 0, any positive count yields flag 1.
        """
        df = pd.DataFrame({"number_emergency": [0, 2, 0, 9]})
        log1p_utilization(df)
        assert str(df["has_any_number_emergency"].dtype) == "Int8"
        assert df["has_any_number_emergency"].tolist() == [0, 1, 0, 1]

    def test_mutates_in_place_and_returns_same_object(self):
        """Mutate-and-return identity contract."""
        df = pd.DataFrame({"number_outpatient": [0, 1]})
        out = log1p_utilization(df)
        assert out is df


# ===========================================================================
# build_train_mask_for_rollup
# ===========================================================================
class TestBuildTrainMaskForRollup:
    """`build_train_mask_for_rollup` reproduces NB05's 70/10/20 patient split.

    It returns a boolean Series, aligned to the input index, that is True on
    rows whose patient lands in NB05's train cohort. The split is patient
    grouped (no patient straddles the train boundary), stratified on the
    any-positive patient-level target, and deterministic for a fixed seed.
    """

    def test_returns_boolean_series_aligned_to_index(self):
        """Output is a bool Series of df length carrying the df's own index.

        Index alignment is essential because the mask is passed straight into
        `df.loc[train_mask]` inside the rollup helpers.
        """
        df = _grouped_patient_cohort()
        mask = build_train_mask_for_rollup(df)
        assert isinstance(mask, pd.Series)
        assert mask.dtype == bool
        assert len(mask) == len(df)
        assert mask.index.equals(df.index)

    def test_is_deterministic_for_a_fixed_seed(self):
        """Two calls with the same seed produce identical masks.

        Reproducibility is the whole point of pinning `random_state` to SEED,
        so the mask must be bit-for-bit stable across calls.
        """
        df = _grouped_patient_cohort()
        first = build_train_mask_for_rollup(df, seed=SEED)
        second = build_train_mask_for_rollup(df, seed=SEED)
        assert first.equals(second)

    def test_different_seed_changes_the_split(self):
        """A different seed yields a different partition.

        This confirms the seed drives the randomness rather than the split being
        accidentally fixed, which would make the seed argument a lie.
        """
        df = _grouped_patient_cohort()
        base = build_train_mask_for_rollup(df, seed=SEED)
        other = build_train_mask_for_rollup(df, seed=SEED + 5)
        assert not base.equals(other)

    def test_all_rows_of_a_patient_share_the_same_split(self):
        """Patient-grouped invariant: no patient straddles the train boundary.

        This is the no-leakage guarantee. Every encounter row for a given
        patient_nbr must carry the same mask value, otherwise the same patient
        would appear in both train and not-train, leaking information across the
        split. We assert each patient resolves to exactly one distinct mask
        value.
        """
        df = _grouped_patient_cohort()
        mask = build_train_mask_for_rollup(df)
        per_patient = (
            pd.DataFrame({"patient_nbr": df["patient_nbr"].values, "in_train": mask.values})
            .groupby("patient_nbr")["in_train"]
            .nunique()
        )
        # nunique == 1 for every patient means no patient is split across the
        # boundary.
        assert (per_patient == 1).all()

    def test_train_fraction_is_approximately_seventy_percent(self):
        """The train cohort is roughly 70 percent of rows.

        The two-stage split is 80 percent trainval then 87.5 percent of that to
        train, landing at 0.80 * 0.875 = 0.70. With a balanced, evenly grouped
        cohort the row-level fraction should sit close to that target. We allow
        a generous tolerance because patient grouping makes the exact row count
        depend on how many encounters each selected patient carries.
        """
        df = _grouped_patient_cohort(n_patients=40)
        mask = build_train_mask_for_rollup(df)
        assert 0.55 <= mask.mean() <= 0.85

    def test_mask_drives_a_rollup_to_train_only_statistics(self):
        """Integration: feeding the mask into a rollup makes it train-only.

        This ties the splitter to its purpose. We attach a specialty column that
        is rare in the (small) train cohort and dominant in the rest, then run
        `rollup_specialty_top10` with the derived mask. Because the rollup only
        sees train rows, the not-train-dominant specialty is at risk of being
        excluded from the top-10. The assertion we can make robustly is that the
        grouped column exists and only ever contains specialties that appear
        in the train slice (or the literal "Other"), which is precisely
        the train-only contract the mask exists to enforce.
        """
        df = _grouped_patient_cohort()
        # Give train patients one specialty and the rest a different one.
        mask = build_train_mask_for_rollup(df)
        df = df.copy()
        df["medical_specialty"] = np.where(mask.values, "TrainSpec", "OtherSpec")
        rollup_specialty_top10(df, train_mask=mask)

        train_specialties = set(df.loc[mask.values, "medical_specialty"].unique())
        grouped_values = set(df["medical_specialty_grouped"].unique())
        # Every grouped label is either a genuinely train-visible specialty or
        # the catch-all Other, never a label seen only outside train.
        assert grouped_values <= (train_specialties | {"Other"})
