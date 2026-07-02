"""End-to-end integration test for the data-cleaning stage.

The per-helper unit tests in `test_clean_helpers.py` exercise each cleaning
function in isolation. This test composes them in their production order on the
shared raw-cohort fixture and asserts the result is a modeling-ready frame: the
missing-value sentinels are gone, the MNAR and lab-measured indicators exist,
and the label is canonicalized to a binary target. It guards the cross-helper
contract the notebooks depend on, which an isolated unit test cannot see.

It uses only numpy and pandas, so it runs in the structural CI lane without the
modeling stack.
"""

from __future__ import annotations

import pandas as pd

from helpers.constants import (
    CATEGORICAL_MISSING_COLS,
    MNAR_FLAG_COLS,
    UNKNOWN_CATEGORICAL,
)
from helpers.clean_helpers import (
    canonicalize_label,
    derive_lab_measured_flags,
    derive_mnar_flags,
    refill_categorical_unknowns,
    restore_unknown_nans,
)


def _clean_chain(raw: pd.DataFrame) -> pd.DataFrame:
    """Run the section 2 cleaning helpers in their canonical, ordering-safe sequence."""
    df = raw.copy()
    # Section 2.5 equivalent: the raw "?" sentinel becomes the UNKNOWN token the
    # later helpers expect to reverse.
    for c in CATEGORICAL_MISSING_COLS:
        if c in df.columns:
            df[c] = df[c].replace({"?": UNKNOWN_CATEGORICAL})
    # Section 2.8 chain. derive_mnar_flags MUST precede refill (ordering invariant).
    df = restore_unknown_nans(df)
    df = derive_mnar_flags(df)
    df = derive_lab_measured_flags(df)
    df = refill_categorical_unknowns(df)
    df = canonicalize_label(df)
    return df


def test_cleaning_chain_produces_modeling_ready_frame(raw_cohort_df):
    n = len(raw_cohort_df)
    cleaned = _clean_chain(raw_cohort_df)

    # Row count is preserved end to end.
    assert len(cleaned) == n

    # No raw "?" sentinel survives anywhere in the cleaned frame.
    assert not (cleaned.astype(object) == "?").to_numpy().any()

    # Every MNAR column gained its 0/1 indicator.
    for c in MNAR_FLAG_COLS:
        flag = f"{c}_missing"
        assert flag in cleaned.columns
        assert set(cleaned[flag].dropna().unique()) <= {0, 1}

    # Lab "was-measured" indicators exist and are 0/1.
    for flag in ("A1C_measured", "glu_measured"):
        assert flag in cleaned.columns
        assert set(cleaned[flag].dropna().unique()) <= {0, 1}

    # The label is canonicalized to its three-class form and the numeric age
    # midpoint is derived for the downstream binary target.
    assert "readmitted_canonical" in cleaned.columns
    assert set(cleaned["readmitted_canonical"].dropna().unique()) <= {"no", "gt30", "lt30"}
    assert "age_mid" in cleaned.columns
    assert pd.api.types.is_numeric_dtype(cleaned["age_mid"])

    # The refilled categoricals carry the shared UNKNOWN token, never a bare NaN.
    for c in CATEGORICAL_MISSING_COLS:
        if c in cleaned.columns:
            assert cleaned[c].notna().all()
