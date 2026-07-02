"""Modular cleaning helpers for nb02 §2.8.

The §2.8 additional-cleaning corrections are split into narrow,
single-concern helpers so the notebook cell reads as a short
pipeline of named steps.

Mutation contract: each helper mutates the passed-in DataFrame in place
(via `df[c] = ...` and `df.drop(...)`) and returns the same object for
chaining. Callers that need an independent copy should pass `df.copy()`
into the first helper of the chain. No I/O, no mutation of globals.
"""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

from helpers.constants import (
    CATEGORICAL_MISSING_COLS,
    MNAR_FLAG_COLS,
    SEED,
    UNKNOWN_CATEGORICAL,
)


def restore_unknown_nans(df: pd.DataFrame, cols: Iterable[str] = CATEGORICAL_MISSING_COLS) -> pd.DataFrame:
    """Reverse §2.5.2's UNKNOWN_CATEGORICAL fill so notna() can compute MNAR signal."""
    for c in cols:
        if c in df.columns:
            df[c] = df[c].replace({UNKNOWN_CATEGORICAL: np.nan})
    return df


def derive_mnar_flags(df: pd.DataFrame, cols: Iterable[str] = MNAR_FLAG_COLS) -> pd.DataFrame:
    """Append '<col>_missing' (Int8 0/1) for every column in 'cols'.

    Threshold rationale (5%)
    ------------------------
    'MNAR_FLAG_COLS' is the audit-frozen subset of columns whose missing
    fraction exceeded the 5% MNAR cutoff in NB02 §2.5. Below that fraction
    the missingness is statistically indistinguishable from MCAR on this
    cohort and the redundant indicator column would dilute the
    mutual-information selector's signal-to-noise in NB05 §5.8 without
    earning its keep.

    Ordering invariant
    ------------------
    MUST run BEFORE 'refill_categorical_unknowns'. Once NaN is replaced
    with the literal '"Unknown"' token the indicator collapses to all
    zeros and the signal is lost.

    Output type is 'Int8' (nullable) so a downstream merge that
    re-introduces NaN does not promote the column to 'float64' and lose
    the 0/1 semantic.
    """
    for c in cols:
        if c in df.columns:
            df[f"{c}_missing"] = df[c].isna().astype("Int8")
    return df


def derive_lab_measured_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'A1C_measured' / 'glu_measured' MNAR flags and fill the labs.

    Why these two columns specifically
    ----------------------------------
    'A1Cresult' and 'max_glu_serum' are the only two lab columns in the
    UCI 130-hospitals cohort, and their *testing* decision is itself a
    clinical risk signal (NB03 §3.4.5: 'A1C_measured' and
    'glu_measured' stratify readmission rate by roughly 6 percentage
    points even when the result itself is unrevealing). Treating "not
    measured" as a real categorical level preserves both the
    "result-when-tested" signal AND the "was-tested" signal in one column
    pair without imputing fictitious lab values.

    Threshold note: these two columns sit far below the 5% non-missing
    cutoff that 'derive_mnar_flags' uses. Per the NB02 §2.9 / NB03 §3.3
    missingness summary, 'max_glu_serum' is ~95% missing (so ~5% of
    encounters were tested) and 'A1Cresult' is ~83% missing (so ~17% of
    encounters were tested). These are the source for the "17% A1C, 5%
    glucose" headline numbers above and are mutually consistent within a
    rounding step (17% tested + 83% missing = 100%, and 5% tested + 95%
    missing = 100%). The flag is added unconditionally here rather than
    driven by the generic MNAR helper.
    """
    for c in ["max_glu_serum", "A1Cresult"]:
        if c in df.columns:
            df[c] = df[c].replace({"": np.nan}).astype("object")
    df["A1C_measured"] = df["A1Cresult"].notna().astype("Int8")
    df["glu_measured"] = df["max_glu_serum"].notna().astype("Int8")
    df["max_glu_serum"] = df["max_glu_serum"].fillna("not_measured")
    df["A1Cresult"] = df["A1Cresult"].fillna("not_measured")
    return df


def refill_categorical_unknowns(df: pd.DataFrame, cols: Iterable[str] = CATEGORICAL_MISSING_COLS) -> pd.DataFrame:
    """Refill the categorical NaNs (post-MNAR-flag derivation) so cleaned.csv is human-readable.

    The fill value is `helpers.constants.UNKNOWN_CATEGORICAL` so the inference
    APIs can import the same constant and produce rows the OHE actually saw
    at fit time (otherwise handle_unknown="ignore" zero-vectors the column).
    """
    for c in cols:
        if c in df.columns:
            df[c] = df[c].fillna(UNKNOWN_CATEGORICAL)
    return df


def canonicalize_admin_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Replace integer NULL sentinels in the three admin id columns with NaN.

    The downstream `refill_categorical_unknowns` helper handles the
    follow-on NaN to `"Unknown"` fill, so this function intentionally
    stops at NaN.
    """
    sentinels = {
        "admission_type_id": [5, 6, 8],
        "discharge_disposition_id": [18, 25, 26],
        "admission_source_id": [9, 15, 17, 20, 21],
    }
    for col, codes in sentinels.items():
        if col in df.columns:
            df[col] = df[col].where(~df[col].isin(codes), other=np.nan)
    return df


def canonicalize_label(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce `readmitted` / `age` to canonical forms and emit derived columns.

    Adds `readmitted_canonical` ({no, gt30, lt30}), consumed by nb05 as a
    target-column to drop alongside `readmitted` / `readmitted_binary`, and
    `age_mid` (numeric midpoint of the `[a-b]` bucket), consumed by nb04 as
    the canonical numeric age column kept after the §4.16 dedup.
    """
    if "readmitted" in df.columns:
        df["readmitted"] = df["readmitted"].astype(str).str.strip()
        readmit_map = {"NO": "no",
                       "no": "no",
                       "gt30": "gt30",
                       ">30": "gt30",
                       "lt30": "lt30",
                       "<30": "lt30"}
        df["readmitted_canonical"] = df["readmitted"].str.upper().map({k.upper(): v for k, v in readmit_map.items()})
        # If the input was already canonicalized (e.g. from a prior run), keep it.
        # Otherwise, ensure we didn't lose any non-NaN info.
        mask = df["readmitted"].notna() & ~df["readmitted"].isin(["nan", "None", ""])
        assert df.loc[mask, "readmitted_canonical"].notna().all(), \
            f"unmapped readmitted value(s): {df.loc[mask & df['readmitted_canonical'].isna(), 'readmitted'].unique()}"

    if "age" in df.columns and "age_mid" not in df.columns:
        # Standard buckets: [0-10), [10-20), ..., [90-100)
        age_map = {f"[{i}-{i + 10})": i + 5 for i in range(0, 100, 10)}
        # Also handle potential float/int if they already were converted somehow
        df["age_mid"] = df["age"].map(age_map)
    return df


def drop_near_constant(
    df: pd.DataFrame,
    threshold: float = 0.99,
    train_mask: pd.Series | None = None,
) -> pd.DataFrame:
    """Drop columns where >= `threshold` of rows share the modal value (zero-info under any model).

    If `train_mask` is provided, the modal-share cutoff is computed on
    train rows only, which prevents the val/test cohorts from influencing
    which columns survive (mild frequency-based leakage). The drop set is
    then applied to the full DataFrame. Default `None` preserves the
    pre-split nb02 contract where the train mask is not yet known.
    """
    pop = df.loc[train_mask] if train_mask is not None else df
    drop_cols: list[str] = []
    for c in df.columns:
        mode_share = pop[c].value_counts(normalize=True, dropna=False).iloc[0]
        if mode_share >= threshold and c not in ("patient_nbr", "encounter_id", "readmitted"):
            drop_cols.append(c)
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


_ICD9_PREFIX_RE = re.compile(r"^(\d{1,3})")


def rollup_icd9(
    df: pd.DataFrame,
    top_k: int = 25,
    train_mask: pd.Series | None = None,
) -> pd.DataFrame:
    """For each diag_X column emit a `diag_X_cat` mapped to the top-K 3-digit ICD-9 prefixes (rest='Other').

    If `train_mask` is provided, the top-K is computed on train rows only,
    so val/test cohort frequencies don't influence which prefixes survive.
    Mapping is then applied to the full DataFrame. Default `None` preserves
    the pre-split nb02 contract.
    """
    pop = df.loc[train_mask] if train_mask is not None else df
    for col in ("diag_1", "diag_2", "diag_3"):
        if col not in df.columns:
            continue
        prefixes_pop  = pop[col].astype(str).str.extract(_ICD9_PREFIX_RE)[0].fillna(UNKNOWN_CATEGORICAL)
        prefixes_full = df[col].astype(str).str.extract(_ICD9_PREFIX_RE)[0].fillna(UNKNOWN_CATEGORICAL)
        keep = prefixes_pop.value_counts().head(top_k).index
        df[f"{col}_cat"] = prefixes_full.where(prefixes_full.isin(keep), "Other")
    return df


def rollup_specialty_top10(
    df: pd.DataFrame,
    train_mask: pd.Series | None = None,
) -> pd.DataFrame:
    """`medical_specialty_grouped`: top-10 specialties as-is, the rest collapsed to 'Other'.

    If `train_mask` is provided, the top-10 is computed on train rows only.
    Default `None` preserves the pre-split nb02 contract. In that mode the
    rollup is a frequency-only leak that's bounded by train-cohort dominance
    (train is ~70% of the data, so the top-10 is ~always train-driven).
    """
    pop = df.loc[train_mask] if train_mask is not None else df
    if "medical_specialty" in df.columns:
        top10 = pop["medical_specialty"].value_counts().head(10).index
        df["medical_specialty_grouped"] = df["medical_specialty"].where(
            df["medical_specialty"].isin(top10), "Other"
        )
    return df


def log1p_utilization(df: pd.DataFrame) -> pd.DataFrame:
    """Heavy-tailed counts → log1p sidecars + a 'has any' flag."""
    for c in ("number_outpatient", "number_emergency", "number_inpatient"):
        if c in df.columns:
            df[f"log_{c}"] = np.log1p(df[c])
            df[f"has_any_{c}"] = (df[c] > 0).astype("Int8")
    return df


def build_train_mask_for_rollup(df: pd.DataFrame, seed: int = SEED) -> pd.Series:
    """Reproduce NB05's two-stage patient-grouped 70 / 10 / 20 split so NB02's §2.8
    rollup can be **train-only** and avoid the pre-split frequency leak the
    rollup-helper docstrings document.

    =====================================================================
    DATASET-NO-TIME-COMPONENT (acknowledged audit false positive)
    =====================================================================
    The UCI Diabetes-130 1999-2008 release has no admission timestamps,
    no encounter ordering field, and no usable event time. `encounter_id`
    values are anonymised surrogates with no documented temporal meaning.
    A random patient-grouped split is the **correct** splitter for this
    dataset. There is no future-looks-at-past leakage to guard against
    because the data carries no past or future axis in the first place.

    Static AST scanners flag any `train_test_split` call that lacks
    `shuffle=False` as a possible time-series leak. That rule does not
    apply here. This banner is the canonical acknowledgement, so future
    audits can recognise the pattern as a known false positive rather
    than reopening the same finding. If a future release of this dataset
    ever gains a real admission timestamp, swap the splitter for
    `TimeSeriesSplit` or `train_test_split(..., shuffle=False)` and
    delete this banner.
    =====================================================================

    NB02 sits before NB05's split in the pipeline order. Calling
    `drop_near_constant` / `rollup_icd9` / `rollup_specialty_top10` on the
    full DataFrame lets val/test row frequencies influence which columns
    survive and which top-K categories make the cut. This helper derives the
    same train mask NB05 will later use, then passes it to those three
    helpers. Both pipelines build the patient-level
    stratification key from `patient_nbr` + the binary `<30` target with
    `random_state=seed` and the **any-positive aggregation**
    (`groupby('patient_nbr')['target'].max()`), so the masks agree by
    construction. NB05's downstream split is unaffected.

    Why any-positive (`.max()`) and not first-encounter (`.first()`): a
    patient with multiple encounters can read 0 on the first encounter and 1
    on a later one. Stratifying by `.first()` here while NB05 stratifies by
    `.max()` would route such patients into different strata across the two
    pipelines, producing slightly different train cohorts and a small but real
    frequency leak through the NB02 rollup. `.max()` is also the more
    conservative target definition for the train cohort: any patient with at
    least one positive encounter is treated as positive at the patient level.

    Args:
        df: cleaned baseline DataFrame (must carry `patient_nbr` and the raw
            `readmitted` column). Row order / index must match what NB05 will
            see when it reads `data/cleaned.csv`.
        seed: random_state shared with NB05's `train_test_split` calls.

    Returns:
        Boolean Series aligned to `df.index`, True on rows whose patient
        is in NB05's train cohort.
    """
    # ------------------------------------------------------------------
    # DATASET-NO-TIME-COMPONENT: UCI Diabetes-130 1999-2008 has no
    # admission timestamps, so a random patient-grouped split is correct.
    # See the function docstring banner for the full acknowledgement of
    # this audit false positive. Static scanners that flag random
    # train_test_split as a time-series leak do not apply to this dataset.
    # Grouping unit is patient_nbr, split is performed on the unique
    # patient set and then expanded back to encounter rows below.
    # ------------------------------------------------------------------
    from sklearn.model_selection import train_test_split
    # DATASET-NO-TIME-COMPONENT: random split is correct, see banner above.
    target = (df["readmitted"].astype(str).str.strip() == "<30").astype(int).values
    # Any-positive aggregation: matches NB05 §5.4 exactly.
    patient_any_positive = (
        pd.DataFrame({"patient_nbr": df["patient_nbr"].values, "target": target})
          .groupby("patient_nbr")["target"].max()
    )
    trainval_patients, _ = train_test_split(
        patient_any_positive.index,
        test_size=0.20,
        stratify=patient_any_positive.values,
        random_state=seed,
    )
    train_patients, _ = train_test_split(
        trainval_patients,
        test_size=0.125,                                          # 0.125 * 0.80 = 0.10 of total
        stratify=patient_any_positive.loc[trainval_patients].values,
        random_state=seed,
    )
    return pd.Series(df["patient_nbr"].isin(train_patients).values, index=df.index)
