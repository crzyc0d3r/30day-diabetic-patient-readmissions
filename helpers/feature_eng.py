"""Feature-engineering primitives used by notebook 04 §4.14.

Each function takes a DataFrame and returns the same DataFrame with one
conceptual group of derived columns added in place. Splitting the §4.14
work into these primitives lets tests/test_clean_helpers.py exercise
each group independently and lets the notebook re-derive a single group
without copy-pasting the whole cell.
"""

from __future__ import annotations

import pandas as pd


def map_icd9(code) -> str:
    """Collapse an ICD-9 diagnosis code to its three-digit chapter, or
    one of the sentinel buckets '"V-codes"' / '"E-codes"' /
    '"Unknown"' / '"Other"'.

    NB04 §4.x calls this on each of 'diag_1' / 'diag_2' / 'diag_3'.
    The retrain DAG runs the same notebook via papermill, so the
    encoding must stay deterministic between research and production.
    Pulled out of the cell so a fresh inference path can apply the same
    mapping without launching a notebook kernel.
    """
    if pd.isna(code) or str(code).strip() in ("Unknown", ""):
        return "Unknown"
    s = str(code).strip()
    if s[:1].upper() == "V":
        return "V-codes"
    if s[:1].upper() == "E":
        return "E-codes"
    try:
        return f"{int(float(s)):03d}"
    except ValueError:
        return "Other"


PER_DAY_FEATURES = (
    "labs_per_day",
    "procedures_per_day",
    "meds_per_day",
    "utilization_per_day",
)

CROSS_PRODUCT_FEATURES = (
    "inpatient_x_emergency",
    "inpatient_x_diagnoses",
    "inpatient_x_meds",
    "inpatient_x_time",
    "emergency_x_diagnoses",
    "meds_x_diagnoses",
)

DISCHARGE_AGE_FEATURES = (
    "inpatient_x_discharge_home",
    "inpatient_x_discharge_transfer",
    "inpatient_x_discharge_snf",
    "inpatient_x_age",
    "young_high_utilizer",
)

POLYNOMIAL_FEATURES = (
    "inpatient_sq",
    "emergency_sq",
    "total_visits_sq",
)

THRESHOLD_FLAG_FEATURES = (
    "high_utilizer",
    "frequent_inpatient",
    "any_emergency",
    "long_stay",
    "many_meds",
    "many_diagnoses",
)

ALL_INTERACTION_FEATURES = (
    PER_DAY_FEATURES
    + CROSS_PRODUCT_FEATURES
    + DISCHARGE_AGE_FEATURES
    + POLYNOMIAL_FEATURES
    + THRESHOLD_FLAG_FEATURES
    + (
        "n_diabetes_meds", "insulin_plus_oral", "on_3plus_diabetes_meds",
        "a1c_risk", "a1c_measured_high", "glu_risk"
    )
)

def add_per_day_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise high-volume counts by length of stay.
    A 2-day stay with 50 labs is very different from a 14-day stay. Length of stay
    is clipped at 1 to avoid divide-by-zero (encounters with `time_in_hospital == 0`
    are extremely rare day-zero discharges).
    """
    los = df["time_in_hospital"].clip(lower=1)
    df["labs_per_day"] = df["num_lab_procedures"] / los
    df["procedures_per_day"] = df["num_procedures"] / los
    df["meds_per_day"] = df["num_medications"] / los
    df["utilization_per_day"] = df["service_utilization"] / los
    return df


def add_cross_product_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Add pairwise count-by-count interactions anchored on `number_inpatient`.

    Input
    -----
    `df` must carry the cleaned-and-encoded count columns
    `number_inpatient`, `number_emergency`, `number_diagnoses`,
    `num_medications`, `time_in_hospital` (all non-negative integers
    after NB02 cleaning). NaNs are NOT imputed here. Upstream NB02 §2.6.6
    drops or fills before this point. Passing a frame with NaNs in any of
    these columns silently propagates NaN through the products.

    Output columns appended
    -----------------------
    `inpatient_x_emergency`, `inpatient_x_diagnoses`,
    `inpatient_x_meds`, `inpatient_x_time`,
    `emergency_x_diagnoses`, `meds_x_diagnoses`.

    Rationale
    ---------
    NB03 §3.6 surfaced `number_inpatient` as the strongest univariate
    predictor. Tree ensembles infer pairwise splits on demand but linear
    models cannot, so this helper exists to give the L2-regularised LR a
    fair shot at the interaction structure the trees already see.
    """
    df["inpatient_x_emergency"] = df["number_inpatient"] * df["number_emergency"]
    df["inpatient_x_diagnoses"] = df["number_inpatient"] * df["number_diagnoses"]
    df["inpatient_x_meds"] = df["number_inpatient"] * df["num_medications"]
    df["inpatient_x_time"] = df["number_inpatient"] * df["time_in_hospital"]
    df["emergency_x_diagnoses"] = df["number_emergency"] * df["number_diagnoses"]
    df["meds_x_diagnoses"] = df["num_medications"] * df["number_diagnoses"]
    return df


def add_discharge_and_age_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Add discharge-disposition x utilization and age x utilization features.

    Input
    -----
    `df` must carry `discharge_group` (the NB02 rollup with the literal
    levels `"Home"` / `"Transfer"` / `"SNF/Rehab"`), `number_inpatient`
    (non-negative int), and `age_numeric` (the NB04 mid-point map of the
    age bracket string). Other `discharge_group` levels (e.g. `"Other"`,
    `"Unknown"`) produce all-zero indicator columns rather than raising,
    so the helper is safe to call on a frame that lost a level during
    cleaning.

    Output columns appended
    -----------------------
    `inpatient_x_discharge_home`, `inpatient_x_discharge_transfer`,
    `inpatient_x_discharge_snf`, `inpatient_x_age`,
    `young_high_utilizer` (binary: age < 50 AND number_inpatient >= 2).

    Rationale
    ---------
    EDA §3.6.8 showed the discharge disposition modulates the readmission-
    rate-per-inpatient-visit slope (a "Home" discharge after high prior
    inpatient is the steepest risk pocket). The `young_high_utilizer`
    flag isolates the small but high-risk under-50 frequent-flier pocket.
    """
    df["inpatient_x_discharge_home"] = df["number_inpatient"] * (df["discharge_group"] == "Home").astype(int)
    df["inpatient_x_discharge_transfer"] = df["number_inpatient"] * (df["discharge_group"] == "Transfer").astype(int)
    df["inpatient_x_discharge_snf"] = df["number_inpatient"] * (df["discharge_group"] == "SNF/Rehab").astype(int)
    df["inpatient_x_age"] = df["number_inpatient"] * df["age_numeric"]
    df["young_high_utilizer"] = ((df["age_numeric"] < 50) & (df["number_inpatient"] >= 2)).astype(int)
    return df


def add_polynomial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append squared terms for the three heaviest-tailed count predictors.

    Input
    -----
    `df` must carry `number_inpatient`, `number_emergency`,
    `total_prior_visits` (all non-negative ints). Squared columns are
    plain element-wise `** 2` on the raw counts (NOT on the `log1p`
    sidecars from NB04 §4.10), so the linear models see *both* a
    log-compressed representation and a raw squared representation and
    the §5.8 mutual-information selector picks whichever the data
    actually favours.

    Output columns appended
    -----------------------
    `inpatient_sq`, `emergency_sq`, `total_visits_sq`.
    """
    df["inpatient_sq"] = df["number_inpatient"] ** 2
    df["emergency_sq"] = df["number_emergency"] ** 2
    df["total_visits_sq"] = df["total_prior_visits"] ** 2
    return df


def add_threshold_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Binary upper-tail flags chosen to match the EDA §3.6 distribution cuts.
    These are heuristic risk markers. The binary flags reduce noise from heavy-tailed
    counts before tree models see them and give linear models a cleaner step signal.
    """
    df["high_utilizer"] = (df["total_prior_visits"] >= 3).astype(int)
    df["frequent_inpatient"] = (df["number_inpatient"] >= 2).astype(int)
    df["any_emergency"] = (df["number_emergency"] > 0).astype(int)
    df["long_stay"] = (df["time_in_hospital"] >= 7).astype(int)
    df["many_meds"] = (df["num_medications"] >= 15).astype(int)
    df["many_diagnoses"] = (df["number_diagnoses"] >= 7).astype(int)
    return df


def add_all_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Apply every interaction group in the canonical §4.14 order.
    The order matters for reproducibility of the persisted CSV. Tests that exercise
    a single group should call the per-group function directly. Production callers
    that want the full interaction block call this one.
    """
    df = add_per_day_ratios(df)
    df = add_cross_product_interactions(df)
    df = add_discharge_and_age_interactions(df)
    df = add_polynomial_features(df)
    df = add_threshold_flags(df)
    df = add_diabetes_complication_flags(df)
    df = add_med_complexity(df)
    df = add_lab_risk_flags(df)
    return df


def add_diabetes_complication_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Diabetes-specific complication flags from raw diag codes.
    These carry higher signal than the coarse 3-digit chapter map for readmission risk.
    Drops any columns it creates that end up constant (these codes are rare in the data,
    so the individuals and any_ flag are usually all-zero and would fail the notebook's
    post-dedup constant-column assertion).
    """
    def _is_diab_comp(code):
        if pd.isna(code):
            return 0
        s = str(code).strip()
        if not s.startswith("250"):
            return 0
        # 250.4x = nephropathy, 250.5x = ophthalmic, 250.6x = neuro,
        # 250.7x = circulatory, 250.8x = other specified complications
        return 1 if any(s.startswith(f"250.{x}") for x in "45678") else 0

    created = []
    for i in (1, 2, 3):
        col = f"diag_{i}"
        if col in df.columns:
            df[f"diag{i}_diab_comp"] = df[col].apply(_is_diab_comp).astype(int)
            created.append(f"diag{i}_diab_comp")
        else:
            df[f"diag{i}_diab_comp"] = 0
            created.append(f"diag{i}_diab_comp")

    df["any_diab_comp"] = (df[[f"diag{i}_diab_comp" for i in (1, 2, 3)]].sum(axis=1) > 0).astype(int)
    created.append("any_diab_comp")

    # Drop any we created that turned out constant (rare events → mostly zero)
    for c in created:
        if c in df.columns and df[c].nunique() <= 1:
            df = df.drop(columns=[c])

    return df


def add_med_complexity(df: pd.DataFrame) -> pd.DataFrame:
    """Count of distinct diabetes meds + clinically meaningful combo flags.
    Stronger signal than the single binary 'change' column.
    """
    med_cols = [
        "metformin", "repaglinide", "glimepiride", "glipizide",
        "glyburide", "pioglitazone", "rosiglitazone", "insulin"
    ]
    present = [c for c in med_cols if c in df.columns]

    if not present:
        df["n_diabetes_meds"] = 0
        df["insulin_plus_oral"] = 0
        df["on_3plus_diabetes_meds"] = 0
        return df

    df["n_diabetes_meds"] = df[present].apply(
        lambda r: (r != "No").sum(), axis=1
    )
    insulin = (df.get("insulin", "No") != "No").astype(int)
    any_oral = df[[c for c in present if c != "insulin"]].apply(
        lambda r: (r != "No").any(), axis=1
    ).astype(int)

    df["insulin_plus_oral"] = (insulin & any_oral).astype(int)
    df["on_3plus_diabetes_meds"] = (df["n_diabetes_meds"] >= 3).astype(int)
    return df


def add_lab_risk_flags(df: pd.DataFrame) -> pd.DataFrame:
    """A1Cresult and max_glu_serum turned into ordered risk scores + informative missingness."""
    if "A1Cresult" in df.columns:
        a1c_map = {">8": 3, ">7": 2, "Norm": 1, "None": 0}
        df["a1c_risk"] = df["A1Cresult"].map(a1c_map).fillna(0).astype(int)
        df["a1c_measured_high"] = df["A1Cresult"].isin([">7", ">8"]).astype(int)

    if "max_glu_serum" in df.columns:
        glu_map = {">300": 3, ">200": 2, "Norm": 1, "None": 0}
        df["glu_risk"] = df["max_glu_serum"].map(glu_map).fillna(0).astype(int)
    return df
