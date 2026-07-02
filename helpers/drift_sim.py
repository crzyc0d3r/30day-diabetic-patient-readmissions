"""drift_sim: synthetic "newly-arrived encounter batch" generator and the
PSI/KS drift report used by the medi-watch drift -> retrain demonstration.

In a real deployment the drift-check DAG would compare the champion's training
reference against a batch of genuinely new patient encounters streamed from the
EHR. We have no live feed, so this module manufactures that batch: it takes the
engineered feature matrix (`data/features.csv`, the same artefact NB04 writes
and the champion trains on) and applies a named, seeded perturbation that
mimics a real-world drift driver. The perturbed batch is schema-identical to
`features.csv` so the drift detector can compare it column-for-column.

Each scenario targets specific columns so the resulting drift is explainable.
A coding change moves the diagnosis categoricals (PSI), a length-of-stay policy
moves the continuous utilisation columns (KS), and so on. `severity` scales
the magnitude so a scenario can be tuned to land in the WARN (>0.1) or ALERT
(>0.2) band defined in `helpers.constants`.

This module owns two things that BOTH the authoring notebook (NB09) and the
timer drift-check DAG import, so the logic is written once:

  * `make_scenario` / `write_scenarios`: generate the perturbed batch.
  * `column_drift_report`: route each monitored column to PSI or KS and
    return a per-column table plus an overall WARN/ALERT verdict.

NOTE ON FIDELITY: perturbations are applied at the engineered-feature level for
speed. In production the driver (e.g. ICD-9 -> ICD-10) would shift the raw
data and propagate through NB04's feature engineering. Here we simulate the
resulting feature drift directly. Derived/interaction columns are left as-is.
The detector monitors the interpretable base columns in `MONITORED_COLUMNS`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

from helpers.constants import (
    DRIFT_KS_UNIQUE_THRESHOLD,
    DRIFT_PSI_ALERT,
    DRIFT_PSI_WARN,
    SEED,
)

# The modelling target carried in features.csv (and every drifted batch). The
# perturbations never touch it, so the batches stay genuinely labeled and the
# champion's harm under drift is measurable rather than guessed.
TARGET = "readmitted_binary"

# Columns the drift-check DAG monitors. Chosen to span both statistics and the
# clinically meaningful axes of the cohort, so the report covers a diverse set
# of columns. Continuous columns route to KS, categorical
# to PSI (see column_drift_report).
MONITORED_CONTINUOUS = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_inpatient",
    "number_emergency",
    "number_diagnoses",
]
MONITORED_CATEGORICAL = [
    "diag_1_cat",
    "diag_2_cat",
    "diag_3_cat",
    "admission_type",
    "discharge_group",
    "payer_grouped",
    "race",
    "gender",
    "insulin",
    "metformin",
    "A1Cresult",
    "change",
]
MONITORED_COLUMNS = MONITORED_CONTINUOUS + MONITORED_CATEGORICAL

# The five demonstration scenarios plus the no-drift control. Each maps to a
# real-world driver (see module docstring) and a target column set so a firing
# is explainable rather than mysterious.
SCENARIOS = (
    "none",                   # control: clean resample, expected verdict OK
    "coding_shift",           # ICD-9 -> ICD-10 recoding   -> PSI on diag_*
    "casemix_shift",          # population / referral shift -> PSI + age KS
    "los_utilization_shift",  # LOS / utilisation policy    -> KS on continuous
    "formulary_shift",        # guideline / formulary change-> PSI on meds
    "mixed_severe",           # EHR migration: everything moves -> PSI + KS
)


# --------------------------------------------------------------------------- #
# Drift statistics (PSI + KS). Single source of truth: the notebook and the
# timer DAG both call column_drift_report rather than reimplementing the math.
# --------------------------------------------------------------------------- #
def _psi_from_fractions(ref_frac: np.ndarray, cur_frac: np.ndarray, eps: float = 1e-6) -> float:
    """Population Stability Index from two aligned bin-fraction vectors."""
    ref_frac = np.clip(ref_frac, eps, None)
    cur_frac = np.clip(cur_frac, eps, None)
    return float(np.sum((cur_frac - ref_frac) * np.log(cur_frac / ref_frac)))


def psi_numeric(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    """PSI for a numeric column, binned on the reference's quantiles."""
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # near-constant column: no meaningful bins
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_c = np.histogram(ref, bins=edges)[0] / max(len(ref), 1)
    cur_c = np.histogram(cur, bins=edges)[0] / max(len(cur), 1)
    return _psi_from_fractions(ref_c, cur_c)


def psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    """PSI for a categorical column over the union of observed categories."""
    ref_s, cur_s = ref.astype(str), cur.astype(str)
    cats = pd.Index(ref_s.unique()).union(cur_s.unique())
    ref_c = ref_s.value_counts(normalize=True).reindex(cats, fill_value=0.0).to_numpy()
    cur_c = cur_s.value_counts(normalize=True).reindex(cats, fill_value=0.0).to_numpy()
    return _psi_from_fractions(ref_c, cur_c)


def ks_statistic(ref: np.ndarray, cur: np.ndarray) -> float:
    """Kolmogorov-Smirnov two-sample statistic (max CDF gap)."""
    return float(ks_2samp(ref, cur).statistic)


# --------------------------------------------------------------------------- #
# Wasserstein + KL: added alongside PSI/KS so the Driftly dashboard (and any
# future pipeline use) draws all four drift statistics from one place. Numeric
# binning mirrors psi_numeric exactly (reference quantile edges) so PSI and KL
# are computed over identical bins and stay directly comparable.
# --------------------------------------------------------------------------- #
def wasserstein_numeric(ref: np.ndarray, cur: np.ndarray) -> float:
    """Wasserstein-1 (earth-mover) distance between two numeric samples,
    normalized by the reference standard deviation so the value is scale-free
    and comparable across features (and threshold-able). A near-constant
    reference (std ~ 0) returns 0.0 rather than dividing by zero. Numeric only;
    categorical columns have no natural ground distance, so the dashboard
    reports `null` for them rather than calling this."""
    ref = np.asarray(ref, dtype=float)
    cur = np.asarray(cur, dtype=float)
    if len(ref) == 0 or len(cur) == 0:
        return 0.0
    raw = float(wasserstein_distance(ref, cur))
    std = float(np.std(ref))
    return raw / std if std > 1e-12 else 0.0


def _kl_from_fractions(cur_frac: np.ndarray, ref_frac: np.ndarray, eps: float = 1e-6) -> float:
    """KL(current || reference) from two aligned bin-fraction vectors. The
    direction is deliberate: it measures the information lost when the (stable)
    reference distribution is used to approximate the (incoming) current one, so
    a bin the current batch populates but the reference rarely did dominates the
    sum. eps-smoothing mirrors `_psi_from_fractions` so empty bins never yield
    log(0) or division by zero."""
    cur_frac = np.clip(cur_frac, eps, None)
    ref_frac = np.clip(ref_frac, eps, None)
    return float(np.sum(cur_frac * np.log(cur_frac / ref_frac)))


def kl_numeric(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    """KL(current || reference) for a numeric column, binned on the reference's
    quantiles (identical binning to psi_numeric)."""
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # near-constant column: no meaningful bins
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_c = np.histogram(ref, bins=edges)[0] / max(len(ref), 1)
    cur_c = np.histogram(cur, bins=edges)[0] / max(len(cur), 1)
    return _kl_from_fractions(cur_c, ref_c)


def kl_categorical(ref: pd.Series, cur: pd.Series) -> float:
    """KL(current || reference) for a categorical column over the union of
    observed categories (identical category handling to psi_categorical)."""
    ref_s, cur_s = ref.astype(str), cur.astype(str)
    cats = pd.Index(ref_s.unique()).union(cur_s.unique())
    ref_c = ref_s.value_counts(normalize=True).reindex(cats, fill_value=0.0).to_numpy()
    cur_c = cur_s.value_counts(normalize=True).reindex(cats, fill_value=0.0).to_numpy()
    return _kl_from_fractions(cur_c, ref_c)


def column_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, str]:
    """Per-column PSI/KS report plus an overall verdict.

    Routing mirrors `evidently_drift_dag`: a numeric column with more than
    `DRIFT_KS_UNIQUE_THRESHOLD` distinct values uses KS, and everything else
    uses PSI. The WARN/ALERT cutoffs (`DRIFT_PSI_WARN` / `DRIFT_PSI_ALERT`) are
    applied to both statistics, matching the convention documented in
    `helpers.constants`.

    Returns `(report_df, verdict)` where `verdict` is the worst per-column
    status ("ALERT" > "WARN" > "OK"). The timer DAG branches on this verdict.
    """
    if columns is None:
        columns = [c for c in MONITORED_COLUMNS if c in reference.columns and c in current.columns]

    rows = []
    for col in columns:
        ref = reference[col].dropna()
        cur = current[col].dropna()
        if len(ref) == 0 or len(cur) == 0:
            continue
        numeric = pd.api.types.is_numeric_dtype(ref)
        if numeric and ref.nunique() > DRIFT_KS_UNIQUE_THRESHOLD:
            method, stat = "KS", ks_statistic(ref.to_numpy(), cur.to_numpy())
        elif numeric:
            method, stat = "PSI", psi_numeric(ref.to_numpy(), cur.to_numpy())
        else:
            method, stat = "PSI", psi_categorical(ref, cur)

        if stat >= DRIFT_PSI_ALERT:
            status = "ALERT"
        elif stat >= DRIFT_PSI_WARN:
            status = "WARN"
        else:
            status = "OK"
        rows.append({"column": col, "method": method, "statistic": round(stat, 4), "status": status})

    report = pd.DataFrame(rows, columns=["column", "method", "statistic", "status"])
    if (report["status"] == "ALERT").any():
        verdict = "ALERT"
    elif (report["status"] == "WARN").any():
        verdict = "WARN"
    else:
        verdict = "OK"
    return report, verdict


# --------------------------------------------------------------------------- #
# Champion-impact: the drifted batch keeps the real `readmitted_binary` label
# (perturbations touch only feature columns), so we can measure how much each
# scenario actually degrades the live champion rather than only flagging that
# the inputs moved. Production triggers retraining on measured harm. This is
# the same signal, computed honestly on labeled data.
# --------------------------------------------------------------------------- #
def load_champion_scorer(
    model_bundle_path: str = "data/final_model.joblib",
    pipeline_path: str = "data/full_inference_pipeline.joblib",
):
    """Return `(predict_proba_fn, threshold)` for the deployed champion.

    `predict_proba_fn(df)` takes a raw engineered-feature frame (targets may
    be present, and only the preprocessing pipeline's expected feature columns
    are used) and returns positive-class probabilities. The champion classifier and
    its recommended decision threshold come from the `final_model.joblib`
    bundle. The preprocessing (encode + scale + select) comes from
    `full_inference_pipeline.joblib`.

    joblib is imported lazily so importing this module never requires the
    artefacts to be present. These are the project's own model artefacts (the
    same ones the inference API and conclusion pipeline load), not untrusted
    input, so joblib deserialisation is safe here.
    """
    import joblib

    bundle = joblib.load(model_bundle_path)
    model = bundle["model"]
    threshold = float(bundle.get("recommended_threshold", 0.5))
    prep = joblib.load(pipeline_path)
    cols = list(prep.feature_names_in_)

    def predict_proba_fn(df: pd.DataFrame) -> np.ndarray:
        transformed = prep.transform(df[cols])
        return model.predict_proba(transformed)[:, 1]

    return predict_proba_fn, threshold


def champion_impact(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    predict_proba_fn,
    *,
    threshold: float = 0.5,
    target: str = TARGET,
) -> dict:
    """Champion metric panel on reference vs current, plus per-metric delta.

    Returns `{"reference": {...}, "current": {...}, "delta": {...},
    "threshold": ...}` where each panel is `helpers.evaluation.metric_panel`
    and `delta` is `current - reference` per metric. Because the drifted
    batch is genuinely labeled, the delta is measured harm, not a guess.
    """
    from helpers.evaluation import metric_panel

    def panel(df: pd.DataFrame) -> dict:
        proba = np.asarray(predict_proba_fn(df), dtype=float)
        pred = (proba >= threshold).astype(int)
        y_true = np.asarray(df[target])
        return metric_panel(y_true, pred, proba)

    ref_panel, cur_panel = panel(reference), panel(current)
    delta = {k: float(cur_panel[k] - ref_panel[k]) for k in ref_panel}
    return {
        "reference": ref_panel,
        "current": cur_panel,
        "delta": delta,
        "threshold": float(threshold),
    }


def build_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    scenario: str = "current",
    predict_proba_fn=None,
    threshold: float = 0.5,
) -> dict:
    """Bundle the per-column PSI/KS report, verdict, metadata, and (when a
    champion scorer is supplied and the target is present in both frames) the
    champion-impact block into one JSON-serialisable dict.

    A scoring failure is captured under `champion_impact_error` rather than
    raised, so the drift verdict survives even when the champion cannot be
    scored (e.g. a missing artefact).
    """
    report_df, verdict = column_drift_report(reference, current)
    out = {
        "scenario": scenario,
        "verdict": verdict,
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "thresholds": {"warn": float(DRIFT_PSI_WARN), "alert": float(DRIFT_PSI_ALERT)},
        "columns": report_df.to_dict(orient="records"),
    }
    if predict_proba_fn is not None and TARGET in reference.columns and TARGET in current.columns:
        try:
            out["champion_impact"] = champion_impact(
                reference, current, predict_proba_fn, threshold=threshold)
        except Exception as e:  # noqa: BLE001 (report must survive a scoring failure)
            out["champion_impact_error"] = str(e)
    return out


# --------------------------------------------------------------------------- #
# Perturbation primitives: small, column-agnostic, guarded against missing
# columns so a scenario degrades gracefully if the schema changes.
# --------------------------------------------------------------------------- #
def _widen_for(s: pd.Series, value) -> pd.Series:
    """Return `s` widened to a dtype that can hold `value`.

    Writing a label the column's dtype can't represent (e.g. the string
    `"Ch"` into the int-coded `change` column) triggers a pandas
    `FutureWarning` and will eventually raise. We promote
    to `object` up front so the assignment is explicit and lossless for the
    values already present, rather than leaning on the deprecated coercion.
    """
    try:
        fits = np.can_cast(np.min_scalar_type(value), s.dtype, casting="safe")
    except TypeError:
        fits = False
    return s if fits else s.astype(object)


def _reassign_to(s: pd.Series, rng: np.random.Generator, value, frac: float) -> pd.Series:
    """Set `frac` of randomly chosen rows to `value`."""
    s = s.copy()
    k = int(round(min(max(frac, 0.0), 1.0) * len(s)))
    if k > 0:
        idx = rng.choice(len(s), size=k, replace=False)
        s = _widen_for(s, value)
        s.iloc[idx] = value
    return s


def _reassign_to_rare(s: pd.Series, rng: np.random.Generator, frac: float) -> pd.Series:
    """Push `frac` of rows onto the rarest observed category (max PSI move)."""
    counts = s.astype(str).value_counts()
    if counts.empty:
        return s
    return _reassign_to(s, rng, counts.index[-1], frac)


def _flip_value(s: pd.Series, rng: np.random.Generator, frm, to, frac: float) -> pd.Series:
    """Convert `frac` of the rows currently equal to `frm` into `to`."""
    s = s.copy()
    cand = np.where(s.astype(str).to_numpy() == str(frm))[0]
    k = int(round(min(max(frac, 0.0), 1.0) * len(cand)))
    if k > 0:
        idx = rng.choice(cand, size=k, replace=False)
        s = _widen_for(s, to)
        s.iloc[idx] = to
    return s


def _shift_numeric(
    s: pd.Series,
    mult: float = 1.0,
    add: float = 0.0,
    integer: bool = False,
    lo: float | None = None,
    hi: float | None = None,
) -> pd.Series:
    """Scale-and-shift a numeric column, clipped and optionally re-integerised."""
    out = s.astype(float) * mult + add
    if lo is not None:
        out = out.clip(lower=lo)
    if hi is not None:
        out = out.clip(upper=hi)
    if integer:
        out = out.round().astype("int64")
    return out


# --------------------------------------------------------------------------- #
# Scenario generator
# --------------------------------------------------------------------------- #
def make_scenario(
    reference: pd.DataFrame,
    scenario: str,
    *,
    severity: float = 1.0,
    seed: int = SEED,
    batch_size: int | None = None,
) -> pd.DataFrame:
    """Return a schema-identical batch perturbed per `scenario`.

    `reference` is the engineered feature frame (`data/features.csv`).
    `severity` scales every perturbation linearly (≈0.4 lands in WARN, ≈1.0+
    in ALERT for the targeted columns). `batch_size` defaults to the
    reference row count. The batch is a seeded bootstrap resample so even the
    `none` control is a fresh sample rather than the identity.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; expected one of {SCENARIOS}")

    rng = np.random.default_rng(seed)
    n = batch_size or len(reference)
    idx = rng.integers(0, len(reference), size=n)
    df = reference.iloc[idx].reset_index(drop=True).copy()

    if scenario == "none":
        return df

    if scenario in ("coding_shift", "mixed_severe"):
        # ICD-9 -> ICD-10: a block of encounters arrives coded under a new
        # scheme the model has never seen. A fresh category moves PSI hard
        # (the reference puts zero mass on it) and is the faithful signature
        # of a coding-system migration rather than a within-vocabulary shift.
        for col in ("diag_1_cat", "diag_2_cat", "diag_3_cat"):
            if col in df.columns:
                df[col] = _reassign_to(df[col], rng, "ICD10_recode", 0.30 * severity)

    if scenario in ("casemix_shift", "mixed_severe"):
        # Older, more acute, differently-insured population through a new
        # referral channel.
        if "age_mid" in df.columns:
            df["age_mid"] = _shift_numeric(df["age_mid"], add=12.0 * severity, lo=0, hi=100)
        if "admission_type" in df.columns:
            df["admission_type"] = _reassign_to(df["admission_type"], rng, "Emergency", 0.25 * severity)
        if "admission_source" in df.columns:
            df["admission_source"] = _reassign_to(df["admission_source"], rng, "Emergency", 0.20 * severity)
        if "payer_grouped" in df.columns:
            df["payer_grouped"] = _reassign_to_rare(df["payer_grouped"], rng, 0.25 * severity)
        if "gender" in df.columns:
            df["gender"] = _reassign_to(df["gender"], rng, "Female", 0.15 * severity)

    if scenario in ("los_utilization_shift", "mixed_severe"):
        # Sicker cohort: longer stays, more labs, more meds, more inpatient.
        if "time_in_hospital" in df.columns:
            df["time_in_hospital"] = _shift_numeric(df["time_in_hospital"], mult=1.0 + 0.6 * severity, integer=True, lo=0)
        if "num_lab_procedures" in df.columns:
            df["num_lab_procedures"] = _shift_numeric(df["num_lab_procedures"], mult=1.0 + 0.4 * severity, integer=True, lo=0)
        if "num_medications" in df.columns:
            df["num_medications"] = _shift_numeric(df["num_medications"], mult=1.0 + 0.4 * severity, integer=True, lo=0)
        if "number_inpatient" in df.columns:
            df["number_inpatient"] = _shift_numeric(df["number_inpatient"], add=1.0 * severity, integer=True, lo=0)

    if scenario in ("formulary_shift", "mixed_severe"):
        # New guideline pushes insulin up and starts more patients on metformin.
        if "metformin" in df.columns:
            df["metformin"] = _flip_value(df["metformin"], rng, frm="No", to="Steady", frac=0.40 * severity)
        if "insulin" in df.columns:
            df["insulin"] = _reassign_to(df["insulin"], rng, "Up", 0.25 * severity)
        if "change" in df.columns:
            # `change` is the 0/1 "any medication change" indicator in
            # features.csv (numeric, scaled by the champion preprocessor). A
            # formulary change pushes more encounters into "changed", so drift
            # it toward 1, never the raw string "Ch", which would corrupt the
            # numeric column and make the batch unscoreable by the champion.
            df["change"] = _reassign_to(df["change"], rng, 1, 0.30 * severity)
        if "n_med_changes" in df.columns:
            df["n_med_changes"] = _shift_numeric(df["n_med_changes"], mult=1.0 + 0.5 * severity, integer=True, lo=0)

    return df


def write_scenarios(
    reference_path: str = "data/features.csv",
    out_dir: str = "data/incoming",
    scenarios=SCENARIOS,
    severity: float = 1.0,
    seed: int = SEED,
) -> dict[str, str]:
    """Generate every scenario batch and write `<out_dir>/<scenario>.csv`.

    Returns a mapping of scenario -> written path. Used by NB09 to materialise
    the demonstration batches the timer DAG later consumes.
    """
    import os

    os.makedirs(out_dir, exist_ok=True)
    reference = pd.read_csv(reference_path, low_memory=False)
    written: dict[str, str] = {}
    for i, scenario in enumerate(scenarios):
        # Vary the seed per scenario so the bootstrap draws differ, but stay
        # deterministic across runs.
        batch = make_scenario(reference, scenario, severity=severity, seed=seed + i)
        path = os.path.join(out_dir, f"{scenario}.csv")
        batch.to_csv(path, index=False)
        written[scenario] = path
    return written
