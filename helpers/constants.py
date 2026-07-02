"""Shared constants for the medi-watch pipeline.

Centralises the threshold knobs (overfit gate, near-constant cut-off),
the canonical RNG seed, and the column groups that the cleaning /
feature-engineering / publishing layers all need to agree on. Nothing in
this module imports project code, so it is safe to import from anywhere,
including pipeline, helper modules, Airflow DAGs, the inference service,
and the GenAI testing lab.
"""

# Train/val F1 (or AUC) gap above which we flag a model as overfit.
# Calibrated against §6 baselines on the cleaned cohort: 0.15 surfaces RF
# (near-perfect train F1, val ≈ 0.00), XGBoost, CatBoost. Tighter (0.10)
# is the right cutoff once the §6.3 audit has been seen and trusted.
OVERFIT_THRESHOLD = 0.15
TIGHT_OVERFIT_THRESHOLD = 0.10  # for §7.5 CV-fold reads where folds are smaller and noisier

# Number of HPO trials per model in nb07 §7.3 / §7.3.1. 10 random/ASHA trials
# per model keep the wall clock under ~5 min on a 2-GPU cluster while sparsely
# covering the 5-7 dim search space. Bump for a more thorough sweep. Lifted
# out of nb07 §7.3's else-branch so the §7.3.1 Tuner path (default when
# MEDIWATCH_USE_TUNER=1) can see it without depending on the skipped branch.
NUM_SAMPLES = 50

# HPO CV splitter and ASHAScheduler knobs (helpers/hpo_pipeline.py). The
# splitter (StratifiedGroupKFold over patient_nbr) is set to N_SPLITS_HPO
# folds, and ASHA's max_t is pinned to the same value so the scheduler runs
# exactly one rung per CV fold. Changing N_SPLITS_HPO propagates to both
# the splitter construction and the scheduler max_t automatically.
N_SPLITS_HPO = 3
ASHA_MAX_T = N_SPLITS_HPO
ASHA_GRACE_PERIOD = 1
ASHA_REDUCTION_FACTOR = 3

# Sentinel for missing categorical values, applied identically in
# nb02 cleaning (helpers.clean_helpers.refill_categorical_unknowns) and at
# inference-time in infra/inference-api.
# Must match the value the OneHotEncoder saw at fit time, otherwise
# handle_unknown="ignore" silently zero-vectors the column at inference.
UNKNOWN_CATEGORICAL = "Unknown"

# Canonical RNG seed for the whole pipeline. Every train/test split, every
# stratified shuffle, every bootstrap resampling MUST take its randomness from
# this value (or a derived child seed), so reruns are bit-for-bit reproducible.
SEED = 42

# Columns that the UCI Diabetes-130 source codes as the literal string "?".
# nb02 §2.5.2 fills them with UNKNOWN_CATEGORICAL so OHE has a stable level,
# and the MNAR subset below is then used to compute per-column missingness
# flags before the fill is reapplied for downstream encoding.
CATEGORICAL_MISSING_COLS = (
    "race", "payer_code", "medical_specialty",
    "diag_1", "diag_2", "diag_3",
)
MNAR_FLAG_COLS = ("race", "payer_code", "medical_specialty")


# ---------------------------------------------------------------------------
# Retrain / promotion gate thresholds.
#
# Lifted out of infra/airflow/dags/retrain_on_drift_dag.py so the gate
# math is shared by the DAG, helpers/evaluation.py, and the unit tests.
# Changing a number here changes the gate everywhere.
# ---------------------------------------------------------------------------

# Bootstrap resamples used by helpers.evaluation.bootstrap_lift_ci.
# 1000 is the sweet spot, enough to stabilise the CI tails to within
# the precision the gate compares against, while keeping each retrain's
# CI computation under a second on the current test set size.
BOOTSTRAP_RESAMPLES = 1000

# Two-sided CI coverage for bootstrap_lift_ci. 0.95 means the gate's
# 'lower bound > LIFT_FLOOR' check is a 95% confidence claim.
BOOTSTRAP_CI_ALPHA = 0.95

# Minimum F1 lift (candidate - prior) the lower CI bound must exceed
# for the retrain gate to promote. 0.005 reflects the smallest delta
# the operational SLA considers meaningful.
LIFT_FLOOR = 0.005

# Minimum days between @champion swaps. The retrain gate refuses to
# promote a candidate within this window so cool-down telemetry has
# time to surface latent regressions before the next rollover.
COOLDOWN_DAYS = 7

# Maximum tolerated drop in per-subgroup recall (candidate vs prior)
# before the equity gate rejects the candidate. 0.05 is the operational
# expression of NB08 §8.11.5's "no subgroup made materially worse off".
EQUITY_RECALL_TOL = 0.05

# Canonical threshold grid for best-F1 selection on the validation set.
# NB07 and NB08 both sweep this same grid so the per-model operating
# threshold that ends up in the registered-model tags lines up with
# what the conclusion notebook plots.
import numpy as np  # noqa: E402 (local to this block)

THRESHOLD_SWEEP_GRID = np.linspace(0.05, 0.95, 91)


# ---------------------------------------------------------------------------
# Drift detection thresholds (Evidently routing in evidently_drift_dag.py).
#
# 0.1 is the WARN cutoff for PSI / KS statistics. 0.2 is the ALERT cutoff
# at which the DAG raises an MLflow tag for downstream gating.
# ---------------------------------------------------------------------------

DRIFT_PSI_WARN = 0.1
DRIFT_PSI_ALERT = 0.2

# Numeric-column cardinality above which Evidently routes through KS
# instead of PSI bucketing. 20 unique values is the operational floor
# below which PSI's discretisation behaves better. Above it, KS on the
# continuous distribution is more sensitive.
DRIFT_KS_UNIQUE_THRESHOLD = 20
