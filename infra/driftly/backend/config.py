"""Driftly backend configuration.

Paths are exposed as *functions* (not module-level constants) so they read the
environment at call time. That lets the test-suite point ``MEDIWATCH_DATA_DIR``
at a temp directory without re-importing the app, and matches how the rest of
the stack resolves the data root (see ``scheduled_drift_check_dag.py``).

Thresholds are read once at import: PSI reuses the pipeline's shared cutoffs so
Driftly and the Airflow drift check agree on what "drift" means; Wasserstein and
KL get their own env-overridable cutoffs since they live on different scales.
"""
from __future__ import annotations

import os

from helpers.constants import DRIFT_PSI_ALERT, DRIFT_PSI_WARN


# --- data locations (env-resolved at call time) ----------------------------- #
def data_dir() -> str:
    return os.environ.get("MEDIWATCH_DATA_DIR", "/workspace/data")


def reference_path() -> str:
    """The champion's training reference matrix (NB04's features.csv)."""
    return os.path.join(data_dir(), "features.csv")


def incoming_dir() -> str:
    """Where NB09 writes scenario batches (and `current.csv` staging)."""
    return os.path.join(data_dir(), "incoming")


def history_dir() -> str:
    return os.path.join(data_dir(), "driftly")


def history_db() -> str:
    return os.path.join(history_dir(), "history.db")


def db_url() -> str | None:
    """The shared mediwatch Postgres URL (helpers/db.py's single data seam).
    When unset (hermetic tests, bare-local runs), history.py falls back to a
    self-contained SQLite file under data/driftly/."""
    return os.environ.get("MEDIWATCH_DB_URL") or None


# --- thresholds ------------------------------------------------------------- #
def _f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# PSI reuses the pipeline's canonical WARN/ALERT bands (0.1 / 0.2) so the two
# drift surfaces never disagree. Wasserstein is normalized by reference std, so
# its bands are expressed in reference-std units; KL is in nats.
PSI_WARN = DRIFT_PSI_WARN
PSI_ALERT = DRIFT_PSI_ALERT
WASSERSTEIN_WARN = _f("DRIFTLY_WASSERSTEIN_WARN", 0.10)
WASSERSTEIN_ALERT = _f("DRIFTLY_WASSERSTEIN_ALERT", 0.25)
KL_WARN = _f("DRIFTLY_KL_WARN", 0.10)
KL_ALERT = _f("DRIFTLY_KL_ALERT", 0.25)

# Display histogram resolution (the metric binning is fixed at 10 to match
# psi_numeric; this only governs the overlay chart).
HIST_BINS = int(os.environ.get("DRIFTLY_HIST_BINS", "12"))


def thresholds() -> dict:
    """The threshold block echoed back in every compute response."""
    return {
        "psi": {"warn": PSI_WARN, "alert": PSI_ALERT},
        "wasserstein": {"warn": WASSERSTEIN_WARN, "alert": WASSERSTEIN_ALERT},
        "kl": {"warn": KL_WARN, "alert": KL_ALERT},
    }


def current_path() -> str:
    """The live staging slot the Airflow drift check watches. The Monitor reads
    drift on this file; the Simulator overwrites it to inject drift."""
    return os.path.join(incoming_dir(), "current.csv")


# --- Airflow integration (Simulator -> pipeline trigger) -------------------- #
# When AIRFLOW_API_URL is unset the Simulator still injects drift (writes
# current.csv); only the optional "trigger the pipeline now" step is disabled.
def airflow_api_url() -> str | None:
    return os.environ.get("AIRFLOW_API_URL") or None


def airflow_api_user() -> str:
    return os.environ.get("AIRFLOW_API_USER", "admin")


def airflow_api_password() -> str:
    return os.environ.get("AIRFLOW_API_PASSWORD", "")


def airflow_ui_url() -> str:
    return os.environ.get("AIRFLOW_UI_URL", "http://localhost:8080")


# The gated drift-check DAG the Simulator fires; it re-validates drift and
# cascades to retrain_on_drift only on a confirmed ALERT.
DRIFT_CHECK_DAG = os.environ.get("DRIFTLY_DRIFT_CHECK_DAG", "scheduled_drift_check")
