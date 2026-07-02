"""Drift simulator: inject a chosen drift scenario into the running dataset.

Reuses ``helpers.drift_sim.make_scenario`` (the same seeded perturbations NB09 and
the Airflow demo DAG use) to turn the champion reference into a perturbed batch,
then atomically overwrites the live slot ``data/incoming/current.csv``. That is
the file the Monitor reads and the Airflow ``scheduled_drift_check`` DAG watches,
so an injection is immediately visible to both. Triggering the pipeline is a
separate, optional step (see ``airflow_client``).
"""
from __future__ import annotations

import os
import tempfile

import pandas as pd

import config
from helpers.drift_sim import SCENARIOS, make_scenario

# Short human descriptions for the Simulator dropdown. Mirrors the scenario
# docstrings in helpers/drift_sim.py (single source of truth for the names).
_DESCRIPTIONS = {
    "none": "Clean resample (control) — resets the running batch to OK.",
    "coding_shift": "ICD-9 → ICD-10 recoding; shifts the diagnosis categoricals (PSI).",
    "casemix_shift": "Population / referral shift; moves age, admission type, payer.",
    "los_utilization_shift": "Length-of-stay / utilisation policy change (KS on continuous).",
    "formulary_shift": "Guideline / formulary change; shifts the medication columns (PSI).",
    "mixed_severe": "EHR migration: every monitored axis moves at once (PSI + KS).",
}


def list_scenarios() -> list[dict]:
    """Available scenarios (from drift_sim.SCENARIOS) with short descriptions."""
    return [{"name": s, "description": _DESCRIPTIONS.get(s, "")} for s in SCENARIOS]


def inject(reference: pd.DataFrame, scenario: str, severity: float = 1.0,
           *, seed: int | None = None) -> dict:
    """Generate ``scenario`` from ``reference`` and overwrite the live slot.

    ``severity`` scales the perturbation magnitude (1.0 = the canonical band).
    The write is atomic (temp file + ``os.replace``) so a Monitor poll or an
    Airflow run never reads a half-written current.csv. Returns metadata about
    the injected batch.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario '{scenario}'; expected one of {list(SCENARIOS)}")

    kwargs = {"severity": severity}
    if seed is not None:
        kwargs["seed"] = seed
    batch = make_scenario(reference, scenario, **kwargs)

    dest = config.current_path()
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".csv", dir=os.path.dirname(dest))
    os.close(fd)
    try:
        batch.to_csv(tmp, index=False)
        # mkstemp creates 0600; make it world-readable so the host and the
        # Airflow worker (a different uid) can read the live slot.
        os.chmod(tmp, 0o644)
        os.replace(tmp, dest)  # atomic swap of the live slot
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return {"scenario": scenario, "severity": float(severity),
            "rows": int(len(batch)), "path": dest}
