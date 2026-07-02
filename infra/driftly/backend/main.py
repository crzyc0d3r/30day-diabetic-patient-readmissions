"""Driftly API — on-demand model-drift computation.

Computes Wasserstein / PSI / KL between the champion's training reference
(``data/features.csv``) and a chosen current batch (``data/incoming/*.csv`` or an
uploaded CSV), persists each run to a SQLite history store, and serves it all to
the React dashboard. No MLflow dependency.

Routes (all under /api):
  GET  /api/health           liveness
  GET  /api/batches          available current batches in data/incoming
  GET  /api/features         monitored columns present in the reference (+ type)
  POST /api/compute          compute drift for a named batch  (JSON {batch})
  POST /api/compute/upload   compute drift for an uploaded CSV (multipart file)
  GET  /api/history          past runs for the trend chart (optional ?feature=)

The service is unauthenticated and intended for a trusted internal network,
matching the inference API's posture. It reads CSVs but never persists row-level
data beyond the aggregate drift metrics.
"""
from __future__ import annotations

import io
import os
from datetime import datetime, timezone

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import airflow_client
import config
import history
import simulator
from drift_metrics import compute_drift
from helpers.drift_sim import MONITORED_COLUMNS

app = FastAPI(title="Driftly", version="1.0.0",
              description="On-demand model-drift dashboard (Wasserstein/PSI/KL).")

# Same-origin in production (nginx proxies /api), but allow any origin so the API
# is also reachable directly on :8003 during development.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# Reference cache keyed on file mtime: features.csv is ~71k rows, so re-reading
# it on every /features and /compute call is wasteful. Invalidates automatically
# when the pipeline rewrites the file.
_ref_cache: dict[str, object] = {"key": None, "df": None}


def _load_reference() -> pd.DataFrame:
    path = config.reference_path()
    if not os.path.exists(path):
        raise HTTPException(
            status_code=503,
            detail=(f"reference {path} not present yet — run notebooks 01-05 "
                    "(or the prepare_data DAG) to materialise features.csv."),
        )
    key = (path, os.path.getmtime(path))  # path-aware so tests can swap data dirs
    if _ref_cache["key"] != key:
        _ref_cache["df"] = pd.read_csv(path, low_memory=False)
        _ref_cache["key"] = key
    return _ref_cache["df"]  # type: ignore[return-value]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _mtime_iso(path: str) -> str:
    return (datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc)
            .isoformat(timespec="seconds").replace("+00:00", "Z"))


def _run(current: pd.DataFrame, batch: str) -> dict:
    reference = _load_reference()
    monitored = [c for c in MONITORED_COLUMNS if c in reference.columns]
    present = [c for c in monitored if c in current.columns]
    if not present:
        raise HTTPException(
            status_code=422,
            detail=("current batch has none of the monitored columns; "
                    f"expected some of {monitored}"),
        )
    result = compute_drift(reference, current, batch)
    result["computed_at"] = _now_iso()
    missing = [c for c in monitored if c not in current.columns]
    if missing:
        result["warning"] = (
            f"current batch missing {len(missing)} monitored column(s): "
            f"{missing}; computed on the intersection.")
    history.record_run(result)
    return result


class ComputeRequest(BaseModel):
    batch: str = "current"


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/batches")
def batches() -> list[dict]:
    out: list[dict] = []
    inc = config.incoming_dir()
    if os.path.isdir(inc):
        for fname in sorted(os.listdir(inc)):
            if not fname.endswith(".csv"):
                continue
            name = fname[:-4]
            path = os.path.join(inc, fname)
            try:
                with open(path) as fh:
                    rows = max(sum(1 for _ in fh) - 1, 0)  # minus header
            except OSError:
                rows = None
            out.append({"name": name, "rows": rows, "is_current": name == "current"})
    return out


@app.get("/api/features")
def features() -> list[dict]:
    reference = _load_reference()
    return [
        {"name": c,
         "type": "numeric" if pd.api.types.is_numeric_dtype(reference[c]) else "categorical"}
        for c in MONITORED_COLUMNS if c in reference.columns
    ]


@app.post("/api/compute")
def compute(req: ComputeRequest) -> dict:
    name = req.batch or "current"
    fname = "current" if name in ("", "current") else name
    path = os.path.join(config.incoming_dir(), f"{fname}.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"batch '{name}' not found at {path}")
    current = pd.read_csv(path, low_memory=False)
    return _run(current, fname)


@app.post("/api/compute/upload")
async def compute_upload(file: UploadFile = File(...)) -> dict:
    raw = await file.read()
    try:
        current = pd.read_csv(io.BytesIO(raw), low_memory=False)
    except Exception as exc:  # noqa: BLE001 — surface a clean 400, not a 500
        raise HTTPException(status_code=400, detail=f"could not parse CSV: {exc}")
    return _run(current, f"upload:{file.filename}")


@app.get("/api/history")
def get_history(feature: str | None = None) -> list[dict]:
    return history.list_runs(feature=feature)


# --------------------------------------------------------------------------- #
# Monitor: live drift of the running dataset (data/incoming/current.csv).
# --------------------------------------------------------------------------- #
@app.get("/api/monitor")
def monitor() -> dict:
    """Drift of the running batch (the live current.csv slot) vs the reference.

    This is what the Monitor view polls on an interval. It always reflects the
    current live batch, including whatever the Simulator most recently injected.
    """
    path = config.current_path()
    if not os.path.exists(path):
        raise HTTPException(
            status_code=503,
            detail=(f"running batch {path} not present yet — inject a scenario from "
                    "the Simulator, or stage data/incoming/current.csv."),
        )
    current = pd.read_csv(path, low_memory=False)
    result = _run(current, "current")
    result["running_batch"] = "current.csv"
    result["running_batch_mtime"] = _mtime_iso(path)
    return result


# --------------------------------------------------------------------------- #
# Simulator: inject a drift scenario into the running dataset, then (optionally)
# trigger the gated Airflow drift check.
# --------------------------------------------------------------------------- #
@app.get("/api/simulator/scenarios")
def simulator_scenarios() -> list[dict]:
    return simulator.list_scenarios()


@app.get("/api/airflow/status")
def airflow_status() -> dict:
    return {"configured": airflow_client.configured(),
            "ui_url": config.airflow_ui_url(),
            "dag_id": config.DRIFT_CHECK_DAG}


class InjectRequest(BaseModel):
    scenario: str
    severity: float = 1.0
    trigger: bool = True


@app.post("/api/simulator/inject")
def simulator_inject(req: InjectRequest) -> dict:
    """Generate `scenario` and overwrite the live slot (always), then trigger the
    drift-check DAG when `trigger` is set and Airflow is configured. The data
    write and the trigger are decoupled so the Monitor reflects the injection
    even when Airflow is unavailable."""
    reference = _load_reference()
    try:
        info = simulator.inject(reference, req.scenario, req.severity)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    out: dict = {"injected": info, "triggered": False}
    if req.trigger:
        if airflow_client.configured():
            try:
                out["airflow"] = airflow_client.trigger_dag(
                    config.DRIFT_CHECK_DAG,
                    {"scenario": "current", "source": "driftly:sim",
                     "origin_scenario": req.scenario},
                )
                out["triggered"] = True
            except airflow_client.AirflowTriggerError as exc:
                out["trigger_error"] = str(exc)
        else:
            out["trigger_note"] = (
                "Airflow not configured (AIRFLOW_API_URL unset); current.csv was "
                "updated — the scheduled drift check will pick it up, or trigger it "
                "manually from the Airflow UI.")
    return out
