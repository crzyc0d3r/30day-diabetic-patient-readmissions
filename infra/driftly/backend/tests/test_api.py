"""API tests for the Driftly backend via FastAPI's TestClient."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

from helpers.drift_sim import MONITORED_CATEGORICAL, MONITORED_CONTINUOUS


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_batches_lists_incoming(client):
    r = client.get("/api/batches")
    assert r.status_code == 200
    names = {b["name"]: b for b in r.json()}
    assert "current" in names and "spike" in names
    assert names["current"]["is_current"] is True
    assert names["current"]["rows"] == 800


def test_features_reports_types(client):
    r = client.get("/api/features")
    assert r.status_code == 200
    types = {f["name"]: f["type"] for f in r.json()}
    assert "numeric" in types.values() and "categorical" in types.values()


def test_compute_named_batch_no_drift(client):
    r = client.post("/api/compute", json={"batch": "current"})
    assert r.status_code == 200
    body = r.json()
    assert body["verdict"] == "OK"
    assert body["batch"] == "current"
    assert "computed_at" in body and body["features"]


def test_compute_strong_drift_escalates(client):
    r = client.post("/api/compute", json={"batch": "spike"})
    assert r.status_code == 200
    assert r.json()["verdict"] in ("WARN", "ALERT")


def test_compute_unknown_batch_404(client):
    r = client.post("/api/compute", json={"batch": "does_not_exist"})
    assert r.status_code == 404


def test_compute_upload(client):
    # build a correctly-typed upload: numeric continuous columns, string cats
    rng = np.random.default_rng(11)
    data = {c: rng.normal(5.0, 2.0, 50).clip(min=0) for c in MONITORED_CONTINUOUS}
    for c in MONITORED_CATEGORICAL:
        data[c] = rng.choice(["a", "b", "c", "d"], 50)
    buf = io.BytesIO()
    pd.DataFrame(data).to_csv(buf, index=False)
    buf.seek(0)
    r = client.post("/api/compute/upload",
                    files={"file": ("mybatch.csv", buf, "text/csv")})
    assert r.status_code == 200
    assert r.json()["batch"] == "upload:mybatch.csv"


def test_compute_upload_no_monitored_columns_422(client):
    df = pd.DataFrame({"totally_unrelated": [1, 2, 3]})
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    r = client.post("/api/compute/upload",
                    files={"file": ("bad.csv", buf, "text/csv")})
    assert r.status_code == 422


def test_history_accumulates_and_orders(client):
    client.post("/api/compute", json={"batch": "current"})
    client.post("/api/compute", json={"batch": "spike"})
    runs = client.get("/api/history").json()
    assert len(runs) >= 2
    assert [r["id"] for r in runs] == sorted(r["id"] for r in runs)  # oldest-first
    # per-feature series
    feat = client.get("/api/features").json()[0]["name"]
    series = client.get(f"/api/history?feature={feat}").json()
    assert all("feature" in run for run in series)


# --------------------------------------------------------------------------- #
# Monitor + Simulator routes
# --------------------------------------------------------------------------- #
def test_monitor_computes_on_running_slot(client):
    r = client.get("/api/monitor")
    assert r.status_code == 200
    b = r.json()
    assert b["batch"] == "current"
    assert b["running_batch"] == "current.csv"
    assert "running_batch_mtime" in b and b["features"]


def test_simulator_scenarios_lists(client):
    r = client.get("/api/simulator/scenarios")
    assert r.status_code == 200
    names = [s["name"] for s in r.json()]
    assert "mixed_severe" in names and "none" in names


def test_airflow_status_unconfigured(client, monkeypatch):
    monkeypatch.delenv("AIRFLOW_API_URL", raising=False)
    r = client.get("/api/airflow/status")
    assert r.status_code == 200 and r.json()["configured"] is False


def test_inject_writes_current_and_notes_trigger_when_unconfigured(client, monkeypatch):
    import os

    import config
    monkeypatch.delenv("AIRFLOW_API_URL", raising=False)
    r = client.post("/api/simulator/inject",
                    json={"scenario": "mixed_severe", "severity": 1.0, "trigger": True})
    assert r.status_code == 200
    b = r.json()
    assert b["injected"]["scenario"] == "mixed_severe"
    assert b["triggered"] is False and "trigger_note" in b
    assert os.path.exists(config.current_path())


def test_inject_triggers_airflow_when_configured(client, monkeypatch):
    import main
    monkeypatch.setattr(main.airflow_client, "configured", lambda env=None: True)
    monkeypatch.setattr(main.airflow_client, "trigger_dag",
                        lambda dag_id, conf, **kw: {"dag_id": dag_id, "dag_run_id": "driftly__t",
                                                    "state": "queued", "run_url": "http://x"})
    r = client.post("/api/simulator/inject", json={"scenario": "coding_shift", "trigger": True})
    assert r.status_code == 200
    b = r.json()
    assert b["triggered"] is True and b["airflow"]["dag_run_id"] == "driftly__t"


def test_inject_unknown_scenario_400(client):
    r = client.post("/api/simulator/inject", json={"scenario": "nope"})
    assert r.status_code == 400
