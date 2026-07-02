"""Unit tests for airflow_client — the Simulator's Airflow trigger (JWT auth).
Hermetic: the HTTP calls are replaced with a fake poster, so nothing hits a
network."""
from __future__ import annotations

import json

import pytest

import airflow_client


def test_configured_reflects_env(monkeypatch):
    monkeypatch.delenv("AIRFLOW_API_URL", raising=False)
    assert airflow_client.configured() is False
    monkeypatch.setenv("AIRFLOW_API_URL", "http://airflow-api-server:8080/api/v2")
    assert airflow_client.configured() is True


def test_token_url_strips_api_path(monkeypatch):
    monkeypatch.setenv("AIRFLOW_API_URL", "http://af:8080/api/v2")
    assert airflow_client._token_url() == "http://af:8080/auth/token"


def test_get_token_posts_credentials(monkeypatch):
    monkeypatch.setenv("AIRFLOW_API_URL", "http://af:8080/api/v2")
    monkeypatch.setenv("AIRFLOW_API_USER", "admin")
    monkeypatch.setenv("AIRFLOW_API_PASSWORD", "secret")
    seen = {}

    def fake_poster(url, headers, data):
        seen.update(url=url, data=data)
        return 201, '{"access_token": "JWT123"}'

    tok = airflow_client.get_token(poster=fake_poster)
    assert tok == "JWT123"
    assert seen["url"] == "http://af:8080/auth/token"
    assert json.loads(seen["data"]) == {"username": "admin", "password": "secret"}


def test_trigger_dag_uses_bearer_and_returns(monkeypatch):
    monkeypatch.setenv("AIRFLOW_API_URL", "http://af:8080/api/v2")
    monkeypatch.setenv("AIRFLOW_UI_URL", "http://localhost:8080")
    seen = {}

    def fake_poster(url, headers, data):
        seen.update(url=url, headers=headers, data=data)
        return 200, '{"state": "queued"}'

    # token supplied -> no auth round-trip; assert the Bearer header + body.
    res = airflow_client.trigger_dag(
        "scheduled_drift_check", {"scenario": "current"},
        run_id="driftly__x", token="JWT123", poster=fake_poster)

    assert seen["url"] == "http://af:8080/api/v2/dags/scheduled_drift_check/dagRuns"
    assert seen["headers"]["Authorization"] == "Bearer JWT123"
    body = json.loads(seen["data"])
    assert body["dag_run_id"] == "driftly__x" and body["conf"] == {"scenario": "current"}
    assert res["dag_id"] == "scheduled_drift_check" and res["state"] == "queued"
    assert "dag_run_id=driftly__x" in res["run_url"]


def test_trigger_dag_unconfigured_raises(monkeypatch):
    monkeypatch.delenv("AIRFLOW_API_URL", raising=False)
    with pytest.raises(airflow_client.AirflowTriggerError):
        airflow_client.trigger_dag("d", {})


def test_trigger_dag_non_2xx_raises(monkeypatch):
    monkeypatch.setenv("AIRFLOW_API_URL", "http://af/api/v2")
    with pytest.raises(airflow_client.AirflowTriggerError):
        airflow_client.trigger_dag("d", {}, token="tok", poster=lambda u, h, b: (403, "forbidden"))
