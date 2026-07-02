"""Trigger Airflow DAGs from Driftly's Simulator.

The Simulator injects drift into the running dataset and then (optionally) kicks
the gated ``scheduled_drift_check`` DAG, which re-validates drift and cascades to
``retrain_on_drift`` on a confirmed ALERT. This is the same REST handoff NB09's
demo cell uses: ``POST {AIRFLOW_API_URL}/dags/{dag_id}/dagRuns`` with HTTP Basic
auth and a ``{"conf": ...}`` body.

Stdlib-only (urllib + base64) so it needs no extra dependency, mirroring
``helpers/cicd_trigger.py``. The single network call goes through ``_http_post``,
which tests replace with a fake ``poster`` so the suite is hermetic.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from datetime import datetime, timezone

import config


class AirflowTriggerError(RuntimeError):
    """Raised on a missing config or a non-2xx response from the Airflow API."""


def configured(env: Mapping[str, str] | None = None) -> bool:
    """True when an Airflow API URL is set, so the trigger step is available."""
    return bool(config.airflow_api_url())


def _http_post(url: str, headers: dict[str, str], data: bytes) -> tuple[int, str]:
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")


def _run_id() -> str:
    return "driftly__" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")


def _token_url() -> str:
    # Airflow 3.x issues JWTs at {host}/auth/token (NOT under /api/v2). The API
    # URL is the host + /api/v2, so strip back to the host and append /auth/token.
    base = config.airflow_api_url() or ""
    root = base.rsplit("/api/", 1)[0] if "/api/" in base else base.rstrip("/")
    return root.rstrip("/") + "/auth/token"


def get_token(*, poster: Callable[[str, dict, bytes], tuple[int, str]] = _http_post) -> str:
    """Exchange the admin username/password for a short-lived JWT (Airflow 3.x)."""
    body = json.dumps({
        "username": config.airflow_api_user(),
        "password": config.airflow_api_password(),
    }).encode()
    status, text = poster(_token_url(), {"Content-Type": "application/json"}, body)
    if not (200 <= status < 300):
        raise AirflowTriggerError(f"Airflow auth failed: HTTP {status}: {text[:200]}")
    try:
        return json.loads(text)["access_token"]
    except (ValueError, KeyError) as exc:
        raise AirflowTriggerError("Airflow auth: no access_token in response") from exc


def trigger_dag(
    dag_id: str,
    conf: dict,
    *,
    run_id: str | None = None,
    token: str | None = None,
    poster: Callable[[str, dict, bytes], tuple[int, str]] = _http_post,
) -> dict:
    """Create a DAG run and return ``{dag_id, dag_run_id, state, run_url}``.

    Authenticates with a JWT (fetched via ``get_token`` unless ``token`` is
    supplied). Raises ``AirflowTriggerError`` when unconfigured, auth fails, or
    the API returns a non-2xx status.
    """
    base = config.airflow_api_url()
    if not base:
        raise AirflowTriggerError("Airflow API not configured (AIRFLOW_API_URL unset)")

    token = token or get_token(poster=poster)
    run_id = run_id or _run_id()
    url = f"{base.rstrip('/')}/dags/{dag_id}/dagRuns"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    body = json.dumps({"dag_run_id": run_id, "logical_date": None, "conf": conf}).encode()

    status, text = poster(url, headers, body)
    if not (200 <= status < 300):
        raise AirflowTriggerError(f"Airflow trigger failed: HTTP {status}: {text[:300]}")

    try:
        state = json.loads(text).get("state", "queued")
    except (ValueError, AttributeError):
        state = "queued"
    ui = config.airflow_ui_url().rstrip("/")
    run_url = f"{ui}/dags/{dag_id}?dag_run_id={urllib.parse.quote(run_id)}"
    return {"dag_id": dag_id, "dag_run_id": run_id, "state": state, "run_url": run_url}
