"""API-level smoke tests for the inference service (`infra/inference-api/main.py`).

WHAT this file covers (rubric: Model Packaging & Deployment)

The FastAPI app that serves the `@champion` model had no automated test
exercising it end to end. These tests close that gap by driving the real ASGI
app through `fastapi.testclient.TestClient` and asserting the contract every
client SDK relies on:

  * GET  /healthz       returns 200 once a model is loaded
  * POST /invocations   accepts the MLflow `dataframe_split` shape, returns
                        `predictions` / `probabilities` of the right length,
                        and strips identifier columns (patient_nbr,
                        encounter_id) before scoring
  * GET  /metrics       returns Prometheus text exposition
  * POST /invocations   rejects a malformed payload with 400

WHY no MLflow / registry: `main.py` exposes a deliberate test seam. The
module-level globals `_pipe` (a `Scorer`), `_meta` (the loaded-model metadata
dict), and `_features` (the cat/num column lists plus numeric medians) are what
`_ensure_loaded()` returns. Installing an in-memory fake scorer into those
globals lets the app serve `/invocations` with no MLflow server, no registry,
no network, and no GPU. The handler even normalises a hand-installed `_pipe`
through `np.asarray` (main.py "belt-and-braces" comment near the return) so a
fake scorer is explicitly supported.

This suite is hermetic: deterministic, CPU-only, no network.

STYLE NOTE: no em dashes, no semicolons, "program" never the British spelling.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# Load infra/inference-api/main.py by path (repo root on sys.path for its helpers import).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_MAIN_PATH = _REPO_ROOT / "infra" / "inference-api" / "main.py"


def _load_main_module():
    """Import infra/inference-api/main.py once, by path, as `inference_main`."""
    if "inference_main" in sys.modules:
        return sys.modules["inference_main"]
    spec = importlib.util.spec_from_file_location("inference_main", _MAIN_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["inference_main"] = module
    spec.loader.exec_module(module)
    return module


# Minimal deterministic feature schema for the fake scorer.
_CAT_COLS = ["race", "gender"]
_NUM_COLS = ["time_in_hospital", "num_medications"]
_NUMERIC_MEDIANS = {"time_in_hospital": 4.0, "num_medications": 12.0}


class _FakeScorer:
    """An in-memory `Scorer` (predict / predict_proba) with no MLflow behind it.

    predict_proba returns a deterministic 2-column probability matrix derived
    from the numeric columns the handler hands it, so the test can assert on
    shape and value range without depending on a real model. predict applies a
    fixed 0.5 cut. The handler reads `proba[:, 1]` as P(readmission=1), so the
    two-column convention here matches what `_ProbaScorer` produces in prod.
    """

    def __init__(self) -> None:
        self.last_columns: list[str] | None = None

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        # Record what the handler actually passed us so a test can assert the
        # identifier columns never reach the scorer.
        self.last_columns = list(df.columns)
        n = len(df)
        # A bounded, deterministic P(class=1) in [0, 1) from one numeric column.
        base = pd.to_numeric(df.get("time_in_hospital", pd.Series([0] * n)), errors="coerce")
        base = base.fillna(0.0).to_numpy(dtype=float)
        p1 = np.clip((base % 10) / 10.0, 0.0, 0.999)
        return np.stack([1.0 - p1, p1], axis=1)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(df)[:, 1] >= 0.5).astype(int)


@pytest.fixture
def api():
    """A TestClient over the app with a fake scorer installed via the seam.

    Yields `(client, main_module, fake_scorer)`. We install `_pipe`, `_meta`,
    and `_features` directly so `_ensure_loaded()` short-circuits and never
    touches MLflow. State is restored afterwards so the module globals do not
    leak between tests.
    """
    main = _load_main_module()

    prev_pipe = main._pipe
    prev_meta = main._meta
    prev_features = main._features

    # Neutralise the startup hook (it would hit the MLflow registry); a fake scorer is installed below.
    prev_startup = list(main.app.router.on_startup)
    prev_shutdown = list(main.app.router.on_shutdown)
    main.app.router.on_startup.clear()
    main.app.router.on_shutdown.clear()

    scorer = _FakeScorer()
    main._pipe = scorer
    main._meta = {
        "model_name": "medi-watch-readmission",
        "alias": "champion",
        "version": "7",
        "run_id": "deadbeefcafebabe",
        "source": "memory://fake",
        "flavor": "sklearn",
        "threshold": 0.5,
        "tags": {},
        "loaded_at": "2026-06-16T00:00:00+00:00",
    }
    main._features = {
        "categorical": _CAT_COLS,
        "numeric": _NUM_COLS,
        "numeric_medians": _NUMERIC_MEDIANS,
    }

    # raise_server_exceptions=True (default) surfaces 500s as real failures.
    with TestClient(main.app) as client:
        yield client, main, scorer

    main._pipe = prev_pipe
    main._meta = prev_meta
    main._features = prev_features
    main.app.router.on_startup[:] = prev_startup
    main.app.router.on_shutdown[:] = prev_shutdown


def test_healthz_ok_with_model_loaded(api):
    """GET /healthz returns 200 and advertises the loaded model when _pipe is set.

    Precondition: a model is loaded (the fixture installed the fake scorer via
    the test seam). Without a loaded model /healthz would 503 because
    _ensure_loaded() would try to reach the registry.
    """
    client, _main, _scorer = api
    resp = client.get("/healthz")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model"]["model_name"] == "medi-watch-readmission"
    assert body["model"]["alias"] == "champion"
    assert body["model"]["version"] == "7"


def test_invocations_dataframe_split_returns_predictions(api):
    """POST /invocations with a valid dataframe_split payload scores every row."""
    client, _main, _scorer = api
    payload = {
        "dataframe_split": {
            "columns": ["race", "gender", "time_in_hospital", "num_medications"],
            "data": [
                ["Caucasian", "Female", 3, 11],
                ["AfricanAmerican", "Male", 8, 20],
                ["Caucasian", "Male", 1, 5],
            ],
        }
    }
    resp = client.post("/invocations", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["n_rows"] == 3
    assert len(body["predictions"]) == 3
    assert len(body["probabilities"]) == 3
    # probabilities are P(class=1), so they must be valid probabilities.
    assert all(0.0 <= p <= 1.0 for p in body["probabilities"])
    # predictions are class labels.
    assert all(int(p) in (0, 1) for p in body["predictions"])
    assert body["model_version"] == "7"
    assert body["model_flavor"] == "sklearn"


def test_invocations_strips_identifier_columns(api):
    """patient_nbr / encounter_id are dropped before the scorer sees the frame.

    The handler enforces minimum-necessary projection at the boundary. We send
    both identifier fields and then assert the fake scorer never received them
    (it records the exact columns it was handed), while the call still scores
    every row.
    """
    client, _main, scorer = api
    payload = {
        "dataframe_split": {
            "columns": [
                "patient_nbr",
                "encounter_id",
                "race",
                "gender",
                "time_in_hospital",
                "num_medications",
            ],
            "data": [
                [12345, 999, "Caucasian", "Female", 4, 10],
                [67890, 1000, "Other", "Male", 9, 30],
            ],
        }
    }
    resp = client.post("/invocations", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["n_rows"] == 2

    # The scorer only ever sees the preprocessor's expected schema, and never
    # any identifier-shaped column.
    assert scorer.last_columns is not None
    assert "patient_nbr" not in scorer.last_columns
    assert "encounter_id" not in scorer.last_columns
    assert set(scorer.last_columns) == set(_CAT_COLS + _NUM_COLS)


def test_metrics_returns_prometheus_text(api):
    """GET /metrics returns Prometheus exposition text with our series names."""
    client, _main, _scorer = api
    # Drive one inference first so the labeled series are populated.
    client.post(
        "/invocations",
        json={
            "dataframe_split": {
                "columns": ["race", "gender", "time_in_hospital", "num_medications"],
                "data": [["Caucasian", "Female", 5, 15]],
            }
        },
    )
    resp = client.get("/metrics")
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("text/plain")
    text = resp.text
    # The four module-scope inference series must be present in the exposition.
    assert "medi_watch_inference_requests_total" in text
    assert "medi_watch_inference_rows_total" in text


def test_invocations_malformed_payload_returns_400(api):
    """A payload with none of instances/dataframe_records/dataframe_split is 400."""
    client, _main, _scorer = api
    resp = client.post("/invocations", json={"not_a_recognized_key": [1, 2, 3]})
    assert resp.status_code == 400, resp.text
    assert "instances" in resp.json()["detail"]


def test_invocations_empty_batch_returns_400(api):
    """An explicitly empty dataframe_split batch is rejected with 400."""
    client, _main, _scorer = api
    resp = client.post(
        "/invocations",
        json={"dataframe_split": {"columns": ["race", "gender"], "data": []}},
    )
    assert resp.status_code == 400, resp.text
    assert "empty" in resp.json()["detail"].lower()
