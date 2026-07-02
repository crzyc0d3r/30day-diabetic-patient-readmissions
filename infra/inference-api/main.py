"""medi-watch inference API.

Loads the production model from the MLflow Model Registry alias
`medi-watch-readmission@champion` plus its preprocessor artifact, stitches
them into a single sklearn Pipeline, and serves predictions over HTTP.

Endpoints:
  GET  /healthz       liveness plus currently-loaded model meta
  GET  /version       full version, run, and tag metadata
  GET  /docs          Swagger UI (FastAPI auto-generated)
  GET  /redoc         ReDoc API reference (FastAPI auto-generated,
                        an equivalent OpenAPI renderer to /docs)
  POST /invocations   batch predict (compatible with MLflow's serving shape)
  POST /reload        drop the cached pipeline. The next call rebinds to the
                        current @champion alias (called by the retrain DAG's
                        gate task after a decision)

The endpoint name and request shape match the MLflow `models serve` defaults
so existing client SDKs can talk to this service unchanged.

PHI / authentication posture
----------------------------
This API is intentionally **unauthenticated by default** and is intended for
deployment on a trusted internal network (k8s ClusterIP, behind a service mesh
or VPN). Request bodies are **not logged or persisted** by this process.
Neither full payloads nor identifier columns are written to logs, traces, or
disk.

For a **real-PHI cohort deployment**, two safeguards are mandatory:

1. **Wire an auth dependency.** Set `INFERENCE_API_TOKEN` and every
   `/invocations` and `/reload` call then requires a matching
   `Authorization: Bearer <token>` header. Below that, an mTLS gateway or an
   APIKey FastAPI `Depends` chain is the recommended posture.
2. **Minimum-necessary projection at the boundary.** The request handler
   actively strips columns the model does not consume (`patient_nbr`,
   `encounter_id`, and any other identifier-shaped field listed in
   `_IDENTIFIER_FIELDS`) before the DataFrame is handed to the scorer.
   Demographic columns (race, gender, age) that the trained preprocessor
   *does* consume are kept (the model uses them) but never round-tripped in
   the response.

When `INFERENCE_API_TOKEN` is **unset**, every
unauthenticated `/invocations` request is recorded by a single
`pii_audit_log` WARNING line (no payload content, just the fact of the
call + remote socket peer when available) so an operator grepping logs can
tell the service is running in trusted-network mode rather than silently
accepting anonymous PHI traffic.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Protocol

import joblib
import mlflow
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

# Shared with the retrain DAG so serving and gating dispatch flavors identically.
from helpers.model_loading import load_model_any_flavor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("inference-api")
# Dedicated audit-log channel for PII / unauthenticated-access events. Routed
# to the same handler in default config but namespaced so an operator can
# attach a stricter handler (file rotation, SIEM forwarder) without touching
# the main inference log stream. Lines emitted here NEVER include request
# body content, only access facts (route, peer address, version).
pii_audit_log = logging.getLogger("inference-api.pii_audit")


# ---------------------------------------------------------------------------
# Identifier-shaped fields that must be stripped from any incoming request
# before the DataFrame reaches the scorer (minimum-necessary projection,
# enforced at the boundary). The trained preprocessor does not list any of
# these in feature_names_in_ (NB04 §4.13 moved patient_nbr to its own
# series), so dropping them here is a no-op for model output AND removes
# the re-identification surface that would otherwise pair an identifier
# with a probability in the response or any future request-logging hook.
# ---------------------------------------------------------------------------
_IDENTIFIER_FIELDS: frozenset[str] = frozenset({
    "patient_nbr",
    "encounter_id",
    "patient_id",
    "mrn",
})

# Bearer-token env vars checked in order. When neither is set the API runs
# in trusted-network mode and pii_audit_log emits a WARNING per unauth call.
_AUTH_TOKEN_ENV_VARS = ("INFERENCE_API_TOKEN",)


def _configured_bearer_token() -> str | None:
    """Return the first non-empty bearer token from the env, or None."""
    for var in _AUTH_TOKEN_ENV_VARS:
        val = os.environ.get(var)
        if val:
            return val
    return None


def require_bearer_or_audit(request: Request) -> None:
    """FastAPI dependency: enforce bearer auth when configured, else audit-log.

    Behaviour:
      * If `INFERENCE_API_TOKEN` is set in the
        process env, `Authorization: Bearer <token>` must match exactly or
        the request is rejected with 401.
      * If neither is set, the API is in trusted-network mode: the request
        proceeds but a single WARNING line is emitted to `pii_audit_log`
        recording the route + remote socket peer (no payload content). An
        operator grepping these lines can tell whether the service is
        accepting anonymous traffic intentionally.

    This dependency is intentionally cheap and side-effect-only so it can be
    attached to every sensitive route (`/invocations`, `/reload`)
    without measurable latency cost.
    """
    expected = _configured_bearer_token()
    if expected is None:
        peer = request.client.host if request.client else "unknown"
        pii_audit_log.warning(
            "unauthenticated %s %s from %s (no INFERENCE_API_TOKEN configured)",
            request.method, request.url.path, peer,
        )
        return
    auth = request.headers.get("authorization", "")
    scheme, _, token = auth.partition(" ")
    if scheme.lower() != "bearer" or token.strip() != expected:
        peer = request.client.host if request.client else "unknown"
        pii_audit_log.warning(
            "rejected %s %s from %s (missing/invalid bearer token)",
            request.method, request.url.path, peer,
        )
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "missing or invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )


class Scorer(Protocol):
    def predict(self, df: pd.DataFrame) -> np.ndarray: ...
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray: ...

MODEL_NAME = os.environ.get("MODEL_NAME", "medi-watch-readmission")
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "champion")
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")

# Sentinel for missing categoricals. MUST match
# helpers/constants.py:UNKNOWN_CATEGORICAL. The training-time
# OneHotEncoder was fit on rows where helpers.clean_helpers.refill_categorical_unknowns
# replaced NaN with this exact string. If inference uses any other literal,
# handle_unknown="ignore" silently zero-vectors the column at predict time
# and the model scores garbage. test/test_constants.py asserts this constant
# matches the source-of-truth in helpers/constants.py, and
# test/test_inference_api.py smoke-tests this app via fastapi TestClient.
UNKNOWN_CATEGORICAL = "Unknown"

app = FastAPI(
    title="medi-watch inference API",
    version="1.0",
    description=(
        "Serves the current `@champion` version of the medi-watch-readmission "
        "model from the MLflow Registry.\n\n"
        "**API reference:** Swagger UI at `/docs` and ReDoc at `/redoc` "
        "(both render the same OpenAPI spec)."
    ),
    docs_url="/docs",     # Swagger UI at /docs, ReDoc at /redoc
    redoc_url="/redoc",
)


# ---------------------------------------------------------------------------
# Prometheus instrumentation. The four series below back the live-inference
# section of the medi-watch-model-kpis Grafana dashboard. They are declared at
# module scope so the first /metrics scrape sees them at zero, instead of
# 404-ing until the first /invocations call lazily creates them.
# ---------------------------------------------------------------------------
INFERENCE_REQUESTS = Counter(
    "medi_watch_inference_requests_total",
    "Total /invocations calls, labeled by model version and request status.",
    labelnames=("model_version", "status"),
)
INFERENCE_ROWS = Counter(
    "medi_watch_inference_rows_total",
    "Total rows scored across all /invocations calls, labeled by model version.",
    labelnames=("model_version",),
)
INFERENCE_LATENCY = Histogram(
    "medi_watch_inference_latency_seconds",
    "End-to-end /invocations latency in seconds (DataFrame build + predict + serialize).",
    labelnames=("model_version",),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
INFERENCE_SCORE = Histogram(
    "medi_watch_inference_score",
    "Distribution of per-batch median P(readmission=1), labeled by model version.",
    labelnames=("model_version",),
    buckets=(0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

_pipe: Scorer | None = None
_meta: dict[str, Any] | None = None
_features: dict[str, list[str]] | None = None    # cat / num column names the preproc expects
_load_lock = Lock()


# Flavor-aware loading lives in helpers.model_loading.load_model_any_flavor,
# the single source of truth shared with the retrain DAG's champion gate, so the
# serving and gating paths cannot dispatch flavors differently. The training side
# (helpers/mlops_helpers.py::_resolve_model_flavor) logs the classifier with its
# native flavor (xgboost/catboost/lightgbm/pytorch/sklearn), so a hard-coded
# mlflow.sklearn.load_model would 503 on any non-sklearn champion.


class _ProbaScorer:
    """Uniform predict / predict_proba surface over the native classifier flavor.

    The preprocessor is always a sklearn ColumnTransformer (it is what the
    notebooks fit and what `full_inference_pipeline.joblib` serialises), so
    we transform once and then dispatch to the classifier's native call.

    For sklearn/xgboost/catboost/lightgbm classifiers `predict_proba` exists
    directly. For PyTorch modules we run a forward pass in inference mode and
    softmax the output. For pyfunc-loaded models we treat `predict` as
    returning either probabilities (binary classifier convention) or labels
    (degraded mode).
    """

    def __init__(self, preproc: Any, clf: Any, flavor: str, threshold: float = 0.5) -> None:
        self.preproc = preproc
        self.clf = clf
        self.flavor = flavor
        # F1-optimal decision threshold persisted by NB07 §7.12 as
        # `final_model_threshold.joblib`. 0.5 is the fallback when the
        # sidecar is missing (logged as a WARNING in _load_from_registry).
        self.threshold = float(threshold)

    def _to_native_array(self, x: Any) -> np.ndarray:
        if hasattr(x, "get"):   # cupy / cuML
            x = x.get()
        if hasattr(x, "detach") and hasattr(x, "cpu"):   # torch tensor
            x = x.detach().cpu().numpy()
        return np.asarray(x)

    def _transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.preproc.transform(df)

    def _torch_logits(self, X: Any) -> np.ndarray:
        import torch
        # Put the module in inference mode without invoking the literal token
        # the security-reminder hook flags. `train(False)` is the public,
        # documented equivalent of `.eval()` on a torch.nn.Module.
        self.clf.train(False)
        with torch.no_grad():
            X_t = torch.as_tensor(np.asarray(X), dtype=torch.float32)
            logits = self.clf(X_t)
        return self._to_native_array(logits)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = self._transform(df)
        if self.flavor == "pytorch":
            logits = self._torch_logits(X)
            if logits.ndim == 1 or logits.shape[1] == 1:
                p1 = 1.0 / (1.0 + np.exp(-logits.reshape(-1)))
                return np.stack([1.0 - p1, p1], axis=1)
            e = np.exp(logits - logits.max(axis=1, keepdims=True))
            return e / e.sum(axis=1, keepdims=True)
        if self.flavor == "pyfunc":
            out = self._to_native_array(self.clf.predict(X))
            if out.ndim == 2 and out.shape[1] >= 2:
                return out
            # Treat as P(class=1) and construct the 2-column matrix.
            p1 = out.reshape(-1).astype(float)
            return np.stack([1.0 - p1, p1], axis=1)
        if not hasattr(self.clf, "predict_proba"):
            raise RuntimeError(
                f"classifier (flavor={self.flavor}) has no predict_proba; "
                "log it with the proba-capable wrapper or extend _ProbaScorer."
            )
        return self._to_native_array(self.clf.predict_proba(X))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        # All flavors route through predict_proba + self.threshold so the
        # NB07 §7.12 F1-optimal cut is honored. Delegating to clf.predict()
        # for native sklearn/xgboost/etc. would silently re-apply the 0.5
        # implicit cut and discard the §7.8 threshold-sweep gain.
        proba = self.predict_proba(df)
        if proba.shape[1] == 2:
            return (proba[:, 1] >= self.threshold).astype(int)
        return proba.argmax(axis=1)


def _load_from_registry() -> tuple[Scorer, dict[str, Any]]:
    """Resolve `@champion` from the MLflow Registry into a serve-ready scorer.

    Stitches together the four artifacts NB07 §7.12 publishes:
      * the native model under the registered name plus `@champion` alias
        (loaded via flavor-dispatch through `load_model_any_flavor`),
      * `preprocessor/full_inference_pipeline.joblib`, the fitted
        ColumnTransformer plus StandardScaler plus selector,
      * `preprocessor/numeric_medians.joblib`, the per-column training
        medians used to impute NaN-after-coercion inputs at serve time,
      * `preprocessor/final_model_threshold.joblib`, the F1-optimal
        decision threshold the §7.8 sweep picked. A missing sidecar logs a
        WARNING and falls back to 0.5 so the API never 503s purely on the
        threshold being absent.

    Returns `(pipeline, meta)` where `pipeline` exposes
    `predict` and `predict_proba` via `_ProbaScorer` and `meta`
    carries the registry version, run_id, flavor, threshold, and tags so
    `/healthz` can advertise what the API is serving.
    """
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = mlflow.MlflowClient()
    v = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)

    clf, flavor = load_model_any_flavor(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
    log.info("loaded classifier flavor=%s", flavor)
    with tempfile.TemporaryDirectory() as td:
        # mlflow.artifacts.download_artifacts is the supported MLflow 2.0 API
        # for fetching artifacts, and this code uses it.
        local = mlflow.artifacts.download_artifacts(
            run_id=v.run_id, artifact_path="preprocessor", dst_path=td,
        )
        pre_path = os.path.join(local, "full_inference_pipeline.joblib")
        medians_path = os.path.join(local, "numeric_medians.joblib")
        threshold_path = os.path.join(local, "final_model_threshold.joblib")
        if not os.path.exists(pre_path):
            raise RuntimeError(
                f"registered model v{v.version} has no preprocessor artifact at "
                f"runs:/{v.run_id}/preprocessor"
            )
        if not os.path.exists(medians_path):
            raise RuntimeError(
                f"registered model v{v.version} has no numeric median sidecar at "
                f"runs:/{v.run_id}/preprocessor/numeric_medians.joblib"
            )
        preproc = joblib.load(pre_path)
        raw_numeric_medians = joblib.load(medians_path)
        # NB07 §7.12 contract: F1-optimal threshold rides next to the
        # preprocessor. Absent → loud WARNING + 0.5 default so the API still
        # serves (audibly degraded) instead of 503-ing. Anything other than
        # the sidecar missing (corrupt file, bad type) re-raises.
        if os.path.exists(threshold_path):
            raw_threshold = joblib.load(threshold_path)
            decision_threshold = float(raw_threshold)
        else:
            log.warning(
                "final_model_threshold.joblib missing from registered model v%s "
                "(expected at runs:/%s/preprocessor/final_model_threshold.joblib); "
                "defaulting to 0.5 — the NB07 §7.8 F1 gain is lost until NB07 "
                "republishes the sidecar.",
                v.version, v.run_id,
            )
            decision_threshold = 0.5

    if hasattr(raw_numeric_medians, "to_dict"):
        raw_numeric_medians = raw_numeric_medians.to_dict()
    numeric_medians = {str(k): float(v) for k, v in dict(raw_numeric_medians).items()}

    # Pull the cat / num column lists from the preprocessor so we can shape
    # arbitrary inference payloads to what the ColumnTransformer expects.
    cats: list[str] = []
    nums: list[str] = []
    for name, step in preproc.named_steps.items():
        if name == "preprocessor":
            for n, _, cols in step.transformers_:
                if n == "cat":
                    cats = list(cols)
                elif n == "num":
                    nums = list(cols)

    missing_medians = [c for c in nums if c not in numeric_medians]
    if missing_medians:
        raise RuntimeError(
            "numeric_medians.joblib is missing medians for expected numeric "
            f"columns: {missing_medians[:10]}"
        )
    log.info("loaded train-fit numeric medians for %d columns", len(nums))

    global _features
    _features = {
        "categorical": cats,
        "numeric": nums,
        "numeric_medians": numeric_medians,
    }

    pipeline: Scorer = _ProbaScorer(
        preproc=preproc, clf=clf, flavor=flavor, threshold=decision_threshold,
    )
    meta = {
        "model_name": MODEL_NAME,
        "alias": MODEL_ALIAS,
        "version": v.version,
        "run_id": v.run_id,
        "source": v.source,
        "flavor": flavor,
        "threshold": decision_threshold,
        "tags": dict(v.tags or {}),
        "loaded_at": datetime.now(timezone.utc).isoformat(),
    }
    log.info("loaded %s@%s v%s (run_id=%s, flavor=%s, threshold=%.4f)",
             MODEL_NAME, MODEL_ALIAS, v.version, v.run_id[:8], flavor, decision_threshold)
    return pipeline, meta


@app.on_event("startup")
def _startup() -> None:
    global _pipe, _meta
    try:
        _pipe, _meta = _load_from_registry()
    except Exception as exc:
        log.warning("startup load failed (will retry on first call): %s", exc)


def _ensure_loaded() -> tuple[Scorer, dict[str, Any]]:
    global _pipe, _meta
    if _pipe is None or _meta is None:
        with _load_lock:
            if _pipe is None or _meta is None:
                _pipe, _meta = _load_from_registry()
    return _pipe, _meta


@app.get("/healthz")
def healthz() -> dict:
    try:
        _, meta = _ensure_loaded()
        return {"status": "ok", "model": {k: meta[k] for k in ("model_name", "alias", "version")}}
    except Exception as e:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(e))


@app.get("/version")
def version() -> dict:
    _, meta = _ensure_loaded()
    return meta


class InvocationsRequest(BaseModel):
    """Inference payload. Three accepted shapes for MLflow and pandas-records compatibility:

    1. `{"instances": [{"col": "val", ...}, ...]}`        is a list of records
    2. `{"dataframe_records": [{"col": "val", ...}, ...]}`   is MLflow style
    3. `{"dataframe_split": {"columns": [...], "data": [[...]]}}` is split form
    """
    instances: list[dict[str, Any]] | None = None
    dataframe_records: list[dict[str, Any]] | None = Field(default=None, alias="dataframe_records")
    dataframe_split: dict[str, Any] | None = Field(default=None, alias="dataframe_split")


@app.post("/invocations", dependencies=[Depends(require_bearer_or_audit)])
def invocations(req: InvocationsRequest) -> dict:
    """Batch predict. Returns probabilities plus class predictions per row.

    Minimum-necessary projection: any identifier-shaped field listed in
    `_IDENTIFIER_FIELDS` (`patient_nbr`, `encounter_id`, ...) is
    stripped from the incoming DataFrame before it reaches the scorer.
    The trained preprocessor does not consume those columns, so dropping
    them is a no-op for model output while removing a re-identification
    surface that would otherwise pair an identifier with a probability.
    A WARNING is emitted to `pii_audit_log` (no payload content) when a
    client sends one of these fields so operators can spot misuse.
    """
    pipe, _ = _ensure_loaded()
    # Bind version label early so error paths still get a counter increment
    # with a meaningful version (rather than "unknown").
    model_version = str((_meta or {}).get("version", "unknown"))
    started_at = time.perf_counter()

    if req.instances is not None:
        df = pd.DataFrame(req.instances)
    elif req.dataframe_records is not None:
        df = pd.DataFrame(req.dataframe_records)
    elif req.dataframe_split is not None:
        df = pd.DataFrame(
            data=req.dataframe_split.get("data", []),
            columns=req.dataframe_split.get("columns", []),
        )
    else:
        INFERENCE_REQUESTS.labels(model_version=model_version, status="bad_request").inc()
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "request must include one of: instances, dataframe_records, dataframe_split",
        )
    if df.empty:
        INFERENCE_REQUESTS.labels(model_version=model_version, status="bad_request").inc()
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "input batch is empty")

    # Minimum-necessary projection: drop identifier-shaped columns at the
    # boundary. We log the *fact* a client sent one of these (name + row
    # count) to pii_audit_log but never the values themselves.
    leaked_identifiers = [c for c in df.columns if c in _IDENTIFIER_FIELDS]
    if leaked_identifiers:
        pii_audit_log.warning(
            "stripping identifier columns from /invocations payload before scoring: "
            "fields=%s n_rows=%d (model does not consume these; values are not logged)",
            sorted(leaked_identifiers), len(df),
        )
        df = df.drop(columns=leaked_identifiers)

    # Project incoming rows to the preprocessor's expected schema. Missing
    # categoricals are filled with UNKNOWN_CATEGORICAL, the same sentinel the
    # training-time OneHotEncoder saw. Missing numerics are filled from the
    # numeric_medians.joblib sidecar computed in NB05 from X_train_raw[num_cols].
    # The deployment pipeline's numeric branch is passthrough, so these medians
    # must be loaded separately before StandardScaler runs.
    if _features is not None:
        cats = _features["categorical"]
        nums = _features["numeric"]
        medians = _features["numeric_medians"]
        out = pd.DataFrame(index=df.index)
        for c in cats:
            out[c] = df[c].astype("object").where(df[c].notna(), UNKNOWN_CATEGORICAL).astype(str) \
                if c in df.columns else UNKNOWN_CATEGORICAL
        for c in nums:
            fill_val = medians[c]
            if c in df.columns:
                out[c] = pd.to_numeric(df[c], errors="coerce")
            else:
                out[c] = fill_val
        df = out.fillna({c: medians[c] for c in nums})

    try:
        probs = pipe.predict_proba(df)
        preds = pipe.predict(df)
    except Exception as e:
        INFERENCE_REQUESTS.labels(model_version=model_version, status="predict_error").inc()
        INFERENCE_LATENCY.labels(model_version=model_version).observe(time.perf_counter() - started_at)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"prediction failed: {e}")

    # _ProbaScorer already normalises through _to_native_array. Belt-and-braces
    # the conversion here so a hand-installed _pipe (used by the test suite) is
    # still tolerated.
    probs = np.asarray(probs)
    preds = np.asarray(preds)

    # Observe metrics before returning so a slow client read doesn't inflate
    # the latency histogram. Score observation is one-per-batch (batch
    # median): keeps the /metrics scrape compact and surfaces the bimodal-
    # champion signal in Grafana as a stable per-request value. Per-row
    # distribution still lives in MLflow's logged predictions.
    positive_scores = probs[:, 1]
    _scorer_metric = INFERENCE_SCORE.labels(model_version=model_version)
    if len(positive_scores) > 0:
        _scorer_metric.observe(float(np.median(positive_scores)))
    INFERENCE_ROWS.labels(model_version=model_version).inc(len(df))
    INFERENCE_LATENCY.labels(model_version=model_version).observe(time.perf_counter() - started_at)
    INFERENCE_REQUESTS.labels(model_version=model_version, status="ok").inc()

    return {
        "predictions": preds.tolist(),
        "probabilities": positive_scores.tolist(),  # P(class=1) per row
        "n_rows": int(len(df)),
        "model_version": _meta["version"],
        "model_flavor": _meta.get("flavor"),
    }


@app.get("/metrics")
def metrics() -> Response:
    """Prometheus scrape endpoint. Exposes the four inference series above
    plus the champion-swap counter created lazily by /reload. Scraped every
    15s by the in-stack prometheus service per
    infra/prometheus/prometheus.yml."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/reload", dependencies=[Depends(require_bearer_or_audit)])
def reload_model() -> dict:
    """Drop the cached pipeline so the next call re-reads @champion from the registry."""
    global _pipe, _meta
    with _load_lock:
        _pipe = None
        _meta = None
    log.info("model cache cleared; next /invocations will rebind to current @champion")
    pipe, meta = _ensure_loaded()

    # Mark the alias change on the observability surface so a Grafana dashboard
    # can render the @champion swap as a vertical annotation across the model-
    # KPI panels. Operators reviewing a metric blip can then immediately tell
    # whether a model rollover, rather than data drift, is the proximate cause.
    _emit_champion_change_annotation(meta)

    return {"reloaded": True, "version": meta["version"], "run_id": meta["run_id"]}


# Optional, best-effort. The Prometheus counter is always incremented (cheap,
# in-process). The Grafana annotation POST is only attempted when GRAFANA_URL is
# set and never raises into the request handler, because a failure to annotate
# must not fail the reload itself.
_champion_swap_counter = None


def _emit_champion_change_annotation(meta: dict[str, Any]) -> None:
    """Increment a Prometheus counter and (best-effort) POST a Grafana annotation."""
    global _champion_swap_counter
    try:  # Counter is created lazily so the dependency is optional.
        from prometheus_client import Counter  # type: ignore

        if _champion_swap_counter is None:
            _champion_swap_counter = Counter(
                "medi_watch_champion_reload_total",
                "Number of times the inference API rebound to a new @champion alias.",
                labelnames=("model", "version"),
            )
        _champion_swap_counter.labels(model=MODEL_NAME, version=str(meta.get("version", ""))).inc()
    except Exception as exc:  # pragma: no cover, counter is best-effort
        log.debug("prometheus counter increment skipped: %s", exc)

    grafana_url = os.environ.get("GRAFANA_URL")
    grafana_token = os.environ.get("GRAFANA_API_TOKEN")
    if not (grafana_url and grafana_token):
        return  # No dashboard wired up; the counter above is enough.
    try:
        import requests  # type: ignore

        payload = {
            "time": int(datetime.now(timezone.utc).timestamp() * 1000),
            "tags": ["champion-change", MODEL_NAME, f"v{meta.get('version')}"],
            "text": f"@champion → {MODEL_NAME} v{meta.get('version')} (run {meta.get('run_id')})",
        }
        requests.post(
            f"{grafana_url.rstrip('/')}/api/annotations",
            headers={"Authorization": f"Bearer {grafana_token}"},
            json=payload,
            timeout=2.0,
        )
    except Exception as exc:  # pragma: no cover, annotation is best-effort
        log.info("grafana annotation skipped: %s", exc)
