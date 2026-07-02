"""Drive real traffic through the inference API so the Grafana model-KPIs
dashboard (Prometheus-backed) has live data to render.

Samples rows from the champion's feature matrix and POSTs them to the inference
API's /invocations endpoint in batches over a few minutes, which populates the
medi_watch_inference_* Prometheus series (request rate, latency histogram, score
distribution, rows scored) that panels 8-12 plot.

Run inside a container on the compose network (reaches inference-api:8002 and has
pandas):

  docker exec -i mlops-airflow-api-server-1 python - \
      < infra/monitoring/generate_inference_traffic.py
"""
from __future__ import annotations

import json
import time
import urllib.request

import pandas as pd

URL = "http://inference-api:8002/invocations"
SOURCE = "/workspace/data/features.csv"
N_ITERS = 48
BATCH = 200
SLEEP_S = 4  # ~48 * (req + 4s) ≈ 4 min, comfortably filling rate[1m]/[5m] windows

df = pd.read_csv(SOURCE, low_memory=False)
df = df.drop(columns=[c for c in ("readmitted_binary",) if c in df.columns])
print(f"loaded {len(df):,} rows x {df.shape[1]} cols; sending {N_ITERS} batches of {BATCH}")

ok = err = rows = 0
for i in range(N_ITERS):
    sample = df.sample(BATCH, random_state=i)
    # NaN -> None so the payload is valid JSON (json can't encode NaN).
    sample = sample.astype(object).where(pd.notnull(sample), None)
    payload = {"dataframe_split": {"columns": list(sample.columns), "data": sample.values.tolist()}}
    data = json.dumps(payload).encode()
    req = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            status = resp.status
        ok += 1
        rows += BATCH
    except Exception as exc:  # noqa: BLE001 — keep driving traffic on a transient error
        status = f"ERR {exc}"
        err += 1
    if i % 6 == 0 or status != 200:
        print(f"  batch {i:>3}: {status}  (ok={ok} err={err} rows={rows})")
    time.sleep(SLEEP_S)

print(f"DONE: {ok} ok, {err} err, {rows:,} rows scored")
