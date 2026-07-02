# Monitoring artifacts

Committed monitoring outputs, produced by scoring every drift scenario in
`data/incoming/*.csv` against the champion's training reference
(`data/features.csv`, 99,340 rows) with the `@champion` scorer
(`final_model.joblib` + `full_inference_pipeline.joblib`).

Regenerate with:

```bash
docker exec -i mlops-airflow-api-server-1 python - \
    < infra/monitoring/generate_monitoring_artifacts.py
cp -r data/monitoring/* infra/monitoring/
```

## Contents

| Path | What |
|---|---|
| `evidently/<scenario>.html` | Evidently data-drift and data-summary report (PSI on bucketed columns, KS on continuous, plus prediction drift), reference versus the scenario batch. Open directly in a browser. |
| `champion_impact.csv` / `.md` | Champion F1 impact per scenario: reference versus current precision, recall, F1, and AUC, with the delta. |
| `summary.json` | Per-scenario drift report (verdict, per-column PSI/KS, champion-impact block). |

## Champion F1 impact (measured harm)

Each scenario batch is genuinely labeled, so the delta is measured harm rather
than an estimate. The champion is robust to these feature-level perturbations.
The largest F1 drops are `los_utilization_shift` and `mixed_severe`, near −0.016
and −0.019, while the `none` control barely moves, near −0.0007.

| scenario | verdict | ref F1 | cur F1 | Δ F1 | Δ AUC-ROC |
|---|---|---|---|---|---|
| none (control) | OK | 0.3271 | 0.3265 | −0.0007 | +0.0005 |
| coding_shift | ALERT | 0.3271 | 0.3280 | +0.0009 | −0.0043 |
| formulary_shift | ALERT | 0.3271 | 0.3235 | −0.0036 | −0.0064 |
| casemix_shift | ALERT | 0.3271 | 0.3206 | −0.0065 | −0.0076 |
| los_utilization_shift | ALERT | 0.3271 | 0.3109 | −0.0163 | −0.0178 |
| mixed_severe | ALERT | 0.3271 | 0.3080 | −0.0191 | −0.0287 |

The verdicts trip on distribution change, not on harm, so a batch can ALERT while
the champion's F1 barely moves.

## Grafana renders (`grafana/`)

Server-side PNG renders of the MediWatch model-KPI dashboard.

| File | Panel |
|---|---|
| `dashboard-full.png` | the whole dashboard |
| `panel-08-requests.png` | inference requests per minute, by model version |
| `panel-09-latency.png` | scoring latency, p50 / p95 / p99 |
| `panel-10-score.png` | predicted P(readmit=1), mean and p95 |
| `panel-11-rows24h.png` | rows scored, last 24h |
| `panel-12-reloads24h.png` | champion reloads, last 24h |
| `panel-05-champion-age.png` | champion age since promotion |
| `panel-06-registered-versions.png` | the registered-version timeline |

Re-render any panel with:

```bash
U=$(curl -s "http://localhost:3003/api/search?query=MediWatch" | python3 -c "import sys,json;print(json.load(sys.stdin)[0]['uid'])")
curl -s "http://localhost:3003/render/d-solo/$U/medi-watch-model-kpis?panelId=8&from=now-30m&to=now&width=1000&height=420&tz=UTC" -o out.png
```
