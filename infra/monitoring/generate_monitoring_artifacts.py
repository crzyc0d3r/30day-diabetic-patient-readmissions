"""Generate real monitoring artifacts from the executed drift scenarios.

Produces, for every scenario batch in ``data/incoming/*.csv`` compared against
the champion's training reference (``data/features.csv``):

  1. An Evidently data-drift + data-summary HTML report (PSI on bucketed columns,
     KS on continuous), mirroring infra/airflow/dags/evidently_drift_dag.py.
  2. The champion F1 impact: the real @champion scorer (final_model.joblib +
     full_inference_pipeline.joblib) scored on reference vs the drifted batch,
     with the per-metric delta. Because each batch is genuinely labeled, the F1
     delta is measured harm, not an estimate.

Run it inside the Airflow image (which has evidently + the scorer) by piping it
to the container's python; it writes to ``$MEDIWATCH_DATA_DIR/monitoring`` which
the caller copies into the tracked ``infra/monitoring/`` tree:

  docker exec -i mlops-airflow-api-server-1 python - \
      < infra/monitoring/generate_monitoring_artifacts.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

from helpers.constants import DRIFT_KS_UNIQUE_THRESHOLD, DRIFT_PSI_WARN
from helpers.drift_sim import (
    MONITORED_COLUMNS,
    TARGET,
    build_drift_report,
    load_champion_scorer,
)

DATA = Path(os.environ.get("MEDIWATCH_DATA_DIR", "/workspace/data"))
OUT = DATA / "monitoring"
EVI = OUT / "evidently"
EVI.mkdir(parents=True, exist_ok=True)

REFERENCE = DATA / "features.csv"
INCOMING = DATA / "incoming"

print(f"loading reference {REFERENCE}")
reference = pd.read_csv(REFERENCE, low_memory=False)

# Real @champion scorer (model + preprocessor sidecar).
predict_proba_fn = threshold = None
try:
    predict_proba_fn, threshold = load_champion_scorer(
        model_bundle_path=str(DATA / "final_model.joblib"),
        pipeline_path=str(DATA / "full_inference_pipeline.joblib"),
    )
    print(f"champion scorer loaded (threshold={threshold:.4f})")
except Exception as exc:  # noqa: BLE001 — impact is best-effort
    print(f"WARNING: champion scorer unavailable: {exc}")

# Evidently report runs on the monitored subset (focused + fast). Champion impact
# uses the full frames because the scorer transforms every feature.
report_cols = [c for c in MONITORED_COLUMNS if c in reference.columns]
if TARGET in reference.columns:
    report_cols = report_cols + [TARGET]

per_column_method: dict[str, str] = {}
for col in report_cols:
    ser = reference[col]
    if pd.api.types.is_numeric_dtype(ser) and ser.nunique(dropna=True) > DRIFT_KS_UNIQUE_THRESHOLD:
        per_column_method[col] = "ks"
    else:
        per_column_method[col] = "psi"

scenarios = sorted(p.stem for p in INCOMING.glob("*.csv"))
print(f"scenarios: {scenarios}")

rows: list[dict] = []
summary: dict[str, dict] = {}

for scen in scenarios:
    current = pd.read_csv(INCOMING / f"{scen}.csv", low_memory=False)
    print(f"\n=== {scen}: ref={len(reference)} cur={len(current)} ===")

    # 1) Evidently HTML report, plus prediction drift (KS on predicted P) when the scorer is available.
    ref_report = reference[report_cols].copy()
    cur_report = current[report_cols].copy()
    scen_pcm = dict(per_column_method)
    if predict_proba_fn is not None:
        try:
            ref_report["prediction"] = predict_proba_fn(reference)
            cur_report["prediction"] = predict_proba_fn(current)
            scen_pcm["prediction"] = "ks"
            print("  prediction-drift column added")
        except Exception as exc:  # noqa: BLE001 — prediction drift is best-effort
            print(f"  prediction-drift column unavailable: {exc}")
    report = Report(metrics=[
        DataDriftPreset(method="psi", per_column_method=scen_pcm, threshold=DRIFT_PSI_WARN),
        DataSummaryPreset(),
    ])
    snapshot = report.run(reference_data=ref_report, current_data=cur_report)
    html_path = EVI / f"{scen}.html"
    snapshot.save_html(str(html_path))
    print(f"  evidently -> {html_path}")

    # 2) Champion F1 impact + drift verdict.
    dr = build_drift_report(
        reference, current, scenario=scen,
        predict_proba_fn=predict_proba_fn, threshold=(threshold or 0.5),
    )
    summary[scen] = dr
    row = {"scenario": scen, "verdict": dr["verdict"], "current_rows": dr["current_rows"]}
    ci = dr.get("champion_impact")
    if ci:
        r, c, d = ci["reference"], ci["current"], ci["delta"]
        row.update({
            "ref_f1": r["f1_pos"], "cur_f1": c["f1_pos"], "delta_f1": d["f1_pos"],
            "ref_precision": r["precision_pos"], "cur_precision": c["precision_pos"], "delta_precision": d["precision_pos"],
            "ref_recall": r["recall_pos"], "cur_recall": c["recall_pos"], "delta_recall": d["recall_pos"],
            "ref_auc_roc": r["auc_roc"], "cur_auc_roc": c["auc_roc"], "delta_auc_roc": d["auc_roc"],
        })
        print(f"  champion F1: {r['f1_pos']:.4f} -> {c['f1_pos']:.4f} (delta {d['f1_pos']:+.4f})")
    else:
        print(f"  champion impact unavailable: {dr.get('champion_impact_error')}")
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(OUT / "champion_impact.csv", index=False)


def _md_table(frame: pd.DataFrame) -> str:
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in frame.iterrows():
        cells = [f"{v:+.4f}" if isinstance(v, float) and str(c).startswith("delta")
                 else (f"{v:.4f}" if isinstance(v, float) else str(v))
                 for c, v in r.items()]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


(OUT / "champion_impact.md").write_text(
    "# Champion F1 impact under executed drift scenarios\n\n"
    f"Reference: `data/features.csv` ({len(reference):,} rows). "
    "Scorer: @champion (`final_model.joblib` + `full_inference_pipeline.joblib`). "
    "Each scenario batch is genuinely labeled, so the F1 delta is **measured harm**, "
    "not an estimate. Verdict bands (PSI WARN/ALERT) come from `helpers.constants`.\n\n"
    + _md_table(df) + "\n"
)
(OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

print(f"\nWROTE {OUT}/champion_impact.csv + .md + summary.json and "
      f"{len(scenarios)} Evidently reports under {EVI}")
