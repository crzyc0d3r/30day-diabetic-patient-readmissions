"""evidently_drift produces a wide Evidently drift and data-quality report on a
drift-scenario batch.

Drift statistics
----------------
This DAG emits both PSI (population stability index) and KS
(Kolmogorov-Smirnov) drift tests in a single Evidently report. Each
column carries exactly one statistic, because 'per_column_method' is a
'column -> method' dict rather than a list of methods. The DAG partitions
columns by a dtype-plus-cardinality heuristic. Continuous numerics (more
than DRIFT_KS_UNIQUE_THRESHOLD unique values) get KS. Everything else
(low-cardinality numerics and categorical) gets PSI. PSI is the report-wide
default for any column not in the override map. PSI thresholds follow the
standard banked-finance and healthcare convention (mild drift above 0.1,
significant drift above 0.2). The Evidently report HTML and JSON summaries
therefore carry PSI for the bucketed columns and KS for the continuous
columns. Both statistics appear in the same artefact set the downstream
tooling reads from MLflow.

Data source
-----------
This DAG belongs to the drift-simulator chain and consumes the same scenario
batches as `scheduled_drift_check`. It compares the champion's training
reference (`data/features.csv`) against one drift-scenario batch
(`data/incoming/<scenario>.csv`, written by NB09) and renders a Data Drift
plus Data Quality report. The HTML lands under
`/workspace/data/evidently_reports/<scenario>_<run-ts>.html` and is logged
as an MLflow artifact under the shared `medi-watch-drift-reports`
experiment, tagged `drift_scenario=<scenario>`.

This is the wide statistical surface (PSI, KS, value distributions per
column) for human review, complementing `scheduled_drift_check`'s
PSI/KS verdict and champion-impact JSON. The two coexist and neither
replaces the other. Fan it out across all five scenarios by triggering
`drift_scenarios_demo`, which fires one `evidently_drift` run per
scenario alongside the matching `scheduled_drift_check` run.

Retry / SLA policy:
  - Schedule: None. The DAG is event-driven, triggered per scenario by
    `drift_scenarios_demo` (or manually with a `scenario` param). There
    is nothing to compare on a timer without a staged batch.
  - Retries: 1 (default_args.retries). A retry covers transient
    artifact-upload or MLflow-tracking-server hiccups.
  - SLA: implicit. A run that exceeds 30 min is anomalous, and the
    render task's logs are the canonical place to look.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pendulum
from airflow.sdk import dag, task

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")

# /workspace is the in-container mount of the project root (see docker-compose).
DATA_DIR = os.environ.get("MEDIWATCH_DATA_DIR", "/workspace/data")
REFERENCE_PATH = f"{DATA_DIR}/features.csv"
INCOMING_DIR = f"{DATA_DIR}/incoming"
REPORT_DIR = f"{DATA_DIR}/evidently_reports"
EXPERIMENT_NAME = "medi-watch-drift-reports"


def _batch_path(scenario: str) -> str:
    """Resolve the batch file for a scenario. `"current"` resolves to the
    `data/incoming/current.csv` staging path. Any other name maps to the
    NB09-written `data/incoming/<scenario>.csv`. This mirrors
    `scheduled_drift_check` so both consume the same batches."""
    name = "current" if scenario in (None, "", "current") else scenario
    return f"{INCOMING_DIR}/{name}.csv"


@dag(
    dag_id="evidently_drift",
    description="Evidently drift + data-quality report on a drift-scenario batch.",
    start_date=pendulum.datetime(2026, 4, 27, tz="UTC"),
    schedule=None,                 # event-driven: triggered per scenario
    catchup=False,
    max_active_runs=1,
    default_args={
        "owner": "medi-watch",
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    tags=["medi-watch", "drift", "evidently"],
    params={"scenario": "current"},
)
def evidently_drift():

    @task
    def load_reference_and_current() -> dict:
        """Load the reference (`data/features.csv`) and the scenario's batch
        (`data/incoming/<scenario>.csv`).

        The scenario comes from the triggering run conf (set by
        `drift_scenarios_demo` or a manual trigger) or the DAG param,
        defaulting to `"current"`. A missing batch skips the run, since
        nothing arrived to assess. This mirrors `scheduled_drift_check`'s
        OK-on-missing behavior.
        """
        import tempfile

        import pandas as pd
        from airflow.sdk import get_current_context
        from airflow.sdk.exceptions import AirflowSkipException

        ctx = get_current_context()
        scenario = (ctx["dag_run"].conf or {}).get("scenario") \
            or ctx["params"].get("scenario") or "current"
        current_path = _batch_path(scenario)

        if not os.path.exists(REFERENCE_PATH):
            raise FileNotFoundError(
                f"reference {REFERENCE_PATH} missing — run notebooks 01-05 first."
            )
        if not os.path.exists(current_path):
            raise AirflowSkipException(
                f"no incoming batch at {current_path} (scenario={scenario}); nothing to report"
            )

        reference = pd.read_csv(REFERENCE_PATH, low_memory=False)
        current = pd.read_csv(current_path, low_memory=False)

        # Per-run tempdir keyed off run_id so concurrent backfills do not
        # collide. Not cleaned up here because render_report reads from it.
        # Cleanup happens at task exit.
        try:
            _run_id = ctx["run_id"]
        except Exception:
            _run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        _td = tempfile.mkdtemp(prefix=f"evidently_{scenario}_{_run_id}_")
        ref_path = os.path.join(_td, "ref.parquet")
        cur_path = os.path.join(_td, "cur.parquet")
        reference.to_parquet(ref_path)
        current.to_parquet(cur_path)

        return {
            "scenario": scenario,
            "reference_path": ref_path,
            "current_path": cur_path,
            "reference_rows": len(reference),
            "current_rows": len(current),
        }

    @task
    def render_report(payload: dict) -> dict:
        """Build the Evidently report and dump the HTML and JSON summary.

        Drift-method note: the DAG computes both PSI and Kolmogorov-Smirnov
        drift tests. 'per_column_method' in Evidently 0.7.x maps each column
        to exactly one method (it is a dict, not a list), so the report
        carries PSI on the bucketed columns and KS on the continuous columns.
        Both statistics appear across the full column set, on disjoint
        columns. The report-wide default 'method="psi"' covers any column
        omitted from the override map.

        API note: Evidently 0.7.x names these kwargs 'method',
        'per_column_method', and 'threshold'. This DAG uses those names.
        """
        import pandas as pd
        from evidently import Report
        from evidently.presets import DataDriftPreset, DataSummaryPreset

        scenario = payload["scenario"]
        ref = pd.read_parquet(payload["reference_path"])
        cur = pd.read_parquet(payload["current_path"])

        # Partition columns: continuous numerics (>DRIFT_KS_UNIQUE_THRESHOLD
        # unique values) get KS, everything else gets PSI. Each column ends
        # up with exactly one method in per_column_method, because Evidently's
        # API is one-method-per-column rather than a list of methods per
        # column. The cardinality cutoff and the PSI WARN/ALERT thresholds
        # live in helpers.constants, so every drift consumer (this DAG,
        # scheduled_drift_check, an ad-hoc notebook) agrees on what counts
        # as "drifted".
        from helpers.constants import (
            DRIFT_KS_UNIQUE_THRESHOLD,
            DRIFT_PSI_WARN,
        )

        continuous_cols: list[str] = []
        bucketed_cols: list[str] = []
        for col in ref.columns:
            ser = ref[col]
            if (
                pd.api.types.is_numeric_dtype(ser)
                and ser.nunique(dropna=True) > DRIFT_KS_UNIQUE_THRESHOLD
            ):
                continuous_cols.append(col)
            else:
                bucketed_cols.append(col)

        per_column_method = {col: "psi" for col in bucketed_cols}
        per_column_method.update({col: "ks" for col in continuous_cols})

        report = Report(metrics=[
            DataDriftPreset(
                method="psi",                                # default for any column not in the override map
                per_column_method=per_column_method,         # one method per column: PSI on bucketed, KS on continuous
                threshold=DRIFT_PSI_WARN,                    # PSI > DRIFT_PSI_WARN = mild drift, DRIFT_PSI_ALERT = significant
            ),
            DataSummaryPreset(),
        ])
        snapshot = report.run(reference_data=ref, current_data=cur)

        os.makedirs(REPORT_DIR, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        html_path = os.path.join(REPORT_DIR, f"{scenario}_{ts}.html")
        json_path = os.path.join(REPORT_DIR, f"{scenario}_{ts}.json")

        snapshot.save_html(html_path)
        # JSON summary feeds downstream tooling and SQL dashboards.
        snapshot.save_json(json_path)

        # Pull headline metrics out of the snapshot for the task return value.
        # Evidently 0.7.x emits a flat `metrics` list; the dataset-level drift
        # count is the single metric whose name starts with "DriftedColumnsCount",
        # carrying value {"count": <n drifted>, "share": <fraction>}. Match by
        # metric NAME, not value shape: every MissingValueCount metric also
        # carries a {"count", "share"} value, so a shape-only match collides.
        #
        # Snapshot-shape guard: a silent zero on shape change is worse than a
        # failed task, because it tells downstream the model is stable when in
        # reality monitoring is broken. Fail loud, dump enough of
        # snapshot.dict() to debug what changed, and let Airflow surface the
        # alert.
        result = snapshot.dict()
        n_drifted: int | None = None
        for m in result.get("metrics", []):
            name = str(m.get("metric_name", ""))
            val = m.get("value")
            if name.startswith("DriftedColumnsCount") and isinstance(val, dict) \
                    and "count" in val:
                n_drifted = int(val["count"])
                break
        if n_drifted is None:
            top_keys = sorted(result.keys())
            metric_names = [str(m.get("metric_name", "")) for m in result.get("metrics", [])]
            raise RuntimeError(
                "Evidently snapshot.dict() shape changed — no `DriftedColumnsCount` "
                "metric with an integer `count` was found. The DAG cannot decide "
                "whether drift occurred. Check the Evidently pin in "
                "infra/airflow/requirements.txt (built against 0.7.x). "
                f"Snapshot top-level keys: {top_keys}. "
                f"First 10 metric names: {metric_names[:10]}. "
                f"Saved snapshot JSON for offline inspection: {json_path}"
            )

        print("=== evidently_drift summary ===")
        print(f"  scenario: {scenario}")
        print(f"  reference ({payload['reference_rows']} rows)")
        print(f"  current   ({payload['current_rows']} rows)")
        print(f"  drifted columns: {n_drifted}")
        print(f"  html: {html_path}")
        print(f"  json: {json_path}")

        return {
            "scenario": scenario,
            "html_path": html_path,
            "json_path": json_path,
            "drifted_columns": int(n_drifted),
            "reference_rows": payload["reference_rows"],
            "current_rows": payload["current_rows"],
        }

    @task
    def log_to_mlflow(report_meta: dict) -> dict:
        """Persist the report as an MLflow run artifact, tagged by scenario.

        Besides the self-contained Evidently HTML (which MLflow shows in a
        sandboxed iframe that blocks its Plotly scripts), this logs static PNG
        charts under `charts/` — a per-column drift-score bar chart and
        reference-vs-current distribution overlays — so the drift signal is
        visible inline in the MLflow run without leaving the UI.
        """
        import matplotlib.pyplot as plt
        import mlflow

        from helpers.evidently_charts import build_figures
        scenario = report_meta["scenario"]
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        run_name = f"evidently_drift[{scenario}]"
        with mlflow.start_run(run_name=run_name,
                              tags={"phase": "monitoring", "tool": "evidently"}) as run, \
                mlflow.start_span(name=f"run_lifecycle:{run_name}") as _span:
            mlflow.set_tag("drift_scenario", scenario)
            try:
                _span.set_attributes({
                    "mlflow.runId": run.info.run_id,
                    "drift_scenario": scenario,
                    "drifted_columns": float(report_meta["drifted_columns"]),
                })
            except Exception:
                pass
            mlflow.log_metrics({
                "drifted_columns": float(report_meta["drifted_columns"]),
                "reference_rows":  float(report_meta["reference_rows"]),
                "current_rows":    float(report_meta["current_rows"]),
            })
            mlflow.log_artifact(report_meta["html_path"], artifact_path="report")
            mlflow.log_artifact(report_meta["json_path"], artifact_path="report")
            # Inline-renderable PNG charts derived from the same report JSON
            # plus the reference/batch CSVs this run compared.
            figs = build_figures(
                report_meta["json_path"], scenario,
                REFERENCE_PATH, _batch_path(scenario),
            )
            for name, fig in figs.items():
                mlflow.log_figure(fig, name)
                plt.close(fig)
            run_id = run.info.run_id
        print(f"  logged to MLflow run_id={run_id} (scenario={scenario})")
        return {**report_meta, "mlflow_run_id": run_id}

    payload = load_reference_and_current()
    report = render_report(payload)
    log_to_mlflow(report)


# Registered unconditionally. This DAG runs in the plain core stack and
# consumes the same drift-scenario batches as scheduled_drift_check.
evidently_drift()
