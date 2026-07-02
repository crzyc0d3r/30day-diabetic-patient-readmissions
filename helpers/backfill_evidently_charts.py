"""Backfill PNG drift charts onto existing evidently_drift MLflow runs.

For each scenario's most-recent `evidently_drift[...]` run in the
`medi-watch-drift-reports` experiment, this finds the report JSON the run
logged, rebuilds the matching static charts (helpers.evidently_charts), and
re-opens the finished run to attach them under `charts/`. Idempotent: re-running
overwrites the same artifact paths.

Run from the project root with the MLflow server reachable on localhost:

    MLFLOW_TRACKING_URI=http://127.0.0.1:5000 python3 -m helpers.backfill_evidently_charts
"""
from __future__ import annotations

import os

import mlflow
import matplotlib.pyplot as plt

from helpers.evidently_charts import build_figures

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = "medi-watch-drift-reports"
DATA_DIR = os.environ.get("MEDIWATCH_DATA_DIR", "data")
REFERENCE_PATH = os.path.join(DATA_DIR, "features.csv")
REPORT_DIR = os.path.join(DATA_DIR, "evidently_reports")
INCOMING_DIR = os.path.join(DATA_DIR, "incoming")


def _batch_path(scenario: str) -> str:
    name = "current" if scenario in (None, "", "current") else scenario
    return os.path.join(INCOMING_DIR, f"{name}.csv")


def _local_json_for_run(client: mlflow.tracking.MlflowClient, run_id: str) -> str | None:
    """Find the report JSON this run logged and resolve it to a local file."""
    for art in client.list_artifacts(run_id, "report"):
        if art.path.endswith(".json"):
            return os.path.join(REPORT_DIR, os.path.basename(art.path))
    return None


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        raise SystemExit(f"experiment {EXPERIMENT_NAME!r} not found at {TRACKING_URI}")

    runs = client.search_runs(
        [exp.experiment_id],
        filter_string="tags.`mlflow.runName` LIKE 'evidently_drift%'",
        order_by=["attribute.start_time DESC"],
        max_results=500,
    )

    # Keep only the newest run per scenario.
    latest: dict[str, object] = {}
    for run in runs:
        scenario = run.data.tags.get("drift_scenario", "")
        latest.setdefault(scenario, run)

    if not latest:
        raise SystemExit("no evidently_drift runs found to backfill")

    for scenario, run in sorted(latest.items()):
        run_id = run.info.run_id
        json_path = _local_json_for_run(client, run_id)
        if not json_path or not os.path.exists(json_path):
            print(f"  SKIP {scenario:22s} run={run_id[:8]} — report JSON not found locally")
            continue
        figs = build_figures(json_path, scenario, REFERENCE_PATH, _batch_path(scenario))
        with mlflow.start_run(run_id=run_id):
            for name, fig in figs.items():
                mlflow.log_figure(fig, name)
                plt.close(fig)
        print(f"  OK   {scenario:22s} run={run_id[:8]} -> {', '.join(figs)}")


if __name__ == "__main__":
    main()
