"""scheduled_drift_check runs a periodic data-drift assessment that decides
whether to retrain.

It answers the production question of whether the incoming patient population
has drifted far enough from the champion's training cohort to justify a retrain,
on a schedule and on demand from the Airflow UI. It compares the champion's
training reference (`data/features.csv`) against the latest production batch
(`data/incoming/current.csv`) using PSI and KS over a hard-coded set of
monitored columns, and on an ALERT verdict triggers `retrain_on_drift`.

The monitored columns and thresholds live in code (`helpers.drift_sim` plus
`helpers.constants`), so it registers and runs in the plain core stack.

The 'current' batch is staged by copying one of NB09's scenario files over
`data/incoming/current.csv`. For example, `cp coding_shift.csv current.csv`
demonstrates a firing, and `none.csv` is the no-retrain control. The schedule
provides the periodic assessment, and manual UI triggers drive the on-camera
demo.

Heavy imports (pandas, helpers) are deferred into the task bodies so the
Airflow dag-processor can import this module without the data stack present.
"""
from __future__ import annotations

import os
from datetime import timedelta

import pendulum
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sdk import dag, task

# /workspace is the in-container mount of the project root (see docker-compose).
DATA_DIR = os.environ.get("MEDIWATCH_DATA_DIR", "/workspace/data")
REFERENCE_PATH = f"{DATA_DIR}/features.csv"
INCOMING_DIR = f"{DATA_DIR}/incoming"
REPORTS_DIR = f"{DATA_DIR}/drift_reports"


def _batch_path(scenario: str) -> str:
    """Resolve the batch file for a scenario. `"current"` resolves to the
    `data/incoming/current.csv` staging path. Any other name maps to the
    NB09-written `data/incoming/<scenario>.csv`."""
    name = "current" if scenario in (None, "", "current") else scenario
    return f"{INCOMING_DIR}/{name}.csv"


def _log_to_mlflow(report: dict, scenario: str) -> None:
    """Best-effort: record the drift report, verdict, and champion-impact
    metrics in MLflow, tagged by scenario, so the Monitoring surface has an
    experiment-tracked, per-scenario history. Never fails the DAG, because
    MLflow being down must not block the drift decision."""
    try:
        import json
        import os as _os
        import tempfile

        import mlflow
        from helpers.mlops_helpers import init_mlflow

        init_mlflow()
        mlflow.set_experiment("medi-watch-drift-reports")
        with mlflow.start_run(run_name=f"scheduled_drift_check[{scenario}]"):
            mlflow.set_tag("source", "scheduled_drift_check")
            mlflow.set_tag("drift_scenario", scenario)
            mlflow.log_param("verdict", report["verdict"])
            cols = report["columns"]
            mlflow.log_metric("cols_alert", sum(c["status"] == "ALERT" for c in cols))
            mlflow.log_metric("cols_warn", sum(c["status"] == "WARN" for c in cols))
            impact = report.get("champion_impact")
            if impact:
                for k, v in impact["delta"].items():
                    mlflow.log_metric(f"delta_{k}", float(v))
            with tempfile.TemporaryDirectory() as td:
                path = _os.path.join(td, "drift_report.json")
                with open(path, "w") as fh:
                    json.dump(report, fh, indent=2)
                mlflow.log_artifact(path, artifact_path="drift")
        print(f"[scheduled_drift_check] drift report logged to MLflow (scenario={scenario})")
    except Exception as e:  # noqa: BLE001
        print(f"[scheduled_drift_check] MLflow log skipped (non-fatal): {e}")


@dag(
    dag_id="scheduled_drift_check",
    description="Periodic PSI/KS drift assessment; triggers retrain_on_drift on an ALERT verdict.",
    start_date=pendulum.datetime(2026, 4, 27, tz="UTC"),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
    default_args={
        "owner": "medi-watch",
        "retries": 1,
        "retry_delay": timedelta(minutes=2),
    },
    tags=["medi-watch", "drift", "monitoring"],
    params={"scenario": "current"},
)
def scheduled_drift_check():

    @task
    def check_drift() -> str:
        """Compare reference against the scenario's batch, emit a per-scenario
        champion-impact report, and return the worst verdict (OK/WARN/ALERT).

        The scenario comes from the triggering run conf (set by
        `drift_scenarios_demo` or the NB09 REST cell) or the DAG param,
        defaulting to `"current"`. A missing batch is treated as OK, since
        nothing arrived to assess. The champion-impact block is best-effort,
        and a scoring failure never blocks the drift verdict.
        """
        import json

        import pandas as pd
        from airflow.sdk import get_current_context
        from airflow.sdk.exceptions import AirflowSkipException

        from helpers.drift_sim import build_drift_report, load_champion_scorer

        ctx = get_current_context()
        scenario = (ctx["dag_run"].conf or {}).get("scenario") \
            or ctx["params"].get("scenario") or "current"
        current_path = _batch_path(scenario)

        # No reference baseline yet means the champion's training cohort hasn't
        # been materialised (fresh stack, or the data pipeline is mid-regenerate
        # — features.csv is rewritten atomically-ish and a @daily run can fire in
        # that window). For an autonomous monitor that is "nothing to assess
        # yet", not a failure, so skip rather than hard-fail-and-retry. Running
        # notebooks 01-05 (or the prepare_data DAG) produces the reference.
        if not os.path.exists(REFERENCE_PATH):
            raise AirflowSkipException(
                f"reference {REFERENCE_PATH} not present yet — skipping drift "
                f"check (run notebooks 01-05 / prepare_data to materialise it)."
            )
        if not os.path.exists(current_path):
            print(f"no incoming batch at {current_path} (scenario={scenario}); -> OK")
            return "OK"

        reference = pd.read_csv(REFERENCE_PATH, low_memory=False)
        current = pd.read_csv(current_path, low_memory=False)

        predict_proba_fn = threshold = None
        try:
            predict_proba_fn, threshold = load_champion_scorer(
                model_bundle_path=f"{DATA_DIR}/final_model.joblib",
                pipeline_path=f"{DATA_DIR}/full_inference_pipeline.joblib",
            )
        except Exception as e:  # noqa: BLE001, impact is best-effort
            print(f"[scheduled_drift_check] champion scorer unavailable: {e}")

        report = build_drift_report(
            reference, current, scenario=scenario,
            predict_proba_fn=predict_proba_fn, threshold=(threshold or 0.5))
        verdict = report["verdict"]

        os.makedirs(REPORTS_DIR, exist_ok=True)
        report_path = f"{REPORTS_DIR}/{scenario}.json"
        with open(report_path, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"DRIFT VERDICT ({scenario}): {verdict}; report -> {report_path}")
        if "champion_impact" in report:
            d = report["champion_impact"]["delta"]
            print(f"  champion delta: dRecall={d['recall_pos']:+.4f} "
                  f"dF1={d['f1_pos']:+.4f} dAUC={d['auc_roc']:+.4f}")

        _log_to_mlflow(report, scenario)
        return verdict

    @task
    def gate_on_alert(verdict: str) -> str:
        """The informed retrain decision: skip everything downstream unless the
        verdict is ALERT, so a retrain only fires on material drift."""
        from airflow.sdk.exceptions import AirflowSkipException

        if verdict != "ALERT":
            raise AirflowSkipException(f"verdict={verdict}; no retrain needed")
        print(f"verdict={verdict}; firing retrain_on_drift")
        return verdict

    trigger_retrain = TriggerDagRunOperator(
        task_id="trigger_retrain",
        trigger_dag_id="retrain_on_drift",
        wait_for_completion=False,
        reset_dag_run=False,
        conf={
            "source": "scheduled_drift_check",
            # Read the scenario from this run's conf (set by the demo DAG / NB09
            # REST cell), falling back to the DAG param so a plain manual trigger
            # still threads a scenario through to the retrain.
            "scenario": "{{ dag_run.conf.get('scenario', params.scenario) }}",
        },
    )

    gate_on_alert(check_drift()) >> trigger_retrain


# Registered unconditionally. Runs in the plain core stack.
scheduled_drift_check()
