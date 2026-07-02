"""drift_scenarios_demo is a one-click fan-out that exercises all five drift
scenarios end-to-end.

Triggering this DAG fires the following once per scenario, each tagged with its
scenario via the run conf:

  - scheduled_drift_check produces a PSI/KS verdict and champion-impact report.
    The five drift scenarios produce an ALERT verdict and cascade into five
    scenario-tagged retrain_on_drift runs, while the `none` control stays OK
    and skips.
  - evidently_drift produces a wide Evidently PSI/KS drift and data-quality
    HTML report on the same scenario batch, logged to MLflow under
    medi-watch-drift-reports.

This is the demo entry point that makes "five separate scenarios" visible in
the Airflow UI without manually swapping data/incoming/current.csv.
"""
from __future__ import annotations

from datetime import timedelta

import pendulum
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sdk import dag

# The five drift scenarios plus the OK control, mirroring
# helpers.drift_sim.SCENARIOS. Hard-coded here (rather than imported) so the
# dag-processor can parse this module without the data stack present.
DRIFT_SCENARIOS = (
    "none",
    "coding_shift",
    "casemix_shift",
    "los_utilization_shift",
    "formulary_shift",
    "mixed_severe",
)


@dag(
    dag_id="drift_scenarios_demo",
    description="Fan-out: trigger scheduled_drift_check once per drift scenario.",
    start_date=pendulum.datetime(2026, 4, 27, tz="UTC"),
    schedule=None,
    catchup=False,
    default_args={"owner": "medi-watch", "retries": 0, "retry_delay": timedelta(minutes=1)},
    tags=["medi-watch", "drift", "demo"],
)
def drift_scenarios_demo():
    for scenario in DRIFT_SCENARIOS:
        # Verdict plus champion-impact. Cascades to retrain_on_drift on ALERT.
        TriggerDagRunOperator(
            task_id=f"check_{scenario}",
            trigger_dag_id="scheduled_drift_check",
            wait_for_completion=False,
            reset_dag_run=True,
            conf={"scenario": scenario},
        )
        # Wide Evidently PSI/KS drift + data-quality report for the same batch.
        TriggerDagRunOperator(
            task_id=f"evidently_{scenario}",
            trigger_dag_id="evidently_drift",
            wait_for_completion=False,
            reset_dag_run=True,
            conf={"scenario": scenario},
        )


drift_scenarios_demo()
