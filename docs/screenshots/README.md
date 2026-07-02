# Screenshots

Captured against the running stack. Each image documents a part of the running system.

## Large-scale HPO + experiment tracking
- `hpo-asha-250-trials.png`: 250 RayTune trials (50 each × 5 model families), ASHA early-stopping (159 trials pruned at fold 1, 91 survive to fold 3).
- `mlflow-hpo-runs-table.png`: 54 tracked runs in the MLflow experiment (hpo_tuner / refit / default / champion_promotion).
- `mlflow-hpo-run-detail.png`: one HPO run: 9 hyperparameters, 7 metrics, logged model, and the RayTune backend tags (`hpo_backend=ray_tune`, `ray_address=ray://ray-head:10001`).
- `mlflow-champion-artifacts.png`: data-transformation artifacts logged with the model (`ohe`, `scaler`, `full_inference_pipeline`, `feature_selector`, `numeric_medians`).
- `airflow-run06-hpo.png`: `run_06_hyperparameter_tuning` orchestrated in `retrain_on_drift`, green.

## Orchestration / drift → retrain loop
- `airflow-retrain-on-drift.png`: `retrain_on_drift` run auto-triggered by a drift ALERT, full task chain.
- `airflow-dags-overview.png`: the DAGs registered in Airflow.

## Monitoring / drift
- `evidently-mixed_severe-drift.png`: Evidently report, 13/78 columns drifted on the `mixed_severe` scenario.

## Serving / registry
- `mlflow-champion-registry.png`: `medi-watch-readmission` v5 with the `@champion` alias; v2-v4 rejected by the lift gate.
- `mlflow-experiment-runs.png`: experiment run history.
