"""prepare_data orchestrates the data-prep stages (NB02-NB05) that the
retrain / drift DAGs assume already ran.

Each task shells out to pipeline/run_pipeline.py (the project's canonical
notebook runner) scoped to its stage, writing artefacts under DATA_DIR:

  clean                     -> 02_data_cleaning.ipynb            -> cleaned.csv
  feature_engineer          -> 03_exploratory_data_analysis.ipynb,
                               04_feature_engineering.ipynb       -> features.csv (+ patient_ids.csv)
  split_encode_scale_select -> 05_split_encode_scale_select.ipynb -> train_test.npz,
                               ohe/scaler/feature_selector/numeric_medians.joblib

Runs under /opt/ray-driver/bin/python (the venv that carries the data stack),
like retrain_on_drift's HPO driver. schedule=None: data prep runs on a data
refresh, not a clock. Heavy imports stay inside the tasks so the dag-processor
parses this module with the stdlib + airflow SDK alone.
"""
from __future__ import annotations

import os
from datetime import timedelta

import pendulum
from airflow.sdk import dag, task

# /workspace maps to the project root in-container (see infra/docker-compose.yml).
DATA_DIR = os.environ.get("MEDIWATCH_DATA_DIR", "/workspace/data")
PROJECT_ROOT = os.environ.get("MEDIWATCH_PROJECT_ROOT", "/workspace")
PIPELINE_DIR = f"{PROJECT_ROOT}/pipeline"
RUN_PIPELINE = f"{PIPELINE_DIR}/run_pipeline.py"

# ray-driver venv carries the data stack + nbconvert (the default airflow one does not).
STAGE_PYTHON = os.environ.get("MEDIWATCH_STAGE_PYTHON", "/opt/ray-driver/bin/python")

# Per-cell nbconvert timeout (seconds); generous cap for heavy EDA cells.
STAGE_TIMEOUT = int(os.environ.get("MEDIWATCH_PREP_TIMEOUT", "7200"))


def _run_stage(from_nb: str, to_nb: str) -> dict:
    """Run a notebook range through pipeline/run_pipeline.py under STAGE_PYTHON,
    cwd=pipeline/ so the notebooks' ../data and ../orig_dataset reads resolve.

    check=True fails the task on any notebook error.
    """
    import subprocess

    cmd = [
        STAGE_PYTHON,
        RUN_PIPELINE,
        "--from", from_nb,
        "--to", to_nb,
        "--timeout", str(STAGE_TIMEOUT),
    ]
    print(f"[prepare_data] running stage {from_nb} -> {to_nb}")
    print(f"[prepare_data] cmd: {' '.join(cmd)} (cwd={PIPELINE_DIR})")
    # PYTHONPATH=/workspace so the notebooks' `from helpers.x import y` cells
    # resolve under the subprocess interpreter, mirroring the compose env.
    env = {**os.environ, "PYTHONPATH": PROJECT_ROOT}
    subprocess.run(cmd, cwd=PIPELINE_DIR, env=env, check=True)
    return {"stage": f"{from_nb}->{to_nb}", "status": "ok"}


@dag(
    dag_id="prepare_data",
    description=(
        "Orchestrate data preparation (NB02-NB05): clean -> feature_engineer -> "
        "split/encode/scale/select, materialising cleaned.csv, features.csv, and "
        "train_test.npz under DATA_DIR. Manual / on-demand."
    ),
    start_date=pendulum.datetime(2026, 4, 27, tz="UTC"),
    schedule=None,            # manual / on-demand; do not re-split on a clock
    catchup=False,
    max_active_runs=1,        # one prep at a time; artefacts are shared files
    default_args={
        "owner": "medi-watch",
        "retries": 0,         # a silent retry of a partial write is worse than
                              # a human-triaged failure (matches retrain_on_drift)
        "retry_delay": timedelta(minutes=5),
    },
    tags=["data-prep", "medi-watch"],
)
def prepare_data():

    @task(task_id="clean")
    def clean() -> dict:
        """NB02 stage: load raw `orig_dataset/diabetic_data.csv`, apply the
        multi-pass cleaning + §2.10/§2.11 corrections (via helpers.clean_helpers),
        and write `data/cleaned.csv`.

        Runs `pipeline/02_data_cleaning.ipynb` through the project runner.
        """
        return _run_stage("02_data_cleaning.ipynb", "02_data_cleaning.ipynb")

    @task(task_id="feature_engineer")
    def feature_engineer() -> dict:
        """NB03+NB04 stage: EDA then feature engineering. NB04 reads
        `data/cleaned.csv`, applies the §4.14 interaction block (via
        helpers.feature_eng.add_all_interactions and the ICD-9 rollups), and
        writes `data/features.csv` (+ `data/patient_ids.csv`).

        The range includes NB03 (EDA) so the canonical 02->03->04->05 notebook
        order is preserved; NB03 produces no modelling artefact but is part of
        the documented pipeline sequence.
        """
        return _run_stage(
            "03_exploratory_data_analysis.ipynb",
            "04_feature_engineering.ipynb",
        )

    @task(task_id="split_encode_scale_select")
    def split_encode_scale_select() -> dict:
        """NB05 stage: patient-grouped 70/10/20 split, one-hot encode, scale,
        and mutual-information feature selection. Reads `data/features.csv` and
        writes `data/train_test.npz` plus the standalone `ohe.joblib`,
        `scaler.joblib`, `feature_selector.joblib`, and `numeric_medians.joblib`
        bundles the inference path and NB06-NB09 consume.

        Runs `pipeline/05_split_encode_scale_select.ipynb` through the runner.
        """
        return _run_stage(
            "05_split_encode_scale_select.ipynb",
            "05_split_encode_scale_select.ipynb",
        )

    # Sequential: each stage reads the prior stage's artefact off disk.
    clean() >> feature_engineer() >> split_encode_scale_select()


# Registered unconditionally; runs in the plain core (LocalExecutor) stack.
prepare_data()
