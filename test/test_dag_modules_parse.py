"""Parse-import guard for the drift orchestration DAG modules.

Each DAG module must import cleanly as a plain Python file with
`DeprecationWarning` promoted to an error. This catches both import breakage
and any deprecated Airflow import, which must route through
`airflow.sdk.exceptions`.

When Airflow is not installed in the local test environment (it lives in the
`mlops-airflow-*` containers), these tests skip. In that case the in-container
`python -W error::DeprecationWarning <dag>.py` parse is the authoritative
check.
"""
from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

import pytest

DAGS = Path("infra/airflow/dags")
MODULES = [
    "scheduled_drift_check_dag.py",
    "retrain_on_drift_dag.py",
    "drift_scenarios_demo_dag.py",
]

airflow_available = importlib.util.find_spec("airflow") is not None


@pytest.mark.skipif(not airflow_available, reason="airflow not installed in this env")
@pytest.mark.parametrize("module_file", MODULES)
def test_dag_module_imports_without_warnings(module_file):
    path = DAGS / module_file
    assert path.exists(), f"missing DAG module {path}"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        spec.loader.exec_module(mod)
