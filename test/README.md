# Unit tests for the `helpers/` package

This directory holds the unit suite for every module under `helpers/`. The
suite exercises the production import path (`from helpers.<module> import ...`),
the same path the notebooks, the Airflow DAGs, and the inference service use.

## What is covered

A `test_<module>.py` file accompanies every module under `helpers/`, plus two
tests that guard code outside `helpers/`: a parse-import check for the drift
orchestration DAGs (`test_dag_modules_parse.py`) and an API-level smoke test for
the inference service (`test_inference_api.py`). Twenty-two test modules in all.

The `helpers/` modules split into three tiers by how much of the outside world
they touch, and the tests match that tier:

- **Pure** (`check_requirements_pins`, `clean_helpers`, `constants`,
  `drift_sim`, `eda_stats`, `evaluation`, `fairness`, `feature_eng`): driven
  directly with small in-memory fixtures and hand-computed expected values.
- **Pure-ish** (`hpo`, `mlp_train`, `models`, `pipeline_parity`): real sklearn
  and torch objects on tiny CPU-only data, kept to a few epochs so the suite
  stays fast.
- **Impure** (`cicd_trigger`, `conclusion_pipeline`, `db`, `hpo_pipeline`,
  `migrate_to_postgres`, `model_loading`, `mlops_helpers`, `training_pipeline`):
  the external boundary (MLflow, Ray, Postgres, a CI provider's REST API,
  `nvidia-smi`) is replaced with `monkeypatch` fakes, so no server, cluster, or
  GPU is required. The deterministic fallback paths run for real.

Shared fixtures live in `conftest.py`: a seeded RNG, a synthetic
binary-classification problem, and a miniature UCI Diabetes-130 cohort.

## Running the suite

The repository ships no virtual environment. The tests need `pytest` plus the
scientific stack the modules import (numpy, pandas, scikit-learn, scipy,
mlflow, torch, ray, xgboost, catboost). The quickest path is a venv that
inherits an interpreter where the scientific stack is already installed:

```bash
python3 -m venv --system-site-packages .venv-test
.venv-test/bin/python -m pip install pytest
.venv-test/bin/python -m pytest test/
```

`.venv-test/` is git-ignored. If a conda environment with the stack and pytest
is already active, run `pytest test/` directly instead.

Configuration lives in the repository-root `pytest.ini`: it confines collection
to `test/` and puts the repository root on `sys.path`.
