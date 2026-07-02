"""Shared pytest fixtures and import wiring for the helpers/ unit suite.

pytest loads this module automatically before any test in the `test/`
tree. It serves two purposes.

1. It guarantees the repository root is importable, so every test file can
   write `from helpers.<module> import ...` exactly as the notebooks, the
   Airflow DAGs, and the inference service do. `pytest.ini` already sets
   `pythonpath = .`. The explicit insert here covers the case where a
   test is invoked from outside the configured rootdir.

2. It provides small, deterministic fixtures shared across modules: a
   seeded random generator, a synthetic binary-classification dataset, and
   a miniature raw-cohort DataFrame shaped like the UCI Diabetes-130
   source. Centralizing these prevents each test file from re-rolling its
   own slightly different toy data, which is how fixtures silently drift
   apart.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Import wiring.
# Resolve the repository root (the parent of this test/ directory) and place
# it at the front of sys.path. Front, not back, so a stray site-packages
# module named "helpers" cannot shadow the package we mean.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@pytest.fixture
def seed() -> int:
    """The canonical RNG seed the whole pipeline pins its randomness to.

    Mirrors `helpers.constants.SEED` so tests that need reproducibility
    use the same value the production code uses, without importing it
    everywhere.
    """
    return 42


@pytest.fixture
def rng(seed: int) -> np.random.Generator:
    """A freshly seeded NumPy generator for deterministic synthetic data."""
    return np.random.default_rng(seed)


@pytest.fixture
def binary_classification_data(rng: np.random.Generator):
    """A small, near-linearly-separable binary problem as (y_true, y_proba).

    Two hundred rows keep sklearn metrics well-defined (no empty
    confusion-matrix cells) while staying fast. The probabilities correlate
    with the labels but remain deliberately imperfect, so threshold sweeps
    and AUC calculations operate on a non-degenerate signal.
    """
    n = 200
    y_true = rng.integers(0, 2, size=n)
    # Push probabilities toward the true label but leave overlap, so metrics
    # land in a realistic mid-range rather than a perfect 1.0.
    noise = rng.normal(0.0, 0.25, size=n)
    y_proba = np.clip(0.30 + 0.40 * y_true + noise, 0.0, 1.0)
    return y_true, y_proba


@pytest.fixture
def raw_cohort_df(rng: np.random.Generator) -> pd.DataFrame:
    """A miniature DataFrame shaped like the UCI Diabetes-130 raw cohort.

    It carries just enough of the real schema for the cleaning and
    feature-engineering helpers to operate on: the categorical columns that
    get "?"-coded, the lab columns, the utilization counts, the age bucket,
    and the readmission label. Patient ids repeat across rows so the
    patient-grouped splitter has groups to honour. Values vary deliberately,
    including the literal "?" sentinel, to exercise the missing-value paths.
    """
    n = 60
    return pd.DataFrame(
        {
            "patient_nbr": rng.integers(1000, 1010, size=n),
            "encounter_id": np.arange(n),
            "race": rng.choice(["Caucasian", "AfricanAmerican", "?"], size=n),
            "gender": rng.choice(["Male", "Female"], size=n),
            "age": rng.choice(["[0-10)", "[40-50)", "[70-80)"], size=n),
            "payer_code": rng.choice(["MC", "HM", "?"], size=n),
            "medical_specialty": rng.choice(["Cardiology", "?", "Surgery"], size=n),
            "diag_1": rng.choice(["250.83", "428", "V57", "?"], size=n),
            "diag_2": rng.choice(["401", "276", "?"], size=n),
            "diag_3": rng.choice(["250", "414", "?"], size=n),
            "A1Cresult": rng.choice(["None", ">7", "Norm"], size=n),
            "max_glu_serum": rng.choice(["None", ">200", "Norm"], size=n),
            "time_in_hospital": rng.integers(1, 14, size=n),
            "num_lab_procedures": rng.integers(1, 80, size=n),
            "num_procedures": rng.integers(0, 6, size=n),
            "num_medications": rng.integers(1, 40, size=n),
            "number_outpatient": rng.integers(0, 5, size=n),
            "number_emergency": rng.integers(0, 5, size=n),
            "number_inpatient": rng.integers(0, 5, size=n),
            "number_diagnoses": rng.integers(1, 16, size=n),
            "readmitted": rng.choice(["NO", ">30", "<30"], size=n),
        }
    )
