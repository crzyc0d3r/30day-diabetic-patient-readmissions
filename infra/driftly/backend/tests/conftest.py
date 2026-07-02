"""Shared fixtures for the Driftly backend tests.

These tests are hermetic: they never read the repository's real ``data/``.
Instead each test gets a temp ``MEDIWATCH_DATA_DIR`` containing a small
synthetic reference matrix plus a couple of incoming batches, so the API and the
metric math run against data we fully control.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Make the backend modules (config, drift_metrics, history, main) and the repo's
# `helpers` package importable, regardless of pytest's rootdir.
_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_BACKEND)))
for p in (_BACKEND, _REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from helpers.drift_sim import MONITORED_CATEGORICAL, MONITORED_CONTINUOUS, TARGET


def _frame(n: int, rng: np.random.Generator, *, shift: float = 0.0) -> pd.DataFrame:
    """A schema-valid synthetic batch over the monitored columns. ``shift``
    pushes the continuous columns and skews the categoricals to inject drift."""
    data: dict[str, object] = {}
    for col in MONITORED_CONTINUOUS:
        data[col] = rng.normal(loc=5.0 + shift, scale=2.0, size=n).clip(min=0)
    cats = ["a", "b", "c", "d"]
    # weights tilt toward the tail as `shift` grows, moving categorical mass
    weights = np.array([4.0, 3.0, 2.0, 1.0]) + np.array([0, 0, shift, shift * 2])
    weights = weights / weights.sum()
    for col in MONITORED_CATEGORICAL:
        data[col] = rng.choice(cats, size=n, p=weights)
    data[TARGET] = rng.integers(0, 2, size=n)
    return pd.DataFrame(data)


@pytest.fixture()
def data_dir(tmp_path, monkeypatch):
    """A temp data root wired into the backend via MEDIWATCH_DATA_DIR, holding
    features.csv (reference) + incoming/current.csv + incoming/spike.csv."""
    rng = np.random.default_rng(7)
    root = tmp_path / "data"
    incoming = root / "incoming"
    incoming.mkdir(parents=True)

    _frame(3000, rng).to_csv(root / "features.csv", index=False)
    _frame(800, rng).to_csv(incoming / "current.csv", index=False)            # no drift
    _frame(800, rng, shift=6.0).to_csv(incoming / "spike.csv", index=False)   # strong drift

    monkeypatch.setenv("MEDIWATCH_DATA_DIR", str(root))

    # The reference cache in main is a module global; clear it so a prior test's
    # (different) reference can't leak across the env swap.
    import importlib
    main = importlib.import_module("main")
    main._ref_cache.update({"key": None, "df": None})
    return root


@pytest.fixture()
def client(data_dir):
    from fastapi.testclient import TestClient

    import main
    return TestClient(main.app)
