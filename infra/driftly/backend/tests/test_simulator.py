"""Unit tests for simulator — scenario injection into the live current.csv slot."""
from __future__ import annotations

import os

import pandas as pd
import pytest

import config
import simulator
from helpers.drift_sim import SCENARIOS


def test_list_scenarios_matches_drift_sim():
    items = simulator.list_scenarios()
    assert [x["name"] for x in items] == list(SCENARIOS)
    assert all(x["description"] for x in items)


def test_inject_overwrites_current_slot(data_dir):
    ref = pd.read_csv(config.reference_path(), low_memory=False)
    info = simulator.inject(ref, "mixed_severe", severity=1.5)

    assert info["scenario"] == "mixed_severe"
    assert info["severity"] == 1.5
    assert info["rows"] > 0
    assert info["path"] == config.current_path()
    assert os.path.exists(config.current_path())

    written = pd.read_csv(config.current_path(), low_memory=False)
    assert set(written.columns) == set(ref.columns)   # schema preserved


def test_inject_unknown_scenario_raises(data_dir):
    ref = pd.read_csv(config.reference_path(), low_memory=False)
    with pytest.raises(ValueError):
        simulator.inject(ref, "not_a_scenario")
