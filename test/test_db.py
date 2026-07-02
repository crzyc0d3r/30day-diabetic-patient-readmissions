"""Tests for the shared Postgres data-access layer (helpers/db.py).

These are integration tests against the live `mediwatch` Postgres. If the DB is
unreachable the whole module is skipped, so CI without Postgres stays green.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text

from helpers import db


@pytest.fixture(scope="module")
def engine():
    try:
        eng = db.get_engine()
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:  # pragma: no cover - env-dependent
        pytest.skip(f"mediwatch Postgres not reachable: {exc}")
    return eng


def test_get_engine_is_cached(engine):
    assert db.get_engine() is db.get_engine()


def test_engine_targets_mediwatch_db(engine):
    with engine.connect() as conn:
        name = conn.execute(text("SELECT current_database()")).scalar()
    assert name == "mediwatch"


def test_write_read_table_roundtrip(engine):
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    try:
        n = db.write_table(df, "_test_roundtrip", if_exists="replace")
        assert n == 3
        out = db.read_table("_test_roundtrip")
        pd.testing.assert_frame_equal(
            out.sort_index(axis=1).reset_index(drop=True),
            df.sort_index(axis=1).reset_index(drop=True),
        )
    finally:
        with engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS _test_roundtrip"))


def test_write_table_replace_overwrites(engine):
    try:
        db.write_table(pd.DataFrame({"a": [1]}), "_test_replace", if_exists="replace")
        db.write_table(pd.DataFrame({"a": [9, 9]}), "_test_replace", if_exists="replace")
        out = db.read_table("_test_replace")
        assert out["a"].tolist() == [9, 9]
    finally:
        with engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS _test_replace"))


def test_save_load_arrays_roundtrip(engine):
    a = np.arange(6, dtype="float64").reshape(2, 3)
    b = np.array([1, 0, 1], dtype="int64")
    try:
        db.save_arrays("_test_arrays", X=a, y=b)
        out = db.load_arrays("_test_arrays")
        np.testing.assert_array_equal(out["X"], a)
        np.testing.assert_array_equal(out["y"], b)
        assert out["X"].dtype == a.dtype
    finally:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM artifacts WHERE name = '_test_arrays'"))


def test_save_arrays_overwrites(engine):
    try:
        db.save_arrays("_test_arr_ow", X=np.array([1.0]))
        db.save_arrays("_test_arr_ow", X=np.array([2.0, 2.0]))
        out = db.load_arrays("_test_arr_ow")
        np.testing.assert_array_equal(out["X"], np.array([2.0, 2.0]))
    finally:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM artifacts WHERE name = '_test_arr_ow'"))


def test_dump_load_joblib_roundtrip(engine):
    obj = {"threshold": 0.42, "labels": ["<30", ">30", "NO"]}
    try:
        db.dump_joblib(obj, "_test_joblib")
        out = db.load_joblib("_test_joblib")
        assert out == obj
    finally:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM artifacts WHERE name = '_test_joblib'"))


def test_load_missing_artifact_raises(engine):
    with pytest.raises(KeyError):
        db.load_joblib("_definitely_missing_artifact_xyz")
