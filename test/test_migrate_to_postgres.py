"""Integration test for the flat-file -> Postgres importer (raw slice)."""
from __future__ import annotations

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


def test_import_raw_diabetic_from_small_csv(tmp_path, engine):
    """import_raw_diabetic loads the given CSV verbatim into the target table."""
    from helpers.migrate_to_postgres import import_raw_diabetic

    csv = tmp_path / "tiny.csv"
    pd.DataFrame(
        {"encounter_id": [1, 2], "race": ["Caucasian", "?"], "readmitted": ["NO", "<30"]}
    ).to_csv(csv, index=False)

    table = "_test_raw_diabetic"
    try:
        n = import_raw_diabetic(csv_path=csv, table=table)
        assert n == 2
        out = db.read_table(table)
        assert out.shape == (2, 3)
        assert list(out.columns) == ["encounter_id", "race", "readmitted"]
        assert set(out["readmitted"]) == {"NO", "<30"}
    finally:
        with engine.begin() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {table}"))
