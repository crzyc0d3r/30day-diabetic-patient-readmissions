"""Importer: flat files -> the shared `mediwatch` Postgres database.

Phase 1 (raw-dataset-first): load orig_dataset/diabetic_data.csv into the
`raw_diabetic` table. Re-running replaces the table, so it is idempotent.
Later phases (cleaned/features/drift batches/artifacts) extend this module.

Usage:
    python -m helpers.migrate_to_postgres        # imports raw_diabetic
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from helpers import db

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "orig_dataset" / "diabetic_data.csv"


def import_raw_diabetic(csv_path: Path = RAW_CSV, table: str = "raw_diabetic") -> int:
    """Load the original dataset CSV into the given table verbatim.

    Returns the number of rows written.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    return db.write_table(df, table, if_exists="replace")


def main() -> None:
    n = import_raw_diabetic()
    print(f"[migrate] raw_diabetic: {n} rows -> mediwatch")


if __name__ == "__main__":
    main()
