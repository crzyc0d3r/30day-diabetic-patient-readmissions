"""Shared Postgres data-access layer for MediWatch.

Single seam through which every system (notebooks, helpers, Airflow DAGs, the
inference API, Driftly) reads and writes data. Backed by the `mediwatch`
logical database in the shared Postgres container.

Connection comes from the MEDIWATCH_DB_URL env var (see infra/.env).

Security note: load_arrays / load_joblib deserialize numpy/joblib payloads,
which can execute arbitrary code on a malicious blob. Every blob in the
`artifacts` table is produced by THIS pipeline and stored in OUR OWN database --
a trusted, first-party round-trip, never untrusted input.
"""
from __future__ import annotations

import io
import os
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.exc import ProgrammingError

DEFAULT_DB_URL = "postgresql+psycopg2://mlops:mlops-local@localhost:5432/mediwatch"


def db_url() -> str:
    return os.environ.get("MEDIWATCH_DB_URL", DEFAULT_DB_URL)


def ensure_database() -> None:
    """Create the target database if it does not already exist.

    The postgres-init script only runs on a fresh volume, so on an
    already-running container we create `mediwatch` on demand via an AUTOCOMMIT
    connection to the always-present `postgres` maintenance database.
    """
    url = make_url(db_url())
    target = url.database
    admin = create_engine(
        url.set(database="postgres"), isolation_level="AUTOCOMMIT", future=True
    )
    try:
        with admin.connect() as conn:
            exists = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :n"),
                {"n": target},
            ).scalar()
            if not exists:
                # Identifier can't be parameterized; target comes from our own
                # env var, not user input.
                try:
                    conn.execute(text(f'CREATE DATABASE "{target}"'))
                except ProgrammingError:
                    # Lost a creation race with another process; the DB now
                    # exists, which is all we wanted.
                    pass
    finally:
        admin.dispose()


@lru_cache(maxsize=None)
def get_engine() -> Engine:
    """Process-level cached SQLAlchemy engine for the mediwatch DB."""
    ensure_database()
    return create_engine(db_url(), pool_pre_ping=True, future=True)


def read_table(name: str) -> "pd.DataFrame":
    """Read an entire table into a DataFrame."""
    return pd.read_sql_table(name, get_engine())


def write_table(df: "pd.DataFrame", name: str, if_exists: str = "replace") -> int:
    """Write a DataFrame to a table; returns the row count written.

    Chunked to keep memory bounded. Uses pandas' default (executemany) insert
    rather than method="multi" so wide tables never exceed the 65535-parameter
    statement limit.
    """
    df.to_sql(name, get_engine(), if_exists=if_exists, index=False, chunksize=5000)
    return len(df)


_ARTIFACTS_DDL = text(
    """
    CREATE TABLE IF NOT EXISTS artifacts (
        name        TEXT PRIMARY KEY,
        kind        TEXT NOT NULL,
        payload     BYTEA NOT NULL,
        created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """
)


def _put_blob(name: str, kind: str, payload: bytes) -> None:
    with get_engine().begin() as conn:
        conn.execute(_ARTIFACTS_DDL)
        conn.execute(
            text(
                """
                INSERT INTO artifacts (name, kind, payload)
                VALUES (:name, :kind, :payload)
                ON CONFLICT (name) DO UPDATE
                SET kind = EXCLUDED.kind,
                    payload = EXCLUDED.payload,
                    created_at = now()
                """
            ),
            {"name": name, "kind": kind, "payload": payload},
        )


def _get_blob(name: str) -> bytes:
    with get_engine().begin() as conn:
        conn.execute(_ARTIFACTS_DDL)
        row = conn.execute(
            text("SELECT payload FROM artifacts WHERE name = :n"), {"n": name}
        ).fetchone()
    if row is None:
        raise KeyError(f"artifact not found: {name!r}")
    return bytes(row[0])


def save_arrays(name: str, **arrays) -> None:
    """Serialize named numpy arrays into one compressed blob in `artifacts`."""
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    _put_blob(name, "arrays", buf.getvalue())


def load_arrays(name: str) -> dict:
    """Load arrays saved by save_arrays. allow_pickle=False: numeric-only, safe."""
    buf = io.BytesIO(_get_blob(name))
    with np.load(buf, allow_pickle=False) as npz:
        return {k: npz[k] for k in npz.files}


def dump_joblib(obj, name: str) -> None:
    """Serialize any Python object with joblib into one blob in `artifacts`."""
    buf = io.BytesIO()
    joblib.dump(obj, buf)
    _put_blob(name, "model", buf.getvalue())


def load_joblib(name: str):
    """Load an object saved by dump_joblib. Trusted first-party blob (see module
    docstring) -- joblib.load executes arbitrary code, but the source is our own
    pipeline writing to our own DB."""
    return joblib.load(io.BytesIO(_get_blob(name)))
