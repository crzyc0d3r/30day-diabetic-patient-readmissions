"""Run-history store for Driftly, backed by the shared mediwatch Postgres.

Each on-demand compute is appended so the dashboard's trend view can plot how
drift evolves over time. The store uses the same Postgres database as the rest
of the stack via ``MEDIWATCH_DB_URL`` -- ``helpers/db.py`` names Driftly as one
of the consumers of that single data seam. When no database URL is configured
(the hermetic test suite and bare-local runs), it falls back to a self-contained
SQLite file under ``data/driftly/`` through the same SQLAlchemy code path, so the
backend never hard-depends on a running Postgres just to be tested.
"""
from __future__ import annotations

from sqlalchemy import (
    Column, Float, Integer, MetaData, String, Table, create_engine, insert, select,
    text,
)
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.exc import ProgrammingError

import config

_metadata = MetaData()

_runs = Table(
    "driftly_runs", _metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("computed_at", String, nullable=False),
    Column("batch", String, nullable=False),
    Column("verdict", String, nullable=False),
    Column("ok", Integer, nullable=False),
    Column("warn", Integer, nullable=False),
    Column("alert", Integer, nullable=False),
    Column("reference_rows", Integer, nullable=False),
    Column("current_rows", Integer, nullable=False),
)

_feature_metrics = Table(
    "driftly_feature_metrics", _metadata,
    Column("run_id", Integer, nullable=False, index=True),
    Column("feature", String, nullable=False, index=True),
    Column("type", String, nullable=False),
    Column("wasserstein", Float),
    Column("psi", Float),
    Column("kl", Float),
    Column("status", String, nullable=False),
)

_engine: Engine | None = None


def _ensure_database(url: str) -> None:
    """Create the target Postgres database if it does not exist.

    The postgres-init script only runs on a fresh volume, so on an
    already-running cluster the `mediwatch` database may be absent. The target
    name comes from our own env var, not user input.
    """
    u = make_url(url)
    if u.get_backend_name() != "postgresql":
        return
    admin = create_engine(
        u.set(database="postgres"), isolation_level="AUTOCOMMIT", future=True
    )
    try:
        with admin.connect() as conn:
            exists = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :n"),
                {"n": u.database},
            ).scalar()
            if not exists:
                try:
                    conn.execute(text(f'CREATE DATABASE "{u.database}"'))
                except ProgrammingError:
                    # Lost a creation race; the DB now exists, which is all we want.
                    pass
    finally:
        admin.dispose()


def _engine_for() -> Engine:
    """Process-cached engine. Postgres when MEDIWATCH_DB_URL is set, otherwise a
    self-contained SQLite file under data/driftly/. Tables are created on first use."""
    global _engine
    if _engine is None:
        url = config.db_url()
        if url:
            _ensure_database(url)
        else:
            import os
            os.makedirs(config.history_dir(), exist_ok=True)
            url = f"sqlite:///{config.history_db()}"
        _engine = create_engine(url, pool_pre_ping=True, future=True)
        _metadata.create_all(_engine)
    return _engine


def init_db() -> None:
    """Idempotently ensure the history tables exist."""
    _engine_for()


def record_run(result: dict) -> int:
    """Persist one compute result and return its run id."""
    s = result["summary"]
    eng = _engine_for()
    with eng.begin() as conn:
        run_id = conn.execute(
            insert(_runs).values(
                computed_at=result["computed_at"],
                batch=result["batch"],
                verdict=result["verdict"],
                ok=s["ok"], warn=s["warn"], alert=s["alert"],
                reference_rows=result["reference_rows"],
                current_rows=result["current_rows"],
            )
        ).inserted_primary_key[0]
        if result["features"]:
            conn.execute(
                insert(_feature_metrics),
                [
                    {
                        "run_id": run_id, "feature": f["name"], "type": f["type"],
                        "wasserstein": f["wasserstein"], "psi": f["psi"], "kl": f["kl"],
                        "status": f["status"],
                    }
                    for f in result["features"]
                ],
            )
    return int(run_id)


def list_runs(feature: str | None = None) -> list[dict]:
    """Return runs oldest-first for the trend chart. When *feature* is given,
    each run also carries that feature's metric values for a per-feature series."""
    eng = _engine_for()
    with eng.connect() as conn:
        runs = [dict(r._mapping) for r in
                conn.execute(select(_runs).order_by(_runs.c.id.asc()))]
        if feature:
            by_run = {
                r._mapping["run_id"]: dict(r._mapping)
                for r in conn.execute(
                    select(
                        _feature_metrics.c.run_id, _feature_metrics.c.wasserstein,
                        _feature_metrics.c.psi, _feature_metrics.c.kl,
                        _feature_metrics.c.status,
                    ).where(_feature_metrics.c.feature == feature)
                )
            }
            for run in runs:
                fm = by_run.get(run["id"])
                run["feature"] = (
                    {
                        "name": feature, "wasserstein": fm["wasserstein"],
                        "psi": fm["psi"], "kl": fm["kl"], "status": fm["status"],
                    }
                    if fm else None
                )
    return runs
