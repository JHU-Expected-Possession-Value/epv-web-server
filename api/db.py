"""Database connection utilities for the FastAPI backend.

Runtime source of truth:
- **AWS RDS PostgreSQL** (or any Postgres reachable from the API) populated by
  `EPV_SARG/AWS/fillTables.py` (`teams`, `players`, `matches`, `frame`, `detection`,
  `events`, `shots`, `goals`, `carries`, `passes`, …).
- Website requests use **fresh SQLAlchemy sessions** per request (`get_db`); new rows ingested
  into RDS are visible on the next API call (no in-memory cache of table contents).

Connection parity with `fillTables.py`:
- Ingest script typically builds:
  `create_engine("postgresql://USER:PASSWORD@HOST:5432/DATABASE")` (see that file).
- Here we use `postgresql+psycopg2://...` (SQLAlchemy 2 + psycopg2 driver) or a full
  **`DATABASE_URL`** env var. Set **`PGHOST` / `PGDATABASE` / `PGUSER` / `PGPASSWORD`**
  (and optional **`PGPORT`**) to match the same host/db/user as `fillTables.py`, without
  hardcoding secrets in code.
- For RDS outside the VPC, append SSL query params to **`DATABASE_URL`** if required
  (e.g. `?sslmode=require`).

Local SkillCorner file trees (`EPV_DATA_DIR`) are **not** used by `get_db`; see `api/utils/paths.py`
for offline-only helpers.

**Env file:** `epv-web-server/.env` is loaded from disk at import time (see `_load_backend_env`)
so `SessionLocal` / `get_engine()` see `DATABASE_URL` / `PG*` even when the process CWD is not
`epv-web-server` (e.g. repo root or Docker WORKDIR).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Generator

from dotenv import load_dotenv
from sqlalchemy import create_engine


def _load_backend_env() -> None:
    """Load `epv-web-server/.env` into `os.environ` before any DB URL is read.

    Path: this file is `epv-web-server/api/db.py` → parent.parent is `epv-web-server`.
    """
    backend_root = Path(__file__).resolve().parent.parent
    load_dotenv(backend_root / ".env")


_load_backend_env()
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker


def _build_database_url_from_env() -> str:
    """Build a postgres URL using the same pieces as `fillTables.py`.

    Required env:
    - PGHOST
    - PGDATABASE
    - PGUSER
    - PGPASSWORD
    Optional:
    - PGPORT (defaults to 5432)
    """
    host = (os.getenv("PGHOST") or "").strip()
    db = (os.getenv("PGDATABASE") or "").strip()
    user = (os.getenv("PGUSER") or "").strip()
    password = (os.getenv("PGPASSWORD") or "").strip()
    port = (os.getenv("PGPORT") or "5432").strip()

    if not host or not db or not user or not password:
        raise RuntimeError(
            "Postgres env vars must be set (PGHOST, PGDATABASE, PGUSER, PGPASSWORD). "
            "Alternatively set DATABASE_URL."
        )

    # Keep it simple and compatible with the `fillTables.py` create_engine() style.
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Create (and cache) a SQLAlchemy engine from env vars.

    Preferred env (matches `fillTables.py` pieces):
    - PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD

    Alternative:
    - DATABASE_URL (full SQLAlchemy URL)
    """

    database_url = (os.getenv("DATABASE_URL") or "").strip()
    if not database_url:
        database_url = _build_database_url_from_env()

    # pool_pre_ping helps with RDS/NAT idle timeouts.
    return create_engine(database_url, pool_pre_ping=True)


def get_sessionmaker() -> sessionmaker:
    engine = get_engine()
    return sessionmaker(bind=engine, autocommit=False, autoflush=False)


SessionLocal = get_sessionmaker()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

