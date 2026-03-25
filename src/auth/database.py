"""
src/auth/database.py — SQLite / SQLAlchemy setup
=================================================
Creates the database engine, session factory, and base class.
Call init_db() once at app startup to create all tables.
Database file: medverify.db (project root, gitignored).
"""
from __future__ import annotations

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

# ── Path ──────────────────────────────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DB_PATH = os.path.join(_ROOT, "medverify.db")
DATABASE_URL = f"sqlite:///{_DB_PATH}"

# ── Engine ────────────────────────────────────────────────────────────────────
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # required for SQLite + threads
    echo=False,
)

# ── Session factory ───────────────────────────────────────────────────────────
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ── Declarative base ──────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    pass


# ── Public helpers ────────────────────────────────────────────────────────────
def get_db():
    """Yield a database session; always closes on exit."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables defined on Base.metadata (idempotent)."""
    # Import models here to register them on Base before create_all
    from src.auth import models  # noqa: F401
    Base.metadata.create_all(bind=engine)
