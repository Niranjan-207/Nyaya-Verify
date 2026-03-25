"""
src/auth/models.py — SQLAlchemy ORM Models
==========================================
User table schema:
  id             — auto-increment primary key
  full_name      — display name (required)
  email          — unique login identifier (required)
  password_hash  — PBKDF2-SHA256 hash (never store plaintext)
  payment_status — False until QR payment confirmed
  created_at     — UTC timestamp of registration
"""
from __future__ import annotations

from datetime import datetime, timezone
from sqlalchemy import Boolean, Column, DateTime, Integer, String

from src.auth.database import Base


class User(Base):
    __tablename__ = "users"

    id             = Column(Integer,  primary_key=True, index=True, autoincrement=True)
    full_name      = Column(String,   nullable=False)
    email          = Column(String,   unique=True, index=True, nullable=False)
    password_hash  = Column(String,   nullable=False)
    payment_status = Column(Boolean,  default=False, nullable=False)
    created_at     = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    def __repr__(self) -> str:
        return (
            f"<User id={self.id} email={self.email!r} "
            f"paid={self.payment_status}>"
        )
