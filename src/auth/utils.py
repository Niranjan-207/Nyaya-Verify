"""
src/auth/utils.py — Auth Business Logic
========================================
Password hashing  : PBKDF2-HMAC-SHA256 (stdlib only — no extra deps).
                    Format stored:  "<hex-salt>:<hex-digest>"
CRUD helpers      : create_user, get_user_by_email, update_payment_status.
"""
from __future__ import annotations

import hashlib
import os
from typing import Optional

from sqlalchemy.orm import Session

from src.auth.models import User


# ─────────────────────────────────────────────────────────────────────────────
# Password hashing (stdlib PBKDF2 — no bcrypt dep)
# ─────────────────────────────────────────────────────────────────────────────
_ITERATIONS = 260_000   # OWASP 2023 recommended minimum for PBKDF2-SHA256


def hash_password(password: str) -> str:
    """Return a salted PBKDF2-SHA256 hash safe to store in the DB."""
    salt = os.urandom(16).hex()           # 32-char hex string
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        _ITERATIONS,
    ).hex()
    return f"{salt}:{digest}"


def verify_password(plain: str, stored_hash: str) -> bool:
    """Return True when `plain` matches `stored_hash`."""
    try:
        salt, digest = stored_hash.split(":", 1)
        check = hashlib.pbkdf2_hmac(
            "sha256",
            plain.encode("utf-8"),
            salt.encode("utf-8"),
            _ITERATIONS,
        ).hex()
        return check == digest
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# CRUD helpers
# ─────────────────────────────────────────────────────────────────────────────
def create_user(
    db: Session,
    full_name: str,
    email: str,
    password: str,
) -> User:
    """
    Create and persist a new User with payment_status=False.
    Raises ValueError if the email is already registered.
    """
    if get_user_by_email(db, email):
        raise ValueError(f"Email already registered: {email}")

    user = User(
        full_name=full_name.strip(),
        email=email.strip().lower(),
        password_hash=hash_password(password),
        payment_status=False,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Return the User with this email, or None."""
    return (
        db.query(User)
        .filter(User.email == email.strip().lower())
        .first()
    )


def update_payment_status(db: Session, user_id: int, status: bool = True) -> None:
    """Flip the payment_status flag for a user (called after QR confirmation)."""
    user = db.query(User).filter(User.id == user_id).first()
    if user:
        user.payment_status = status
        db.commit()
