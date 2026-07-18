"""Authentication service — Firebase + Flask-Login integration."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import firebase_admin
from firebase_admin import auth as firebase_auth
from flask import session
from flask_login import UserMixin

from webapp.config import Config

logger = logging.getLogger(__name__)


@dataclass
class User(UserMixin):
    id: str
    email: str
    display_name: str
    firebase_uid: str
    subscription_tier: str = "free"
    stripe_customer_id: Optional[str] = None
    stripe_subscription_id: Optional[str] = None
    credits_remaining: int = 100
    created_at: Optional[datetime] = None
    last_login: Optional[datetime] = None

    def get_id(self) -> str:
        return self.id

    @property
    def tier_config(self) -> dict:
        return Config.SUBSCRIPTION_TIERS.get(self.subscription_tier, Config.SUBSCRIPTION_TIERS["free"])

    @staticmethod
    def get(user_id: str) -> Optional["User"]:
        try:
            import sqlite3
            conn = sqlite3.connect(Config.DATABASE_PATH)
            row = conn.execute(
                "SELECT * FROM users WHERE id = ?", (int(user_id),)
            ).fetchone()
            conn.close()
            if row:
                cols = [desc[0] for desc in conn.execute("SELECT * FROM users LIMIT 0").description] if False else _get_user_columns()
                cols = ["id", "firebase_uid", "email", "display_name", "subscription_tier",
                       "stripe_customer_id", "stripe_subscription_id", "credits_remaining",
                       "created_at", "last_login"]
                data = dict(zip(cols, row))
                return User(
                    id=str(data["id"]),
                    email=data.get("email") or "",
                    display_name=data.get("display_name") or "",
                    firebase_uid=data.get("firebase_uid") or "",
                    subscription_tier=data.get("subscription_tier") or "free",
                    stripe_customer_id=data.get("stripe_customer_id"),
                    stripe_subscription_id=data.get("stripe_subscription_id"),
                    credits_remaining=data.get("credits_remaining", 100),
                    created_at=data.get("created_at"),
                    last_login=data.get("last_login"),
                )
        except Exception as exc:
            logger.error(f"User.get failed: {exc}")
        return None

    @staticmethod
    def get_by_firebase_uid(uid: str) -> Optional["User"]:
        try:
            import sqlite3
            conn = sqlite3.connect(Config.DATABASE_PATH)
            row = conn.execute(
                "SELECT * FROM users WHERE firebase_uid = ?", (uid,)
            ).fetchone()
            conn.close()
            if row:
                data = dict(zip(
                    ["id", "firebase_uid", "email", "display_name", "subscription_tier",
                     "stripe_customer_id", "stripe_subscription_id", "credits_remaining",
                     "created_at", "last_login"],
                    row,
                ))
                return User(
                    id=str(data["id"]),
                    email=data.get("email") or "",
                    display_name=data.get("display_name") or "",
                    firebase_uid=data.get("firebase_uid") or "",
                    subscription_tier=data.get("subscription_tier") or "free",
                    stripe_customer_id=data.get("stripe_customer_id"),
                    stripe_subscription_id=data.get("stripe_subscription_id"),
                    credits_remaining=data.get("credits_remaining", 100),
                    created_at=data.get("created_at"),
                    last_login=data.get("last_login"),
                )
        except Exception:
            pass
        return None

    @staticmethod
    def create_or_update(firebase_uid: str, email: str = "", display_name: str = "") -> Optional["User"]:
        try:
            import sqlite3
            conn = sqlite3.connect(Config.DATABASE_PATH)
            existing = conn.execute(
                "SELECT id FROM users WHERE firebase_uid = ?", (firebase_uid,)
            ).fetchone()
            if existing:
                conn.execute(
                    """UPDATE users SET email=?, display_name=?, last_login=?
                       WHERE firebase_uid=?""",
                    (email, display_name, datetime.now(timezone.utc).isoformat(), firebase_uid),
                )
            else:
                conn.execute(
                    """INSERT INTO users (firebase_uid, email, display_name, last_login)
                       VALUES (?, ?, ?, ?)""",
                    (firebase_uid, email, display_name, datetime.now(timezone.utc).isoformat()),
                )
            conn.commit()
            user = User.get_by_firebase_uid(firebase_uid)
            conn.close()
            return user
        except Exception as exc:
            logger.error(f"User.create_or_update failed: {exc}")
        return None

    @staticmethod
    def update_tier(user_id: str, tier: str, stripe_customer_id: str = "", stripe_subscription_id: str = "") -> None:
        try:
            import sqlite3
            conn = sqlite3.connect(Config.DATABASE_PATH)
            conn.execute(
                """UPDATE users SET subscription_tier=?, stripe_customer_id=?, stripe_subscription_id=?
                   WHERE id=?""",
                (tier, stripe_customer_id, stripe_subscription_id, int(user_id)),
            )
            conn.commit()
            conn.close()
        except Exception as exc:
            logger.error(f"User.update_tier failed: {exc}")

    @staticmethod
    def verify_firebase_token(id_token: str) -> Optional[dict]:
        if not Config.FIREBASE_AUTH_ENABLED:
            return {"uid": "dev_user", "email": "dev@quantflow.local"}
        try:
            decoded = firebase_auth.verify_id_token(id_token)
            return decoded
        except Exception as exc:
            logger.warning(f"Firebase token verification failed: {exc}")
            return None

    @staticmethod
    def create_db_tables(db_path: str) -> None:
        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                firebase_uid TEXT UNIQUE NOT NULL,
                email TEXT,
                display_name TEXT,
                subscription_tier TEXT DEFAULT 'free',
                stripe_customer_id TEXT,
                stripe_subscription_id TEXT,
                credits_remaining INTEGER DEFAULT 100,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS model_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                version TEXT NOT NULL,
                onnx_path TEXT,
                pytorch_path TEXT,
                metrics_json TEXT,
                is_active BOOLEAN DEFAULT 0,
                trained_at TIMESTAMP,
                deployed_at TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS ml_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                model_version_id INTEGER REFERENCES model_versions(id),
                prediction_date DATE NOT NULL,
                prob_inter_event REAL,
                prob_pre_event REAL,
                prob_onset REAL,
                forecast_tau_days REAL,
                uncertainty_sigma REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker, prediction_date, model_version_id)
            );
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER REFERENCES users(id),
                snapshot_date DATE NOT NULL,
                ticker TEXT NOT NULL,
                shares REAL,
                cost_basis REAL,
                current_price REAL,
                allocation_pct REAL,
                ml_signal TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS agent_conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER REFERENCES users(id),
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                tool_calls_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS backtest_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER REFERENCES users(id),
                status TEXT DEFAULT 'queued',
                config_json TEXT NOT NULL,
                result_json TEXT,
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS usage_tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER REFERENCES users(id),
                action TEXT NOT NULL,
                detail TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()
        conn.close()
        logger.info("Database tables created")


def _get_user_columns() -> list:
    return [
        "id", "firebase_uid", "email", "display_name", "subscription_tier",
        "stripe_customer_id", "stripe_subscription_id", "credits_remaining",
        "created_at", "last_login",
    ]
