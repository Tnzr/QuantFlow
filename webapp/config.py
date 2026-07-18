"""Central configuration for QuantFlow WebApp."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEBAPP_ROOT = Path(__file__).resolve().parent


class Config:
    SECRET_KEY: str = os.getenv("QF_SECRET_KEY", os.urandom(32).hex())
    DATABASE_PATH: str = os.getenv("QF_DB_PATH", str(PROJECT_ROOT / "quantflow.db"))
    SQLALCHEMY_DATABASE_URI: str = f"sqlite:///{DATABASE_PATH}"

    FIREBASE_AUTH_ENABLED: bool = os.getenv("QF_FIREBASE_AUTH_ENABLED", "false").lower() == "true"
    FIREBASE_CREDENTIALS_PATH: str = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    FIREBASE_API_KEY: str = os.getenv("QF_FIREBASE_API_KEY", "")
    FIREBASE_AUTH_DOMAIN: str = os.getenv("QF_FIREBASE_PROJECT_ID", "") + ".firebaseapp.com"
    FIREBASE_PROJECT_ID: str = os.getenv("QF_FIREBASE_PROJECT_ID", "")

    STRIPE_SECRET_KEY: str = os.getenv("STRIPE_SECRET_KEY", "")
    STRIPE_PUBLISHABLE_KEY: str = os.getenv("STRIPE_PUBLISHABLE_KEY", "")
    STRIPE_WEBHOOK_SECRET: str = os.getenv("STRIPE_WEBHOOK_SECRET", "")
    STRIPE_PRO_PRICE_ID: str = os.getenv("STRIPE_PRO_PRICE_ID", "")
    STRIPE_ENTERPRISE_PRICE_ID: str = os.getenv("STRIPE_ENTERPRISE_PRICE_ID", "")

    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", REDIS_URL)
    CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL)

    ONNX_MODEL_PATH: str = os.getenv(
        "QF_ONNX_MODEL_PATH",
        str(PROJECT_ROOT / "checkpoints" / "bilstm_full_universe_v2.onnx"),
    )
    PYTORCH_MODEL_PATH: str = os.getenv(
        "QF_PYTORCH_MODEL_PATH",
        str(PROJECT_ROOT / "checkpoints" / "bilstm_full_universe_v2.pt"),
    )

    API_BASE_URL: str = os.getenv("QF_API_BASE_URL", "http://127.0.0.1:5001")
    APP_NAME: str = "QuantFlow"
    APP_VERSION: str = "1.0.0-beta"

    SUBSCRIPTION_TIERS: Dict[str, dict] = {
        "free": {
            "name": "Free",
            "scanner_presets": 3,
            "ticker_limit": 5,
            "ml_requests_daily": 3,
            "backtest_daily": 1,
            "agent_queries_daily": 10,
            "data_refresh_hours": 168,
            "api_requests_daily": 0,
            "broker_access": "readonly",
            "export_formats": [],
        },
        "pro": {
            "name": "Pro",
            "price_id": STRIPE_PRO_PRICE_ID,
            "price_monthly": 29,
            "scanner_presets": 999,
            "ticker_limit": 50,
            "ml_requests_daily": 50,
            "backtest_daily": 10,
            "agent_queries_daily": 100,
            "data_refresh_hours": 24,
            "api_requests_daily": 1000,
            "broker_access": "readwrite",
            "export_formats": ["pdf"],
        },
        "enterprise": {
            "name": "Enterprise",
            "price_id": STRIPE_ENTERPRISE_PRICE_ID,
            "price_monthly": 199,
            "scanner_presets": 9999,
            "ticker_limit": 9999,
            "ml_requests_daily": 99999,
            "backtest_daily": 9999,
            "agent_queries_daily": 99999,
            "data_refresh_hours": 0.016,
            "api_requests_daily": 999999,
            "broker_access": "full_mcp",
            "export_formats": ["pdf", "csv", "api"],
        },
    }

    RATE_LIMIT_WINDOW: int = 3600
    RATE_LIMIT_MAX: int = 1000
