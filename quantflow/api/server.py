from __future__ import annotations

from typing import Optional
import re
import uuid
from threading import Lock, Thread
from datetime import datetime, timezone
import multiprocessing as mp
from queue import Empty
import numpy as np
import pandas as pd
import yfinance as yf
import yaml

import os

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

from ..data.persistence import create_schema
from ..data.queries import recent_execution_intents, latest_universe, latest_recommendations_per_ticker
from ..broker.factory import make_broker, BROKER_MODE_MCP
from ..broker.mcp_client import (
    clear_runtime_mcp_auth_config,
    get_runtime_mcp_auth_config_summary,
    set_runtime_mcp_auth_config,
)
from ..portfolio.evaluator import evaluate_positions
from ..execution.policy import ExecutionPolicy, evaluate_order_intent
from ..execution.audit import audit_intent, audit_event, audit_policy
from .auth import require_write_auth, require_firebase_user
from .user_settings import load_user_settings, save_user_settings, UserSettingsStoreError
from ..data.finviz_client import PRESETS, run_screener, run_screeners_parallel, get_scraping_runtime_config
from ..data.persistence import save_finviz_snapshot, save_recommendations, save_analytics_leaderboard
from ..data.persistence import save_news_articles
from ..data.universe import HIGH_INTEREST
from ..data.queries import latest_analytics_leaderboard, recent_news_articles
from ..recommend.engine import RuleEngine
from ..recommend.options_picker import pick_affordable_contracts
from ..features.analytics import support_resistance_from_distribution, forecast_prices
from ..features.earnings_temporal import build_earnings_temporal_macro, build_earnings_temporal_profile
from ..features.signals import generate_signals
from ..features.news import build_news_summary, build_news_timeline
from ..features.training import build_training_frame
from ..features.indicators import fetch_ohlcv, compute_indicators
from ..features.seasonality import seasonality_by_doy
from ..backtest.signal_backtest import backtest_short_term
from ..ops.reporting import default_json_report_path, read_json_report


app = FastAPI(title="QuantFlow API", version="0.1.0")


def _safe_records(df: "pd.DataFrame") -> list[dict]:
    """Convert a DataFrame to JSON-safe records (replacing NaN/Infinity with None)."""
    clean = df.replace([np.inf, -np.inf], np.nan).astype(object).where(pd.notnull(df.replace([np.inf, -np.inf], np.nan)), None)
    return clean.to_dict(orient="records")


_SCAN_JOBS: dict[str, dict] = {}
_SCAN_JOBS_LOCK = Lock()

_SECTOR_TO_ETF = {
    "technology": "XLK",
    "healthcare": "XLV",
    "financial": "XLF",
    "financial services": "XLF",
    "industrials": "XLI",
    "consumer cyclical": "XLY",
    "consumer defensive": "XLP",
    "energy": "XLE",
    "real estate": "XLRE",
    "utilities": "XLU",
    "communication services": "XLC",
    "basic materials": "XLB",
    "materials": "XLB",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _scan_job_snapshot(job: dict) -> dict:
    total = max(1, int(job.get("preset_count", 0) or 0))
    done = max(0, int(job.get("presets_completed", 0) or 0))
    progress_pct = round((done / total) * 100.0, 2) if total else 0.0

    started_at = job.get("started_at")
    finished_at = job.get("finished_at")
    elapsed_seconds = None
    eta_seconds = None

    try:
        if started_at:
            start_dt = datetime.fromisoformat(str(started_at))
            end_dt = datetime.fromisoformat(str(finished_at)) if finished_at else datetime.now(timezone.utc)
            elapsed_seconds = max(0.0, (end_dt - start_dt).total_seconds())
            if done > 0 and done < total and job.get("status") in {"queued", "running", "cancel_requested"}:
                avg_seconds = elapsed_seconds / done
                eta_seconds = max(0.0, avg_seconds * (total - done))
    except Exception:
        elapsed_seconds = None
        eta_seconds = None

    return {
        "job_id": job["job_id"],
        "status": job.get("status"),
        "cancel_requested": bool(job.get("cancel_requested")),
        "created_at": job.get("created_at"),
        "started_at": started_at,
        "finished_at": finished_at,
        "updated_at": job.get("updated_at"),
        "selected_presets": list(job.get("selected_presets") or []),
        "preset_count": int(job.get("preset_count", 0) or 0),
        "presets_completed": done,
        "rows_saved": int(job.get("rows_saved", 0) or 0),
        "failed_presets": list(job.get("failed_presets") or []),
        "current_preset": job.get("current_preset"),
        "message": job.get("message"),
        "last_error": job.get("last_error"),
        "db": job.get("db"),
        "parallel": bool(job.get("parallel")),
        "max_workers": int(job.get("max_workers", 1) or 1),
        "progress_pct": progress_pct,
        "elapsed_seconds": elapsed_seconds,
        "eta_seconds": eta_seconds,
    }


def _infer_sector(ticker: str) -> str | None:
    try:
        info = yf.Ticker(ticker).info or {}
        sector = info.get("sector")
        if isinstance(sector, str) and sector.strip():
            return sector.strip()
    except Exception:
        pass
    return None


def _sector_to_etf(sector: str | None) -> str:
    if not sector:
        return "SPY"
    return _SECTOR_TO_ETF.get(str(sector).strip().lower(), "SPY")


def _seasonality_alignment_metrics(stock_df: pd.DataFrame, sector_df: pd.DataFrame) -> dict:
    merged = stock_df.merge(sector_df, on="doy", suffixes=("_stock", "_sector"))
    if merged.empty:
        return {"corr": None, "mean_abs_diff": None, "follow_rate": None}

    stock = merged["avg_ret_stock"].fillna(0.0)
    sector = merged["avg_ret_sector"].fillna(0.0)
    corr = float(stock.corr(sector)) if len(merged) > 2 else None
    mad = float((stock - sector).abs().mean())
    same_sign = ((np.sign(stock) == np.sign(sector)).astype(float)).mean()
    return {
        "corr": corr,
        "mean_abs_diff": mad,
        "follow_rate": float(same_sign),
    }


def _run_scan_job(job_id: str):
    with _SCAN_JOBS_LOCK:
        job = _SCAN_JOBS.get(job_id)
        if not job:
            return
        job["status"] = "running"
        job["started_at"] = _utc_now_iso()
        job["updated_at"] = job["started_at"]
        job["message"] = "Scan job started"

    try:
        create_schema(job.get("db") or "sqlite:///quantflow.db")
        names = list(job.get("selected_presets") or [])

        for preset_name in names:
            with _SCAN_JOBS_LOCK:
                current = _SCAN_JOBS.get(job_id)
                if not current:
                    return
                if current.get("cancel_requested"):
                    current["status"] = "canceled"
                    current["message"] = "Scan canceled"
                    current["finished_at"] = _utc_now_iso()
                    current["updated_at"] = current["finished_at"]
                    return
                current["current_preset"] = preset_name
                current["message"] = f"Running preset: {preset_name}"
                current["updated_at"] = _utc_now_iso()

            try:
                result_q: mp.Queue = mp.Queue(maxsize=1)
                db_path = job.get("db") or "sqlite:///quantflow.db"
                proc = mp.Process(target=_scan_preset_worker, args=(preset_name, db_path, result_q), daemon=True)
                proc.start()

                canceled = False
                while proc.is_alive():
                    with _SCAN_JOBS_LOCK:
                        current = _SCAN_JOBS.get(job_id)
                        if not current:
                            proc.terminate()
                            proc.join(timeout=3)
                            return
                        if current.get("cancel_requested"):
                            canceled = True
                    if canceled:
                        proc.terminate()
                        proc.join(timeout=3)
                        with _SCAN_JOBS_LOCK:
                            current = _SCAN_JOBS.get(job_id)
                            if not current:
                                return
                            current["status"] = "canceled"
                            current["message"] = "Scan canceled"
                            current["finished_at"] = _utc_now_iso()
                            current["updated_at"] = current["finished_at"]
                            current["current_preset"] = None
                        return
                    proc.join(timeout=0.3)

                if proc.exitcode not in (0, None):
                    raise RuntimeError(f"Preset worker failed with exit code {proc.exitcode}")

                try:
                    worker_payload = result_q.get_nowait()
                except Empty:
                    worker_payload = {"ok": False, "error": "No worker payload returned"}

                if not worker_payload.get("ok"):
                    raise RuntimeError(str(worker_payload.get("error") or "Unknown preset worker error"))

                row_count = int(worker_payload.get("rows_saved", 0) or 0)
                with _SCAN_JOBS_LOCK:
                    current = _SCAN_JOBS.get(job_id)
                    if not current:
                        return
                    current["rows_saved"] = int(current.get("rows_saved", 0)) + int(row_count)
                    current["presets_completed"] = int(current.get("presets_completed", 0)) + 1
                    current["message"] = f"Completed preset: {preset_name} ({row_count} rows)"
                    current["updated_at"] = _utc_now_iso()
            except Exception as e:
                with _SCAN_JOBS_LOCK:
                    current = _SCAN_JOBS.get(job_id)
                    if not current:
                        return
                    failures = list(current.get("failed_presets") or [])
                    failures.append(preset_name)
                    current["failed_presets"] = failures
                    current["presets_completed"] = int(current.get("presets_completed", 0)) + 1
                    current["last_error"] = str(e)
                    current["message"] = f"Failed preset: {preset_name}"
                    current["updated_at"] = _utc_now_iso()

        with _SCAN_JOBS_LOCK:
            current = _SCAN_JOBS.get(job_id)
            if not current:
                return
            if current.get("cancel_requested"):
                current["status"] = "canceled"
                current["message"] = "Scan canceled"
            else:
                current["status"] = "completed"
                current["message"] = "Scan completed"
            current["finished_at"] = _utc_now_iso()
            current["updated_at"] = current["finished_at"]
            current["current_preset"] = None
    except Exception as e:
        with _SCAN_JOBS_LOCK:
            current = _SCAN_JOBS.get(job_id)
            if not current:
                return
            current["status"] = "failed"
            current["last_error"] = str(e)
            current["message"] = "Scan job failed"
            current["finished_at"] = _utc_now_iso()
            current["updated_at"] = current["finished_at"]
            current["current_preset"] = None


def _scan_preset_worker(preset_name: str, db_path: str, result_q: mp.Queue):
    try:
        df = run_screener(preset_name)
        save_finviz_snapshot(df, preset_name, db_path=db_path)
        result_q.put({"ok": True, "rows_saved": len(df)})
    except Exception as e:
        result_q.put({"ok": False, "error": str(e)})

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict:
    return {
        "name": "QuantFlow API",
        "status": "ok",
        "health": "/health",
        "docs": "/docs",
    }


class ProposeOrderRequest(BaseModel):
    ticker: str = Field(min_length=1)
    action: str = Field(default="buy")
    qty: float = Field(default=1.0)
    notional: float = Field(gt=0)
    orders_today: int = Field(default=0, ge=0)
    position_notional_after: float = Field(default=0.0, ge=0)
    broker_mode: str = Field(default=BROKER_MODE_MCP)
    auto: bool = Field(default=False)
    max_daily_notional: float = Field(default=5000.0, gt=0)
    max_orders_per_day: int = Field(default=20, gt=0)
    max_position_notional: float = Field(default=2000.0, gt=0)
    db: str = Field(default="sqlite:///quantflow.db")


class RunScanRequest(BaseModel):
    db: str = Field(default="sqlite:///quantflow.db")
    parallel: bool = Field(default=True)
    max_workers: int = Field(default=4, ge=1, le=16)
    presets: Optional[list[str]] = None


class RunRecommendRequest(BaseModel):
    db: str = Field(default="sqlite:///quantflow.db")
    tickers: Optional[list[str]] = None
    save: bool = Field(default=True)


class RunAnalyticsRequest(BaseModel):
    db: str = Field(default="sqlite:///quantflow.db")
    tickers: list[str] = Field(default_factory=list)
    period: str = Field(default="2y")
    interval: str = Field(default="1d")
    limit: int = Field(default=8, ge=1, le=20)
    save: bool = Field(default=True)
    forecast_horizon: int = Field(default=20, ge=5, le=90)


class NewsArticleRequest(BaseModel):
    articles: list[dict] = Field(default_factory=list)


class TrainingFeaturesRequest(BaseModel):
    tickers: list[str] = Field(default_factory=list)
    period: str = Field(default="2y")
    interval: str = Field(default="1d")
    lookback_days: int = Field(default=30, ge=1, le=365)
    forecast_horizon: int = Field(default=20, ge=5, le=90)


class AssistantQueryRequest(BaseModel):
    message: str = Field(min_length=1)
    db: str = Field(default="sqlite:///quantflow.db")


class UserSettingsRequest(BaseModel):
    settings: dict = Field(default_factory=dict)


class MCPAuthConfigRequest(BaseModel):
    auth_header: str = Field(default="Authorization")
    bearer_token: Optional[str] = None
    api_key_header: str = Field(default="X-API-Key")
    api_key: Optional[str] = None
    cookie: Optional[str] = None
    custom_headers: dict[str, str] = Field(default_factory=dict)
    secret_provider: Optional[str] = None
    secret_refs: dict[str, str] = Field(default_factory=dict)
    provider_options: dict[str, str] = Field(default_factory=dict)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "auth_enabled": os.getenv("QF_API_AUTH_ENABLED", "false").lower() in {"1", "true", "yes"},
        "firebase_auth_enabled": os.getenv("QF_FIREBASE_AUTH_ENABLED", "false").lower() in {"1", "true", "yes"},
    }


@app.get("/ops/latest-report")
def ops_latest_report(report_out: str = "docs/LOCAL_SYSTEM_REPORT.md"):
    try:
        return read_json_report(default_json_report_path(report_out))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Ops report JSON not found. Run ops-report or ops-refresh with report output first.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/auth/me")
def auth_me(auth_payload: dict = Depends(require_write_auth)):
    return {
        "authenticated": True,
        "mode": auth_payload.get("mode", "unknown"),
        "claims": auth_payload.get("claims", {}),
    }


@app.get("/scanner/presets")
def scanner_presets():
    names = list(PRESETS.keys())
    return {"count": len(names), "items": names}


@app.get("/scanner/presets/details")
def scanner_presets_details():
    names = sorted(list(PRESETS.keys()))
    weekly = [n for n in names if "weekly" in n]
    monthly = [n for n in names if "monthly" in n]
    midterm = [n for n in names if "midterm" in n]
    reversal = [n for n in names if "reversal" in n]
    leaps = [n for n in names if "leaps" in n]
    return {
        "count": len(names),
        "items": names,
        "categories": {
            "weekly": weekly,
            "monthly": monthly,
            "midterm": midterm,
            "reversal": reversal,
            "leaps": leaps,
        },
        "profiles": PRESETS,
    }


@app.post("/scanner/presets/update")
def scanner_presets_update(payload: dict, _auth: dict = Depends(require_write_auth)):
    """Update or create a preset in the YAML config file."""
    try:
        name = str(payload.get("name", "")).strip()
        filters = payload.get("filters", {})
        
        if not name or not isinstance(filters, dict):
            raise HTTPException(status_code=400, detail="Invalid preset name or filters")
        
        # Import yaml here for safety
        import yaml
        import os
        
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "configs", "finviz_presets.yml"
        )
        
        # Load current presets
        try:
            with open(config_path, "r") as f:
                data = yaml.safe_load(f) or {}
        except Exception:
            data = {}
        
        # Update the specific preset
        data[name] = {str(k): str(v) for k, v in filters.items()}
        
        # Write back
        try:
            with open(config_path, "w") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save preset: {e}")
        
        # Update in-memory PRESETS dict
        PRESETS[name] = {str(k): str(v) for k, v in filters.items()}
        
        return {
            "ok": True,
            "message": f"Preset '{name}' saved successfully",
            "name": name,
            "filters": PRESETS[name],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/scanner/scraping/status")
def scanner_scraping_status():
    cfg = get_scraping_runtime_config()
    return {
        "provider": cfg.provider,
        "mode": cfg.mode,
        "proxy_configured": cfg.proxy_configured,
        "proxy_env_source": cfg.proxy_env_source,
        "api_endpoint": cfg.api_endpoint,
        "scraping_api_key_set": cfg.scraping_api_key_set,
        "scrapingbee_api_key_set": cfg.scrapingbee_api_key_set,
        "notes": cfg.notes,
    }


@app.get("/ops/env/validate")
def ops_env_validate():
    cfg = get_scraping_runtime_config()
    provider = cfg.provider

    missing: list[str] = []
    warnings: list[str] = []

    if cfg.mode == "direct":
        warnings.append("Scraping mode is direct. Configure proxy or API provider for resilient Finviz access.")

    if provider in {"smartproxy", "proxy"} and not cfg.proxy_configured:
        missing.append("QF_SCRAPING_PROXY_URL (or compatibility proxy alias)")

    if provider in {"scrape_do", "scrapedo"} and not cfg.proxy_configured and not cfg.scraping_api_key_set:
        missing.append("SCRAPING_API_KEY")

    if provider == "scrapingbee" and not cfg.proxy_configured and not cfg.scrapingbee_api_key_set:
        missing.append("SCRAPINGBEE_API_KEY")

    if os.getenv("QF_API_AUTH_ENABLED", "false").lower() in {"1", "true", "yes"}:
        firebase_enabled = os.getenv("QF_FIREBASE_AUTH_ENABLED", "false").lower() in {"1", "true", "yes"}
        token_set = bool(os.getenv("QF_API_TOKEN", "").strip())
        if not firebase_enabled and not token_set:
            missing.append("QF_API_TOKEN or QF_FIREBASE_AUTH_ENABLED=true")

    if os.getenv("QF_FIREBASE_AUTH_ENABLED", "false").lower() in {"1", "true", "yes"}:
        if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip():
            warnings.append("GOOGLE_APPLICATION_CREDENTIALS is not set; default app credentials must be available in runtime.")

    return {
        "ok": len(missing) == 0,
        "provider": provider,
        "scraping_mode": cfg.mode,
        "missing": missing,
        "warnings": warnings,
        "notes": cfg.notes,
        "auth": {
            "api_auth_enabled": os.getenv("QF_API_AUTH_ENABLED", "false").lower() in {"1", "true", "yes"},
            "firebase_auth_enabled": os.getenv("QF_FIREBASE_AUTH_ENABLED", "false").lower() in {"1", "true", "yes"},
            "api_token_set": bool(os.getenv("QF_API_TOKEN", "").strip()),
        },
    }


@app.get("/user/settings")
def user_settings_get(auth_payload: dict = Depends(require_firebase_user)):
    uid = str(auth_payload.get("uid", "")).strip()
    try:
        payload = load_user_settings(uid)
        return {
            "ok": True,
            "uid": payload.get("uid"),
            "settings": payload.get("settings", {}),
            "created_at": payload.get("created_at"),
            "updated_at": payload.get("updated_at"),
        }
    except UserSettingsStoreError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.put("/user/settings")
def user_settings_put(req: UserSettingsRequest, auth_payload: dict = Depends(require_firebase_user)):
    uid = str(auth_payload.get("uid", "")).strip()
    if not isinstance(req.settings, dict):
        raise HTTPException(status_code=400, detail="settings must be a JSON object")
    if len(str(req.settings)) > 100_000:
        raise HTTPException(status_code=400, detail="settings payload too large")

    try:
        payload = save_user_settings(uid, req.settings)
        return {
            "ok": True,
            "uid": payload.get("uid"),
            "settings": payload.get("settings", {}),
            "created_at": payload.get("created_at"),
            "updated_at": payload.get("updated_at"),
        }
    except UserSettingsStoreError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/scanner/latest-universe")
def scanner_latest_universe(db: str = "sqlite:///quantflow.db"):
    try:
        df = latest_universe(db_path=db)
        if df.empty:
            return {"count": 0, "items": []}
        return {"count": len(df), "items": _safe_records(df)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/recommend/latest")
def recommend_latest(db: str = "sqlite:///quantflow.db"):
    try:
        df = latest_recommendations_per_ticker(db_path=db)
        if df.empty:
            return {"count": 0, "items": []}
        return {"count": len(df), "items": _safe_records(df)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/recommend/analyze")
def recommend_analyze(
    tickers: str,
    period: str = "2y",
    interval: str = "1d",
    limit: int = 8,
):
    try:
        names = [t.strip().upper() for t in tickers.split(",") if t.strip()]
        if not names:
            raise ValueError("Provide at least one ticker")
        eng = RuleEngine()
        ranked = eng.rank_tickers(names[: max(1, min(limit, 20))], period=period, interval=interval)
        return {"count": len(ranked), "items": ranked}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/analytics/leaderboard")
def analytics_leaderboard(req: RunAnalyticsRequest, _auth: dict = Depends(require_write_auth)):
    try:
        names = [t.strip().upper() for t in (req.tickers or []) if str(t).strip()]
        if not names:
            raise ValueError("Provide at least one ticker")
        create_schema(req.db)
        eng = RuleEngine()
        ranked = eng.rank_tickers(names[: req.limit], period=req.period, interval=req.interval)
        batch_id = None
        if req.save and ranked:
            batch_id = save_analytics_leaderboard(ranked, db_path=req.db, period=req.period, interval=req.interval)
        return {
            "count": len(ranked),
            "saved": bool(req.save and ranked),
            "batch_id": batch_id,
            "items": ranked,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/analytics/leaderboard/latest")
def analytics_leaderboard_latest(db: str = "sqlite:///quantflow.db"):
    try:
        df = latest_analytics_leaderboard(db_path=db)
        if df.empty:
            return {"count": 0, "items": []}
        return {
            "count": len(df),
            "batch_id": df.iloc[0]["batch_id"],
            "items": _safe_records(df),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/analytics/training-features")
def analytics_training_features(
    tickers: str,
    period: str = "2y",
    interval: str = "1d",
    forecast_horizon: int = 20,
    limit: int = 12,
):
    try:
        names = [t.strip().upper() for t in tickers.split(",") if t.strip()]
        if not names:
            raise ValueError("Provide at least one ticker")
        eng = RuleEngine()
        rows = [eng.training_features(ticker, period=period, interval=interval, forecast_horizon=forecast_horizon) for ticker in names[:limit]]
        rows.sort(key=lambda row: row.get("composite_score", 0.0), reverse=True)
        return {"count": len(rows), "items": rows}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/signals/generate")
def signals_generate(
    ticker: str,
    timeframe: str = "5Min",
    lookback_bars: int = 100,
    detectors: str = "breakout,rsi,macd,pullback",
    rsi_oversold: float = 30,
    rsi_overbought: float = 70,
    breakout_window: int = 20,
    ema_span: int = 21,
    trend_span: int = 50,
):
    """Generate live trading signals for a ticker.

    Uses Alpaca real-time bars if ALPACA_API_KEY + ALPACA_SECRET_KEY are set,
    otherwise falls back to yfinance delayed data.
    """
    try:
        detector_list = [d.strip() for d in detectors.split(",") if d.strip()]
        result = generate_signals(
            ticker=ticker.strip().upper(),
            timeframe=timeframe,
            lookback_bars=lookback_bars,
            detectors=detector_list,
            rsi_oversold=rsi_oversold,
            rsi_overbought=rsi_overbought,
            breakout_window=breakout_window,
            ema_span=ema_span,
            trend_span=trend_span,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/analytics/earnings-temporal")
def analytics_earnings_temporal(
    ticker: str,
    years: int = 8,
    sentiment_window_days: int = 3,
    db: str = "sqlite:///quantflow.db",
):
    try:
        create_schema(db)
        payload = build_earnings_temporal_profile(
            ticker=ticker.upper(),
            years=years,
            sentiment_window_days=sentiment_window_days,
            db_path=db,
        )
        return payload
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/analytics/earnings-temporal/macro")
def analytics_earnings_temporal_macro(
    tickers: str | None = None,
    years: int = 8,
    sentiment_window_days: int = 3,
    db: str = "sqlite:///quantflow.db",
    limit: int = 16,
):
    try:
        create_schema(db)
        names = [t.strip().upper() for t in str(tickers or "").split(",") if t.strip()]
        if not names:
            ranked = latest_analytics_leaderboard(db_path=db)
            if not ranked.empty:
                names = [str(t).upper() for t in ranked["ticker"].head(max(4, min(limit, 32))).tolist()]
        if not names:
            uni = latest_universe(db_path=db)
            if not uni.empty:
                names = [str(t).upper() for t in uni["ticker"].head(max(4, min(limit, 32))).tolist()]

        payload = build_earnings_temporal_macro(
            tickers=names[: max(4, min(limit, 32))],
            years=years,
            sentiment_window_days=sentiment_window_days,
            db_path=db,
        )
        return payload
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/news/articles")
def news_articles(
    db: str = "sqlite:///quantflow.db",
    ticker: str | None = None,
    days: int = 30,
    limit: int = 100,
):
    try:
        create_schema(db)
        df = recent_news_articles(days=days, db_path=db, ticker=ticker.upper() if ticker else None, limit=limit)
        if df.empty:
            return {"count": 0, "items": []}
        return {"count": len(df), "items": _safe_records(df)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/news/summary")
def news_summary(
    db: str = "sqlite:///quantflow.db",
    ticker: str | None = None,
    days: int = 30,
    sentiment: str = "all",
    max_groups: int = 10,
):
    try:
        create_schema(db)
        payload = build_news_summary(
            ticker=ticker.upper() if ticker else None,
            days=days,
            db_path=db,
            sentiment=sentiment,
            max_groups=max_groups,
        )
        return payload
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/news/timeline")
def news_timeline(
    ticker: str,
    db: str = "sqlite:///quantflow.db",
    days: int = 30,
    period: str = "2y",
    interval: str = "1d",
    sentiment: str = "all",
    max_articles: int = 40,
):
    try:
        create_schema(db)
        payload = build_news_timeline(
            ticker=ticker.upper(),
            db_path=db,
            days=days,
            period=period,
            interval=interval,
            sentiment=sentiment,
            max_articles=max_articles,
        )
        return payload
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/news/articles")
def news_articles_post(req: NewsArticleRequest, _auth: dict = Depends(require_write_auth)):
    try:
        create_schema()
        saved = save_news_articles(req.articles, db_path="sqlite:///quantflow.db")
        return {"saved": saved}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/training/features")
def training_features_post(req: TrainingFeaturesRequest, _auth: dict = Depends(require_write_auth)):
    try:
        names = [t.strip().upper() for t in req.tickers if str(t).strip()]
        if not names:
            raise ValueError("Provide at least one ticker")
        frame = build_training_frame(
            tickers=names,
            db_path="sqlite:///quantflow.db",
            period=req.period,
            interval=req.interval,
            lookback_days=req.lookback_days,
            forecast_horizon=req.forecast_horizon,
        )
        return {"count": len(frame), "items": _safe_records(frame)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/training/features")
def training_features_get(
    tickers: str,
    period: str = "2y",
    interval: str = "1d",
    lookback_days: int = 30,
    forecast_horizon: int = 20,
    db: str = "sqlite:///quantflow.db",
):
    try:
        names = [t.strip().upper() for t in tickers.split(",") if t.strip()]
        if not names:
            raise ValueError("Provide at least one ticker")
        create_schema(db)
        frame = build_training_frame(
            tickers=names,
            db_path=db,
            period=period,
            interval=interval,
            lookback_days=lookback_days,
            forecast_horizon=forecast_horizon,
        )
        return {"count": len(frame), "items": _safe_records(frame)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/portfolio/signals")
def portfolio_signals(
    broker_mode: str = BROKER_MODE_MCP,
):
    try:
        broker = make_broker(broker_mode)
        sigs = evaluate_positions(broker)
        return {
            "broker_mode": broker_mode,
            "count": len(sigs),
            "signals": [s.__dict__ for s in sigs],
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/broker/mcp/status")
def broker_mcp_status():
    status = {
        "broker_mode": BROKER_MODE_MCP,
        "connected": False,
        "authenticated": False,
        "account_available": False,
        "positions_available": False,
        "positions_count": 0,
        "fixture_account_loaded": bool(os.getenv("QF_MCP_ACCOUNT_JSON", "").strip()),
        "fixture_positions_loaded": bool(os.getenv("QF_MCP_POSITIONS_JSON", "").strip()),
        "runtime_auth_override": get_runtime_mcp_auth_config_summary(),
    }

    try:
        broker = make_broker(BROKER_MODE_MCP)
        if hasattr(broker, "client") and getattr(broker, "client") is not None:
            status["endpoint"] = getattr(getattr(broker, "client"), "config", {}).endpoint if hasattr(getattr(broker, "client"), "config") else None
            if hasattr(getattr(broker, "client"), "auth_summary"):
                status["auth"] = getattr(broker, "client").auth_summary()
    except Exception as e:
        status["error"] = str(e)
        return status

    try:
        broker.login()
        status["connected"] = True
    except Exception as e:
        status["connect_error"] = str(e)

    try:
        acct = broker.account()
        status["authenticated"] = True
        status["account_available"] = True
        status["account"] = {
            "equity": acct.equity,
            "cash": acct.cash,
            "buying_power": acct.buying_power,
        }
    except Exception as e:
        status["account_error"] = str(e)

    try:
        rows = broker.positions()
        status["positions_available"] = True
        status["positions_count"] = len(rows)
    except Exception as e:
        status["positions_error"] = str(e)

    return status


def _mcp_client_login_steps() -> list[dict]:
    endpoint = "https://agent.robinhood.com/mcp/trading"
    return [
        {"client": "Claude Code", "steps": [f"Run: claude mcp add robinhood-trading --transport http {endpoint}", "Run /mcp, select robinhood-trading, then complete Robinhood auth."]},
        {"client": "Claude Desktop", "steps": ["Open Settings -> Connectors -> Add custom connector.", f"Add MCP URL: {endpoint} and complete Robinhood auth."]},
        {"client": "ChatGPT", "steps": ["Enable Developer Mode.", "Open Settings -> Apps -> Create app.", f"Add MCP URL: {endpoint} and complete Robinhood auth."]},
        {"client": "Codex", "steps": ["Open Settings -> MCP servers -> Streamable HTTP.", f"Add MCP URL: {endpoint} and complete Robinhood auth."]},
        {"client": "Codex CLI", "steps": [f"Run: codex mcp add robinhood-trading --url {endpoint}", "Run /mcp, select robinhood-trading, then complete Robinhood auth."]},
        {"client": "Cursor", "steps": [f"Provide MCP URL to the agent: {endpoint}", "Open Settings -> Cursor Settings -> Tools & MCPs -> Connect and complete auth."]},
    ]

@app.post("/broker/mcp/login")
def broker_mcp_login():
    status = broker_mcp_status()
    readiness = broker_mcp_readiness()

    if status.get("authenticated"):
        message = "Robinhood MCP is authenticated and account data is available."
        ok = True
    elif status.get("connected"):
        message = "Robinhood MCP transport is reachable, but account auth is still required in your MCP client session."
        ok = False
    else:
        message = "Robinhood MCP is not connected yet. Check MCP endpoint/config and retry."
        ok = False

    return {
        "ok": ok,
        "message": message,
        "auth_instructions": _mcp_client_login_steps(),
        "status": status,
        "readiness": readiness,
    }


@app.get("/broker/mcp/config")
def broker_mcp_config(_auth: dict = Depends(require_write_auth)):
    summary = get_runtime_mcp_auth_config_summary()
    return {
        "configured": bool(summary.get("configured")),
        "runtime": summary,
        "message": "Runtime MCP auth override summary (secrets are never returned).",
    }


@app.put("/broker/mcp/config")
def broker_mcp_config_update(req: MCPAuthConfigRequest, _auth: dict = Depends(require_write_auth)):
    payload = {
        "auth_header": req.auth_header,
        "bearer_token": req.bearer_token or "",
        "api_key_header": req.api_key_header,
        "api_key": req.api_key or "",
        "cookie": req.cookie or "",
        "custom_headers": req.custom_headers or {},
        "secret_provider": (req.secret_provider or "").strip().lower(),
        "secret_refs": req.secret_refs or {},
        "provider_options": req.provider_options or {},
    }
    set_runtime_mcp_auth_config(payload)
    summary = get_runtime_mcp_auth_config_summary()
    return {
        "ok": True,
        "message": "Runtime MCP auth override updated.",
        "runtime": summary,
    }


@app.delete("/broker/mcp/config")
def broker_mcp_config_clear(_auth: dict = Depends(require_write_auth)):
    clear_runtime_mcp_auth_config()
    return {
        "ok": True,
        "message": "Runtime MCP auth override cleared.",
        "runtime": get_runtime_mcp_auth_config_summary(),
    }


@app.get("/broker/mcp/readiness")
def broker_mcp_readiness():
    status = broker_mcp_status()

    fixture_account = bool(status.get("fixture_account_loaded"))
    fixture_positions = bool(status.get("fixture_positions_loaded"))
    has_fixtures = fixture_account and fixture_positions
    connected = bool(status.get("connected"))
    authenticated = bool(status.get("authenticated"))
    auth_configured = bool((status.get("auth") or {}).get("configured"))
    runtime_cfg = status.get("runtime_auth_override") or {}
    runtime_active = bool(runtime_cfg.get("configured"))
    runtime_provider = runtime_cfg.get("secret_provider")

    checks = [
        {
            "id": "auth_config",
            "label": "Backend MCP auth wiring",
            "state": "pass" if auth_configured else "warn",
            "details": "Backend has MCP auth headers/token configured."
            if auth_configured
            else "No MCP auth token/header configured in backend env. Set QF_MCP_BEARER_TOKEN or QF_MCP_HEADERS_JSON.",
        },
        {
            "id": "runtime_auth",
            "label": "Runtime secure MCP config",
            "state": "pass" if runtime_active else "warn",
            "details": "Runtime MCP auth override is configured via backend API."
            if runtime_active
            else "No runtime MCP auth override is configured. Use PUT /broker/mcp/config for non-env secret wiring.",
        },
        {
            "id": "secret_provider",
            "label": "Secret manager provider",
            "state": "pass" if runtime_provider else "warn",
            "details": f"Runtime secret provider configured: {runtime_provider}."
            if runtime_provider
            else "No secret provider configured (vault/aws/gcp/azure).",
        },
        {
            "id": "transport",
            "label": "MCP transport reachable",
            "state": "pass" if connected else "fail",
            "details": "QuantFlow can reach Robinhood MCP transport endpoint." if connected else "Transport not reachable from current runtime.",
        },
        {
            "id": "auth",
            "label": "Agent account authentication",
            "state": "pass" if authenticated else "fail",
            "details": "Agent account auth is active and account payload is readable."
            if authenticated
            else "Authentication not complete yet (typically 401 until the agent account auth flow is completed).",
        },
        {
            "id": "fixtures",
            "label": "Local fixture fallback",
            "state": "pass" if has_fixtures else "warn",
            "details": "Fixture env vars are set for account and positions."
            if has_fixtures
            else "Optional local fixture env vars are not fully set (QF_MCP_ACCOUNT_JSON and QF_MCP_POSITIONS_JSON).",
        },
        {
            "id": "positions",
            "label": "Positions visibility",
            "state": "pass" if bool(status.get("positions_available")) else "warn",
            "details": "Positions payload is readable from broker adapter."
            if bool(status.get("positions_available"))
            else "Positions are not readable yet. This is expected when auth is incomplete.",
        },
    ]

    next_steps = [
        "Configure MCP auth via PUT /broker/mcp/config (preferred for runtime) or env vars as fallback.",
        "For production, point secret_refs to Vault/AWS/GCP/Azure and avoid raw token payloads.",
        "Authenticate the Robinhood Agentic account in your MCP-capable client session (see /broker/mcp/login auth_instructions).",
        "Re-check MCP status from Overview to confirm authenticated=yes.",
        "Optional: set QF_MCP_ACCOUNT_JSON and QF_MCP_POSITIONS_JSON for local fixture-based testing.",
        "Once authenticated, validate account and positions read before enabling execution pathways.",
    ]

    return {
        "overall_ready": connected and (authenticated or has_fixtures),
        "checks": checks,
        "next_steps": next_steps,
        "status": status,
    }


@app.post("/assistant/query")
def assistant_query(req: AssistantQueryRequest):
    text = req.message.strip()
    low = text.lower()
    normalized = re.sub(r"[^a-z0-9\s]+", " ", low)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    try:
        recs = latest_recommendations_per_ticker(db_path=req.db)
        intents_df = recent_execution_intents(limit=10, db_path=req.db)
        uni = latest_universe(db_path=req.db)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    summary = {
        "recommendations": 0 if recs.empty else len(recs),
        "recent_intents": 0 if intents_df.empty else len(intents_df),
        "latest_universe": 0 if uni.empty else len(uni),
        "presets": len(PRESETS),
    }

    if re.fullmatch(r"[\W_]+", text or "") or normalized in {
        "hi",
        "hello",
        "hey",
        "sup",
        "whats up",
        "what s up",
        "yo",
        "how are you",
        "how r you",
        "thanks",
        "thank you",
        "ok",
        "okay",
        "cool",
    }:
        return {
            "answer": (
                "Ready when you are. Ask for high-conviction rankings, earnings temporal analysis, "
                "scanner status, MCP readiness, or a technical/seasonality/sentiment workflow for a ticker."
            ),
            "summary": summary,
            "suggested_tool_calls": [],
        }

    capability_tokens = {
        "what else can you do",
        "what can you do",
        "what do you do",
        "help",
        "capabilities",
        "what else",
        "what else can you say",
        "what can you say",
        "say more",
    }
    is_capability_prompt = (
        normalized in capability_tokens
        or normalized.startswith("what else")
        or ("what else" in normalized and ("do" in normalized or "say" in normalized))
    )

    if is_capability_prompt:
        return {
            "answer": (
                "I can work with local QuantFlow data for rankings, scanner and universe status, earnings temporal analysis, "
                "technical and seasonality workflows, sentiment timelines, execution intents, and MCP readiness. "
                "Try: top high conviction ideas, technical plus sentiment on AAPL, scanner status, or MCP readiness."
            ),
            "summary": summary,
            "suggested_tool_calls": [],
        }

    actions = [
        {"name": "refresh_overview", "method": "GET", "path": "/health"},
        {"name": "run_scan", "method": "POST", "path": "/pipeline/scan", "payload": {"parallel": True, "max_workers": 4}},
        {"name": "run_recommend", "method": "POST", "path": "/pipeline/recommend", "payload": {"tickers": ["AAPL"], "save": True}},
        {"name": "check_mcp", "method": "GET", "path": "/broker/mcp/status"},
        {"name": "connect_mcp", "method": "POST", "path": "/broker/mcp/login", "payload": {}},
    ]

    def _top_opportunities(limit: int = 5) -> list[dict]:
        if recs.empty:
            return []

        rows = []
        for _, row in recs.iterrows():
            ticker = str(row.get("ticker") or "").strip().upper()
            bias = str(row.get("bias") or row.get("action") or "neutral").strip().lower()
            horizon = str(row.get("horizon") or "")
            confidence = float(row.get("confidence") or row.get("signal") or 0.0)
            entry = float(row.get("entry") or 0.0)
            target = float(row.get("target") or 0.0)
            stop = float(row.get("stop") or 0.0)

            edge_pct = 0.0
            risk_pct = 0.0
            if entry > 0:
                if bias == "short":
                    edge_pct = (entry - target) / entry if target > 0 else 0.0
                    risk_pct = (stop - entry) / entry if stop > 0 else 0.0
                else:
                    edge_pct = (target - entry) / entry if target > 0 else 0.0
                    risk_pct = (entry - stop) / entry if stop > 0 else 0.0

            conviction = max(0.0, confidence) * 0.7 + max(-1.0, min(1.0, edge_pct)) * 0.3
            rows.append({
                "ticker": ticker,
                "bias": bias,
                "horizon": horizon,
                "confidence": round(confidence, 4),
                "entry": entry,
                "target": target,
                "stop": stop,
                "edge_pct": round(edge_pct, 4),
                "risk_pct": round(risk_pct, 4),
                "conviction": round(conviction, 4),
            })

        rows = [r for r in rows if r["ticker"] and r["bias"] != "neutral"]
        rows.sort(key=lambda r: (r["conviction"], r["confidence"], r["edge_pct"]), reverse=True)

        best_by_ticker = {}
        for row in rows:
            if row["ticker"] not in best_by_ticker:
                best_by_ticker[row["ticker"]] = row
            if len(best_by_ticker) >= max(1, limit):
                break
        return list(best_by_ticker.values())

    if any(k in low for k in ["weekly", "week"]) and any(k in low for k in ["scan", "scanner", "universe"]):
        weekly_presets = [name for name in PRESETS.keys() if "weekly" in name]
        answer = (
            "For a weekly universe pass, run weekly presets first, then inspect refreshed universe rows and recommendations. "
            "I prepared tool calls for that workflow."
        )
        weekly_actions = [
            {
                "name": "run_scan_weekly",
                "method": "POST",
                "path": "/pipeline/scan",
                "payload": {
                    "parallel": True,
                    "max_workers": 4,
                    "presets": weekly_presets,
                },
            },
            {"name": "load_latest_universe", "method": "GET", "path": "/scanner/latest-universe"},
            {
                "name": "run_recommend_weekly",
                "method": "POST",
                "path": "/pipeline/recommend",
                "payload": {"save": True},
            },
        ]
        return {
            "answer": answer,
            "summary": summary,
            "weekly_presets": weekly_presets,
            "suggested_tool_calls": weekly_actions,
        }

    if any(k in low for k in ["mcp", "robinhood", "broker"]):
        mcp = broker_mcp_status()
        if mcp.get("authenticated"):
            answer = "Robinhood MCP appears authenticated and account data is available."
        elif mcp.get("connected"):
            answer = "Robinhood MCP transport is reachable, but account auth/data is not yet available."
        else:
            answer = "Robinhood MCP is not connected yet. Use fixture env vars or platform auth flow, then retry."
        mcp_actions = [
            {"name": "connect_robinhood", "method": "POST", "path": "/broker/mcp/login", "payload": {}},
            {"name": "check_mcp", "method": "GET", "path": "/broker/mcp/status"},
            {"name": "check_mcp_readiness", "method": "GET", "path": "/broker/mcp/readiness"},
            {"name": "portfolio_signals", "method": "GET", "path": "/portfolio/signals?broker_mode=robinhood_mcp"},
        ]
        return {"answer": answer, "summary": summary, "mcp_status": mcp, "suggested_tool_calls": mcp_actions}

    if any(k in low for k in ["earnings", "quarter", "quarterly", "operations report", "ops report"]):
        focus = ""
        match = re.search(r"([A-Z]{1,5})", req.message)
        if match:
            focus = match.group(1).upper()
        if not focus:
            top = _top_opportunities(limit=1)
            focus = top[0]["ticker"] if top else "AAPL"

        answer = (
            f"I can run earnings timing analytics for {focus} and compare quarterly report cadence against archived sentiment. "
            "Use the suggested calls for per-ticker timing and macro aggregation across your active basket."
        )
        return {
            "answer": answer,
            "summary": summary,
            "suggested_tool_calls": [
                {"name": "earnings_temporal_ticker", "method": "GET", "path": f"/analytics/earnings-temporal?ticker={focus}&years=8&sentiment_window_days=3&db={req.db}"},
                {"name": "earnings_temporal_macro", "method": "GET", "path": f"/analytics/earnings-temporal/macro?years=8&sentiment_window_days=3&db={req.db}&limit=16"},
                {"name": "sentiment_timeline", "method": "GET", "path": f"/news/timeline?ticker={focus}&db={req.db}&days=120&period=2y&interval=1d&sentiment=all&max_articles=40"},
            ],
        }

    if any(k in low for k in ["high conviction", "best", "top", "opportunit", "rank"]):
        opps = _top_opportunities(limit=8)
        if not opps:
            return {
                "answer": "No ranked opportunities are available yet. Run scan + recommend first, then ask again.",
                "summary": summary,
                "opportunities": [],
                "suggested_tool_calls": [
                    {"name": "run_scan", "method": "POST", "path": "/pipeline/scan", "payload": {"parallel": True, "max_workers": 4}},
                    {"name": "run_recommend", "method": "POST", "path": "/pipeline/recommend", "payload": {"save": True}},
                    {"name": "load_recommendations", "method": "GET", "path": "/recommend/latest"},
                ],
            }

        top = opps[:5]
        top_tickers = [row["ticker"] for row in top]
        readable = ", ".join([f"{row['ticker']} ({row['bias']}, conf {row['confidence']:.2f}, edge {row['edge_pct']:.1%})" for row in top])
        answer = (
            f"Top high-conviction opportunities right now: {readable}. "
            "These are ranked from local recommendation data using confidence plus expected edge. "
            "Use the suggested calls to run deeper technical, seasonality, and sentiment analysis for this basket."
        )
        return {
            "answer": answer,
            "summary": summary,
            "opportunities": top,
            "suggested_tool_calls": [
                {"name": "refresh_recommendations", "method": "POST", "path": "/pipeline/recommend", "payload": {"tickers": top_tickers, "save": True}},
                {"name": "analyze_basket", "method": "GET", "path": f"/recommend/analyze?tickers={','.join(top_tickers)}&period=1y&interval=1d&limit={len(top_tickers)}"},
                {"name": "seasonality_primary", "method": "GET", "path": f"/charts/seasonality?ticker={top_tickers[0]}&years=10"},
                {"name": "sentiment_primary", "method": "GET", "path": f"/news/summary?ticker={top_tickers[0]}&db={req.db}&days=30&sentiment=all&max_groups=8"},
            ],
        }

    if any(k in low for k in ["technical", "seasonal", "seasonality", "temporal", "sentiment", "inference"]):
        selected = _top_opportunities(limit=3)
        focus = selected[0]["ticker"] if selected else "AAPL"
        answer = (
            f"I can run a fast multi-factor workflow for {focus}: technical indicators, seasonality/temporal profile, and sentiment timeline. "
            "Use the suggested calls to fetch those datasets and compare them in the Charts and Analytics views."
        )
        return {
            "answer": answer,
            "summary": summary,
            "focus_ticker": focus,
            "suggested_tool_calls": [
                {"name": "indicators", "method": "GET", "path": f"/charts/indicators?ticker={focus}&period=1y&interval=1d"},
                {"name": "seasonality", "method": "GET", "path": f"/charts/seasonality-compare?ticker={focus}&years=10"},
                {"name": "sentiment_summary", "method": "GET", "path": f"/news/summary?ticker={focus}&db={req.db}&days=30&sentiment=all&max_groups=8"},
                {"name": "sentiment_timeline", "method": "GET", "path": f"/news/timeline?ticker={focus}&db={req.db}&days=30&period=1y&interval=1d&sentiment=all&max_articles=24"},
            ],
        }

    if any(k in low for k in ["recommend", "signal", "idea"]):
        items = []
        if not recs.empty:
            top = recs.head(5)
            for _, row in top.iterrows():
                items.append({
                    "ticker": row.get("ticker"),
                    "horizon": row.get("horizon"),
                    "action": row.get("action"),
                    "signal": row.get("signal"),
                })
        answer = "Here are the latest offline recommendations from local persistence."
        return {"answer": answer, "summary": summary, "recommendations": items, "suggested_tool_calls": actions}

    if any(k in low for k in ["intent", "execution", "order"]):
        items = []
        if not intents_df.empty:
            for _, row in intents_df.head(5).iterrows():
                items.append({
                    "id": int(row.get("id")),
                    "ticker": row.get("ticker"),
                    "action": row.get("action"),
                    "status": row.get("status"),
                    "notional": row.get("notional"),
                })
        answer = "Here are the most recent execution intents from local persistence."
        return {"answer": answer, "summary": summary, "intents": items, "suggested_tool_calls": actions}

    if re.search(r"scan|universe|preset", low):
        answer = "Scanner data is available. You can run a fresh scan or inspect the latest cached universe rows."
        return {"answer": answer, "summary": summary, "suggested_tool_calls": []}

    # Only surface top ideas when the message contains a financial keyword.
    # Nonsense, conversational, or off-topic input gets a neutral guidance reply.
    _FINANCE_RE = re.compile(
        r"\b(ticker|stock|trade|trading|signal|rank|buy|sell|long|short|portfolio|"
        r"market|price|chart|analysis|analyse|analyze|forecast|recommend|setup|"
        r"alpha|backtest|option|put|call|spread|trend|momentum|indicator|rsi|macd|"
        r"earnings|revenue|eps|idea|ideas|opportunity|opportunities)\b"
    )
    if _FINANCE_RE.search(low):
        top = _top_opportunities(limit=3)
        if top:
            preview = ", ".join([f"{row['ticker']} ({row['bias']}, {row['confidence']:.2f})" for row in top])
            answer = (
                "From offline QuantFlow data, top current ideas are "
                f"{preview}. Ask for high-conviction ranking, MCP status, technical/seasonality/sentiment workflow, or execution intents."
            )
            return {"answer": answer, "summary": summary, "suggested_tool_calls": []}

    answer = (
        "I didn't catch a specific request. You can ask for: high-conviction rankings, "
        "technical/seasonality/sentiment analysis for a ticker, earnings temporal cadence, "
        "scanner or universe status, execution intents, or MCP readiness."
    )
    return {"answer": answer, "summary": summary, "suggested_tool_calls": []}


@app.post("/execution/propose")
def execution_propose(req: ProposeOrderRequest, _auth: dict = Depends(require_write_auth)):
    if req.broker_mode != BROKER_MODE_MCP:
        raise HTTPException(status_code=400, detail="Invalid broker_mode")

    create_schema(req.db)

    policy = ExecutionPolicy(
        require_approval=(not req.auto),
        max_daily_notional=req.max_daily_notional,
        max_orders_per_day=req.max_orders_per_day,
        max_position_notional=req.max_position_notional,
    )
    decision = evaluate_order_intent(
        notional=req.notional,
        orders_today=req.orders_today,
        position_notional_after=req.position_notional_after,
        policy=policy,
    )

    intent_id = audit_intent(
        ticker=req.ticker.upper(),
        action=req.action,
        qty=req.qty,
        notional=req.notional,
        broker_mode=req.broker_mode,
        status="approved" if decision.allow else "blocked",
        db_path=req.db,
    )
    audit_policy(intent_id=intent_id, allow=decision.allow, reason=decision.reason, db_path=req.db)
    audit_event(intent_id=intent_id, event_type="policy_check", message=decision.reason, db_path=req.db)

    return {
        "intent_id": intent_id,
        "allow": decision.allow,
        "reason": decision.reason,
    }


@app.get("/execution/intents")
def execution_intents(limit: int = 100, db: str = "sqlite:///quantflow.db"):
    try:
        limit = max(1, min(int(limit), 500))
        df = recent_execution_intents(limit=limit, db_path=db)
        if df.empty:
            return {"count": 0, "items": []}
        return {"count": len(df), "items": _safe_records(df)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/pipeline/scan/start")
def pipeline_scan_start(req: RunScanRequest, _auth: dict = Depends(require_write_auth)):
    try:
        names = [n for n in (req.presets or list(PRESETS.keys()))]
        if not names:
            raise HTTPException(status_code=400, detail="No presets provided")

        unknown = [n for n in names if n not in PRESETS]
        if unknown:
            raise HTTPException(status_code=400, detail=f"Unknown presets: {', '.join(unknown)}")

        job_id = str(uuid.uuid4())
        job = {
            "job_id": job_id,
            "status": "queued",
            "cancel_requested": False,
            "created_at": _utc_now_iso(),
            "started_at": None,
            "finished_at": None,
            "updated_at": _utc_now_iso(),
            "selected_presets": names,
            "preset_count": len(names),
            "presets_completed": 0,
            "rows_saved": 0,
            "failed_presets": [],
            "current_preset": None,
            "message": "Scan job queued",
            "last_error": None,
            "db": req.db,
            "parallel": req.parallel,
            "max_workers": req.max_workers,
        }
        with _SCAN_JOBS_LOCK:
            _SCAN_JOBS[job_id] = job

        worker = Thread(target=_run_scan_job, args=(job_id,), daemon=True)
        worker.start()

        return _scan_job_snapshot(job)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/pipeline/scan/status/{job_id}")
def pipeline_scan_status(job_id: str, _auth: dict = Depends(require_write_auth)):
    with _SCAN_JOBS_LOCK:
        job = _SCAN_JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Scan job not found")
        return _scan_job_snapshot(job)


@app.post("/pipeline/scan/cancel/{job_id}")
def pipeline_scan_cancel(job_id: str, _auth: dict = Depends(require_write_auth)):
    with _SCAN_JOBS_LOCK:
        job = _SCAN_JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Scan job not found")
        if job.get("status") in {"completed", "failed", "canceled"}:
            return _scan_job_snapshot(job)
        job["cancel_requested"] = True
        job["status"] = "cancel_requested"
        job["message"] = "Cancel requested"
        job["updated_at"] = _utc_now_iso()
        return _scan_job_snapshot(job)


@app.get("/pipeline/scan/history")
def pipeline_scan_history(limit: int = 20, status: str | None = None, _auth: dict = Depends(require_write_auth)):
    try:
        lim = max(1, min(int(limit), 100))
        with _SCAN_JOBS_LOCK:
            snapshots = [_scan_job_snapshot(job) for job in _SCAN_JOBS.values()]

        if status:
            status_norm = status.strip().lower()
            snapshots = [row for row in snapshots if str(row.get("status", "")).lower() == status_norm]

        snapshots.sort(key=lambda row: str(row.get("created_at") or ""), reverse=True)
        items = snapshots[:lim]

        counts = {
            "completed": sum(1 for row in items if row.get("status") == "completed"),
            "running": sum(1 for row in items if row.get("status") == "running"),
            "queued": sum(1 for row in items if row.get("status") == "queued"),
            "cancel_requested": sum(1 for row in items if row.get("status") == "cancel_requested"),
            "canceled": sum(1 for row in items if row.get("status") == "canceled"),
            "failed": sum(1 for row in items if row.get("status") == "failed"),
        }

        return {
            "count": len(items),
            "limit": lim,
            "status_filter": status,
            "counts": counts,
            "items": items,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/pipeline/scan")
def pipeline_scan(req: RunScanRequest, _auth: dict = Depends(require_write_auth)):
    try:
        create_schema(req.db)
        names = [n for n in (req.presets or list(PRESETS.keys()))]
        if not names:
            raise HTTPException(status_code=400, detail="No presets provided")

        unknown = [n for n in names if n not in PRESETS]
        if unknown:
            raise HTTPException(status_code=400, detail=f"Unknown presets: {', '.join(unknown)}")

        saved = 0
        failures = []
        if req.parallel:
            results = run_screeners_parallel(names, max_workers=req.max_workers)
            for name, df in results.items():
                if df is None:
                    failures.append(name)
                    continue
                save_finviz_snapshot(df, name, db_path=req.db)
                saved += len(df)
        else:
            for name in names:
                try:
                    df = run_screener(name)
                    save_finviz_snapshot(df, name, db_path=req.db)
                    saved += len(df)
                except Exception:
                    failures.append(name)
        return {
            "selected_presets": names,
            "preset_count": len(names),
            "rows_saved": saved,
            "failed_presets": failures,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/pipeline/recommend")
def pipeline_recommend(req: RunRecommendRequest, _auth: dict = Depends(require_write_auth)):
    try:
        create_schema(req.db)
        eng = RuleEngine()
        tickers = [t.upper() for t in (req.tickers or HIGH_INTEREST)]
        all_recs = []
        failures = []
        for t in tickers:
            try:
                all_recs.extend(eng.recommend(t))
            except Exception:
                failures.append(t)
        if req.save and all_recs:
            save_recommendations(all_recs, db_path=req.db)
        return {
            "ticker_count": len(tickers),
            "recommendation_count": len(all_recs),
            "failed_tickers": failures,
            "saved": bool(req.save and all_recs),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/options/ideas")
def options_ideas(
    ticker: str,
    horizon: str = "1m",
    bias: str = "long",
    budget: float = 300.0,
):
    try:
        ideas = pick_affordable_contracts(
            ticker=ticker.upper(),
            horizon=horizon,
            bias=bias,
            budget=budget,
        )
        return {"count": len(ideas), "items": [i.__dict__ for i in ideas]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/charts/performance-compare")
def charts_performance_compare(
    tickers: str,
    period: str = "1y",
    interval: str = "1d",
    base: float = 100.0,
):
    try:
        names = [t.strip().upper() for t in tickers.split(",") if t.strip()]
        if len(names) < 1:
            raise ValueError("Provide at least one ticker")

        frames: dict[str, pd.Series] = {}
        summary: list[dict] = []
        for name in names[:10]:
            df = fetch_ohlcv(name, period=period, interval=interval)
            if df.empty:
                continue
            price = df["adj close"].astype(float).dropna()
            if price.empty:
                continue
            indexed = (price / float(price.iloc[0])) * float(base)
            frames[name] = indexed
            summary.append(
                {
                    "ticker": name,
                    "start": float(price.iloc[0]),
                    "last": float(price.iloc[-1]),
                    "return_pct": float((price.iloc[-1] / price.iloc[0]) - 1.0),
                    "volatility": float(price.pct_change().std() * np.sqrt(252)),
                }
            )

        if not frames:
            return {"count": 0, "items": [], "summary": []}

        mat = pd.DataFrame(frames).dropna(how="all").sort_index()
        items = []
        for idx, row in mat.iterrows():
            payload = {"date": str(pd.Timestamp(idx).date())}
            for col in mat.columns:
                val = row.get(col)
                payload[col] = None if pd.isna(val) else float(val)
            items.append(payload)

        summary.sort(key=lambda r: r.get("return_pct", 0.0), reverse=True)
        return {
            "count": len(items),
            "period": period,
            "interval": interval,
            "base": base,
            "tickers": list(mat.columns),
            "items": items,
            "summary": summary,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/charts/seasonality-compare")
def charts_seasonality_compare(
    ticker: str,
    years: int = 10,
    sector: str | None = None,
    sector_etf: str | None = None,
):
    try:
        symbol = ticker.strip().upper()
        if not symbol:
            raise ValueError("ticker is required")

        resolved_sector = sector or _infer_sector(symbol)
        resolved_sector_etf = (sector_etf or _sector_to_etf(resolved_sector)).upper()

        stock = seasonality_by_doy(symbol, years=years)
        bench = seasonality_by_doy(resolved_sector_etf, years=years)

        stock = stock.rename(columns={"avg_ret": "stock_avg_ret", "std_ret": "stock_std_ret", "n": "stock_n"})
        bench = bench.rename(columns={"avg_ret": "sector_avg_ret", "std_ret": "sector_std_ret", "n": "sector_n"})

        merged = stock.merge(bench, on="doy", how="outer").sort_values("doy").fillna(0.0)
        merged["stock_cum"] = (1.0 + merged["stock_avg_ret"]).cumprod() - 1.0
        merged["sector_cum"] = (1.0 + merged["sector_avg_ret"]).cumprod() - 1.0
        merged["spread"] = merged["stock_avg_ret"] - merged["sector_avg_ret"]

        metrics = _seasonality_alignment_metrics(
            stock.rename(columns={"stock_avg_ret": "avg_ret"})[["doy", "avg_ret"]],
            bench.rename(columns={"sector_avg_ret": "avg_ret"})[["doy", "avg_ret"]],
        )

        return {
            "ticker": symbol,
            "sector": resolved_sector,
            "sector_etf": resolved_sector_etf,
            "years": years,
            "metrics": metrics,
            "count": len(merged),
            "items": _safe_records(merged),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/charts/indicators")
def charts_indicators(ticker: str, period: str = "6mo", interval: str = "1d"):
    try:
        df = fetch_ohlcv(ticker.upper(), period=period, interval=interval)
        df = compute_indicators(df)
        chart_cols = [
            "adj close",
            "close",
            "open",
            "high",
            "low",
            "volume",
            "rsi14",
            "sma20",
            "sma50",
            "sma200",
            "atr14",
            "vol20",
        ]
        use_cols = [c for c in chart_cols if c in df.columns]
        out = df[use_cols].reset_index().copy()
        out["date"] = out["date"].astype(str)
        return {"count": len(out), "items": _safe_records(out)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/charts/seasonality")
def charts_seasonality(ticker: str, years: int = 10):
    try:
        df = seasonality_by_doy(ticker.upper(), years=years)
        return {"count": len(df), "items": _safe_records(df)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/charts/price-distribution")
def charts_price_distribution(ticker: str, period: str = "2y", interval: str = "1d", bins: int = 24):
    try:
        df = fetch_ohlcv(ticker.upper(), period=period, interval=interval)
        df = compute_indicators(df)
        profile = support_resistance_from_distribution(df, bins=bins, level_count=4)
        return {
            "ticker": ticker.upper(),
            "period": period,
            "interval": interval,
            **profile,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/charts/forecast")
def charts_forecast(ticker: str, period: str = "2y", interval: str = "1d", horizon: int = 30):
    try:
        df = fetch_ohlcv(ticker.upper(), period=period, interval=interval)
        forecast = forecast_prices(df, horizon=horizon)
        return {
            "ticker": ticker.upper(),
            "period": period,
            "interval": interval,
            "horizon": max(5, min(int(horizon), 90)),
            **forecast,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/backtest/short-term")
def backtest_short_term_api(
    ticker: str,
    start: str = "2018-01-01",
    end: str | None = None,
    entry_rsi_threshold: float = 50.0,
    max_hold_days: int = 7,
    stop_loss_pct: float = 0.0,
    ma_filter: str = "sma20",
    take_profit_pct: float = 0.0,
    ma_trend_filter: str = "none",
):
    try:
        ma_selected = str(ma_filter or "sma20").lower()
        if ma_selected not in {"sma20", "sma50", "sma200"}:
            raise ValueError("ma_filter must be one of: sma20, sma50, sma200")

        ma_trend = str(ma_trend_filter or "none").lower()
        if ma_trend not in {"none", "above_sma50", "above_sma200"}:
            raise ValueError("ma_trend_filter must be one of: none, above_sma50, above_sma200")

        rpt = backtest_short_term(
            ticker=ticker.upper(),
            start=start,
            end=end,
            entry_rsi_threshold=entry_rsi_threshold,
            max_hold_days=max_hold_days,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            ma_trend_filter=ma_trend,
            ma_filter=ma_selected,
        )
        eq = rpt.equity_curve.reset_index()
        eq.columns = ["date", "equity"]
        eq["date"] = eq["date"].astype(str)
        return {
            "summary": {
                "n_trades": rpt.n_trades,
                "win_rate": rpt.win_rate,
                "avg_ret": rpt.avg_ret,
                "median_ret": rpt.median_ret,
                "avg_hold_days": rpt.avg_hold_days,
                "max_dd": rpt.max_dd,
                "sharpe": rpt.sharpe,
                "sortino": rpt.sortino,
                "cagr": rpt.cagr,
                "profit_factor": rpt.profit_factor,
                "params": {
                    "entry_rsi_threshold": float(entry_rsi_threshold),
                    "max_hold_days": int(max_hold_days),
                    "stop_loss_pct": float(stop_loss_pct),
                    "ma_filter": ma_selected,
                    "take_profit_pct": float(take_profit_pct),
                    "ma_trend_filter": ma_trend,
                },
            },
            "equity": _safe_records(eq),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/market/series/cache")
def market_series_cache(
    ticker: str,
    period: str = "2y",
    interval: str = "1d",
):
    try:
        symbol = ticker.strip().upper()
        if not symbol:
            raise ValueError("ticker is required")

        df = fetch_ohlcv(symbol, period=period, interval=interval)
        df = compute_indicators(df)
        if df.empty:
            return {"ok": False, "count": 0, "items": [], "path": None}

        out = df.reset_index().copy()
        out["date"] = out["date"].astype(str)

        cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "database", "market_series")
        os.makedirs(cache_dir, exist_ok=True)
        safe_period = re.sub(r"[^a-zA-Z0-9_-]", "_", str(period))
        safe_interval = re.sub(r"[^a-zA-Z0-9_-]", "_", str(interval))
        file_name = f"{symbol}_{safe_period}_{safe_interval}.csv"
        file_path = os.path.join(cache_dir, file_name)
        out.to_csv(file_path, index=False)

        # FastAPI/Starlette JSON rendering rejects NaN/Infinity values.
        items_df = (
            out.tail(365)
            .replace([np.inf, -np.inf], np.nan)
            .astype(object)
            .where(pd.notnull(out.tail(365)), None)
        )

        return {
            "ok": True,
            "ticker": symbol,
            "period": period,
            "interval": interval,
            "count": len(out),
            "path": file_path,
            "items": _safe_records(items_df),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
