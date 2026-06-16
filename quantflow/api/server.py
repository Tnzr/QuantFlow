from __future__ import annotations

from typing import Optional

import os

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

from ..data.persistence import create_schema
from ..data.queries import recent_execution_intents, latest_universe, latest_recommendations_per_ticker
from ..broker.factory import make_broker, BROKER_MODE_LEGACY, BROKER_MODE_MCP
from ..portfolio.evaluator import evaluate_positions
from ..execution.policy import ExecutionPolicy, evaluate_order_intent
from ..execution.audit import audit_intent, audit_event, audit_policy
from .auth import require_write_auth
from ..data.finviz_client import PRESETS, run_screener, run_screeners_parallel
from ..data.persistence import save_finviz_snapshot, save_recommendations
from ..data.universe import HIGH_INTEREST
from ..recommend.engine import RuleEngine
from ..recommend.options_picker import pick_affordable_contracts
from ..features.indicators import fetch_ohlcv, compute_indicators
from ..features.seasonality import seasonality_by_doy
from ..backtest.signal_backtest import backtest_short_term


app = FastAPI(title="QuantFlow API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ProposeOrderRequest(BaseModel):
    ticker: str = Field(min_length=1)
    action: str = Field(default="buy")
    qty: float = Field(default=1.0)
    notional: float = Field(gt=0)
    orders_today: int = Field(default=0, ge=0)
    position_notional_after: float = Field(default=0.0, ge=0)
    broker_mode: str = Field(default=BROKER_MODE_LEGACY)
    auto: bool = Field(default=False)
    max_daily_notional: float = Field(default=5000.0, gt=0)
    max_orders_per_day: int = Field(default=20, gt=0)
    max_position_notional: float = Field(default=2000.0, gt=0)
    db: str = Field(default="sqlite:///quantflow.db")


class RunScanRequest(BaseModel):
    db: str = Field(default="sqlite:///quantflow.db")
    parallel: bool = Field(default=True)
    max_workers: int = Field(default=4, ge=1, le=16)


class RunRecommendRequest(BaseModel):
    db: str = Field(default="sqlite:///quantflow.db")
    tickers: Optional[list[str]] = None
    save: bool = Field(default=True)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "auth_enabled": os.getenv("QF_API_AUTH_ENABLED", "false").lower() in {"1", "true", "yes"},
        "firebase_auth_enabled": os.getenv("QF_FIREBASE_AUTH_ENABLED", "false").lower() in {"1", "true", "yes"},
    }


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


@app.get("/scanner/latest-universe")
def scanner_latest_universe(db: str = "sqlite:///quantflow.db"):
    try:
        df = latest_universe(db_path=db)
        if df.empty:
            return {"count": 0, "items": []}
        return {"count": len(df), "items": df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/recommend/latest")
def recommend_latest(db: str = "sqlite:///quantflow.db"):
    try:
        df = latest_recommendations_per_ticker(db_path=db)
        if df.empty:
            return {"count": 0, "items": []}
        return {"count": len(df), "items": df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/portfolio/signals")
def portfolio_signals(
    broker_mode: str = BROKER_MODE_LEGACY,
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


@app.post("/execution/propose")
def execution_propose(req: ProposeOrderRequest, _auth: dict = Depends(require_write_auth)):
    if req.broker_mode not in (BROKER_MODE_LEGACY, BROKER_MODE_MCP):
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
        return {"count": len(df), "items": df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/pipeline/scan")
def pipeline_scan(req: RunScanRequest, _auth: dict = Depends(require_write_auth)):
    try:
        create_schema(req.db)
        names = list(PRESETS.keys())
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


@app.get("/charts/indicators")
def charts_indicators(ticker: str, period: str = "6m", interval: str = "1d"):
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
        return {"count": len(out), "items": out.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/charts/seasonality")
def charts_seasonality(ticker: str, years: int = 10):
    try:
        df = seasonality_by_doy(ticker.upper(), years=years)
        return {"count": len(df), "items": df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/backtest/short-term")
def backtest_short_term_api(ticker: str, start: str = "2018-01-01"):
    try:
        rpt = backtest_short_term(ticker=ticker.upper(), start=start)
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
            },
            "equity": eq.to_dict(orient="records"),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
