from __future__ import annotations

from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select, func, desc
from .persistence import get_engine, TickerSnapshot, Recommendation
import pandas as pd


def latest_by_preset(preset: str, db_path: str = "sqlite:///quantflow.db") -> pd.DataFrame:
    engine = get_engine(db_path)
    with Session(engine) as s:
        # get latest timestamp for this preset
        latest_ts = s.execute(select(func.max(TickerSnapshot.created_at)).where(TickerSnapshot.preset == preset)).scalar()
        if not latest_ts:
            return pd.DataFrame()
        q = (
            select(
                TickerSnapshot.ticker,
                TickerSnapshot.company,
                TickerSnapshot.sector,
                TickerSnapshot.industry,
                TickerSnapshot.price,
                TickerSnapshot.change,
                TickerSnapshot.rel_volume,
                TickerSnapshot.atr,
                TickerSnapshot.rsi,
                TickerSnapshot.country,
                TickerSnapshot.created_at,
            )
            .where((TickerSnapshot.preset == preset) & (TickerSnapshot.created_at == latest_ts))
        )
        rows = s.execute(q).all()
        return pd.DataFrame(rows, columns=[c.key for c in q.selected_columns])


def history_for_ticker(ticker: str, db_path: str = "sqlite:///quantflow.db") -> pd.DataFrame:
    engine = get_engine(db_path)
    with Session(engine) as s:
        q = (
            select(
                TickerSnapshot.created_at,
                TickerSnapshot.preset,
                TickerSnapshot.price,
                TickerSnapshot.change,
                TickerSnapshot.rel_volume,
                TickerSnapshot.atr,
                TickerSnapshot.rsi,
            )
            .where(TickerSnapshot.ticker == ticker)
            .order_by(TickerSnapshot.created_at)
        )
        rows = s.execute(q).all()
        return pd.DataFrame(rows, columns=[c.key for c in q.selected_columns])


def latest_universe(db_path: str = "sqlite:///quantflow.db") -> pd.DataFrame:
    engine = get_engine(db_path)
    with Session(engine) as s:
        # per ticker latest record across presets
        sub = select(TickerSnapshot.ticker, func.max(TickerSnapshot.created_at).label("mx")).group_by(TickerSnapshot.ticker).subquery()
        q = (
            select(
                TickerSnapshot.ticker,
                TickerSnapshot.company,
                TickerSnapshot.sector,
                TickerSnapshot.price,
                TickerSnapshot.rsi,
                TickerSnapshot.rel_volume,
                TickerSnapshot.atr,
                TickerSnapshot.preset,
                TickerSnapshot.created_at,
            )
            .join(sub, (TickerSnapshot.ticker == sub.c.ticker) & (TickerSnapshot.created_at == sub.c.mx))
        )
        rows = s.execute(q).all()
        return pd.DataFrame(rows, columns=[c.key for c in q.selected_columns])


def recent_recommendations(days: int = 7, db_path: str = "sqlite:///quantflow.db") -> pd.DataFrame:
    engine = get_engine(db_path)
    with Session(engine) as s:
        cutoff = func.datetime(func.datetime("now"), f"-{days} day")
        q = (
            select(
                Recommendation.created_at,
                Recommendation.ticker,
                Recommendation.horizon,
                Recommendation.bias,
                Recommendation.entry,
                Recommendation.stop,
                Recommendation.target,
                Recommendation.confidence,
                Recommendation.notes,
            ).where(Recommendation.created_at >= cutoff).order_by(desc(Recommendation.created_at))
        )
        rows = s.execute(q).all()
        return pd.DataFrame(rows, columns=[c.key for c in q.selected_columns])


def latest_recommendations_per_ticker(db_path: str = "sqlite:///quantflow.db") -> pd.DataFrame:
    engine = get_engine(db_path)
    with Session(engine) as s:
        sub = select(Recommendation.ticker, func.max(Recommendation.created_at).label("mx")).group_by(Recommendation.ticker).subquery()
        q = (
            select(
                Recommendation.created_at,
                Recommendation.ticker,
                Recommendation.horizon,
                Recommendation.bias,
                Recommendation.entry,
                Recommendation.stop,
                Recommendation.target,
                Recommendation.confidence,
                Recommendation.notes,
            ).join(sub, (Recommendation.ticker == sub.c.ticker) & (Recommendation.created_at == sub.c.mx))
        )
        rows = s.execute(q).all()
        return pd.DataFrame(rows, columns=[c.key for c in q.selected_columns])
