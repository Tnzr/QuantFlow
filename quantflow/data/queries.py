from __future__ import annotations

import json
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select, func, desc
from .persistence import get_engine, TickerSnapshot, Recommendation, NewsArticle, AnalyticsLeaderboardEntry, ExecutionIntent, ExecutionEvent, PolicyDecision
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


def latest_analytics_leaderboard(db_path: str = "sqlite:///quantflow.db") -> pd.DataFrame:
    engine = get_engine(db_path)
    with Session(engine) as s:
        latest_batch = s.execute(
            select(AnalyticsLeaderboardEntry.batch_id)
            .order_by(desc(AnalyticsLeaderboardEntry.created_at), desc(AnalyticsLeaderboardEntry.id))
            .limit(1)
        ).scalar()
        if not latest_batch:
            return pd.DataFrame()
        q = (
            select(
                AnalyticsLeaderboardEntry.created_at,
                AnalyticsLeaderboardEntry.batch_id,
                AnalyticsLeaderboardEntry.ticker,
                AnalyticsLeaderboardEntry.period,
                AnalyticsLeaderboardEntry.interval,
                AnalyticsLeaderboardEntry.bias,
                AnalyticsLeaderboardEntry.primary_horizon,
                AnalyticsLeaderboardEntry.price,
                AnalyticsLeaderboardEntry.composite_score,
                AnalyticsLeaderboardEntry.nearest_support,
                AnalyticsLeaderboardEntry.nearest_resistance,
                AnalyticsLeaderboardEntry.support_gap_pct,
                AnalyticsLeaderboardEntry.resistance_gap_pct,
                AnalyticsLeaderboardEntry.score_breakdown_json,
                AnalyticsLeaderboardEntry.recommendations_json,
            )
            .where(AnalyticsLeaderboardEntry.batch_id == latest_batch)
            .order_by(desc(AnalyticsLeaderboardEntry.composite_score), AnalyticsLeaderboardEntry.ticker)
        )
        rows = s.execute(q).all()
        frame = pd.DataFrame(rows, columns=[c.key for c in q.selected_columns])
        if frame.empty:
            return frame
        frame["score_breakdown"] = frame["score_breakdown_json"].apply(lambda value: json.loads(value) if value else {})
        frame["recommendations"] = frame["recommendations_json"].apply(lambda value: json.loads(value) if value else [])
        return frame.drop(columns=["score_breakdown_json", "recommendations_json"])


def recent_news_articles(
    days: int = 30,
    db_path: str = "sqlite:///quantflow.db",
    ticker: Optional[str] = None,
    limit: int = 250,
) -> pd.DataFrame:
    engine = get_engine(db_path)
    with Session(engine) as s:
        cutoff = func.datetime(func.datetime("now"), f"-{days} day")
        query = select(
            NewsArticle.id,
            NewsArticle.created_at,
            NewsArticle.published_at,
            NewsArticle.source,
            NewsArticle.source_id,
            NewsArticle.url,
            NewsArticle.language,
            NewsArticle.ticker,
            NewsArticle.tickers_json,
            NewsArticle.title,
            NewsArticle.summary,
            NewsArticle.content,
            NewsArticle.sentiment_score,
            NewsArticle.sentiment_label,
            NewsArticle.metadata_json,
        ).where(NewsArticle.created_at >= cutoff)
        if ticker:
            ticker = ticker.upper()
            query = query.where((NewsArticle.ticker == ticker) | (NewsArticle.tickers_json.contains(f'"{ticker}"')))
        query = query.order_by(desc(NewsArticle.created_at)).limit(limit)
        rows = s.execute(query).all()
        return pd.DataFrame(rows, columns=[c.key for c in query.selected_columns])


def recent_execution_intents(limit: int = 100, db_path: str = "sqlite:///quantflow.db") -> pd.DataFrame:
    engine = get_engine(db_path)
    with Session(engine) as s:
        q = (
            select(
                ExecutionIntent.id,
                ExecutionIntent.created_at,
                ExecutionIntent.ticker,
                ExecutionIntent.action,
                ExecutionIntent.qty,
                ExecutionIntent.notional,
                ExecutionIntent.broker_mode,
                ExecutionIntent.source,
                ExecutionIntent.status,
            )
            .order_by(desc(ExecutionIntent.created_at))
            .limit(limit)
        )
        rows = s.execute(q).all()
        return pd.DataFrame(rows, columns=[c.key for c in q.selected_columns])


def events_for_intent(intent_id: int, db_path: str = "sqlite:///quantflow.db") -> pd.DataFrame:
    engine = get_engine(db_path)
    with Session(engine) as s:
        q = (
            select(
                ExecutionEvent.id,
                ExecutionEvent.created_at,
                ExecutionEvent.intent_id,
                ExecutionEvent.event_type,
                ExecutionEvent.message,
            )
            .where(ExecutionEvent.intent_id == intent_id)
            .order_by(ExecutionEvent.created_at)
        )
        rows = s.execute(q).all()
        return pd.DataFrame(rows, columns=[c.key for c in q.selected_columns])


def policy_for_intent(intent_id: int, db_path: str = "sqlite:///quantflow.db") -> pd.DataFrame:
    engine = get_engine(db_path)
    with Session(engine) as s:
        q = (
            select(
                PolicyDecision.id,
                PolicyDecision.created_at,
                PolicyDecision.intent_id,
                PolicyDecision.allow,
                PolicyDecision.reason,
            )
            .where(PolicyDecision.intent_id == intent_id)
            .order_by(desc(PolicyDecision.created_at))
        )
        rows = s.execute(q).all()
        return pd.DataFrame(rows, columns=[c.key for c in q.selected_columns])
