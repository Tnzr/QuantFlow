from __future__ import annotations

from datetime import datetime
import json
from typing import Optional, Iterable
from uuid import uuid4

from sqlalchemy import create_engine, String, Float, Integer, DateTime, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session


class Base(DeclarativeBase):
    pass


class TickerSnapshot(Base):
    __tablename__ = "ticker_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    source: Mapped[str] = mapped_column(String(32))  # e.g., finviz
    preset: Mapped[str] = mapped_column(String(64))

    ticker: Mapped[str] = mapped_column(String(16), index=True)
    company: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    sector: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    industry: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    change: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rel_volume: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    atr: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rsi: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    country: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)


class Recommendation(Base):
    __tablename__ = "recommendations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)

    ticker: Mapped[str] = mapped_column(String(16), index=True)
    horizon: Mapped[str] = mapped_column(String(8), index=True)
    bias: Mapped[str] = mapped_column(String(8))
    entry: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    stop: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    target: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    confidence: Mapped[float] = mapped_column(Float)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class NewsArticle(Base):
    __tablename__ = "news_articles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    published_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True, index=True)

    source: Mapped[str] = mapped_column(String(64), index=True)
    source_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, index=True)
    url: Mapped[Optional[str]] = mapped_column(String(512), nullable=True, index=True)
    language: Mapped[str] = mapped_column(String(16), default="en")

    ticker: Mapped[Optional[str]] = mapped_column(String(16), nullable=True, index=True)
    tickers_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    title: Mapped[str] = mapped_column(Text)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    sentiment_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True, index=True)
    sentiment_label: Mapped[Optional[str]] = mapped_column(String(16), nullable=True, index=True)
    metadata_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class TrainingFeatureSnapshot(Base):
    __tablename__ = "training_feature_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    batch_id: Mapped[str] = mapped_column(String(64), index=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    period: Mapped[str] = mapped_column(String(16), index=True)
    interval: Mapped[str] = mapped_column(String(16), index=True)
    as_of_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True, index=True)
    feature_json: Mapped[str] = mapped_column(Text)


class AnalyticsLeaderboardEntry(Base):
    __tablename__ = "analytics_leaderboard_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    batch_id: Mapped[str] = mapped_column(String(64), index=True)

    ticker: Mapped[str] = mapped_column(String(16), index=True)
    period: Mapped[str] = mapped_column(String(16), index=True)
    interval: Mapped[str] = mapped_column(String(16), index=True)
    bias: Mapped[str] = mapped_column(String(16))
    primary_horizon: Mapped[str] = mapped_column(String(8), index=True)
    price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    composite_score: Mapped[float] = mapped_column(Float, index=True)
    nearest_support: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    nearest_resistance: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    support_gap_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    resistance_gap_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    score_breakdown_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    recommendations_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class ExecutionIntent(Base):
    __tablename__ = "execution_intents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)

    ticker: Mapped[str] = mapped_column(String(16), index=True)
    action: Mapped[str] = mapped_column(String(16))
    qty: Mapped[float] = mapped_column(Float)
    notional: Mapped[float] = mapped_column(Float)
    broker_mode: Mapped[str] = mapped_column(String(32), index=True)
    source: Mapped[str] = mapped_column(String(32), default="quantflow")
    status: Mapped[str] = mapped_column(String(32), default="proposed", index=True)


class ExecutionEvent(Base):
    __tablename__ = "execution_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)

    intent_id: Mapped[int] = mapped_column(Integer, index=True)
    event_type: Mapped[str] = mapped_column(String(32), index=True)
    message: Mapped[str] = mapped_column(Text)


class PolicyDecision(Base):
    __tablename__ = "policy_decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)

    intent_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    allow: Mapped[int] = mapped_column(Integer)  # 1 true, 0 false
    reason: Mapped[str] = mapped_column(Text)


def get_engine(db_path: str = "sqlite:///quantflow.db"):
    return create_engine(db_path, future=True)


def create_schema(db_path: str = "sqlite:///quantflow.db"):
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)


def save_finviz_snapshot(df, preset: str, db_path: str = "sqlite:///quantflow.db"):
    engine = get_engine(db_path)
    with Session(engine) as s:
        for _, row in df.iterrows():
            rec = TickerSnapshot(
                source="finviz",
                preset=preset,
                ticker=str(row.get("Ticker")),
                company=row.get("Company"),
                sector=row.get("Sector"),
                industry=row.get("Industry"),
                price=_safe_float(row.get("Price")),
                change=_parse_percent(row.get("Change")),
                rel_volume=_safe_float(row.get("Rel Volume")),
                atr=_safe_float(row.get("ATR")),
                rsi=_safe_float(row.get("RSI (14)")),
                country=row.get("Country"),
            )
            s.add(rec)
        s.commit()


def save_recommendations(recs: Iterable, db_path: str = "sqlite:///quantflow.db"):
    """Persist a batch of Rec-like objects (ticker, horizon, bias, entry, stop, target, confidence, notes)."""
    engine = get_engine(db_path)
    with Session(engine) as s:
        for r in recs:
            s.add(
                Recommendation(
                    ticker=r.ticker,
                    horizon=r.horizon,
                    bias=r.bias,
                    entry=r.entry,
                    stop=r.stop,
                    target=r.target,
                    confidence=r.confidence,
                    notes=r.notes,
                )
            )
        s.commit()


def save_news_articles(articles: Iterable[dict], db_path: str = "sqlite:///quantflow.db") -> int:
    engine = get_engine(db_path)
    saved = 0
    with Session(engine) as s:
        for article in articles:
            tickers = article.get("tickers") or article.get("symbols") or []
            if isinstance(tickers, str):
                tickers = [t.strip().upper() for t in tickers.split(",") if t.strip()]
            tickers = [str(t).upper() for t in tickers if str(t).strip()]
            ticker = article.get("ticker") or (tickers[0] if tickers else None)
            published_at = article.get("published_at")
            if isinstance(published_at, str) and published_at:
                try:
                    published_at = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                except Exception:
                    published_at = None
            rec = NewsArticle(
                source=str(article.get("source", "manual")),
                source_id=article.get("source_id"),
                url=article.get("url"),
                language=str(article.get("language", "en")),
                ticker=str(ticker).upper() if ticker else None,
                tickers_json=json.dumps(tickers) if tickers else None,
                title=str(article.get("title", "")).strip(),
                summary=article.get("summary"),
                content=article.get("content"),
                published_at=published_at,
                sentiment_score=_safe_float(article.get("sentiment_score")),
                sentiment_label=article.get("sentiment_label"),
                metadata_json=json.dumps(article.get("metadata") or {}),
            )
            s.add(rec)
            saved += 1
        s.commit()
    return saved


def save_analytics_leaderboard(
    items: Iterable[dict],
    db_path: str = "sqlite:///quantflow.db",
    *,
    period: str = "2y",
    interval: str = "1d",
) -> str:
    engine = get_engine(db_path)
    batch_id = f"analytics-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{uuid4().hex[:8]}"
    with Session(engine) as s:
        for item in items:
            support = item.get("support_resistance") or {}
            s.add(
                AnalyticsLeaderboardEntry(
                    batch_id=batch_id,
                    ticker=str(item.get("ticker", "")).upper(),
                    period=period,
                    interval=interval,
                    bias=str(item.get("bias", "neutral")),
                    primary_horizon=str(item.get("primary_horizon", "")),
                    price=_safe_float(item.get("price")),
                    composite_score=float(item.get("composite_score", 0.0)),
                    nearest_support=_safe_float(support.get("nearest_support")),
                    nearest_resistance=_safe_float(support.get("nearest_resistance")),
                    support_gap_pct=_safe_float(support.get("support_gap_pct")),
                    resistance_gap_pct=_safe_float(support.get("resistance_gap_pct")),
                    score_breakdown_json=json.dumps(item.get("score_breakdown") or {}),
                    recommendations_json=json.dumps(item.get("recommendations") or []),
                )
            )
        s.commit()
    return batch_id


def save_execution_intent(
    ticker: str,
    action: str,
    qty: float,
    notional: float,
    broker_mode: str,
    source: str = "quantflow",
    status: str = "proposed",
    db_path: str = "sqlite:///quantflow.db",
) -> int:
    engine = get_engine(db_path)
    with Session(engine) as s:
        rec = ExecutionIntent(
            ticker=ticker,
            action=action,
            qty=qty,
            notional=notional,
            broker_mode=broker_mode,
            source=source,
            status=status,
        )
        s.add(rec)
        s.commit()
        s.refresh(rec)
        return int(rec.id)


def save_execution_event(
    intent_id: int,
    event_type: str,
    message: str,
    db_path: str = "sqlite:///quantflow.db",
) -> None:
    engine = get_engine(db_path)
    with Session(engine) as s:
        s.add(
            ExecutionEvent(
                intent_id=intent_id,
                event_type=event_type,
                message=message,
            )
        )
        s.commit()


def save_policy_decision(
    intent_id: Optional[int],
    allow: bool,
    reason: str,
    db_path: str = "sqlite:///quantflow.db",
) -> None:
    engine = get_engine(db_path)
    with Session(engine) as s:
        s.add(
            PolicyDecision(
                intent_id=intent_id,
                allow=1 if allow else 0,
                reason=reason,
            )
        )
        s.commit()


def save_training_features(rows: Iterable[dict], db_path: str = "sqlite:///quantflow.db") -> str:
    engine = get_engine(db_path)
    batch_id = f"train-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    with Session(engine) as s:
        for row in rows:
            s.add(
                TrainingFeatureSnapshot(
                    batch_id=batch_id,
                    ticker=str(row.get("ticker", "")).upper(),
                    period=str(row.get("period", "")),
                    interval=str(row.get("interval", "")),
                    as_of_date=row.get("as_of_date"),
                    feature_json=json.dumps(row, default=str),
                )
            )
        s.commit()
    return batch_id


def _safe_float(x):
    try:
        if x is None or x == "-":
            return None
        return float(str(x).replace(",", ""))
    except Exception:
        return None


def _parse_percent(x):
    try:
        if x is None or x == "-":
            return None
        s = str(x).strip().replace("%", "")
        return float(s) / 100.0
    except Exception:
        return None
