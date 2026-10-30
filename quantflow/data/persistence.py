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


class ElectionEvent(Base):
    __tablename__ = "election_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    source: Mapped[str] = mapped_column(String(64), index=True)
    ticker_or_topic: Mapped[str] = mapped_column(String(64), index=True)
    polarity_score: Mapped[float] = mapped_column(Float)
    rationale: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    title: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    dedupe_key: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)


class SentimentSnapshot(Base):
    __tablename__ = "sentiment_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    source: Mapped[str] = mapped_column(String(64), index=True)
    ticker_or_topic: Mapped[str] = mapped_column(String(64), index=True)
    polarity_score: Mapped[float] = mapped_column(Float)
    rationale: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    probability: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    n_items: Mapped[int] = mapped_column(Integer, default=0)


class AgentStateSnapshot(Base):
    """Point-in-time agent state, keyed by the bar timestamp it was computed for."""

    __tablename__ = "agent_state_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    as_of: Mapped[datetime] = mapped_column(DateTime, index=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    horizon: Mapped[str] = mapped_column(String(8))
    base_confidence: Mapped[float] = mapped_column(Float)
    agent_confidence: Mapped[float] = mapped_column(Float)
    final_confidence: Mapped[float] = mapped_column(Float)
    override_applied: Mapped[int] = mapped_column(Integer, default=0)
    volatility_triggered: Mapped[int] = mapped_column(Integer, default=0)
    sentiment_multiplier: Mapped[float] = mapped_column(Float, default=1.0)
    state_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class AllocationProposalRecord(Base):
    __tablename__ = "allocation_proposals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    proposal_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    structure: Mapped[str] = mapped_column(String(32))
    payload_json: Mapped[str] = mapped_column(Text)
    signature: Mapped[str] = mapped_column(Text)
    public_key: Mapped[str] = mapped_column(Text)
    published: Mapped[int] = mapped_column(Integer, default=0)
    publish_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


def _event_kwargs(item: dict) -> dict:
    created_at = item.get("created_at")
    if isinstance(created_at, str) and created_at:
        try:
            created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            created_at = None
    elif isinstance(created_at, datetime) and created_at.tzinfo is not None:
        created_at = created_at.replace(tzinfo=None)
    return {
        "created_at": created_at or datetime.utcnow(),
        "source": str(item.get("source", "unknown"))[:64],
        "ticker_or_topic": str(item.get("ticker_or_topic", "macro"))[:64],
        "polarity_score": float(item.get("polarity_score") or 0.0),
        "rationale": item.get("rationale"),
    }


def save_election_events(events: Iterable[dict], db_path: str = "sqlite:///quantflow.db") -> int:
    """Persist election/macro news events; rows with a known ``dedupe_key`` are skipped."""
    engine = get_engine(db_path)
    Base.metadata.create_all(engine, tables=[ElectionEvent.__table__])
    saved = 0
    with Session(engine) as s:
        seen = {k for (k,) in s.query(ElectionEvent.dedupe_key).filter(ElectionEvent.dedupe_key.isnot(None))}
        for item in events:
            key = item.get("dedupe_key")
            if key and key in seen:
                continue
            s.add(ElectionEvent(**_event_kwargs(item), title=item.get("title"), url=item.get("url"), dedupe_key=key))
            if key:
                seen.add(key)
            saved += 1
        s.commit()
    return saved


def save_sentiment_snapshots(snapshots: Iterable[dict], db_path: str = "sqlite:///quantflow.db") -> int:
    engine = get_engine(db_path)
    Base.metadata.create_all(engine, tables=[SentimentSnapshot.__table__])
    saved = 0
    with Session(engine) as s:
        for item in snapshots:
            s.add(
                SentimentSnapshot(
                    **_event_kwargs(item),
                    probability=_safe_float(item.get("probability")),
                    n_items=int(item.get("n_items") or 0),
                )
            )
            saved += 1
        s.commit()
    return saved


def save_agent_states(states: Iterable[dict], db_path: str = "sqlite:///quantflow.db") -> int:
    engine = get_engine(db_path)
    Base.metadata.create_all(engine, tables=[AgentStateSnapshot.__table__])
    saved = 0
    with Session(engine) as s:
        for st in states:
            s.add(
                AgentStateSnapshot(
                    as_of=st["as_of"],
                    run_id=str(st.get("run_id", "live")),
                    ticker=str(st["ticker"]).upper(),
                    horizon=str(st.get("horizon", "1w")),
                    base_confidence=float(st["base_confidence"]),
                    agent_confidence=float(st["agent_confidence"]),
                    final_confidence=float(st["final_confidence"]),
                    override_applied=int(bool(st.get("override_applied"))),
                    volatility_triggered=int(bool(st.get("volatility_triggered"))),
                    sentiment_multiplier=float(st.get("sentiment_multiplier", 1.0)),
                    state_json=json.dumps(st.get("state") or {}, default=str),
                )
            )
            saved += 1
        s.commit()
    return saved


def save_allocation_proposal(
    proposal: dict,
    db_path: str = "sqlite:///quantflow.db",
    *,
    published: bool = False,
    publish_error: Optional[str] = None,
) -> None:
    engine = get_engine(db_path)
    Base.metadata.create_all(engine, tables=[AllocationProposalRecord.__table__])
    payload = {k: v for k, v in proposal.items() if k not in ("signature", "public_key")}
    with Session(engine) as s:
        s.add(
            AllocationProposalRecord(
                proposal_id=str(proposal["proposal_id"]),
                ticker=str(proposal["ticker"]).upper(),
                structure=str(proposal["structure"]),
                payload_json=json.dumps(payload, default=str, sort_keys=True),
                signature=str(proposal["signature"]),
                public_key=str(proposal.get("public_key", "")),
                published=int(published),
                publish_error=publish_error,
            )
        )
        s.commit()


class PlaybookSignal(Base):
    __tablename__ = "playbook_signals"
    symbol: Mapped[str] = mapped_column(String(16), primary_key=True)
    state: Mapped[str] = mapped_column(String(16))
    side: Mapped[str] = mapped_column(String(8), nullable=True)
    bull: Mapped[float] = mapped_column(Float, nullable=True)
    bear: Mapped[float] = mapped_column(Float, nullable=True)
    payload: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class PlaybookAlert(Base):
    __tablename__ = "playbook_alerts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    symbol: Mapped[str] = mapped_column(String(16), index=True)
    kind: Mapped[str] = mapped_column(String(24))
    message: Mapped[str] = mapped_column(Text)
    payload: Mapped[str] = mapped_column(Text)


def load_playbook_states(db_path: str = "sqlite:///quantflow.db") -> dict[str, dict]:
    with Session(get_engine(db_path)) as s:
        return {r.symbol: json.loads(r.payload) for r in s.query(PlaybookSignal).all()}


def save_playbook_state(state: dict, db_path: str = "sqlite:///quantflow.db") -> None:
    with Session(get_engine(db_path)) as s:
        s.merge(PlaybookSignal(symbol=state["symbol"], state=state["state"], side=state.get("side"),
                               bull=state.get("bull"), bear=state.get("bear"),
                               payload=json.dumps(state, default=str), updated_at=datetime.utcnow()))
        s.commit()


def save_playbook_alerts(alerts: Iterable[dict], db_path: str = "sqlite:///quantflow.db") -> int:
    rows = [PlaybookAlert(symbol=a["symbol"], kind=a["kind"], message=a["message"], payload=json.dumps(a, default=str))
            for a in alerts]
    with Session(get_engine(db_path)) as s:
        s.add_all(rows)
        s.commit()
    return len(rows)


def list_playbook_alerts(limit: int = 100, db_path: str = "sqlite:///quantflow.db") -> list[dict]:
    with Session(get_engine(db_path)) as s:
        q = s.query(PlaybookAlert).order_by(PlaybookAlert.id.desc()).limit(limit)
        return [{**json.loads(r.payload), "id": r.id, "created_at": r.created_at.isoformat()} for r in q]
