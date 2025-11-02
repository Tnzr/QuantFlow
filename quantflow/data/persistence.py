from __future__ import annotations

from datetime import datetime
from typing import Optional, Iterable

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
