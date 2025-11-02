from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import yfinance as yf
import pandas as pd


@dataclass
class Contract:
    ticker: str
    expiry: str
    strike: float
    right: str  # C/P
    last: float
    bid: float
    ask: float
    volume: int
    open_interest: int
    iv: float


def load_chain(ticker: str) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    expiries = t.options
    rows: List[Contract] = []
    for ex in expiries[:6]:  # limit for speed
        calls = t.option_chain(ex).calls
        puts = t.option_chain(ex).puts
        for _, r in calls.iterrows():
            try:
                rows.append(Contract(
                    ticker, ex, float(r["strike"]), "C",
                    _safe(r, "lastPrice"),
                    _safe(r, "bid"),
                    _safe(r, "ask"),
                    int(_safe(r, "volume", 0) or 0),
                    int(_safe(r, "openInterest", 0) or 0),
                    float(_safe(r, "impliedVolatility") or 0.0)
                ))
            except Exception:
                continue
        for _, r in puts.iterrows():
            try:
                rows.append(Contract(
                    ticker, ex, float(r["strike"]), "P",
                    _safe(r, "lastPrice"),
                    _safe(r, "bid"),
                    _safe(r, "ask"),
                    int(_safe(r, "volume", 0) or 0),
                    int(_safe(r, "openInterest", 0) or 0),
                    float(_safe(r, "impliedVolatility") or 0.0)
                ))
            except Exception:
                continue
    df = pd.DataFrame([c.__dict__ for c in rows])
    return df


def _safe(r, k, default=None):
    try:
        v = r.get(k, default)
        if v is None:
            return default
        return v
    except Exception:
        return default
