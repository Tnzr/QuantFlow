"""Alpaca Markets data client — real-time and historical bars.

Environment variables (all optional; falls back to yfinance if absent):
    ALPACA_API_KEY      — paper or live key ID
    ALPACA_SECRET_KEY   — corresponding secret
    ALPACA_BASE_URL     — defaults to paper-trading base
                          "https://paper-api.alpaca.markets"

Data endpoints used:
    Historical bars  → https://data.alpaca.markets/v2/stocks/{symbol}/bars
    Latest quote     → https://data.alpaca.markets/v2/stocks/{symbol}/quotes/latest
    Latest trade     → https://data.alpaca.markets/v2/stocks/{symbol}/trades/latest

The free Alpaca plan includes 15-min delayed SIP data and real-time IEX quotes.
A paid data subscription unlocks real-time SIP.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

# Auto-load .env from project root if not already loaded
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass

_ALPACA_DATA_BASE = "https://data.alpaca.markets/v2"

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _get_headers() -> dict[str, str]:
    key = os.environ.get("ALPACA_API_KEY", "").strip()
    secret = os.environ.get("ALPACA_SECRET_KEY", "").strip()
    if not key or not secret:
        raise RuntimeError(
            "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set to use the Alpaca client. "
            "Get a free paper-trading key at https://alpaca.markets."
        )
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


def _get(url: str, params: dict | None = None) -> Any:
    import requests  # guarded import — not a hard dependency

    resp = requests.get(url, headers=_get_headers(), params=params or {}, timeout=15)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def is_configured() -> bool:
    """Return True if both Alpaca env vars are present."""
    return bool(
        os.environ.get("ALPACA_API_KEY", "").strip()
        and os.environ.get("ALPACA_SECRET_KEY", "").strip()
    )


def fetch_bars(
    symbol: str,
    timeframe: str = "1Min",
    start: str | None = None,
    end: str | None = None,
    limit: int = 1000,
    feed: str = "iex",
) -> pd.DataFrame:
    """Fetch OHLCV bars for *symbol* from Alpaca.

    Args:
        symbol:    Ticker, e.g. "AAPL".
        timeframe: Alpaca timeframe string — "1Min", "5Min", "15Min", "1Hour", "1Day".
        start:     ISO-8601 datetime string or None (defaults to 1 day ago for intraday).
        end:       ISO-8601 datetime string or None (defaults to now).
        limit:     Max bars to return (max 10000 per request).
        feed:      "iex" (free real-time) or "sip" (requires paid subscription).

    Returns:
        DataFrame with columns [open, high, low, close, volume, vwap, trade_count].
        Index is a timezone-aware DatetimeIndex named "date".
    """
    symbol = symbol.strip().upper()
    if not start:
        now_utc = datetime.now(timezone.utc)
        start_dt = now_utc - timedelta(days=1)
        start = start_dt.isoformat()
    if not end:
        end = datetime.now(timezone.utc).isoformat()

    url = f"{_ALPACA_DATA_BASE}/stocks/{symbol}/bars"
    params: dict[str, Any] = {
        "timeframe": timeframe,
        "start": start,
        "end": end,
        "limit": limit,
        "feed": feed,
        "sort": "asc",
    }

    rows = []
    page_token: str | None = None
    while True:
        if page_token:
            params["page_token"] = page_token
        data = _get(url, params)
        rows.extend(data.get("bars") or [])
        page_token = data.get("next_page_token")
        if not page_token or len(rows) >= limit:
            break
        time.sleep(0.1)  # respect rate limit

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.rename(columns={"t": "date", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "vw": "vwap", "n": "trade_count"})
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.set_index("date").sort_index()
    for col in ["open", "high", "low", "close", "volume", "vwap", "trade_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["open", "high", "low", "close", "volume"] + [c for c in ["vwap", "trade_count"] if c in df.columns]]


def fetch_latest_quote(symbol: str, feed: str = "iex") -> dict:
    """Return the latest bid/ask quote for *symbol*."""
    symbol = symbol.strip().upper()
    data = _get(f"{_ALPACA_DATA_BASE}/stocks/{symbol}/quotes/latest", {"feed": feed})
    q = data.get("quote") or {}
    return {
        "symbol": symbol,
        "bid": q.get("bp"),
        "ask": q.get("ap"),
        "bid_size": q.get("bs"),
        "ask_size": q.get("as"),
        "timestamp": q.get("t"),
        "conditions": q.get("c"),
    }


def fetch_latest_trade(symbol: str, feed: str = "iex") -> dict:
    """Return the most recent trade print for *symbol*."""
    symbol = symbol.strip().upper()
    data = _get(f"{_ALPACA_DATA_BASE}/stocks/{symbol}/trades/latest", {"feed": feed})
    t = data.get("trade") or {}
    return {
        "symbol": symbol,
        "price": t.get("p"),
        "size": t.get("s"),
        "timestamp": t.get("t"),
        "conditions": t.get("c"),
        "exchange": t.get("x"),
    }


def fetch_latest_bars_multi(
    symbols: list[str],
    timeframe: str = "1Min",
    feed: str = "iex",
) -> dict[str, pd.DataFrame]:
    """Fetch latest bar snapshots for a list of symbols.

    Returns a dict keyed by symbol.  Only symbols with data are included.
    """
    result: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            df = fetch_bars(sym, timeframe=timeframe, feed=feed, limit=2)
            if not df.empty:
                result[sym.upper()] = df
        except Exception:
            pass
    return result
