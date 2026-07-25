"""Live signal generator.

Computes entry/exit signals from recent OHLCV data using a configurable rule
set.  Works with both Alpaca real-time bars (when ALPACA_API_KEY is configured)
and yfinance delayed data as a fallback.

Signals are returned as typed dicts so they can be directly serialised to JSON
and stored in the database.

Signal types
------------
- "breakout"   — close crosses above the N-bar high
- "pullback"   — close pulls back to EMA support within a trend
- "rsi_cross"  — RSI crosses the oversold/overbought boundary
- "macd_cross" — MACD line crosses the signal line
- "alert"      — price crosses a user-defined level (from ``thresholds``)

Each signal dict has:
    ticker, signal_type, direction (long|short), price, ts (UTC ISO string),
    confidence (0-1), details (free-form dict of indicator values).
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def _safe_float(x: Any) -> float | None:
    try:
        v = float(x)
        return None if not math.isfinite(v) else v
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# data loader
# ---------------------------------------------------------------------------


def _load_bars(
    ticker: str,
    timeframe: str,
    lookback_bars: int,
) -> pd.DataFrame:
    """Load OHLCV data, preferring Alpaca if configured, else yfinance."""
    from quantflow.data.alpaca_client import is_configured, fetch_bars

    if is_configured():
        df = fetch_bars(ticker, timeframe=timeframe, limit=lookback_bars + 50)
        if not df.empty:
            return df.tail(lookback_bars + 50)

    # yfinance fallback — map Alpaca timeframe → yfinance interval
    _TF_MAP = {
        "1Min": "1m",
        "5Min": "5m",
        "15Min": "15m",
        "30Min": "30m",
        "1Hour": "1h",
        "1Day": "1d",
    }
    interval = _TF_MAP.get(timeframe, "5m")
    period = "5d" if "Min" in timeframe or "Hour" in timeframe else "1y"

    from quantflow.features.indicators import fetch_ohlcv
    df = fetch_ohlcv(ticker, period=period, interval=interval)
    return df.tail(lookback_bars + 50) if not df.empty else df


# ---------------------------------------------------------------------------
# individual signal detectors
# ---------------------------------------------------------------------------


def _detect_breakout(df: pd.DataFrame, ticker: str, window: int = 20) -> list[dict]:
    if len(df) < window + 2:
        return []
    signals = []
    close = df["close"]
    high = df["high"]
    prev_high = high.shift(1).rolling(window).max().shift(1)

    last = close.iloc[-1]
    prev_breakout_level = prev_high.iloc[-1]
    prev_close = close.iloc[-2]

    if (
        prev_close <= prev_breakout_level
        and last > prev_breakout_level
        and _safe_float(last) is not None
        and _safe_float(prev_breakout_level) is not None
    ):
        signals.append({
            "ticker": ticker,
            "signal_type": "breakout",
            "direction": "long",
            "price": _safe_float(last),
            "ts": datetime.now(timezone.utc).isoformat(),
            "confidence": min(1.0, round((last - prev_breakout_level) / prev_breakout_level * 10, 3)),
            "details": {
                "breakout_level": _safe_float(prev_breakout_level),
                "window": window,
            },
        })
    return signals


def _detect_rsi_cross(df: pd.DataFrame, ticker: str, oversold: float = 30, overbought: float = 70) -> list[dict]:
    if len(df) < 20:
        return []
    close = df["close"]
    rsi = _rsi(close)
    if rsi.isna().iloc[-1] or rsi.isna().iloc[-2]:
        return []

    signals = []
    prev_rsi, last_rsi = rsi.iloc[-2], rsi.iloc[-1]
    last_price = _safe_float(close.iloc[-1])
    if last_price is None:
        return []

    if prev_rsi <= oversold < last_rsi:
        signals.append({
            "ticker": ticker,
            "signal_type": "rsi_cross",
            "direction": "long",
            "price": last_price,
            "ts": datetime.now(timezone.utc).isoformat(),
            "confidence": round(min(1.0, (last_rsi - oversold) / 10), 3),
            "details": {"rsi": _safe_float(last_rsi), "threshold": oversold},
        })
    elif prev_rsi >= overbought > last_rsi:
        signals.append({
            "ticker": ticker,
            "signal_type": "rsi_cross",
            "direction": "short",
            "price": last_price,
            "ts": datetime.now(timezone.utc).isoformat(),
            "confidence": round(min(1.0, (overbought - last_rsi) / 10), 3),
            "details": {"rsi": _safe_float(last_rsi), "threshold": overbought},
        })
    return signals


def _detect_macd_cross(df: pd.DataFrame, ticker: str) -> list[dict]:
    if len(df) < 35:
        return []
    close = df["close"]
    macd_line, signal_line, _ = _macd(close)
    if macd_line.isna().iloc[-1] or signal_line.isna().iloc[-1]:
        return []

    signals = []
    prev_diff = macd_line.iloc[-2] - signal_line.iloc[-2]
    curr_diff = macd_line.iloc[-1] - signal_line.iloc[-1]
    last_price = _safe_float(close.iloc[-1])
    if last_price is None:
        return []

    if prev_diff < 0 < curr_diff:
        signals.append({
            "ticker": ticker,
            "signal_type": "macd_cross",
            "direction": "long",
            "price": last_price,
            "ts": datetime.now(timezone.utc).isoformat(),
            "confidence": round(min(1.0, abs(curr_diff) / (abs(macd_line.iloc[-1]) + 1e-9)), 3),
            "details": {
                "macd": _safe_float(macd_line.iloc[-1]),
                "signal": _safe_float(signal_line.iloc[-1]),
            },
        })
    elif prev_diff > 0 > curr_diff:
        signals.append({
            "ticker": ticker,
            "signal_type": "macd_cross",
            "direction": "short",
            "price": last_price,
            "ts": datetime.now(timezone.utc).isoformat(),
            "confidence": round(min(1.0, abs(curr_diff) / (abs(macd_line.iloc[-1]) + 1e-9)), 3),
            "details": {
                "macd": _safe_float(macd_line.iloc[-1]),
                "signal": _safe_float(signal_line.iloc[-1]),
            },
        })
    return signals


def _detect_pullback(df: pd.DataFrame, ticker: str, ema_span: int = 21, trend_span: int = 50) -> list[dict]:
    if len(df) < trend_span + 5:
        return []
    close = df["close"]
    ema_fast = _ema(close, ema_span)
    ema_slow = _ema(close, trend_span)
    atr = _atr(df)

    last_price = _safe_float(close.iloc[-1])
    last_ema = _safe_float(ema_fast.iloc[-1])
    last_slow = _safe_float(ema_slow.iloc[-1])
    last_atr = _safe_float(atr.iloc[-1])
    if any(v is None for v in [last_price, last_ema, last_slow, last_atr]):
        return []

    in_uptrend = last_slow < last_ema
    at_support = abs(last_price - last_ema) < last_atr * 0.5
    if in_uptrend and at_support:
        return [{
            "ticker": ticker,
            "signal_type": "pullback",
            "direction": "long",
            "price": last_price,
            "ts": datetime.now(timezone.utc).isoformat(),
            "confidence": round(1 - abs(last_price - last_ema) / (last_atr * 0.5 + 1e-9), 3),
            "details": {"ema": last_ema, "atr": last_atr},
        }]
    return []


def _detect_threshold_alerts(df: pd.DataFrame, ticker: str, thresholds: list[dict]) -> list[dict]:
    """Generate alerts when price crosses user-defined levels.

    Each threshold dict: {"price": float, "direction": "above"|"below", "label": str}
    """
    if df.empty or not thresholds:
        return []
    last = _safe_float(df["close"].iloc[-1])
    prev = _safe_float(df["close"].iloc[-2]) if len(df) >= 2 else None
    if last is None:
        return []
    signals = []
    for t in thresholds:
        lvl = _safe_float(t.get("price"))
        if lvl is None:
            continue
        label = str(t.get("label", f"level_{lvl}"))
        direction = str(t.get("direction", "above")).lower()
        if direction == "above" and (prev is None or prev < lvl) and last >= lvl:
            signals.append({
                "ticker": ticker,
                "signal_type": "alert",
                "direction": "long",
                "price": last,
                "ts": datetime.now(timezone.utc).isoformat(),
                "confidence": 1.0,
                "details": {"label": label, "threshold": lvl},
            })
        elif direction == "below" and (prev is None or prev > lvl) and last <= lvl:
            signals.append({
                "ticker": ticker,
                "signal_type": "alert",
                "direction": "short",
                "price": last,
                "ts": datetime.now(timezone.utc).isoformat(),
                "confidence": 1.0,
                "details": {"label": label, "threshold": lvl},
            })
    return signals


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------


def generate_signals(
    ticker: str,
    timeframe: str = "5Min",
    lookback_bars: int = 100,
    detectors: list[str] | None = None,
    thresholds: list[dict] | None = None,
    rsi_oversold: float = 30,
    rsi_overbought: float = 70,
    breakout_window: int = 20,
    ema_span: int = 21,
    trend_span: int = 50,
) -> dict:
    """Run all enabled signal detectors for *ticker* and return results.

    Args:
        ticker:          Ticker symbol.
        timeframe:       Bar timeframe ("1Min","5Min","15Min","30Min","1Hour","1Day").
        lookback_bars:   Number of bars of history to use.
        detectors:       Subset of ["breakout","rsi","macd","pullback","alert"].
                         Defaults to all.
        thresholds:      List of alert threshold dicts for "alert" detector.
        rsi_oversold:    RSI oversold level (default 30).
        rsi_overbought:  RSI overbought level (default 70).
        breakout_window: Rolling high window for breakout detector.
        ema_span:        Fast EMA span for pullback detector.
        trend_span:      Slow EMA span for trend detection.

    Returns:
        Dict with keys:
            ticker, timeframe, bar_count, signals (list), latest_bar (dict),
            data_source ("alpaca" or "yfinance"), generated_at (ISO string).
    """
    from quantflow.data.alpaca_client import is_configured

    detectors = detectors or ["breakout", "rsi", "macd", "pullback", "alert"]
    ticker = ticker.strip().upper()

    df = _load_bars(ticker, timeframe, lookback_bars)
    data_source = "alpaca" if is_configured() else "yfinance"

    if df.empty:
        return {
            "ticker": ticker,
            "timeframe": timeframe,
            "bar_count": 0,
            "signals": [],
            "latest_bar": None,
            "data_source": data_source,
            "error": "No data returned",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    # Ensure expected columns exist
    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            return {
                "ticker": ticker, "timeframe": timeframe, "bar_count": len(df),
                "signals": [], "latest_bar": None, "data_source": data_source,
                "error": f"Missing column: {col}",
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
    df = df.dropna(subset=["open", "high", "low", "close"]).copy()

    all_signals: list[dict] = []
    if "breakout" in detectors:
        all_signals.extend(_detect_breakout(df, ticker, window=breakout_window))
    if "rsi" in detectors:
        all_signals.extend(_detect_rsi_cross(df, ticker, oversold=rsi_oversold, overbought=rsi_overbought))
    if "macd" in detectors:
        all_signals.extend(_detect_macd_cross(df, ticker))
    if "pullback" in detectors:
        all_signals.extend(_detect_pullback(df, ticker, ema_span=ema_span, trend_span=trend_span))
    if "alert" in detectors and thresholds:
        all_signals.extend(_detect_threshold_alerts(df, ticker, thresholds))

    # Latest bar summary
    last = df.iloc[-1]
    ts_val = last.name
    ts_str = ts_val.isoformat() if hasattr(ts_val, "isoformat") else str(ts_val)
    latest_bar = {
        "ts": ts_str,
        "open": _safe_float(last.get("open") if hasattr(last, "get") else last["open"]),
        "high": _safe_float(last.get("high") if hasattr(last, "get") else last["high"]),
        "low": _safe_float(last.get("low") if hasattr(last, "get") else last["low"]),
        "close": _safe_float(last.get("close") if hasattr(last, "get") else last["close"]),
        "volume": _safe_float(last.get("volume") if hasattr(last, "get") else getattr(last, "volume", None)),
    }

    return {
        "ticker": ticker,
        "timeframe": timeframe,
        "bar_count": len(df),
        "signals": all_signals,
        "signal_count": len(all_signals),
        "latest_bar": latest_bar,
        "data_source": data_source,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
