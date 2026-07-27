from __future__ import annotations

import os
import numpy as np
import pandas as pd
import yfinance as yf
import concurrent.futures


_INTERVAL_TO_ALPACA_TF = {
    "1m": "1Min",
    "5m": "5Min",
    "15m": "15Min",
    "30m": "30Min",
    "1h": "1Hour",
    "1d": "1Day",
    "1wk": "1Week",
    "1mo": "1Month",
}

_PERIOD_LOOKBACK_DAYS = {
    "1m": 7,
    "5m": 60,
    "15m": 90,
    "30m": 120,
    "1h": 365,
    "1d": 365 * 5,
    "1wk": 365 * 5,
    "1mo": 365 * 10,
}


def _fetch_ohlcv_alpaca(ticker: str, period: str, interval: str) -> pd.DataFrame | None:
    """Try to fetch OHLCV via Alpaca. Returns None on failure."""
    from quantflow.data.alpaca_client import is_configured, fetch_bars
    from datetime import datetime, timedelta, timezone

    if not is_configured():
        return None

    tf = _INTERVAL_TO_ALPACA_TF.get(interval)
    if tf is None:
        return None

    days = _PERIOD_LOOKBACK_DAYS.get(interval, 365)
    # honor explicit period for very long windows
    if isinstance(period, str) and period.endswith("y"):
        try:
            days = int(period[:-1]) * 365
        except ValueError:
            pass

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    try:
        df = fetch_bars(
            ticker,
            timeframe=tf,
            start=start.isoformat(),
            end=end.isoformat(),
            limit=10000,
            feed="iex",
        )
    except Exception:
        return None
    if df is None or df.empty:
        return None
    return df


def _fetch_ohlcv_yfinance(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """yfinance fallback path."""
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    return normalize_ohlcv(df)


def fetch_ohlcv(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV with Alpaca-first, yfinance fallback.

    Set ALPACA_PREFER_YFINANCE=1 to force the yfinance path.
    """
    prefer_yf = os.environ.get("ALPACA_PREFER_YFINANCE", "").strip() in ("1", "true", "yes")

    if not prefer_yf:
        df = _fetch_ohlcv_alpaca(ticker, period, interval)
        if df is not None and not df.empty:
            return normalize_ohlcv(df)

    return _fetch_ohlcv_yfinance(ticker, period, interval)


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # yfinance may return MultiIndex columns, e.g. ("Adj Close", "AAPL").
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [str(c[0]).strip().lower() for c in out.columns.to_list()]
    else:
        out.columns = [str(c).strip().lower() for c in out.columns]

    # Common column aliases.
    out = out.rename(
        columns={
            "adjclose": "adj close",
            "adj_close": "adj close",
            "adj. close": "adj close",
        }
    )

    # Ensure required OHLCV compatibility.
    if "close" not in out.columns and "adj close" in out.columns:
        out["close"] = out["adj close"]
    if "adj close" not in out.columns and "close" in out.columns:
        out["adj close"] = out["close"]
    if "open" not in out.columns and "close" in out.columns:
        out["open"] = out["close"]
    if "high" not in out.columns and "close" in out.columns:
        out["high"] = out["close"]
    if "low" not in out.columns and "close" in out.columns:
        out["low"] = out["close"]
    if "volume" not in out.columns:
        out["volume"] = 0.0

    out.index = pd.to_datetime(out.index)
    out.index.name = "date"
    return out


def fetch_ohlcv_parallel(tickers: list[str], period: str = "5y", interval: str = "1d", max_workers: int = 8) -> dict[str, pd.DataFrame]:
    """
    Download OHLCV data for multiple tickers in parallel.
    Returns a dict of ticker -> DataFrame.
    """
    def fetch_one(ticker):
        try:
            return ticker, fetch_ohlcv(ticker, period=period, interval=interval)
        except Exception as e:
            return ticker, None
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for ticker, df in executor.map(fetch_one, tickers):
            results[ticker] = df
    return results


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Avoid recomputing if columns already exist
    if "ret" not in out.columns:
        out["ret"] = out["adj close"].pct_change()
    if "rsi14" not in out.columns:
        out["rsi14"] = rsi(out["adj close"], 14)
    if "sma20" not in out.columns:
        out["sma20"] = out["adj close"].rolling(20).mean()
    if "sma50" not in out.columns:
        out["sma50"] = out["adj close"].rolling(50).mean()
    if "sma200" not in out.columns:
        out["sma200"] = out["adj close"].rolling(200).mean()
    if "atr14" not in out.columns:
        out["atr14"] = atr(out["high"], out["low"], out["close"], 14)
    if "vol20" not in out.columns:
        out["vol20"] = out["ret"].rolling(20).std() * np.sqrt(252)
    return out


def compute_indicators_parallel(dfs: dict[str, pd.DataFrame], max_workers: int = 8) -> dict[str, pd.DataFrame]:
    """
    Compute indicators for multiple DataFrames in parallel.
    Returns a dict of ticker -> DataFrame with indicators.
    """
    def compute_one(item):
        ticker, df = item
        if df is not None:
            return ticker, compute_indicators(df)
        return ticker, None
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for ticker, out in executor.map(compute_one, dfs.items()):
            results[ticker] = out
    return results


def rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
    return tr.ewm(alpha=1/period, adjust=False).mean()
