from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
import concurrent.futures


def fetch_ohlcv(ticker: str, period: str = "6m", interval: str = "1wk") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    df = df.rename(columns=str.lower)
    df.index.name = "date"
    return df


def fetch_ohlcv_parallel(tickers: list[str], period: str = "6m", interval: str = "1wk", max_workers: int = 8) -> dict[str, pd.DataFrame]:
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
