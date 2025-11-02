from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf


def fetch_ohlcv(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    df = df.rename(columns=str.lower)
    df.index.name = "date"
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret"] = out["adj close"].pct_change()
    out["rsi14"] = rsi(out["adj close"], 14)
    out["sma20"] = out["adj close"].rolling(20).mean()
    out["sma50"] = out["adj close"].rolling(50).mean()
    out["sma200"] = out["adj close"].rolling(200).mean()
    out["atr14"] = atr(out["high"], out["low"], out["close"], 14)
    out["vol20"] = out["ret"].rolling(20).std() * np.sqrt(252)
    return out


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
