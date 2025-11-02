from __future__ import annotations

import pandas as pd
import yfinance as yf
from datetime import timedelta


def seasonality_by_doy(ticker: str, years: int = 10) -> pd.DataFrame:
    df = yf.download(ticker, period=f"{years}y", interval="1d", progress=False)
    df = df.rename(columns=str.lower)
    df.index.name = "date"
    df["ret"] = df["adj close"].pct_change()
    df["doy"] = df.index.dayofyear
    agg = df.groupby("doy")["ret"].agg(["mean", "std", "count"]).reset_index()
    agg = agg.rename(columns={"mean": "avg_ret", "std": "std_ret", "count": "n"})
    return agg


def next_earnings_hint(ticker: str):
    t = yf.Ticker(ticker)
    try:
        cal = t.calendar
        if cal is not None and "Earnings Date" in cal.index:
            vals = cal.loc["Earnings Date"].values
            if len(vals) >= 1:
                return pd.to_datetime(vals[0]).date()
    except Exception:
        pass
    return None


def proximity_to_earnings(ticker: str, today=None):
    today = pd.Timestamp.today().date() if today is None else today
    d = next_earnings_hint(ticker)
    if d is None:
        return None
    return (d - today).days
