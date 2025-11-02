from __future__ import annotations

import pandas as pd
from ..features.indicators import fetch_ohlcv, compute_indicators


def simulate_equity_trend_follow(ticker: str, start: str = "2018-01-01") -> pd.DataFrame:
    df = fetch_ohlcv(ticker, period="max")
    df = compute_indicators(df)
    df = df[df.index >= pd.to_datetime(start)]

    pos = 0
    entry = 0.0
    pnl = []

    for date, row in df.iterrows():
        price = row["adj close"]
        if pos == 0:
            if price > row["sma50"] and row["rsi14"] > 50:
                pos = 1
                entry = price
        else:
            # exit when close < SMA20
            if price < row["sma20"]:
                pnl.append({"date": date, "pnl": (price/entry - 1)})
                pos = 0
    return pd.DataFrame(pnl)
