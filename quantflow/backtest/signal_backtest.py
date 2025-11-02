from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd

from ..features.indicators import fetch_ohlcv, compute_indicators
from ..features.exit_rules import simple_exit_plan
from .utils import equity_from_returns, max_drawdown, sharpe, sortino, cagr, profit_factor


@dataclass
class Trade:
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    ret: float


@dataclass
class BacktestReport:
    n_trades: int
    win_rate: float
    avg_ret: float
    median_ret: float
    avg_hold_days: float
    max_dd: float
    equity_curve: pd.Series
    sharpe: float
    sortino: float
    cagr: float
    profit_factor: float


def backtest_short_term(ticker: str, start: str = "2018-01-01") -> BacktestReport:
    """Simple backtest of 1w-style logic: enter when close > SMA20 and RSI>50; exit on SMA20 cross down or 5-day time stop.
    Uses daily bars as proxy.
    """
    df = fetch_ohlcv(ticker, period="max")
    df = compute_indicators(df)
    df = df[df.index >= pd.to_datetime(start)]
    df = df.dropna(subset=["sma20","rsi14"]).copy()

    trades: List[Trade] = []
    in_pos = False
    entry_price = 0.0
    entry_date = None

    # Build daily PnL series based on entries/exits (flat when no position)
    daily_ret = pd.Series(0.0, index=df.index, name="ret")

    for i in range(1, len(df)):
        row_prev = df.iloc[i-1]
        row = df.iloc[i]

        if not in_pos:
            if (row_prev["adj close"] <= row_prev["sma20"]) and (row["adj close"] > row["sma20"]) and row["rsi14"] > 50:
                in_pos = True
                entry_price = float(row["adj close"])
                entry_date = df.index[i]
        else:
            # mark daily return while in position
            daily_ret.iloc[i] = row["adj close"]/row_prev["adj close"] - 1
            # exit on close < SMA20 or after ~5 trading days (7 calendar days)
            hold_days = (df.index[i] - entry_date).days
            if (row["adj close"] < row["sma20"]) or hold_days >= 7:
                exit_price = float(row["adj close"])
                trades.append(Trade(entry_date, entry_price, df.index[i], exit_price, exit_price/entry_price - 1))
                in_pos = False

    equity = equity_from_returns(daily_ret)
    max_dd_val = max_drawdown(equity)
    rets = np.array([t.ret for t in trades]) if trades else np.array([])

    return BacktestReport(
        n_trades=len(trades),
        win_rate=float((rets > 0).mean()) if rets.size else 0.0,
        avg_ret=float(rets.mean()) if rets.size else 0.0,
        median_ret=float(np.median(rets)) if rets.size else 0.0,
        avg_hold_days=float(np.mean([(t.exit_date - t.entry_date).days for t in trades])) if trades else 0.0,
        max_dd=max_dd_val,
        equity_curve=equity,
        sharpe=sharpe(daily_ret),
        sortino=sortino(daily_ret),
        cagr=cagr(equity),
        profit_factor=profit_factor(rets) if rets.size else 0.0,
    )
