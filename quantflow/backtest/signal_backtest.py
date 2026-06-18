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


def backtest_short_term(
    ticker: str,
    start: str = "2018-01-01",
    end: str | None = None,
    entry_rsi_threshold: float = 50.0,
    max_hold_days: int = 7,
    stop_loss_pct: float = 0.0,
    ma_filter: str = "sma20",
    take_profit_pct: float = 0.0,
    ma_trend_filter: str = "none",
) -> BacktestReport:
    """Simple backtest of 1w-style logic: enter on close crossing above MA and RSI filter.
    Exit on MA cross down, optional stop-loss, take-profit, or time stop.
    
    Args:
        ma_trend_filter: trend condition for entries. Options: 'none', 'above_sma50', 'above_sma200'
    
    Uses daily bars as proxy.
    """
    ma_col = str(ma_filter or "sma20").lower()
    if ma_col not in {"sma20", "sma50", "sma200"}:
        raise ValueError("ma_filter must be one of: sma20, sma50, sma200")

    if ma_trend_filter not in {"none", "above_sma50", "above_sma200"}:
        raise ValueError("ma_trend_filter must be one of: none, above_sma50, above_sma200")

    df = fetch_ohlcv(ticker, period="max")
    df = compute_indicators(df)
    df = df[df.index >= pd.to_datetime(start)]
    if end:
        df = df[df.index <= pd.to_datetime(end)]
    
    required_cols = [ma_col, "rsi14"]
    if ma_trend_filter != "none":
        trend_ma = "sma50" if ma_trend_filter == "above_sma50" else "sma200"
        required_cols.append(trend_ma)
    
    df = df.dropna(subset=required_cols).copy()

    stop_loss_frac = max(0.0, float(stop_loss_pct)) / 100.0
    take_profit_frac = max(0.0, float(take_profit_pct)) / 100.0

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
            # Check entry conditions
            ma_crossover = (row_prev["adj close"] <= row_prev[ma_col]) and (row["adj close"] > row[ma_col])
            rsi_signal = row["rsi14"] > float(entry_rsi_threshold)
            
            # Apply trend filter if specified
            trend_ok = True
            if ma_trend_filter == "above_sma50":
                trend_ok = row["adj close"] > row["sma50"]
            elif ma_trend_filter == "above_sma200":
                trend_ok = row["adj close"] > row["sma200"]
            
            if ma_crossover and rsi_signal and trend_ok:
                in_pos = True
                entry_price = float(row["adj close"])
                entry_date = df.index[i]
        else:
            # mark daily return while in position
            daily_ret.iloc[i] = row["adj close"]/row_prev["adj close"] - 1
            # exit on close < selected MA, optional stop-loss, take-profit, or time stop
            hold_days = (df.index[i] - entry_date).days
            stop_loss_hit = stop_loss_frac > 0 and (float(row["adj close"]) <= entry_price * (1.0 - stop_loss_frac))
            take_profit_hit = take_profit_frac > 0 and (float(row["adj close"]) >= entry_price * (1.0 + take_profit_frac))
            ma_exit = row["adj close"] < row[ma_col]
            
            if ma_exit or stop_loss_hit or take_profit_hit or hold_days >= int(max_hold_days):
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
