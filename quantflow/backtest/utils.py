from __future__ import annotations

import numpy as np
import pandas as pd


def equity_from_returns(daily_returns: pd.Series, start_equity: float = 1.0) -> pd.Series:
    eq = (1.0 + daily_returns.fillna(0)).cumprod() * start_equity
    eq.name = "equity"
    return eq


def max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    dd = 1.0 - (equity / roll_max).clip(lower=1e-12)
    return float(dd.max()) if len(dd) else 0.0


def sharpe(daily_returns: pd.Series, rf: float = 0.0) -> float:
    if daily_returns.empty:
        return 0.0
    excess = daily_returns - rf / 252.0
    return float(np.sqrt(252) * excess.mean() / (excess.std(ddof=1) + 1e-12))


def sortino(daily_returns: pd.Series, rf: float = 0.0) -> float:
    if daily_returns.empty:
        return 0.0
    excess = daily_returns - rf / 252.0
    downside = excess[excess < 0]
    denom = downside.std(ddof=1) + 1e-12
    return float(np.sqrt(252) * excess.mean() / denom)


def cagr(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def profit_factor(trade_returns: np.ndarray) -> float:
    gains = trade_returns[trade_returns > 0].sum()
    losses = -trade_returns[trade_returns < 0].sum()
    return float(gains / (losses + 1e-12))
