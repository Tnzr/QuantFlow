"""Score-to-portfolio construction with cost-aware turnover control."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantflow.research.costs import CostModel

DEFAULT_CAPITAL = 10_000.0


def smooth_scores(scores: pd.Series, halflife: float) -> pd.Series:
    """Per-ticker EWM of scores over time (causal); reduces turnover."""
    if not halflife:
        return scores
    wide = scores.unstack()
    return wide.ewm(halflife=halflife, min_periods=1).mean().stack().reindex(scores.index)


def quantile_weights(scores: pd.Series, q: float = 0.2, long_only: bool = False) -> pd.Series:
    """Dollar-neutral top/bottom quantile weights (gross 2, or 1 if long-only)."""
    n = max(1, int(len(scores) * q))
    w = pd.Series(0.0, index=scores.index)
    w[scores.nlargest(n).index] = 1.0 / n
    if not long_only:
        w[scores.nsmallest(n).index] = -1.0 / n
    return w


def apply_no_trade_band(target: pd.Series, prev: pd.Series, band: float) -> pd.Series:
    """Keep the previous weight where the change is smaller than ``band``."""
    if not band:
        return target
    prev = prev.reindex(target.index).fillna(0.0)
    return target.where((target - prev).abs() >= band, prev)


def backtest_scores(
    scores: pd.Series,
    fwd_returns: pd.Series,
    cost: CostModel | None = None,
    q: float = 0.2,
    long_only: bool = False,
    halflife: float = 0.0,
    no_trade_band: float = 0.0,
    capital: float = DEFAULT_CAPITAL,
    adv_dollars: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Rebalance at every date in ``scores`` and hold until the next one.

    ``fwd_returns`` must be the return from each rebalance date to the next.
    Returns per-period gross, cost and net returns plus turnover.
    """
    cost = cost or CostModel()
    s = smooth_scores(scores.dropna(), halflife)
    prev = pd.Series(dtype=float)
    rows = []
    for d, g in s.groupby(level=0):
        g = g.droplevel(0)
        target = apply_no_trade_band(quantile_weights(g, q, long_only), prev, no_trade_band)
        trades = target.subtract(prev.reindex(target.index.union(prev.index)).fillna(0.0), fill_value=0.0)
        rr = fwd_returns.xs(d, level=0).reindex(target.index).fillna(0.0)
        adv = adv_dollars.loc[d] if adv_dollars is not None and d in adv_dollars.index else None
        c = cost.trading_cost(trades, capital, adv) + cost.borrow_cost(target)
        gross = float((target * rr).sum())
        rows.append((d, gross, c, gross - c, float(trades.abs().sum())))
        # drift weights with realised returns before the next rebalance
        drifted = target * (1 + rr)
        denom = 1 + gross
        prev = drifted / denom if denom > 0 else drifted
    return pd.DataFrame(rows, columns=["date", "gross", "cost", "net", "turnover"]).set_index("date")


def trend_long_flat(close: pd.Series, window: int = 200, step: int = 5, allow_short: bool = False, cost: CostModel | None = None) -> pd.DataFrame:
    """Single-asset moving-average trend sleeve evaluated every ``step`` bars."""
    cost = cost or CostModel()
    c = close.dropna()
    sig = np.sign(c / c.rolling(window).mean() - 1).reindex(c.index[::step]).dropna()
    pos = sig.clip(lower=-1 if allow_short else 0)
    fwd = c.reindex(c.index[::step]).pct_change().shift(-1).reindex(pos.index)
    to = pos.diff().abs().fillna(pos.abs())
    net = pos * fwd - to * cost.linear_bps / 1e4 - (pos < 0) * cost.borrow_bps_annual / 1e4 / cost.periods_per_year
    return pd.DataFrame({"gross": pos * fwd, "net": net, "turnover": to, "position": pos}).dropna()
