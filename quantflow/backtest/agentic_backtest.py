"""Point-in-time agentic backtest and dynamic-resilience evaluation.

1. ``compute_signal_frame`` runs the LangGraph agents bar-by-bar on data truncated
   to each bar (``created_at``/index <= bar timestamp), so no future bars or
   sentiment records leak into a decision.
2. ``simulate`` turns signals into trades with ATR stops, 1%-of-equity risk sizing
   and a slippage/cost model.
3. ``resilience_report`` stresses the strategy: walk-forward tuning with
   out-of-sample scoring, election windows, high-volatility regimes, cost
   sensitivity, and bootstrap drawdown.
"""
from __future__ import annotations

import itertools
import logging
import math
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from ..agents.gating import WeightAdjuster
from ..agents.graph import AgentGraph, SentimentFn
from ..election.calendar import elections
from ..execution.risk_guards import MAX_RISK_PCT

STRESS_PERIODS = {
    "gfc_2008": ("2008-09-01", "2009-03-31"),
    "euro_2011": ("2011-07-01", "2011-10-31"),
    "covid_2020": ("2020-02-15", "2020-05-31"),
    "bear_2022": ("2022-01-01", "2022-10-31"),
    "tariff_2025": ("2025-03-15", "2025-05-31"),
}

log = logging.getLogger(__name__)


@dataclass
class SimParams:
    entry_threshold: float = 0.6
    exit_threshold: float = 0.45
    max_hold_bars: int = 20
    stop_atr: float = 2.0
    risk_pct: float = MAX_RISK_PCT
    cost_bps: float = 5.0
    allow_short: bool = False


@dataclass
class SimResult:
    params: dict
    trades: int
    win_rate: float
    total_return: float
    cagr: float
    sharpe: float
    max_drawdown: float
    profit_factor: float
    exposure: float
    buy_hold_return: float
    override_share: float
    equity: list = field(default_factory=list)
    trade_log: list = field(default_factory=list)

    def summary(self) -> dict:
        d = asdict(self)
        d.pop("equity")
        d.pop("trade_log")
        return d


def compute_signal_frame(
    ticker: str,
    df: pd.DataFrame,
    start: Optional[str] = None,
    sentiment_fn: Optional[SentimentFn] = None,
    adjuster: Optional[WeightAdjuster] = None,
    warmup: int = 252,
) -> pd.DataFrame:
    graph = AgentGraph(sentiment_fn=sentiment_fn, adjuster=adjuster)
    idx = df.index
    first = max(warmup, int(idx.searchsorted(pd.Timestamp(start))) if start else warmup)
    rows = []
    for i in range(first, len(idx)):
        as_of = idx[i]
        out = graph.run(ticker, df.iloc[: i + 1], as_of=as_of)
        g = out["gate"]
        assert pd.Timestamp(out["as_of"]) <= as_of
        rows.append(
            {
                "as_of": as_of,
                "base": g["base_confidence"],
                "agent": g["agent_confidence"],
                "final": out["final_confidence"],
                "override": g["override_applied"],
                "vol_triggered": g["volatility_triggered"],
                "vol_ratio": out["volatility"]["vol_ratio"],
                "seasonal": out["seasonal"]["score"],
                "sentiment": out["sentiment"]["score"],
            }
        )
    return pd.DataFrame(rows).set_index("as_of")


def simulate(df: pd.DataFrame, signals: pd.DataFrame, p: SimParams, start_equity: float = 100_000.0, confidence_col: str = "final") -> SimResult:
    """Signal at bar t close -> fill at bar t+1 open (no same-bar lookahead)."""
    px = df.loc[signals.index]
    opens = px["open"].values if "open" in px else px["adj close"].values
    closes = px["adj close"].values
    raw_close = px["close"].values if "close" in px else closes
    adj = closes / raw_close
    opens = opens * adj
    highs = (px["high"].values * adj) if "high" in px else closes
    lows = (px["low"].values * adj) if "low" in px else closes
    atr = px["atr14"].values * adj if "atr14" in px else pd.Series(closes).pct_change().rolling(14).std().values * closes
    conf = signals[confidence_col].values
    overrides = signals["override"].values
    n = len(px)
    cash = start_equity
    pos = 0.0
    side = 0
    entry_px = stop = 0.0
    entry_i = 0
    entry_override = False
    equity = np.empty(n)
    trades = []
    cost = p.cost_bps / 1e4
    for i in range(n):
        if pos and i > 0:
            hit_stop = (side > 0 and lows[i] <= stop) or (side < 0 and highs[i] >= stop)
            exit_signal = (side > 0 and conf[i - 1] < p.exit_threshold) or (side < 0 and conf[i - 1] > 1 - p.exit_threshold)
            timed_out = i - entry_i >= p.max_hold_bars
            if hit_stop or exit_signal or timed_out:
                fill = stop if hit_stop else opens[i]
                fill = min(fill, opens[i]) if hit_stop and side > 0 else (max(fill, opens[i]) if hit_stop else fill)
                fill *= (1 - cost) if side > 0 else (1 + cost)
                pnl = side * pos * (fill - entry_px)
                cash += pnl
                trades.append({"entry": str(px.index[entry_i].date()), "exit": str(px.index[i].date()), "side": side,
                               "entry_px": entry_px, "exit_px": fill, "pnl": pnl, "ret": side * (fill / entry_px - 1),
                               "reason": "stop" if hit_stop else ("signal" if exit_signal else "time"), "override": entry_override})
                pos, side = 0.0, 0
        if not pos and i > 0 and i < n - 1 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0:
            want = 1 if conf[i - 1] >= p.entry_threshold else (-1 if p.allow_short and conf[i - 1] <= 1 - p.entry_threshold else 0)
            if want:
                entry_px = opens[i] * ((1 + cost) if want > 0 else (1 - cost))
                risk_per_share = p.stop_atr * atr[i - 1]
                qty = min(p.risk_pct * cash / risk_per_share, cash / entry_px)
                if qty > 0:
                    pos, side, entry_i, entry_override = qty, want, i, bool(overrides[i - 1])
                    stop = entry_px - want * risk_per_share
        if pos and i == n - 1:
            fill = closes[i] * ((1 - cost) if side > 0 else (1 + cost))
            pnl = side * pos * (fill - entry_px)
            cash += pnl
            trades.append({"entry": str(px.index[entry_i].date()), "exit": str(px.index[i].date()), "side": side,
                           "entry_px": entry_px, "exit_px": fill, "pnl": pnl, "ret": side * (fill / entry_px - 1),
                           "reason": "end_of_data", "override": entry_override})
            pos, side = 0.0, 0
        mark = side * pos * (closes[i] - entry_px) if pos else 0.0
        equity[i] = cash + mark
    eq = pd.Series(equity, index=px.index)
    rets = eq.pct_change().dropna()
    years = max((px.index[-1] - px.index[0]).days / 365.25, 1e-9)
    total = float(eq.iloc[-1] / start_equity - 1)
    tr = pd.DataFrame(trades, columns=["entry", "exit", "side", "entry_px", "exit_px", "pnl", "ret", "reason", "override"])
    wins = tr[tr["pnl"] > 0]["pnl"].sum() if len(tr) else 0.0
    losses = -tr[tr["pnl"] < 0]["pnl"].sum() if len(tr) else 0.0
    exposure_bars = sum(max(1, (pd.Timestamp(t["exit"]) - pd.Timestamp(t["entry"])).days) for t in trades)
    return SimResult(
        params=asdict(p),
        trades=len(tr),
        win_rate=float((tr["pnl"] > 0).mean()) if len(tr) else 0.0,
        total_return=total,
        cagr=float((1 + total) ** (1 / years) - 1) if total > -1 else -1.0,
        sharpe=float(rets.mean() / rets.std() * math.sqrt(252)) if rets.std() > 0 else 0.0,
        max_drawdown=float((eq / eq.cummax() - 1).min()),
        profit_factor=float(wins / losses) if losses > 0 else float("inf") if wins > 0 else 0.0,
        exposure=float(min(1.0, exposure_bars / max((px.index[-1] - px.index[0]).days, 1))),
        buy_hold_return=float(closes[-1] / closes[0] - 1),
        override_share=float(tr["override"].mean()) if len(tr) else 0.0,
        equity=[{"date": str(d.date()), "equity": round(float(v), 2)} for d, v in eq.iloc[::5].items()],
        trade_log=trades,
    )


def _objective(r: SimResult) -> float:
    """Risk-adjusted score; penalise deep drawdowns and too few trades."""
    if r.trades < 3:
        return -1e9
    return r.sharpe - 2.0 * max(0.0, -r.max_drawdown - 0.15)


DEFAULT_GRID = {
    "entry_threshold": [0.55, 0.6, 0.65, 0.7],
    "max_hold_bars": [10, 20, 40],
    "stop_atr": [1.5, 2.0, 3.0],
}


def tune(df: pd.DataFrame, signals: pd.DataFrame, grid: dict = DEFAULT_GRID, base: Optional[SimParams] = None) -> tuple[SimParams, SimResult]:
    base = base or SimParams()
    best: Optional[tuple[SimParams, SimResult]] = None
    keys = list(grid)
    for combo in itertools.product(*(grid[k] for k in keys)):
        p = SimParams(**{**asdict(base), **dict(zip(keys, combo))})
        r = simulate(df, signals, p)
        if best is None or _objective(r) > _objective(best[1]):
            best = (p, r)
    return best  # type: ignore[return-value]


def walk_forward(df: pd.DataFrame, signals: pd.DataFrame, train_years: int = 4, test_years: int = 1, grid: dict = DEFAULT_GRID) -> dict:
    """Tune on a rolling train window, then score the chosen params on the next unseen window."""
    folds = []
    start = signals.index[0]
    while True:
        train_end = start + pd.DateOffset(years=train_years)
        test_end = train_end + pd.DateOffset(years=test_years)
        if train_end >= signals.index[-1]:
            break
        tr_sig = signals[(signals.index >= start) & (signals.index < train_end)]
        te_sig = signals[(signals.index >= train_end) & (signals.index < test_end)]
        if len(tr_sig) < 200 or len(te_sig) < 60:
            break
        p, tr_res = tune(df, tr_sig, grid)
        te_res = simulate(df, te_sig, p)
        folds.append({
            "train": f"{tr_sig.index[0].date()}..{tr_sig.index[-1].date()}",
            "test": f"{te_sig.index[0].date()}..{te_sig.index[-1].date()}",
            "params": {k: getattr(p, k) for k in grid},
            "train_sharpe": tr_res.sharpe,
            "test_sharpe": te_res.sharpe,
            "test_return": te_res.total_return,
            "test_buy_hold": te_res.buy_hold_return,
            "test_max_dd": te_res.max_drawdown,
            "test_trades": te_res.trades,
        })
        start = start + pd.DateOffset(years=test_years)
    f = pd.DataFrame(folds)
    return {
        "folds": folds,
        "oos_mean_sharpe": float(f["test_sharpe"].mean()) if len(f) else 0.0,
        "oos_positive_share": float((f["test_return"] > 0).mean()) if len(f) else 0.0,
        "oos_beats_buy_hold_share": float((f["test_return"] > f["test_buy_hold"]).mean()) if len(f) else 0.0,
        "sharpe_decay": float((f["train_sharpe"] - f["test_sharpe"]).mean()) if len(f) else 0.0,
    }


def _window_trades(trades: list[dict], windows: list[tuple[pd.Timestamp, pd.Timestamp]]) -> pd.DataFrame:
    t = pd.DataFrame(trades)
    if t.empty:
        return t
    entry = pd.to_datetime(t["entry"])
    mask = np.zeros(len(t), dtype=bool)
    for a, b in windows:
        mask |= ((entry >= a) & (entry <= b)).values
    return t[mask]


def _trade_stats(t: pd.DataFrame) -> dict:
    if t.empty:
        return {"trades": 0, "win_rate": None, "avg_ret": None, "sum_pnl": 0.0}
    return {"trades": int(len(t)), "win_rate": float((t["pnl"] > 0).mean()), "avg_ret": float(t["ret"].mean()), "sum_pnl": float(t["pnl"].sum())}


def bootstrap_drawdown(trades: list[dict], start_equity: float = 100_000.0, n: int = 1000, seed: int = 7) -> dict:
    pnl = np.array([t["pnl"] for t in trades])
    if len(pnl) < 5:
        return {"p50": None, "p95": None, "n_trades": int(len(pnl))}
    rng = np.random.default_rng(seed)
    dds = []
    for _ in range(n):
        eq = start_equity + np.cumsum(rng.choice(pnl, size=len(pnl), replace=True))
        peak = np.maximum.accumulate(np.concatenate([[start_equity], eq]))[1:]
        dds.append(float((eq / peak - 1).min()))
    return {"p50": float(np.percentile(dds, 50)), "p95": float(np.percentile(dds, 5)), "n_trades": int(len(pnl))}


def resilience_report(
    ticker: str,
    start: str = "2006-01-01",
    sentiment_fn: Optional[SentimentFn] = None,
    df: Optional[pd.DataFrame] = None,
    signals: Optional[pd.DataFrame] = None,
    params: Optional[SimParams] = None,
) -> dict:
    from ..features.indicators import compute_indicators, fetch_ohlcv

    if df is None:
        df = compute_indicators(fetch_ohlcv(ticker, period="max")).dropna(subset=["adj close"])
        if len(df) and df.index[0] > pd.Timestamp(start) + pd.Timedelta(days=30):
            log.warning("%s history starts %s, after requested start %s", ticker, df.index[0].date(), start)
    if signals is None:
        signals = compute_signal_frame(ticker, df, start=start, sentiment_fn=sentiment_fn)
    params = params or SimParams()
    gated = simulate(df, signals, params)
    baseline = simulate(df, signals, params, confidence_col="base")
    wf = walk_forward(df, signals)
    election_windows = [(pd.Timestamp(e.day) - pd.Timedelta(days=60), pd.Timestamp(e.day) + pd.Timedelta(days=60)) for e in elections(resolved_only=True)]
    stress_windows = {k: [(pd.Timestamp(a), pd.Timestamp(b))] for k, (a, b) in STRESS_PERIODS.items()}
    costs = {f"{bps}bps": simulate(df, signals, SimParams(**{**asdict(params), "cost_bps": bps})).summary() for bps in (0, 5, 15, 30)}
    return {
        "ticker": ticker.upper(),
        "period": f"{signals.index[0].date()}..{signals.index[-1].date()}",
        "requested_start": start,
        "params": asdict(params),
        "agentic": gated.summary(),
        "baseline_rules_only": baseline.summary(),
        "agent_uplift_sharpe": gated.sharpe - baseline.sharpe,
        "override_bars_share": float(signals["override"].mean()),
        "walk_forward": wf,
        "election_windows": _trade_stats(_window_trades(gated.trade_log, election_windows)),
        "stress_periods": {k: _trade_stats(_window_trades(gated.trade_log, w)) for k, w in stress_windows.items()},
        "cost_sensitivity": {k: {"total_return": v["total_return"], "sharpe": v["sharpe"]} for k, v in costs.items()},
        "bootstrap_drawdown": bootstrap_drawdown(gated.trade_log),
        "equity": gated.equity,
        "point_in_time": True,
    }
