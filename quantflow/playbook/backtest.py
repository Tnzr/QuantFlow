"""Replay the rules tree after each resolved midterm (leave-one-election-out impact priors).

For election e, the impact screen is recomputed *without* e, the tree runs on daily bars
from the first post-election session for ``window`` bars with the true Senate outcome,
and each confirmed trade is closed by the tree's own exits (or at the window end).
Compared with a naive entry at the first post-election close in the same direction,
held ``time_stop`` bars. Approximate option P&L prices a 45-DTE ATM option with
Black–Scholes at 1.1x trailing 20-day realised vol at entry and exit.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..election.calendar import elections
from ..options.verticals import bs_price
from .impact import event_moves, score_symbol
from .rules import RuleConfig, SignalState, indicators, step


def _option_ret(entry: float, exit_: float, sigma: float, held: int, side: str) -> float:
    right = "C" if side == "long" else "P"
    p0 = bs_price(entry, entry, 45 / 365, sigma, right)
    p1 = bs_price(exit_, entry, max(45 - held * 7 / 5, 0) / 365, sigma, right)
    return p1 / p0 - 1 if p0 > 0 else 0.0


def replay(bars: pd.DataFrame, closes: pd.DataFrame, symbol: str, benchmark: str, group: str,
           window: int = 30, cfg: RuleConfig | None = None) -> list[dict]:
    cfg = cfg or RuleConfig()
    evs = elections(kind="midterm", resolved_only=True)
    bench = closes[benchmark] if benchmark else None
    rows = []
    for ev in evs:
        others = [e for e in evs if e.day != ev.day]
        imp = score_symbol(event_moves(closes, symbol, benchmark, others), symbol, benchmark, group, 0.5)
        if imp is None:
            continue
        imp = imp.to_dict()
        start = int(bars.index.searchsorted(pd.Timestamp(ev.day)))
        if start < 200 or start + window >= len(bars):
            continue
        z = imp["z_dem"] if ev.senate_after == "D" else imp["z_rep"]
        side = "long" if z > 0 else "short"
        sgn = 1 if side == "long" else -1
        c0 = float(bars["close"].iloc[start])
        cN = float(bars["close"].iloc[min(start + cfg.time_stop, len(bars) - 1)])
        sigma = float(bars["close"].pct_change().iloc[start - 20:start].std() * math.sqrt(252) * 1.1)
        naive = sgn * (cN / c0 - 1)
        st, trade = SignalState(symbol), None
        for i in range(start, start + window):
            ind = indicators(bars.iloc[: i + 1], bench, None, cfg)
            st, alerts = step(st, ind, imp, ev.senate_after, cfg)
            for a in alerts:
                if a.kind == "entry_confirmed" and trade is None:
                    trade = {"entry_bar": i - start, "entry": a.price, "stop0": a.stop, "trade_side": a.side}
                elif a.kind == "exit" and trade is not None and "exit" not in trade:
                    trade.update(exit=a.price, reason=st.exit_reason, held=st.bars_held)
            if trade is not None and "exit" in trade:
                break
        if trade is not None and "exit" not in trade:
            trade.update(exit=float(bars["close"].iloc[start + window - 1]), reason="window_end", held=st.bars_held)
        row = {"symbol": symbol, "election": ev.day.year, "senate_after": ev.senate_after, "side": side,
               "naive_ret": naive, "naive_option_ret": _option_ret(c0, cN, sigma, cfg.time_stop, side),
               "confirmed": trade is not None}
        if trade:
            risk = abs(trade["entry"] - trade["stop0"]) or np.nan
            ts = 1 if trade["trade_side"] == "long" else -1
            row.update(trade, ret=ts * (trade["exit"] / trade["entry"] - 1),
                       r_multiple=ts * (trade["exit"] - trade["entry"]) / risk,
                       option_ret=_option_ret(trade["entry"], trade["exit"], sigma, trade["held"], trade["trade_side"]))
        rows.append(row)
    return rows


def summarize(rows: list[dict]) -> dict:
    df = pd.DataFrame(rows)
    if df.empty:
        return {"n_events": 0}
    t = df[df["confirmed"]]
    return {
        "n_events": len(df), "n_trades": len(t), "confirm_rate": len(t) / len(df),
        "tree_hit_rate": float((t["ret"] > 0).mean()) if len(t) else None,
        "tree_mean_ret": float(t["ret"].mean()) if len(t) else None,
        "tree_mean_r": float(t["r_multiple"].mean()) if len(t) else None,
        "tree_mean_option_ret": float(t["option_ret"].mean()) if len(t) else None,
        "naive_hit_rate": float((df["naive_ret"] > 0).mean()),
        "naive_mean_ret": float(df["naive_ret"].mean()),
        "naive_mean_option_ret": float(df["naive_option_ret"].mean()),
    }
