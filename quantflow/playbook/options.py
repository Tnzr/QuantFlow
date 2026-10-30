"""Pick a long call / long put (or a debit vertical fallback) that fits the per-trade risk budget."""
from __future__ import annotations

import math

import pandas as pd

from ..options.greeks import black_scholes_greeks
from ..options.verticals import RISK_FREE, build_vertical

TARGET_DELTA = 0.45


def occ_symbol(ticker: str, expiry: str, right: str, strike: float) -> str:
    return f"{ticker.upper()}{pd.Timestamp(expiry):%y%m%d}{right.upper()}{round(strike * 1000):08d}"


def load_swing_chain(ticker: str, min_dte: int = 21, max_dte: int = 63, today: pd.Timestamp | None = None) -> pd.DataFrame:
    """Calls and puts for expiries inside the 3-week to 2-month swing window."""
    import yfinance as yf

    today = pd.Timestamp(today or pd.Timestamp.today().normalize())
    t = yf.Ticker(ticker)
    frames = []
    for ex in t.options:
        dte = (pd.Timestamp(ex) - today).days
        if dte < min_dte or dte > max_dte:
            continue
        ch = t.option_chain(ex)
        for right, df in (("C", ch.calls), ("P", ch.puts)):
            frames.append(pd.DataFrame({
                "ticker": ticker.upper(), "expiry": ex, "right": right, "strike": df["strike"].astype(float),
                "bid": df["bid"].fillna(0.0), "ask": df["ask"].fillna(0.0), "last": df["lastPrice"].fillna(0.0),
                "iv": df["impliedVolatility"].fillna(0.0), "volume": df["volume"].fillna(0).astype(int),
                "open_interest": df["openInterest"].fillna(0).astype(int),
            }))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def choose_contract(
    chain: pd.DataFrame,
    spot: float,
    side: str,
    budget: float,
    *,
    today: pd.Timestamp | None = None,
    delta_range: tuple[float, float] = (0.30, 0.60),
    max_rel_spread: float = 0.25,
    min_dte: int = 21,
    max_dte: int = 63,
) -> dict:
    """Long call (side=long) or long put (side=short) with |delta| in range and premium <= budget.

    Falls back to a debit vertical in the same direction when no single option is affordable.
    Max loss is always the premium paid, so the 1% cap is enforced by construction.
    """
    right = "C" if side == "long" else "P"
    if chain is None or chain.empty:
        return {"structure": None, "reason": "no option chain"}
    today = pd.Timestamp(today or pd.Timestamp.today().normalize())
    df = chain[(chain["right"] == right) & (chain["bid"] > 0) & (chain["ask"] > 0)].copy()
    df["dte"] = (pd.to_datetime(df["expiry"]) - today).dt.days
    df = df[(df["dte"] >= min_dte) & (df["dte"] <= max_dte)]
    df["mid"] = (df["bid"] + df["ask"]) / 2
    df = df[(df["ask"] - df["bid"]) / df["mid"] <= max_rel_spread]
    if not df.empty:
        df["delta"] = [
            abs(black_scholes_greeks(spot, k, RISK_FREE, 0.0, iv if iv > 0 else 0.3, d / 365, right).delta)
            for k, iv, d in zip(df["strike"], df["iv"], df["dte"])
        ]
        df = df[(df["delta"] >= delta_range[0]) & (df["delta"] <= delta_range[1])]
        df["cost"] = df["ask"] * 100
    affordable = df[df["cost"] <= budget] if not df.empty else df
    if not affordable.empty:
        r = affordable.iloc[(affordable["delta"] - TARGET_DELTA).abs().argsort().iloc[0]]
        qty = int(budget // r["cost"])
        return {
            "structure": "long_call" if right == "C" else "long_put",
            "contract": occ_symbol(r["ticker"], r["expiry"], right, r["strike"]),
            "expiry": r["expiry"], "strike": float(r["strike"]), "dte": int(r["dte"]),
            "limit_price": round(float(r["ask"]), 2), "qty": qty, "max_loss": round(qty * float(r["cost"]), 2),
            "delta": round(float(r["delta"]), 3), "iv": round(float(r["iv"]), 4),
            "premium_stop": round(0.5 * float(r["ask"]), 2),
        }
    spread = None
    for width in (0.05, 0.03, 0.02, 0.01):
        cand = build_vertical(chain, spot, side, min_dte=min_dte, max_dte=max_dte, width_pct=width, today=today)
        if cand is not None and 0 < cand.max_loss <= budget:
            spread = cand
            break
    if spread is not None:
        d = spread.to_dict()
        d["qty"] = int(budget // spread.max_loss)
        d["max_loss_total"] = round(d["qty"] * spread.max_loss, 2)
        d["legs_occ"] = [occ_symbol(spread.ticker, spread.expiry, spread.right, k) for k in (spread.long_strike, spread.short_strike)]
        return {**d, "spread_type": d.get("structure"), "structure": "vertical_debit",
                "reason": "single option over budget; defined-risk vertical in the same direction"}
    cheapest = float(df["cost"].min()) if not df.empty else math.nan
    return {"structure": None,
            "reason": f"cheapest {right} in delta range costs ${cheapest:.0f} vs ${budget:.0f} budget and no vertical fits"}
