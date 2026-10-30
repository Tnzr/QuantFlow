"""Defined-risk vertical debit spreads and post-election IV-crush modelling.

Before an election, implied vol carries an event premium that typically collapses
once results are known. A long single option is short that premium (negative vega);
a debit vertical is roughly vega-neutral, so it is preferred when IV is elevated and
an election falls inside the option's life.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..election.calendar import elections, election_in_window

RISK_FREE = 0.04


def bs_price(S: float, K: float, T: float, sigma: float, right: str, r: float = RISK_FREE) -> float:
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K) if right.upper() == "C" else max(0.0, K - S)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if right.upper() == "C":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def historical_vix_crush(vix: pd.Series, kind: str = "midterm", days_after: int = 5) -> dict:
    """Fractional VIX change from the pre-election close to ``days_after`` sessions later."""
    changes = {}
    if getattr(vix.index, "tz", None) is not None:
        vix = vix.tz_localize(None)
    for ev in elections(kind=kind, resolved_only=True):
        pos = vix.index.searchsorted(pd.Timestamp(ev.day)) - 1
        if pos < 0 or pos + days_after >= len(vix):
            continue
        changes[ev.day.year] = float(vix.iloc[pos + days_after] / vix.iloc[pos] - 1)
    vals = list(changes.values())
    return {
        "by_year": changes,
        "median_change": float(np.median(vals)) if vals else 0.0,
        "mean_change": float(np.mean(vals)) if vals else 0.0,
        "n": len(vals),
    }


@dataclass
class VerticalSpread:
    ticker: str
    expiry: str
    right: str  # C (bull call) or P (bear put)
    long_strike: float
    short_strike: float
    debit: float
    max_loss: float  # per spread (x100)
    max_gain: float
    breakeven: float
    reward_risk: float
    long_iv: float
    short_iv: float
    dte: int
    contracts: int = 1

    @property
    def structure(self) -> str:
        return "bull_call_vertical" if self.right == "C" else "bear_put_vertical"

    def legs(self) -> list[dict]:
        return [
            {"action": "buy", "right": self.right, "strike": self.long_strike, "expiry": self.expiry, "qty": self.contracts},
            {"action": "sell", "right": self.right, "strike": self.short_strike, "expiry": self.expiry, "qty": self.contracts},
        ]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["structure"] = self.structure
        d["legs"] = self.legs()
        return d


def _mid(row) -> float:
    bid, ask = float(row.get("bid") or 0), float(row.get("ask") or 0)
    return (bid + ask) / 2 if bid > 0 and ask > 0 else float(row.get("last") or 0)


def build_vertical(
    chain: pd.DataFrame,
    spot: float,
    bias: str,
    *,
    min_dte: int = 21,
    max_dte: int = 75,
    width_pct: float = 0.05,
    max_rel_spread: float = 0.25,
    today: Optional[pd.Timestamp] = None,
) -> Optional[VerticalSpread]:
    """Near-ATM debit vertical: buy ~ATM, sell ~``width_pct`` OTM, best reward/risk."""
    if chain is None or chain.empty or bias not in ("long", "short"):
        return None
    right = "C" if bias == "long" else "P"
    today = pd.Timestamp(today or pd.Timestamp.today().normalize())
    df = chain[chain["right"] == right].copy()
    df["dte"] = (pd.to_datetime(df["expiry"]) - today).dt.days
    df = df[(df["dte"] >= min_dte) & (df["dte"] <= max_dte)]
    if df.empty:
        return None
    df["mid"] = df.apply(_mid, axis=1)
    df = df[df["mid"] > 0]
    quoted = (df["bid"] > 0) & (df["ask"] > 0)
    tight = (df["ask"] - df["bid"]).abs() / df["mid"].clip(lower=0.01) <= max_rel_spread
    df = df[tight & quoted]
    best: Optional[VerticalSpread] = None
    for expiry, grp in df.groupby("expiry"):
        grp = grp.sort_values("strike")
        long_leg = grp.iloc[(grp["strike"] - spot).abs().argsort().iloc[0]]
        target = spot * (1 + width_pct) if right == "C" else spot * (1 - width_pct)
        side = grp[grp["strike"] > long_leg["strike"]] if right == "C" else grp[grp["strike"] < long_leg["strike"]]
        if side.empty:
            continue
        short_leg = side.iloc[(side["strike"] - target).abs().argsort().iloc[0]]
        width = abs(float(short_leg["strike"]) - float(long_leg["strike"]))
        debit = float(long_leg["mid"]) - float(short_leg["mid"])
        if debit <= 0 or debit >= width:
            continue
        rr = (width - debit) / debit
        be = float(long_leg["strike"]) + debit if right == "C" else float(long_leg["strike"]) - debit
        cand = VerticalSpread(
            ticker=str(long_leg["ticker"]), expiry=str(expiry), right=right,
            long_strike=float(long_leg["strike"]), short_strike=float(short_leg["strike"]),
            debit=round(debit, 2), max_loss=round(debit * 100, 2), max_gain=round((width - debit) * 100, 2),
            breakeven=round(be, 2), reward_risk=round(rr, 3),
            long_iv=float(long_leg.get("iv") or 0), short_iv=float(short_leg.get("iv") or 0), dte=int(long_leg["dte"]),
        )
        if best is None or cand.reward_risk > best.reward_risk:
            best = cand
    return best


def iv_crush_scenario(
    spot: float,
    strike: float,
    short_strike: Optional[float],
    iv: float,
    dte: int,
    right: str,
    crush: float,
    days_after: int = 5,
) -> dict:
    """Value change of a long option vs a debit vertical if IV falls by ``crush``
    (fraction, e.g. -0.2) over ``days_after`` days with spot unchanged."""
    T0, T1 = dte / 365.0, max(dte - days_after, 0) / 365.0
    iv1 = max(iv * (1 + crush), 0.01)
    single0, single1 = bs_price(spot, strike, T0, iv, right), bs_price(spot, strike, T1, iv1, right)
    out = {"iv_before": iv, "iv_after": iv1, "single_pnl_pct": (single1 / single0 - 1) if single0 > 0 else 0.0}
    if short_strike is not None:
        v0 = single0 - bs_price(spot, short_strike, T0, iv, right)
        v1 = single1 - bs_price(spot, short_strike, T1, iv1, right)
        out["vertical_pnl_pct"] = (v1 / v0 - 1) if v0 > 0 else 0.0
    return out


def prefer_defined_risk(atm_iv: float, realized_vol: float, as_of=None, window_days: int = 60, iv_premium: float = 1.15) -> dict:
    """Prefer verticals when an election is inside the window and IV > premium x realized vol."""
    in_window = election_in_window(as_of or pd.Timestamp.today(), window_days)
    ratio = atm_iv / realized_vol if realized_vol > 0 else float("inf")
    prefer = bool(in_window and ratio >= iv_premium) or ratio >= 1.5
    return {"prefer_vertical": prefer, "iv_rv_ratio": float(ratio), "election_window": bool(in_window)}


def recommend_structure(ticker: str, bias: str, equity: float, max_risk_pct: float = 0.01, chain: Optional[pd.DataFrame] = None) -> dict:
    """Live helper: chain -> vertical sized to the portfolio-risk cap, plus crush comparison."""
    from ..features.indicators import fetch_ohlcv
    from .chain import load_chain

    import yfinance as yf

    px = fetch_ohlcv(ticker, period="1y")["adj close"].dropna()
    spot = float(px.iloc[-1])
    rv = float(px.pct_change().tail(20).std() * math.sqrt(252))
    chain = chain if chain is not None else load_chain(ticker)
    spread = build_vertical(chain, spot, bias)
    if spread is None:
        return {"ticker": ticker, "spot": spot, "error": "no liquid vertical found"}
    atm_iv, iv_source = spread.long_iv, "chain"
    if not (0.5 * rv <= atm_iv <= 4 * rv):
        atm_iv, iv_source = rv, "realized_vol_fallback"
    pref = prefer_defined_risk(atm_iv, rv)
    try:
        vix = yf.Ticker("^VIX").history(period="30y")["Close"]
        crush = historical_vix_crush(vix)
    except Exception:
        crush = {"median_change": -0.15, "n": 0, "by_year": {}}
    scen = iv_crush_scenario(spot, spread.long_strike, spread.short_strike, atm_iv, spread.dte, spread.right, crush["median_change"])
    spread.contracts = int(max_risk_pct * equity // spread.max_loss) if spread.max_loss > 0 else 0
    return {
        "ticker": ticker,
        "spot": spot,
        "realized_vol_20d": rv,
        "spread": spread.to_dict(),
        "preference": pref,
        "iv_source": iv_source,
        "iv_crush": {"historical": crush, "scenario": scen},
        "risk": {"equity": equity, "max_risk_pct": max_risk_pct, "risk_amount": spread.contracts * spread.max_loss},
    }
