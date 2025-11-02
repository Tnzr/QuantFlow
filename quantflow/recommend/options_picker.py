from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import math
import pandas as pd
import yfinance as yf

from ..options.chain import load_chain
from ..options.greeks import black_scholes_greeks


@dataclass
class OptionIdea:
    ticker: str
    horizon: str
    bias: str  # long/short
    expiry: str
    right: str  # C/P
    strike: float
    bid: float
    ask: float
    mid: float
    volume: int
    open_interest: int
    iv: float
    delta_band: str
    rationale: str


HORIZON_DTE = {
    "1w": (5, 10),
    "1m": (25, 45),
    "3m": (55, 95),
    "6m": (115, 200),
    "1y": (230, 420),
}


def dte_from_expiry(expiry: str, today: Optional[pd.Timestamp] = None) -> int:
    today = pd.Timestamp.today().normalize() if today is None else today
    ex = pd.Timestamp(expiry).normalize()
    return max(0, (ex - today).days)


def pick_affordable_contracts(
    ticker: str,
    horizon: str,
    bias: str,
    budget: float = 300.0,
    max_spread_abs: float = 0.10,
    max_spread_pct: float = 0.08,
    min_oi: int = 200,
    min_vol: int = 50,
) -> List[OptionIdea]:
    """Pick single-leg contracts (calls for long, puts for short) matching liquidity and budget."""
    chain = load_chain(ticker)
    if chain.empty:
        return []

    # Pull underlying spot
    spot = None
    try:
        spot = float(yf.Ticker(ticker).fast_info.get("last_price"))
    except Exception:
        try:
            spot = float(yf.Ticker(ticker).history(period="5d")["Close"].iloc[-1])
        except Exception:
            spot = None

    dte_lo, dte_hi = HORIZON_DTE.get(horizon, (25, 45))
    chain = chain.copy()
    chain["dte"] = chain["expiry"].apply(lambda x: dte_from_expiry(x))
    right = "C" if bias == "long" else "P"
    side = chain[(chain["right"] == right) & (chain["dte"].between(dte_lo, dte_hi))]

    # Price filters
    side["mid"] = (side["bid"].fillna(0) + side["ask"].fillna(0)) / 2.0
    side = side[(side["ask"] > 0) & (side["mid"] > 0) & (side["ask"] <= budget)]

    # Liquidity filters
    side = side[(side["open_interest"] >= min_oi) & (side["volume"] >= min_vol)]

    # Spread quality
    side["spread"] = (side["ask"] - side["bid"]).clip(lower=0)
    side = side[(side["spread"] <= max_spread_abs) | (side["spread"] / side["mid"] <= max_spread_pct)]

    # Compute delta approximation via Black-Scholes (requires IV and spot)
    if spot is not None:
        r = 0.03  # flat risk-free
        q = 0.0   # dividend yield unknown
        def _delta(row):
            T = max(1e-6, row["dte"]/365.0)
            sigma = max(1e-6, float(row["iv"]) or 0.4)
            g = black_scholes_greeks(spot, float(row["strike"]), r, q, sigma, T, right)
            return g.delta
        side["delta"] = side.apply(_delta, axis=1)
        # Desired band: ~0.30-0.40 OTM for convexity and decay balance
        if right == "C":
            side["delta_score"] = (side["delta"] - 0.35).abs()
        else:
            side["delta_score"] = (abs(side["delta"]) - 0.35).abs()
    else:
        side["delta_score"] = 0.5  # neutral if unknown

    # Rank by: dte closeness to mid of range, lowest spread pct, highest OI, mid price closest to half budget, delta score
    dte_mid = (dte_lo + dte_hi) / 2
    side["rank"] = (
        (side["spread"] / side["mid"]).rank(method="first") * 0.35
        + (side["open_interest"].rank(ascending=False, method="first")) * 0.25
        + (side["volume"].rank(ascending=False, method="first")) * 0.15
        + (side["dte"].sub(dte_mid).abs().rank(method="first")) * 0.1
        + (side["delta_score"].rank(method="first")) * 0.15
    )
    side = side.sort_values("rank")

    ideas: List[OptionIdea] = []
    for _, r in side.head(5).iterrows():
        ideas.append(
            OptionIdea(
                ticker=ticker,
                horizon=horizon,
                bias=bias,
                expiry=str(r["expiry"]),
                right=right,
                strike=float(r["strike"]),
                bid=float(r["bid"] or 0.0),
                ask=float(r["ask"] or 0.0),
                mid=float(r["mid"] or 0.0),
                volume=int(r["volume"] or 0),
                open_interest=int(r["open_interest"] or 0),
                iv=float(r["iv"] or 0.0),
                delta_band="~0.35 target" if spot is not None else "ATM/OTM heuristic",
                rationale="Budget+liquidity+spread filters; delta-targeted with BS greeks",
            )
        )
    return ideas
