from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import pandas as pd


@dataclass
class ExitPlan:
    trailing_stop: float
    hard_stop: float
    time_stop_days: int
    notes: str


def atr_trailing_stop(prices: pd.Series, atr: pd.Series, factor: float = 2.0) -> float:
    """Last price minus ATR-based offset for long; for short, caller can invert.
    Returns trailing stop level for long positions.
    """
    last_price = float(prices.iloc[-1])
    last_atr = float(atr.iloc[-1]) if pd.notna(atr.iloc[-1]) else last_price * 0.02
    return last_price - factor * last_atr


def simple_exit_plan(df: pd.DataFrame, bias: str = "long", atr_factor: float = 2.0, time_stop_days: int = 10) -> ExitPlan:
    """Generate a basic exit plan combining trailing ATR stop, a hard stop via SMA20, and a time stop.
    df must include columns: 'adj close', 'atr14', 'sma20'
    """
    price = df["adj close"]
    atr = df["atr14"]
    sma20 = df["sma20"].iloc[-1]

    if bias == "long":
        trailing = atr_trailing_stop(price, atr, atr_factor)
        hard = float(sma20) if pd.notna(sma20) else trailing
    else:
        # For shorts, mirror levels around price
        last = float(price.iloc[-1])
        trailing = last + atr.iloc[-1] * atr_factor if pd.notna(atr.iloc[-1]) else last * 1.02
        hard = float(sma20) if pd.notna(sma20) else trailing

    return ExitPlan(trailing_stop=trailing, hard_stop=hard, time_stop_days=time_stop_days, notes="ATR trail + SMA20 hard stop + time stop")
