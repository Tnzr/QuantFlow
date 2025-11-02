from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Position:
    ticker: str
    qty: float
    avg_price: float
    side: str  # long/short


@dataclass
class Account:
    equity: float
    cash: float
    buying_power: float
