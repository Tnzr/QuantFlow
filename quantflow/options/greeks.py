from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

from scipy.stats import norm


@dataclass
class Greeks:
    delta: float
    gamma: float
    theta: float
    vega: float


def black_scholes_greeks(S: float, K: float, r: float, q: float, sigma: float, T: float, right: str) -> Greeks:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return Greeks(0.0, 0.0, 0.0, 0.0)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if right.upper() == "C":
        delta = math.exp(-q * T) * norm.cdf(d1)
        theta = (
            - (S * math.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
            - r * K * math.exp(-r * T) * norm.cdf(d2)
            + q * S * math.exp(-q * T) * norm.cdf(d1)
        )
    else:
        delta = -math.exp(-q * T) * norm.cdf(-d1)
        theta = (
            - (S * math.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
            + r * K * math.exp(-r * T) * norm.cdf(-d2)
            - q * S * math.exp(-q * T) * norm.cdf(-d1)
        )

    gamma = math.exp(-q * T) * norm.pdf(d1) / (S * sigma * math.sqrt(T))
    vega = S * math.exp(-q * T) * norm.pdf(d1) * math.sqrt(T) / 100.0  # per 1 vol point = 0.01

    return Greeks(delta=delta, gamma=gamma, theta=theta/365.0, vega=vega)
