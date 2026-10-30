"""Pre-trade guards: slippage cap and per-position portfolio-risk cap."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Optional

MAX_SLIPPAGE_PCT = float(os.getenv("QUANTFLOW_MAX_SLIPPAGE_PCT", "0.02"))
MAX_RISK_PCT = float(os.getenv("QUANTFLOW_MAX_RISK_PCT", "0.01"))


@dataclass
class RiskCheck:
    allow: bool
    reason: str
    slippage_pct: Optional[float]
    risk_amount: float
    risk_pct: Optional[float]
    max_slippage_pct: float
    max_risk_pct: float
    max_qty_allowed: Optional[float]

    def to_dict(self) -> dict:
        return asdict(self)


class RiskGuardError(RuntimeError):
    def __init__(self, check: RiskCheck):
        super().__init__(check.reason)
        self.check = check


def estimate_slippage_pct(side: str, reference_price: float, expected_fill: float) -> float:
    """Adverse move from reference (mid) to expected fill, as a fraction of reference."""
    if reference_price <= 0:
        return float("inf")
    move = (expected_fill - reference_price) if side == "buy" else (reference_price - expected_fill)
    return max(0.0, move / reference_price)


def check_order_risk(
    side: str,
    qty: float,
    entry_price: float,
    equity: float,
    *,
    reference_price: Optional[float] = None,
    stop_price: Optional[float] = None,
    max_loss_per_unit: Optional[float] = None,
    max_slippage_pct: float = MAX_SLIPPAGE_PCT,
    max_risk_pct: float = MAX_RISK_PCT,
) -> RiskCheck:
    """Allow an order only if slippage <= cap and worst-case loss <= ``max_risk_pct`` of equity.

    Worst-case loss per unit: ``max_loss_per_unit`` (defined-risk structures, e.g. spread
    debit x 100), else ``|entry - stop|``, else the full entry price (no stop = whole
    notional at risk).
    """
    side = side.lower()
    slip = estimate_slippage_pct(side, reference_price, entry_price) if reference_price else None
    if max_loss_per_unit is not None:
        per_unit = float(max_loss_per_unit)
    elif stop_price is not None and stop_price > 0:
        per_unit = abs(entry_price - stop_price)
    else:
        per_unit = float(entry_price)
    risk = per_unit * float(qty)
    risk_pct = risk / equity if equity and equity > 0 else None
    max_qty = (max_risk_pct * equity / per_unit) if equity and per_unit > 0 else None

    def result(allow: bool, reason: str) -> RiskCheck:
        return RiskCheck(allow, reason, slip, risk, risk_pct, max_slippage_pct, max_risk_pct, max_qty)

    if qty <= 0 or entry_price <= 0:
        return result(False, "invalid qty or price")
    if slip is not None and slip > max_slippage_pct:
        return result(False, f"slippage {slip:.2%} exceeds cap {max_slippage_pct:.2%}")
    if risk_pct is None:
        return result(False, "account equity unavailable; cannot size risk")
    if risk_pct > max_risk_pct:
        return result(False, f"position risk {risk_pct:.2%} of equity exceeds cap {max_risk_pct:.2%} (max qty {max_qty:.4g})")
    return result(True, "risk checks passed")


def size_for_risk(equity: float, per_unit_risk: float, max_risk_pct: float = MAX_RISK_PCT) -> float:
    if equity <= 0 or per_unit_risk <= 0:
        return 0.0
    return max_risk_pct * equity / per_unit_risk
