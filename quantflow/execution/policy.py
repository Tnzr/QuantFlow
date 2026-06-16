from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExecutionPolicy:
    require_approval: bool = True
    max_daily_notional: float = 5000.0
    max_orders_per_day: int = 20
    max_position_notional: float = 2000.0


@dataclass
class PolicyDecision:
    allow: bool
    reason: str


def evaluate_order_intent(
    notional: float,
    orders_today: int,
    position_notional_after: float,
    policy: ExecutionPolicy,
) -> PolicyDecision:
    if orders_today >= policy.max_orders_per_day:
        return PolicyDecision(False, "max_orders_per_day exceeded")
    if notional > policy.max_daily_notional:
        return PolicyDecision(False, "max_daily_notional exceeded")
    if position_notional_after > policy.max_position_notional:
        return PolicyDecision(False, "max_position_notional exceeded")
    if policy.require_approval:
        return PolicyDecision(False, "manual approval required")
    return PolicyDecision(True, "policy checks passed")
