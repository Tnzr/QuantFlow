from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..broker.base import Broker
from ..broker.types import Position
from ..recommend.engine import RuleEngine


@dataclass
class PositionSignal:
    ticker: str
    side: str
    action: str   # hold/trim/exit/add/flip
    horizon: str
    confidence: float
    notes: str


def evaluate_positions(broker: Broker) -> List[PositionSignal]:
    broker.login()
    eng = RuleEngine()
    signals: List[PositionSignal] = []
    for p in broker.positions():
        recs = eng.recommend(p.ticker)
        # choose the shortest relevant horizon with non-neutral bias
        chosen = next((r for r in recs if r.horizon in ("1w","1m") and r.bias != "neutral"), recs[0])
        action = "hold"
        if p.side == "long":
            if chosen.bias == "short":
                action = "exit"
            elif chosen.bias == "long":
                action = "add" if chosen.confidence >= 0.7 else "hold"
        else:
            if chosen.bias == "long":
                action = "exit"
            elif chosen.bias == "short":
                action = "add" if chosen.confidence >= 0.7 else "hold"
        signals.append(PositionSignal(
            ticker=p.ticker,
            side=p.side,
            action=action,
            horizon=chosen.horizon,
            confidence=chosen.confidence,
            notes=chosen.notes,
        ))
    return signals
