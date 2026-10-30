"""AgentVeto / WeightAdjuster: agents may only override the rule baseline when
election/macro volatility is elevated AND the confidence gap is >= ``min_gap``."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ..election.calendar import election_in_window


@dataclass
class GateDecision:
    base_confidence: float
    agent_confidence: float
    final_confidence: float
    confidence_gap: float
    volatility_triggered: bool
    override_applied: bool
    sentiment_multiplier: float
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def volatility_regime(
    df: pd.DataFrame,
    as_of=None,
    election_window_days: int = 45,
    vol_ratio_elevated: float = 1.2,
    vol_ratio_extreme: float = 1.5,
) -> dict:
    """Realized-vol regime using only bars at or before ``as_of``.

    ``vol_ratio`` = 20d realized vol / trailing-252d median of 20d realized vol.
    Elevated when (election inside window and ratio >= elevated) or ratio >= extreme.
    """
    px = df["adj close"] if "adj close" in df else df["close"]
    if as_of is not None:
        px = px[px.index <= pd.Timestamp(as_of)]
    rets = px.pct_change()
    rv = rets.rolling(20).std() * np.sqrt(252)
    current = float(rv.iloc[-1]) if len(rv) and np.isfinite(rv.iloc[-1]) else float("nan")
    med = float(rv.iloc[-252:].median()) if rv.notna().sum() >= 40 else float("nan")
    ratio = current / med if np.isfinite(current) and np.isfinite(med) and med > 0 else 1.0
    when = px.index[-1] if len(px) else pd.Timestamp.utcnow()
    in_window = election_in_window(when, election_window_days)
    elevated = (in_window and ratio >= vol_ratio_elevated) or ratio >= vol_ratio_extreme
    return {
        "as_of": str(pd.Timestamp(when).date()),
        "realized_vol_20d": current,
        "median_vol_1y": med,
        "vol_ratio": float(ratio),
        "election_window": bool(in_window),
        "elevated": bool(elevated),
    }


@dataclass
class WeightAdjuster:
    min_gap: float = 0.25
    sentiment_cap: float = 0.10  # multiplier stays within [1 - cap, 1 + cap]

    def sentiment_multiplier(self, polarity: float) -> float:
        p = float(np.clip(polarity, -1.0, 1.0))
        return float(1.0 + self.sentiment_cap * p)

    def decide(
        self,
        base_confidence: float,
        agent_confidence: float,
        volatility_triggered: bool,
        sentiment_polarity: float = 0.0,
    ) -> GateDecision:
        gap = abs(agent_confidence - base_confidence)
        mult = self.sentiment_multiplier(sentiment_polarity)
        if volatility_triggered and gap >= self.min_gap:
            final = agent_confidence
            override, reason = True, f"override: volatility elevated and gap {gap:.2f} >= {self.min_gap:.2f}"
        else:
            final = base_confidence * mult
            override = False
            reason = "baseline kept: " + ("gap below threshold" if volatility_triggered else "volatility not elevated")
        return GateDecision(
            base_confidence=float(base_confidence),
            agent_confidence=float(agent_confidence),
            final_confidence=float(np.clip(final, 0.0, 1.0)),
            confidence_gap=float(gap),
            volatility_triggered=bool(volatility_triggered),
            override_applied=override,
            sentiment_multiplier=mult,
            reason=reason,
        )
