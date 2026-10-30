"""Transaction-cost model applied to every backtest in the harness."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostModel:
    """Per-rebalance costs, all in basis points of traded notional.

    ``impact_coef`` scales a square-root market-impact term by participation
    (traded notional / average daily dollar volume). Names with unknown ADV
    are charged as the least liquid known name, capped at
    ``fallback_adv_dollars``. ``borrow_bps_annual`` is charged on short
    notional per period held.
    """

    half_spread_bps: float = 2.0
    slippage_bps: float = 3.0
    commission_bps: float = 0.0
    borrow_bps_annual: float = 50.0
    impact_coef: float = 10.0
    periods_per_year: int = 52
    fallback_adv_dollars: float = 1e6

    @property
    def linear_bps(self) -> float:
        return self.half_spread_bps + self.slippage_bps + self.commission_bps

    def trading_cost(
        self,
        trades: pd.Series,
        capital: float | None = None,
        adv_dollars: pd.Series | None = None,
    ) -> float:
        """Cost as a fraction of capital for weight changes ``trades``."""
        traded = trades.abs()
        cost = traded.sum() * self.linear_bps / 1e4
        if capital and adv_dollars is not None and self.impact_coef:
            # unknown liquidity: assume the thinnest name seen that date, or a conservative floor
            adv = adv_dollars.reindex(traded.index).where(lambda a: a > 0)
            known_min = adv.min()
            floor = min(known_min, self.fallback_adv_dollars) if pd.notna(known_min) else self.fallback_adv_dollars
            part = (traded * capital / adv.fillna(floor)).clip(lower=0)
            cost += float((traded * self.impact_coef / 1e4 * np.sqrt(part)).sum())
        return float(cost)

    def borrow_cost(self, weights: pd.Series) -> float:
        short = weights[weights < 0].abs().sum()
        return float(short * self.borrow_bps_annual / 1e4 / self.periods_per_year)


@dataclass(frozen=True)
class OptionCostModel:
    """Fill model for multi-leg option spreads: pay a fraction of each leg's
    bid/ask width on entry and exit plus a per-contract commission."""

    spread_fill_frac: float = 0.5
    commission_per_contract: float = 0.0

    def round_trip(self, leg_widths: list[float], contracts: int) -> float:
        width = sum(abs(w) for w in leg_widths)
        return 2 * (width * self.spread_fill_frac * 100 + self.commission_per_contract * len(leg_widths)) * contracts
