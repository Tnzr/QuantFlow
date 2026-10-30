"""Research harness: leakage-safe validation, cost models and baseline ladder.

Every strategy or model change must be evaluated here (purged walk-forward,
net-of-cost metrics, trial-count-adjusted Sharpe) before it can be promoted
to paper trading.
"""

from quantflow.research.costs import CostModel
from quantflow.research.splits import CombinatorialPurgedKFold, PurgedWalkForward

__all__ = ["CombinatorialPurgedKFold", "CostModel", "PurgedWalkForward"]
