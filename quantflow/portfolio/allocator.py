"""Portfolio allocation optimizer.

Implements:
- Mean-variance optimization with risk constraints.
- Hierarchical risk parity (HRP) for robustness.
- Volatility-based allocation.
- Constrained optimization (min/max weights, target volatility).
- Expected return forecasting from signals.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class ExpectedReturns:
    """Expected returns by asset."""
    returns: Dict[str, float]  # ticker -> expected return pct
    signals: Dict[str, float]  # ticker -> signal confidence 0-1
    forecasts: Dict[str, float]  # ticker -> forecast return pct


@dataclass
class RiskMetrics:
    """Risk metrics for optimization."""
    covariance: pd.DataFrame  # asset x asset covariance matrix
    volatilities: Dict[str, float]  # ticker -> annualized volatility
    correlations: pd.DataFrame  # asset x asset correlations


@dataclass
class AllocationConstraints:
    """Portfolio constraints."""
    max_weight: float = 0.15  # max 15% per asset
    min_weight: float = 0.0  # min 0% (allow exit)
    target_volatility: Optional[float] = 0.12  # target 12% annual vol
    max_concentration: float = 0.4  # top 3 can't exceed 40% together
    rebalance_threshold: float = 0.02  # rebalance if drift > 2%
    turnover_limit: Optional[float] = 0.05  # max 5% turnover


@dataclass
class AllocationResult:
    """Optimization result."""
    weights: Dict[str, float]  # ticker -> target weight
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    method: str  # "mean_variance", "hrp", "equal_weight"
    rationale: str


# ---------------------------------------------------------------------------
# Allocation strategies
# ---------------------------------------------------------------------------


class PortfolioAllocator:
    """Portfolio allocation optimizer."""

    def __init__(self, constraints: AllocationConstraints = None):
        self.constraints = constraints or AllocationConstraints()

    def optimize_mean_variance(
        self,
        expected_returns: ExpectedReturns,
        risk_metrics: RiskMetrics,
        risk_free_rate: float = 0.03,
    ) -> AllocationResult:
        """Optimize portfolio using mean-variance framework.

        Maximizes (E[R] - rf) / Vol subject to constraints.
        """
        tickers = list(expected_returns.returns.keys())
        n = len(tickers)

        if n < 2:
            return AllocationResult(
                {tickers[0]: 1.0},
                expected_returns.returns[tickers[0]],
                risk_metrics.volatilities[tickers[0]],
                0.0,
                "mean_variance",
                "Single asset."
            )

        # Extract returns and covariance
        ret_vec = np.array([expected_returns.returns[t] for t in tickers])
        cov_matrix = risk_metrics.covariance.loc[tickers, tickers].values

        # Objective: minimize negative Sharpe ratio
        def objective(w):
            port_ret = np.dot(w, ret_vec)
            port_vol = np.sqrt(np.dot(w, np.dot(cov_matrix, w)))
            sharpe = (port_ret - risk_free_rate) / (port_vol + 1e-8)
            return -sharpe

        # Constraints
        constraints_list = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        # Volatility constraint
        if self.constraints.target_volatility:
            def vol_constraint(w):
                return self.constraints.target_volatility - np.sqrt(np.dot(w, np.dot(cov_matrix, w)))
            constraints_list.append({"type": "ineq", "fun": vol_constraint})

        # Bounds per asset
        bounds = tuple((self.constraints.min_weight, self.constraints.max_weight) for _ in range(n))

        # Initial guess: equal weight
        w0 = np.array([1.0 / n] * n)

        result = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=constraints_list)

        if not result.success:
            # Fallback to equal weight
            w_opt = w0
        else:
            w_opt = result.x

        # Compute final metrics
        port_ret = np.dot(w_opt, ret_vec)
        port_vol = np.sqrt(np.dot(w_opt, np.dot(cov_matrix, w_opt)))
        sharpe = (port_ret - risk_free_rate) / (port_vol + 1e-8)

        weights = {tickers[i]: float(w_opt[i]) for i in range(n)}
        return AllocationResult(
            weights=weights,
            expected_return=port_ret,
            expected_volatility=port_vol,
            sharpe_ratio=sharpe,
            method="mean_variance",
            rationale=f"Optimized for Sharpe ratio {sharpe:.2f} with target vol {self.constraints.target_volatility or port_vol:.1%}",
        )

    def optimize_hrp(
        self,
        risk_metrics: RiskMetrics,
    ) -> AllocationResult:
        """Hierarchical Risk Parity allocation.

        Clusters correlated assets and allocates equal risk to each cluster.
        More robust to estimation error than mean-variance.
        """
        from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list

        tickers = list(risk_metrics.volatilities.keys())
        n = len(tickers)

        if n < 2:
            return AllocationResult(
                {tickers[0]: 1.0},
                0.0,
                risk_metrics.volatilities[tickers[0]],
                0.0,
                "hrp",
                "Single asset."
            )

        # Build distance matrix from correlation
        corr = risk_metrics.correlations.loc[tickers, tickers].values
        dist = np.sqrt(0.5 * (1 - corr))
        np.fill_diagonal(dist, 0)

        # Hierarchical clustering
        Z = linkage(dist[np.triu_indices_from(dist, k=1)], method="ward")
        idx = leaves_list(Z)

        # Allocate risk recursively
        vols = np.array([risk_metrics.volatilities[t] for t in tickers])
        weights = self._recursive_bisection(vols, idx)

        weights_dict = {tickers[i]: float(weights[i]) for i in range(n)}
        port_vol = np.sqrt(np.sum([weights[i]**2 * vols[i]**2 for i in range(n)]))

        return AllocationResult(
            weights=weights_dict,
            expected_return=0.0,
            expected_volatility=port_vol,
            sharpe_ratio=0.0,
            method="hrp",
            rationale="Hierarchical Risk Parity: equal risk per cluster, robust to correlation estimation error.",
        )

    def optimize_equal_weight(
        self,
        tickers: List[str],
    ) -> AllocationResult:
        """Simple equal-weight allocation."""
        weight = 1.0 / len(tickers)
        weights = {t: weight for t in tickers}
        return AllocationResult(
            weights=weights,
            expected_return=0.0,
            expected_volatility=0.0,
            sharpe_ratio=0.0,
            method="equal_weight",
            rationale="Equal-weight baseline: 1/N allocation.",
        )

    def optimize_volatility_scaled(
        self,
        risk_metrics: RiskMetrics,
    ) -> AllocationResult:
        """Allocate inverse to volatility (risk parity approximation)."""
        tickers = list(risk_metrics.volatilities.keys())
        vols = np.array([risk_metrics.volatilities[t] for t in tickers])

        # Inverse volatility weight
        inv_vols = 1.0 / (vols + 1e-8)
        weights_raw = inv_vols / inv_vols.sum()

        # Apply constraints
        weights_constrained = np.clip(weights_raw, self.constraints.min_weight, self.constraints.max_weight)
        weights_constrained = weights_constrained / weights_constrained.sum()

        weights_dict = {tickers[i]: float(weights_constrained[i]) for i in range(len(tickers))}
        port_vol = np.sqrt(np.sum([weights_constrained[i]**2 * vols[i]**2 for i in range(len(tickers))]))

        return AllocationResult(
            weights=weights_dict,
            expected_return=0.0,
            expected_volatility=port_vol,
            sharpe_ratio=0.0,
            method="volatility_scaled",
            rationale="Risk parity approximation: allocate inverse to volatility.",
        )

    @staticmethod
    def _recursive_bisection(vols: np.ndarray, idx: np.ndarray, depth: int = 0) -> np.ndarray:
        """Recursively bisect and allocate equal risk to each branch."""
        if len(idx) == 1:
            weights = np.zeros_like(vols, dtype=float)
            weights[idx[0]] = 1.0
            return weights

        if len(idx) == 2:
            v1, v2 = vols[idx[0]], vols[idx[1]]
            w1 = v2 / (v1 + v2)
            w2 = v1 / (v1 + v2)
            weights = np.zeros_like(vols, dtype=float)
            weights[idx[0]] = w1
            weights[idx[1]] = w2
            return weights

        mid = len(idx) // 2
        left_idx = idx[:mid]
        right_idx = idx[mid:]

        left_vol = np.sqrt(np.sum(vols[left_idx]**2))
        right_vol = np.sqrt(np.sum(vols[right_idx]**2))

        w_left = right_vol / (left_vol + right_vol) if (left_vol + right_vol) > 0 else 0.5
        w_right = left_vol / (left_vol + right_vol) if (left_vol + right_vol) > 0 else 0.5

        left_weights = PortfolioAllocator._recursive_bisection(vols, left_idx, depth + 1)
        right_weights = PortfolioAllocator._recursive_bisection(vols, right_idx, depth + 1)

        # Combine weights properly
        weights = np.zeros_like(vols, dtype=float)
        weights += w_left * left_weights
        weights += w_right * right_weights

        return weights


# ---------------------------------------------------------------------------
# Expected returns estimation
# ---------------------------------------------------------------------------


def estimate_returns_from_signals(
    signals: Dict[str, float],  # ticker -> signal confidence
    forecasts: Dict[str, float],  # ticker -> predicted return %
    historical_means: Dict[str, float],  # ticker -> historical mean return
) -> ExpectedReturns:
    """Estimate expected returns from signals + forecasts + historical."""
    returns = {}
    for ticker in signals.keys():
        # Blend: 40% signal, 30% forecast, 30% historical
        signal_ret = signals.get(ticker, 0) * 0.15  # confidence -> return contribution
        forecast_ret = forecasts.get(ticker, 0) * 0.3
        historical_ret = historical_means.get(ticker, 0) * 0.3
        returns[ticker] = signal_ret + forecast_ret + historical_ret

    return ExpectedReturns(
        returns=returns,
        signals=signals,
        forecasts=forecasts,
    )


def estimate_risk_from_history(
    ohlcv_dict: Dict[str, pd.DataFrame],
    lookback_days: int = 252,
) -> RiskMetrics:
    """Estimate risk metrics from historical data."""
    tickers = list(ohlcv_dict.keys())
    returns_dict = {}

    for ticker in tickers:
        df = ohlcv_dict[ticker].tail(lookback_days)
        rets = df["close"].pct_change().dropna()
        returns_dict[ticker] = rets

    # Align returns
    all_dates = pd.DatetimeIndex(set().union(*[s.index for s in returns_dict.values()]))
    aligned_rets = pd.DataFrame(index=all_dates)
    for ticker in tickers:
        aligned_rets[ticker] = returns_dict[ticker]

    aligned_rets = aligned_rets.dropna()

    # Compute metrics
    volatilities = {ticker: aligned_rets[ticker].std() * np.sqrt(252) for ticker in tickers}
    cov = aligned_rets.cov() * 252
    corr = aligned_rets.corr()

    return RiskMetrics(
        covariance=cov,
        volatilities=volatilities,
        correlations=corr,
    )
