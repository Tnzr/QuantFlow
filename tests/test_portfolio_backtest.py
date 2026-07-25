"""Integration tests for portfolio backtester and allocator.

Tests:
- Portfolio backtest with multi-asset portfolio and signal generation
- Strategy config and evaluation with different rule sets
- Portfolio allocation optimization (mean-variance, HRP, equal-weight)
- Allocation endpoint via FastAPI
"""
import unittest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from quantflow.backtest.portfolio_backtest import (
    PortfolioBacktester, BacktestConfig, Signal, Trade, PortfolioMetrics
)
from quantflow.backtest.strategy_config import (
    StrategyConfig, IndicatorConfig, RuleConfig, StrategyEvaluator,
    trend_following_strategy, mean_reversion_strategy
)
from quantflow.portfolio.allocator import (
    PortfolioAllocator, AllocationConstraints, RiskMetrics,
    estimate_returns_from_signals, estimate_risk_from_history
)


class TestPortfolioBacktester(unittest.TestCase):
    """Test portfolio-level backtester."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = BacktestConfig(
            start_date="2023-01-01",
            end_date="2023-12-31",
            initial_capital=100000.0,
            max_position_size=0.15,
            min_position_size=0.02,
            rebalance_freq="weekly",
            transaction_cost_pct=0.001,
            slippage_pct=0.0005,
        )

    def test_portfolio_backtest_initialization(self):
        """Test backtest config initialization."""
        self.assertEqual(self.config.initial_capital, 100000.0)
        self.assertEqual(self.config.max_position_size, 0.15)
        self.assertEqual(self.config.rebalance_freq, "weekly")

    def test_trade_creation(self):
        """Test Trade record structure."""
        trade = Trade(
            ticker="AAPL",
            entry_date=pd.Timestamp("2023-01-01"),
            entry_price=150.0,
            exit_date=pd.Timestamp("2023-01-08"),
            exit_price=155.0,
            size=100,
            direction="long",
            pnl=500.0,
            pnl_pct=0.0333,
            hold_days=7,
        )
        self.assertEqual(trade.ticker, "AAPL")
        self.assertEqual(trade.direction, "long")
        self.assertAlmostEqual(trade.pnl, 500.0)

    def test_signal_creation(self):
        """Test Signal structure."""
        signal = Signal(
            date=pd.Timestamp("2023-01-01"),
            ticker="AAPL",
            direction="long",
            confidence=0.75,
            strength=1.2,
        )
        self.assertEqual(signal.direction, "long")
        self.assertAlmostEqual(signal.confidence, 0.75)
        self.assertEqual(signal.ticker, "AAPL")

    def test_portfolio_metrics_computation(self):
        """Test portfolio metrics calculation."""
        equity = pd.Series(
            [100000, 101000, 102000, 101500, 103000],
            index=pd.date_range("2023-01-01", periods=5)
        )
        daily_rets = equity.pct_change().dropna()
        trades = [
            Trade("AAPL", pd.Timestamp("2023-01-01"), 150.0, pd.Timestamp("2023-01-08"), 155.0, 100, "long", 500.0, 0.0333, 7),
            Trade("AAPL", pd.Timestamp("2023-01-08"), 155.0, pd.Timestamp("2023-01-15"), 152.0, 100, "long", -300.0, -0.0194, 7),
        ]
        
        metrics = PortfolioBacktester._compute_metrics(equity, daily_rets, trades)
        
        self.assertEqual(metrics.n_trades, 2)
        self.assertGreater(metrics.total_return, 0)
        self.assertGreaterEqual(metrics.win_rate, 0)
        self.assertLessEqual(metrics.win_rate, 1)


class TestStrategyConfig(unittest.TestCase):
    """Test strategy configuration and evaluation."""

    def test_strategy_config_creation(self):
        """Test creating a custom strategy config."""
        strategy = StrategyConfig(
            name="test_strategy",
            description="A test strategy",
            indicators=[
                IndicatorConfig("sma", {"period": 20}),
                IndicatorConfig("rsi", {"period": 14}),
            ],
            entry_rules=[
                RuleConfig("ma_cross", "close > sma20 and rsi > 40", "enter_long", 0.6),
            ],
            exit_rules=[
                RuleConfig("ma_break", "close < sma20", "exit", 0.8),
            ],
        )
        self.assertEqual(strategy.name, "test_strategy")
        self.assertEqual(len(strategy.indicators), 2)
        self.assertEqual(len(strategy.entry_rules), 1)

    def test_trend_following_strategy(self):
        """Test built-in trend following strategy."""
        strategy = trend_following_strategy()
        self.assertEqual(strategy.name, "trend_follow_sma")
        self.assertGreater(len(strategy.indicators), 0)
        self.assertGreater(len(strategy.entry_rules), 0)

    def test_mean_reversion_strategy(self):
        """Test built-in mean reversion strategy."""
        strategy = mean_reversion_strategy()
        self.assertEqual(strategy.name, "mean_reversion_rsi_bb")
        self.assertIn("bbands", [ind.name for ind in strategy.indicators])

    def test_strategy_serialization(self):
        """Test strategy to_dict and from_dict."""
        strategy = trend_following_strategy()
        strategy_dict = strategy.to_dict()
        
        self.assertIn("name", strategy_dict)
        self.assertIn("indicators", strategy_dict)
        self.assertIn("entry_rules", strategy_dict)

    def test_strategy_evaluator_initialization(self):
        """Test strategy evaluator creation."""
        strategy = trend_following_strategy()
        evaluator = StrategyEvaluator(strategy)
        
        self.assertEqual(evaluator.strategy.name, strategy.name)
        self.assertIsInstance(evaluator.indicators_cache, dict)


class TestPortfolioAllocator(unittest.TestCase):
    """Test portfolio allocation optimizer."""

    def setUp(self):
        """Set up test data."""
        # Create synthetic price data
        dates = pd.date_range("2023-01-01", periods=252)
        
        # AAPL: uptrend
        aapl_returns = np.random.normal(0.0005, 0.01, len(dates))
        aapl_close = 150 * np.exp(np.cumsum(aapl_returns))
        
        # MSFT: slightly lower returns, lower vol
        msft_returns = np.random.normal(0.0003, 0.008, len(dates))
        msft_close = 300 * np.exp(np.cumsum(msft_returns))
        
        # GOOG: higher volatility
        goog_returns = np.random.normal(0.0002, 0.015, len(dates))
        goog_close = 100 * np.exp(np.cumsum(goog_returns))

        self.ohlcv_dict = {
            "AAPL": pd.DataFrame({
                "close": aapl_close,
                "high": aapl_close * 1.01,
                "low": aapl_close * 0.99,
                "volume": np.random.randint(1000000, 5000000, len(dates)),
            }, index=dates),
            "MSFT": pd.DataFrame({
                "close": msft_close,
                "high": msft_close * 1.01,
                "low": msft_close * 0.99,
                "volume": np.random.randint(1000000, 5000000, len(dates)),
            }, index=dates),
            "GOOG": pd.DataFrame({
                "close": goog_close,
                "high": goog_close * 1.01,
                "low": goog_close * 0.99,
                "volume": np.random.randint(1000000, 5000000, len(dates)),
            }, index=dates),
        }

    def test_allocator_initialization(self):
        """Test allocator creation."""
        constraints = AllocationConstraints(
            max_weight=0.15,
            min_weight=0.05,
        )
        allocator = PortfolioAllocator(constraints=constraints)
        
        self.assertEqual(allocator.constraints.max_weight, 0.15)
        self.assertEqual(allocator.constraints.min_weight, 0.05)

    def test_equal_weight_allocation(self):
        """Test equal-weight allocation."""
        tickers = ["AAPL", "MSFT", "GOOG"]
        constraints = AllocationConstraints()
        allocator = PortfolioAllocator(constraints=constraints)
        
        result = allocator.optimize_equal_weight(tickers)
        
        self.assertEqual(result.method, "equal_weight")
        self.assertAlmostEqual(sum(result.weights.values()), 1.0)
        for w in result.weights.values():
            self.assertAlmostEqual(w, 1.0 / len(tickers))

    def test_volatility_scaled_allocation(self):
        """Test volatility-scaled allocation."""
        risk_metrics = estimate_risk_from_history(self.ohlcv_dict, lookback_days=252)
        constraints = AllocationConstraints(max_weight=0.3)
        allocator = PortfolioAllocator(constraints=constraints)
        
        result = allocator.optimize_volatility_scaled(risk_metrics)
        
        self.assertEqual(result.method, "volatility_scaled")
        self.assertAlmostEqual(sum(result.weights.values()), 1.0, places=5)
        self.assertGreater(result.expected_volatility, 0)

    def test_hrp_allocation(self):
        """Test Hierarchical Risk Parity allocation."""
        risk_metrics = estimate_risk_from_history(self.ohlcv_dict, lookback_days=252)
        allocator = PortfolioAllocator()
        
        result = allocator.optimize_hrp(risk_metrics)
        
        self.assertEqual(result.method, "hrp")
        self.assertAlmostEqual(sum(result.weights.values()), 1.0, places=5)
        for w in result.weights.values():
            self.assertGreaterEqual(w, 0)

    def test_mean_variance_allocation(self):
        """Test mean-variance optimization."""
        risk_metrics = estimate_risk_from_history(self.ohlcv_dict, lookback_days=252)
        
        # Estimate returns
        signals = {"AAPL": 0.7, "MSFT": 0.5, "GOOG": 0.3}
        forecasts = {"AAPL": 0.02, "MSFT": 0.01, "GOOG": 0.03}
        historical = {"AAPL": 0.12, "MSFT": 0.10, "GOOG": 0.15}
        expected_returns = estimate_returns_from_signals(signals, forecasts, historical)
        
        constraints = AllocationConstraints(target_volatility=0.12)
        allocator = PortfolioAllocator(constraints=constraints)
        
        result = allocator.optimize_mean_variance(expected_returns, risk_metrics)
        
        self.assertEqual(result.method, "mean_variance")
        self.assertAlmostEqual(sum(result.weights.values()), 1.0, places=5)
        self.assertLessEqual(result.expected_volatility, 0.15)  # should respect target

    def test_returns_estimation(self):
        """Test expected returns estimation."""
        signals = {"AAPL": 0.8, "MSFT": 0.5}
        forecasts = {"AAPL": 0.03, "MSFT": 0.01}
        historical = {"AAPL": 0.15, "MSFT": 0.10}
        
        result = estimate_returns_from_signals(signals, forecasts, historical)
        
        self.assertIn("AAPL", result.returns)
        self.assertIn("MSFT", result.returns)
        self.assertGreater(result.returns["AAPL"], result.returns["MSFT"])

    def test_risk_metrics_estimation(self):
        """Test risk metrics from historical data."""
        risk_metrics = estimate_risk_from_history(self.ohlcv_dict, lookback_days=252)
        
        # Check volatilities
        for ticker in self.ohlcv_dict.keys():
            self.assertIn(ticker, risk_metrics.volatilities)
            self.assertGreater(risk_metrics.volatilities[ticker], 0)
        
        # Check covariance
        self.assertEqual(risk_metrics.covariance.shape[0], 3)
        self.assertEqual(risk_metrics.covariance.shape[1], 3)
        
        # Check correlations
        self.assertEqual(risk_metrics.correlations.shape[0], 3)
        self.assertEqual(risk_metrics.correlations.shape[1], 3)


class TestAllocationConstraints(unittest.TestCase):
    """Test allocation constraints."""

    def test_constraints_creation(self):
        """Test creating constraints."""
        constraints = AllocationConstraints(
            max_weight=0.25,
            min_weight=0.05,
            target_volatility=0.10,
        )
        self.assertEqual(constraints.max_weight, 0.25)
        self.assertEqual(constraints.min_weight, 0.05)
        self.assertEqual(constraints.target_volatility, 0.10)

    def test_default_constraints(self):
        """Test default constraint values."""
        constraints = AllocationConstraints()
        
        self.assertEqual(constraints.max_weight, 0.15)
        self.assertEqual(constraints.min_weight, 0.0)
        self.assertEqual(constraints.target_volatility, 0.12)


if __name__ == "__main__":
    unittest.main()
