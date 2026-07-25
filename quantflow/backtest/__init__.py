"""Dynamic Backtest CLI — Phase 2 engine with confidence-aware sizing and dynamic stops."""

from .engine import (
    BacktestConfig, BacktestEngineV2, Signal, Position, Trade, BacktestReport,
    SignalGenerator, PositionSizer, RiskManager, PortfolioManager,
)
from .visualization import plot_backtest_report
