"""Strategy configuration schema and builders.

Strategies are defined as JSON-serializable configs with:
- Indicators (moving averages, RSI, MACD, Bollinger Bands, etc.)
- Entry/exit rules (conditions on indicators)
- Position sizing (fixed %, volatility-based, signal-weighted)
- Regimes (trending, mean-reversion, high-vol, etc.)
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import json
import pandas as pd

from .portfolio_backtest import Signal


# ---------------------------------------------------------------------------
# Strategy configuration types
# ---------------------------------------------------------------------------


@dataclass
class IndicatorConfig:
    """Indicator calculation parameters."""
    name: str  # "sma", "ema", "rsi", "macd", "bbands", "atr", "adx"
    params: Dict[str, Any]  # {"period": 20, "multiplier": 2}

    def to_dict(self):
        return asdict(self)


@dataclass
class RuleConfig:
    """Entry/exit rule: condition + action."""
    name: str  # e.g., "rsi_oversold", "ma_crossover", "atr_breakout"
    condition: str  # expression like "rsi < 30 and price > sma20"
    action: str  # "enter_long", "enter_short", "exit"
    confidence_score: float = 0.5  # 0-1

    def to_dict(self):
        return asdict(self)


@dataclass
class StrategyConfig:
    """Full strategy definition."""
    name: str
    description: str
    indicators: List[IndicatorConfig] = field(default_factory=list)
    entry_rules: List[RuleConfig] = field(default_factory=list)
    exit_rules: List[RuleConfig] = field(default_factory=list)
    position_sizing: str = "equal_weight"  # "equal_weight", "volatility_scaled", "signal_weighted"
    max_position_pct: float = 0.15
    min_position_pct: float = 0.02
    holding_period_days: int = 7
    regime_detection: Optional[str] = None  # "adr", "vix_level", "trend"

    def to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "indicators": [ind.to_dict() for ind in self.indicators],
            "entry_rules": [rule.to_dict() for rule in self.entry_rules],
            "exit_rules": [rule.to_dict() for rule in self.exit_rules],
            "position_sizing": self.position_sizing,
            "max_position_pct": self.max_position_pct,
            "min_position_pct": self.min_position_pct,
            "holding_period_days": self.holding_period_days,
            "regime_detection": self.regime_detection,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> StrategyConfig:
        """Load strategy from dict/JSON."""
        return cls(
            name=data["name"],
            description=data["description"],
            indicators=[IndicatorConfig(i["name"], i["params"]) for i in data.get("indicators", [])],
            entry_rules=[RuleConfig(r["name"], r["condition"], r["action"], r.get("confidence_score", 0.5))
                         for r in data.get("entry_rules", [])],
            exit_rules=[RuleConfig(r["name"], r["condition"], r["action"], r.get("confidence_score", 0.5))
                        for r in data.get("exit_rules", [])],
            position_sizing=data.get("position_sizing", "equal_weight"),
            max_position_pct=data.get("max_position_pct", 0.15),
            min_position_pct=data.get("min_position_pct", 0.02),
            holding_period_days=data.get("holding_period_days", 7),
            regime_detection=data.get("regime_detection"),
        )


# ---------------------------------------------------------------------------
# Built-in strategy templates
# ---------------------------------------------------------------------------


def trend_following_strategy() -> StrategyConfig:
    """Simple trend-following strategy: enter on MA crossover, exit on reverse."""
    return StrategyConfig(
        name="trend_follow_sma",
        description="Enter on close > SMA50, exit on SMA20 crossover or time stop",
        indicators=[
            IndicatorConfig("sma", {"period": 20}),
            IndicatorConfig("sma", {"period": 50}),
            IndicatorConfig("rsi", {"period": 14}),
        ],
        entry_rules=[
            RuleConfig("ma_cross", "close > sma50 and rsi > 40", "enter_long", 0.6),
        ],
        exit_rules=[
            RuleConfig("reverse_cross", "close < sma20", "exit", 0.8),
        ],
        position_sizing="volatility_scaled",
        holding_period_days=21,
        regime_detection="trend",
    )


def mean_reversion_strategy() -> StrategyConfig:
    """Mean reversion: enter when RSI oversold within Bollinger Band, exit on recovery."""
    return StrategyConfig(
        name="mean_reversion_rsi_bb",
        description="Enter on RSI < 30 and price < lower BB, exit on RSI > 70 or upper BB touch",
        indicators=[
            IndicatorConfig("bbands", {"period": 20, "multiplier": 2}),
            IndicatorConfig("rsi", {"period": 14}),
            IndicatorConfig("sma", {"period": 20}),
        ],
        entry_rules=[
            RuleConfig("oversold_bb", "rsi < 30 and close < bb_lower", "enter_long", 0.7),
        ],
        exit_rules=[
            RuleConfig("rsi_recovery", "rsi > 70", "exit", 0.8),
        ],
        position_sizing="signal_weighted",
        holding_period_days=5,
        regime_detection="adr",
    )


def breakout_strategy() -> StrategyConfig:
    """Breakout: enter on 52-week high, exit on reversal or time."""
    return StrategyConfig(
        name="breakout_high",
        description="Enter on new 52-week high above ATR, exit on close below entry - 2*ATR",
        indicators=[
            IndicatorConfig("atr", {"period": 14}),
            IndicatorConfig("highest", {"period": 252}),
        ],
        entry_rules=[
            RuleConfig("breakout", "close > highest_252_bar and close > previous_close + atr", "enter_long", 0.5),
        ],
        exit_rules=[
            RuleConfig("reversal", "close < entry_price - 2*atr", "exit", 0.9),
        ],
        position_sizing="equal_weight",
        holding_period_days=14,
        regime_detection="adr",
    )


def multi_timeframe_strategy() -> StrategyConfig:
    """Multi-timeframe: confirm on daily, signal on 4H, exit on daily breakdown."""
    return StrategyConfig(
        name="multi_tf_confirm",
        description="Daily MA trend + 4H RSI confirmation + ATR exit",
        indicators=[
            IndicatorConfig("sma", {"period": 50, "timeframe": "daily"}),
            IndicatorConfig("sma", {"period": 200, "timeframe": "daily"}),
            IndicatorConfig("rsi", {"period": 14, "timeframe": "4h"}),
            IndicatorConfig("atr", {"period": 14, "timeframe": "daily"}),
        ],
        entry_rules=[
            RuleConfig("tf_confirm", "sma50 > sma200 and rsi_4h > 40", "enter_long", 0.75),
        ],
        exit_rules=[
            RuleConfig("daily_break", "close < sma50", "exit", 0.9),
        ],
        position_sizing="volatility_scaled",
        holding_period_days=7,
        regime_detection="trend",
    )


# ---------------------------------------------------------------------------
# Strategy evaluator
# ---------------------------------------------------------------------------


class StrategyEvaluator:
    """Evaluate strategy rules against indicator values."""

    def __init__(self, strategy: StrategyConfig):
        self.strategy = strategy
        self.indicators_cache = {}

    def compute_indicators(self, ticker: str, ohlcv_df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Compute all required indicators for the strategy."""
        indicators = {}

        for ind_config in self.strategy.indicators:
            if ind_config.name == "sma":
                period = ind_config.params.get("period", 20)
                indicators[f"sma{period}"] = ohlcv_df["close"].rolling(period).mean()
            elif ind_config.name == "ema":
                period = ind_config.params.get("period", 20)
                indicators[f"ema{period}"] = ohlcv_df["close"].ewm(span=period).mean()
            elif ind_config.name == "rsi":
                period = ind_config.params.get("period", 14)
                indicators[f"rsi{period}"] = self._compute_rsi(ohlcv_df["close"], period)
            elif ind_config.name == "bbands":
                period = ind_config.params.get("period", 20)
                mult = ind_config.params.get("multiplier", 2)
                mid = ohlcv_df["close"].rolling(period).mean()
                std = ohlcv_df["close"].rolling(period).std()
                indicators[f"bb_middle"] = mid
                indicators[f"bb_upper"] = mid + mult * std
                indicators[f"bb_lower"] = mid - mult * std
            elif ind_config.name == "atr":
                period = ind_config.params.get("period", 14)
                indicators[f"atr{period}"] = self._compute_atr(ohlcv_df, period)
            elif ind_config.name == "macd":
                indicators["macd"], indicators["macd_signal"] = self._compute_macd(ohlcv_df["close"])

        return indicators

    @staticmethod
    def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """Compute RSI."""
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-10)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Compute ATR."""
        h, l, c = df["high"], df["low"], df["close"]
        prev_c = c.shift(1)
        tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
        return tr.ewm(com=period - 1, min_periods=period).mean()

    @staticmethod
    def _compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """Compute MACD."""
        ema_fast = close.ewm(span=fast).mean()
        ema_slow = close.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        return macd_line, signal_line

    def evaluate_signal(self, ticker: str, ohlcv_df: pd.DataFrame) -> Signal:
        """Evaluate all rules and return entry/exit signal."""
        if ohlcv_df.empty:
            return Signal(ohlcv_df.index[-1] if len(ohlcv_df) > 0 else pd.Timestamp.now(),
                         ticker, "neutral", 0.0)

        indicators = self.compute_indicators(ticker, ohlcv_df)
        last_row = ohlcv_df.iloc[-1]
        price = float(last_row["close"])

        # Evaluate entry rules
        direction = "neutral"
        confidence = 0.0
        for rule in self.strategy.entry_rules:
            if self._eval_rule(rule.condition, price, indicators, last_row):
                direction = "long" if rule.action == "enter_long" else "short"
                confidence = max(confidence, rule.confidence_score)

        # Evaluate exit rules (override)
        for rule in self.strategy.exit_rules:
            if self._eval_rule(rule.condition, price, indicators, last_row):
                direction = "neutral"
                confidence = 0.0

        return Signal(
            ohlcv_df.index[-1],
            ticker,
            direction,
            confidence,
            strength=1.0,
        )

    @staticmethod
    def _eval_rule(condition: str, price: float, indicators: Dict, row: pd.Series) -> bool:
        """Evaluate a rule condition (simple expression parsing)."""
        try:
            # Build local namespace for eval
            namespace = {
                "close": price,
                "high": float(row.get("high", 0)),
                "low": float(row.get("low", 0)),
                "volume": float(row.get("volume", 0)),
            }
            namespace.update({k: float(v.iloc[-1]) if hasattr(v, "iloc") else float(v)
                             for k, v in indicators.items()})

            # Simple safe eval
            result = eval(condition, {"__builtins__": {}}, namespace)
            return bool(result)
        except Exception:
            return False
