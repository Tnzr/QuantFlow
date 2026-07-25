from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, List, Optional, Tuple
import math

import pandas as pd

from ..features.analytics import support_resistance_from_distribution
from ..features.analytics import forecast_prices
from ..features.indicators import fetch_ohlcv, compute_indicators
from ..features.seasonality import proximity_to_earnings
from ..features.exit_rules import simple_exit_plan


@dataclass
class Rec:
    ticker: str
    horizon: str  # 1w, 1m, 3m, 6m, 1y
    bias: str     # long/short/neutral
    entry: Optional[float]
    stop: Optional[float]
    target: Optional[float]
    confidence: float
    notes: str


class RuleEngine:
    def __init__(self):
        pass

    def score_short_term(self, df: pd.DataFrame) -> float:
        # 1w: favor RSI 40-60 cross to >50, price above SMA20, positive momentum
        last = df.iloc[-1]
        score = 0.0
        if last["adj close"] > last["sma20"]:
            score += 0.4
        if last["rsi14"] > 50:
            score += 0.3
        if df["ret"].tail(3).mean() > 0:
            score += 0.3
        return score

    def score_medium_term(self, df: pd.DataFrame) -> float:
        # 1-3m: price above SMA50/200, rising averages
        last = df.iloc[-1]
        score = 0.0
        if last["adj close"] > last["sma50"]:
            score += 0.3
        if last["adj close"] > last["sma200"]:
            score += 0.3
        if last["sma50"] > df["sma50"].iloc[-20]:
            score += 0.2
        if last["sma200"] > df["sma200"].iloc[-50]:
            score += 0.2
        return score

    def score_long_term(self, df: pd.DataFrame) -> float:
        # 6-12m: price above SMA200 and SMA200 sloping up, low vol
        last = df.iloc[-1]
        score = 0.0
        if last["adj close"] > last["sma200"]:
            score += 0.5
        if last["sma200"] > df["sma200"].iloc[-100]:
            score += 0.3
        if df["vol20"].iloc[-1] < df["vol20"].quantile(0.6):
            score += 0.2
        return score

    def apply_event_penalty(self, ticker: str, horizon: str, base_score: float) -> Tuple[float, str]:
        days = proximity_to_earnings(ticker)
        note = ""
        if days is None:
            return base_score, note
        # penalize short-dated setups near earnings to reduce overconfidence
        if horizon == "1w" and days <= 10:
            note = f"Earnings in {days}d: reduced confidence"
            return max(0.0, base_score - 0.25), note
        if horizon in ("1m", "3m") and days <= 15:
            note = f"Earnings in {days}d: slight penalty"
            return max(0.0, base_score - 0.1), note
        return base_score, note

    def gen_stop_target(self, df: pd.DataFrame, bias: str) -> Tuple[float, float]:
        last = df.iloc[-1]
        price = float(last["adj close"])
        atr = float(last["atr14"]) if not math.isnan(last["atr14"]) else price * 0.02
        if bias == "long":
            stop = price - 1.5 * atr
            target = price + 2.5 * atr
        else:
            stop = price + 1.5 * atr
            target = price - 2.5 * atr
        return stop, target

    def _score_breakdown(self, df: pd.DataFrame) -> dict[str, float]:
        last = df.iloc[-1]
        price = float(last["adj close"])
        sma20 = float(last["sma20"]) if not pd.isna(last["sma20"]) else price
        sma50 = float(last["sma50"]) if not pd.isna(last["sma50"]) else price
        sma200 = float(last["sma200"]) if not pd.isna(last["sma200"]) else price
        rsi14 = float(last["rsi14"]) if not pd.isna(last["rsi14"]) else 50.0
        atr14 = float(last["atr14"]) if not pd.isna(last["atr14"]) else max(price * 0.02, 0.01)
        vol20 = float(last["vol20"]) if not pd.isna(last["vol20"]) else 0.0

        ret_21 = float(df["adj close"].pct_change(21).iloc[-1]) if len(df) > 21 else 0.0
        ret_63 = float(df["adj close"].pct_change(63).iloc[-1]) if len(df) > 63 else ret_21
        trend_score = max(0.0, min(1.0, 0.45 * (price > sma20) + 0.25 * (price > sma50) + 0.30 * (price > sma200)))
        momentum_score = max(0.0, min(1.0, 0.5 + 1.25 * ret_21 + 0.75 * ret_63))
        rsi_score = max(0.0, min(1.0, 1.0 - abs(rsi14 - 58.0) / 35.0))
        volatility_score = max(0.0, min(1.0, 1.0 - min(vol20, 0.8) / 0.8))
        risk_score = max(0.0, min(1.0, 1.0 - min(atr14 / max(price, 0.01), 0.12) / 0.12))
        return {
            "trend": trend_score,
            "momentum": momentum_score,
            "rsi_quality": rsi_score,
            "volatility_regime": volatility_score,
            "atr_efficiency": risk_score,
        }

    def analyze_ticker(self, ticker: str, period: str = "2y", interval: str = "1d") -> dict[str, Any]:
        df = fetch_ohlcv(ticker, period=period, interval=interval)
        df = compute_indicators(df)
        recs = self.recommend(ticker)
        last = df.iloc[-1]
        support_profile = support_resistance_from_distribution(df, bins=24, level_count=3)
        breakdown = self._score_breakdown(df)
        base_score = (
            0.30 * breakdown["trend"]
            + 0.25 * breakdown["momentum"]
            + 0.15 * breakdown["rsi_quality"]
            + 0.15 * breakdown["volatility_regime"]
            + 0.15 * breakdown["atr_efficiency"]
        )

        nearest_support = support_profile.get("nearest_support")
        nearest_resistance = support_profile.get("nearest_resistance")
        current_price = float(last["adj close"])
        support_bonus = 0.0
        if nearest_support and current_price:
            support_gap = max(0.0, (current_price / nearest_support) - 1.0)
            support_bonus = max(0.0, 0.15 - support_gap)

        resistance_penalty = 0.0
        if nearest_resistance and current_price:
            resistance_gap = max(0.0, (nearest_resistance / current_price) - 1.0)
            resistance_penalty = max(0.0, 0.08 - resistance_gap)

        composite_score = max(0.0, min(1.0, base_score + support_bonus - resistance_penalty))
        rec_payload = [asdict(rec) for rec in recs]
        top_rec = max(rec_payload, key=lambda rec: rec["confidence"])
        top_bias = top_rec["bias"]

        return {
            "ticker": ticker.upper(),
            "price": current_price,
            "bias": top_bias,
            "primary_horizon": top_rec["horizon"],
            "composite_score": composite_score,
            "score_breakdown": breakdown,
            "support_resistance": {
                "nearest_support": nearest_support,
                "nearest_resistance": nearest_resistance,
                "support_gap_pct": support_profile.get("support_gap_pct"),
                "resistance_gap_pct": support_profile.get("resistance_gap_pct"),
                "support_levels": support_profile.get("support_levels", []),
                "resistance_levels": support_profile.get("resistance_levels", []),
            },
            "indicators": {
                "rsi14": float(last["rsi14"]) if not pd.isna(last["rsi14"]) else None,
                "sma20": float(last["sma20"]) if not pd.isna(last["sma20"]) else None,
                "sma50": float(last["sma50"]) if not pd.isna(last["sma50"]) else None,
                "sma200": float(last["sma200"]) if not pd.isna(last["sma200"]) else None,
                "atr14": float(last["atr14"]) if not pd.isna(last["atr14"]) else None,
                "vol20": float(last["vol20"]) if not pd.isna(last["vol20"]) else None,
            },
            "recommendations": rec_payload,
        }

    def training_features(self, ticker: str, period: str = "2y", interval: str = "1d", forecast_horizon: int = 20) -> dict[str, Any]:
        df = fetch_ohlcv(ticker, period=period, interval=interval)
        df = compute_indicators(df)
        analysis = self.analyze_ticker(ticker, period=period, interval=interval)
        forecast = forecast_prices(df, horizon=forecast_horizon)
        last = df.iloc[-1]

        features = {
            "ticker": ticker.upper(),
            "period": period,
            "interval": interval,
            "price": float(last["adj close"]),
            "ret_5d": float(df["adj close"].pct_change(5).iloc[-1]) if len(df) > 5 else 0.0,
            "ret_21d": float(df["adj close"].pct_change(21).iloc[-1]) if len(df) > 21 else 0.0,
            "ret_63d": float(df["adj close"].pct_change(63).iloc[-1]) if len(df) > 63 else 0.0,
            "rsi14": float(last["rsi14"]) if not pd.isna(last["rsi14"]) else None,
            "atr14": float(last["atr14"]) if not pd.isna(last["atr14"]) else None,
            "vol20": float(last["vol20"]) if not pd.isna(last["vol20"]) else None,
            "distance_sma20_pct": float((last["adj close"] / last["sma20"]) - 1.0) if not pd.isna(last["sma20"]) and last["sma20"] else None,
            "distance_sma50_pct": float((last["adj close"] / last["sma50"]) - 1.0) if not pd.isna(last["sma50"]) and last["sma50"] else None,
            "distance_sma200_pct": float((last["adj close"] / last["sma200"]) - 1.0) if not pd.isna(last["sma200"]) and last["sma200"] else None,
            "composite_score": float(analysis["composite_score"]),
            "trend_score": float(analysis["score_breakdown"].get("trend", 0.0)),
            "momentum_score": float(analysis["score_breakdown"].get("momentum", 0.0)),
            "rsi_quality_score": float(analysis["score_breakdown"].get("rsi_quality", 0.0)),
            "volatility_regime_score": float(analysis["score_breakdown"].get("volatility_regime", 0.0)),
            "atr_efficiency_score": float(analysis["score_breakdown"].get("atr_efficiency", 0.0)),
            "nearest_support": analysis["support_resistance"].get("nearest_support"),
            "nearest_resistance": analysis["support_resistance"].get("nearest_resistance"),
            "support_gap_pct": analysis["support_resistance"].get("support_gap_pct"),
            "resistance_gap_pct": analysis["support_resistance"].get("resistance_gap_pct"),
            "forecast_return_pct": float(forecast.get("forecast_return_pct", 0.0)),
            "forecast_band_pct": float(forecast.get("confidence_band_pct", 0.0)),
            "forecast_daily_trend_pct": float(forecast.get("daily_trend_pct", 0.0)),
            "top_bias": analysis.get("bias"),
            "primary_horizon": analysis.get("primary_horizon"),
        }
        return features

    def rank_tickers(self, tickers: list[str], period: str = "2y", interval: str = "1d") -> list[dict[str, Any]]:
        ranked = []
        for ticker in tickers:
            ranked.append(self.analyze_ticker(ticker, period=period, interval=interval))
        ranked.sort(
            key=lambda row: (
                row["composite_score"],
                max((rec["confidence"] for rec in row["recommendations"]), default=0.0),
            ),
            reverse=True,
        )
        return ranked

    def recommend(self, ticker: str) -> List[Rec]:
        df = fetch_ohlcv(ticker)
        df = compute_indicators(df)
        last = df.iloc[-1]

        recs: List[Rec] = []

        # 1w
        s1 = self.score_short_term(df)
        s1, note1 = self.apply_event_penalty(ticker, "1w", s1)
        bias1 = "long" if s1 >= 0.6 else ("short" if last["rsi14"] < 45 and last["adj close"] < last["sma20"] else "neutral")
        stop, tgt = self.gen_stop_target(df, bias1 if bias1 != "neutral" else "long")
        exit1 = simple_exit_plan(df, bias=bias1 if bias1 != "neutral" else "long", atr_factor=2.0, time_stop_days=5)
        recs.append(Rec(ticker, "1w", bias1, float(last["adj close"]), exit1.hard_stop, tgt, s1, f"RSI/SMA20 momentum. {note1} Trailing stop={exit1.trailing_stop:.2f}".strip()))

        # 1m
        s2 = self.score_medium_term(df)
        s2, note2 = self.apply_event_penalty(ticker, "1m", s2)
        bias2 = "long" if s2 >= 0.6 else ("short" if last["adj close"] < last["sma200"] else "neutral")
        stop, tgt = self.gen_stop_target(df, bias2 if bias2 != "neutral" else "long")
        exit2 = simple_exit_plan(df, bias=bias2 if bias2 != "neutral" else "long", atr_factor=2.5, time_stop_days=20)
        recs.append(Rec(ticker, "1m", bias2, float(last["adj close"]), exit2.hard_stop, tgt, s2, f"SMA50/200 trend. {note2} Trailing stop={exit2.trailing_stop:.2f}".strip()))

        # 3m
        s3 = 0.5 * self.score_medium_term(df) + 0.5 * self.score_long_term(df)
        s3, note3 = self.apply_event_penalty(ticker, "3m", s3)
        bias3 = "long" if s3 >= 0.65 else ("short" if last["adj close"] < last["sma50"] else "neutral")
        stop, tgt = self.gen_stop_target(df, bias3 if bias3 != "neutral" else "long")
        exit3 = simple_exit_plan(df, bias=bias3 if bias3 != "neutral" else "long", atr_factor=3.0, time_stop_days=45)
        recs.append(Rec(ticker, "3m", bias3, float(last["adj close"]), exit3.hard_stop, tgt, s3, f"Medium+long trend blend. {note3} Trailing stop={exit3.trailing_stop:.2f}".strip()))

        # 6m
        s4 = self.score_long_term(df)
        bias4 = "long" if s4 >= 0.6 else "neutral"
        stop, tgt = self.gen_stop_target(df, "long")
        exit4 = simple_exit_plan(df, bias="long", atr_factor=3.0, time_stop_days=90)
        recs.append(Rec(ticker, "6m", bias4, float(last["adj close"]), exit4.hard_stop, tgt, s4, "Long-term trend"))

        # 1y
        s5 = self.score_long_term(df)
        bias5 = "long" if s5 >= 0.65 else "neutral"
        stop, tgt = self.gen_stop_target(df, "long")
        exit5 = simple_exit_plan(df, bias="long", atr_factor=3.5, time_stop_days=180)
        recs.append(Rec(ticker, "1y", bias5, float(last["adj close"]), exit5.hard_stop, tgt, s5, "Long-term trend & low vol"))

        return recs
