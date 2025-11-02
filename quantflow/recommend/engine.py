from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import math

import pandas as pd

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
