from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..data.queries import recent_news_articles
from .indicators import fetch_ohlcv, compute_indicators
from .news import build_news_tokens


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return None


def _seasonality_tokens(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "day_of_week": None,
            "day_of_year": None,
            "dow_mean_return": None,
            "dow_win_rate": None,
            "doy_mean_return": None,
            "doy_rank_pct": None,
            "tokens": [],
        }

    work = df.copy()
    work["ret"] = work["adj close"].pct_change()
    work["dow"] = work.index.dayofweek
    work["doy"] = work.index.dayofyear

    last = work.iloc[-1]
    day_of_week = int(last.name.dayofweek)
    day_of_year = int(last.name.dayofyear)

    dow_group = work.groupby("dow")["ret"].agg(["mean", "count", lambda x: float((x > 0).mean())]).rename(columns={"mean": "dow_mean_return", "count": "dow_count", "<lambda_0>": "dow_win_rate"})
    doy_group = work.groupby("doy")["ret"].agg(["mean", "std", "count"]).rename(columns={"mean": "avg_ret", "std": "std_ret", "count": "n"}).reset_index()

    dow_row = dow_group.loc[day_of_week] if day_of_week in dow_group.index else None
    doy_row = doy_group[doy_group["doy"] == day_of_year]
    doy_mean = float(doy_row.iloc[0]["avg_ret"]) if len(doy_row) else None
    all_doy_mean = doy_group["avg_ret"].dropna()
    doy_rank_pct = None
    if doy_mean is not None and len(all_doy_mean):
        doy_rank_pct = float(all_doy_mean.rank(pct=True).loc[day_of_year]) if day_of_year in all_doy_mean.index else float(all_doy_mean.rank(pct=True).iloc[-1])

    dow_name = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"][day_of_week]
    tokens = [
        f"dow_{dow_name}",
        f"dow_{dow_name}_{'bullish' if (float(dow_row['dow_mean_return']) if dow_row is not None else 0.0) >= 0 else 'bearish'}" if dow_row is not None else f"dow_{dow_name}_neutral",
        "weekend_cycle" if day_of_week in {0, 4} else "midweek_cycle",
    ]
    if day_of_week == 4:
        tokens.append("friday_weekend_risk")
    if day_of_week == 0:
        tokens.append("monday_reversal_watch")
    if doy_rank_pct is not None:
        tokens.append("doy_tailwind" if doy_rank_pct >= 0.66 else ("doy_headwind" if doy_rank_pct <= 0.33 else "doy_neutral"))

    return {
        "day_of_week": day_of_week,
        "day_of_year": day_of_year,
        "dow_mean_return": _safe_float(dow_row["dow_mean_return"]) if dow_row is not None else None,
        "dow_win_rate": _safe_float(dow_row["dow_win_rate"]) if dow_row is not None else None,
        "doy_mean_return": doy_mean,
        "doy_rank_pct": doy_rank_pct,
        "tokens": tokens,
    }


def _news_sentiment_features(ticker: str, db_path: str, lookback_days: int = 30) -> dict[str, Any]:
    articles = recent_news_articles(db_path=db_path, days=lookback_days, ticker=ticker, limit=250)
    if articles.empty:
        return {
            "article_count": 0,
            "avg_sentiment": 0.0,
            "sentiment_std": 0.0,
            "bullish_ratio": 0.0,
            "bearish_ratio": 0.0,
            "tokens": ["news_archive_empty"],
        }

    scores = []
    tokens: list[str] = []
    for row in articles.to_dict(orient="records"):
        article = {
            "title": row.get("title"),
            "summary": row.get("summary"),
            "content": row.get("content"),
            "sentiment_score": row.get("sentiment_score"),
            "sentiment_label": row.get("sentiment_label"),
            "tickers": row.get("tickers_json"),
        }
        built = build_news_tokens(article)
        score = float(built["sentiment_score"])
        scores.append(score)
        tokens.extend(built["tokens"])

    scores_array = np.asarray(scores, dtype=float)
    return {
        "article_count": int(len(scores_array)),
        "avg_sentiment": float(scores_array.mean()) if len(scores_array) else 0.0,
        "sentiment_std": float(scores_array.std(ddof=0)) if len(scores_array) else 0.0,
        "bullish_ratio": float((scores_array >= 0.2).mean()) if len(scores_array) else 0.0,
        "bearish_ratio": float((scores_array <= -0.2).mean()) if len(scores_array) else 0.0,
        "tokens": sorted(set(tokens)) or ["news_archive_empty"],
    }


def build_training_row(
    ticker: str,
    db_path: str = "sqlite:///quantflow.db",
    period: str = "2y",
    interval: str = "1d",
    lookback_days: int = 30,
    forecast_horizon: int = 20,
) -> dict[str, Any]:
    from .news import simple_sentiment_score  # local import keeps module light for no-news use
    from .analytics import forecast_prices, support_resistance_from_distribution
    from ..recommend.engine import RuleEngine

    df = fetch_ohlcv(ticker, period=period, interval=interval)
    df = compute_indicators(df)
    if df.empty:
        raise ValueError(f"No market history available for {ticker}")

    last = df.iloc[-1]
    engine = RuleEngine()
    analysis = engine.analyze_ticker(ticker, period=period, interval=interval)
    forecast = forecast_prices(df, horizon=forecast_horizon)
    distribution = support_resistance_from_distribution(df, bins=24, level_count=4)

    seasonality = _seasonality_tokens(df)
    news = _news_sentiment_features(ticker.upper(), db_path=db_path, lookback_days=lookback_days)

    technical_tokens = [
        "price_above_sma20" if float(last["adj close"]) >= float(last["sma20"]) else "price_below_sma20",
        "price_above_sma50" if float(last["adj close"]) >= float(last["sma50"]) else "price_below_sma50",
        "price_above_sma200" if float(last["adj close"]) >= float(last["sma200"]) else "price_below_sma200",
        "rsi_over_60" if float(last["rsi14"]) >= 60 else ("rsi_under_40" if float(last["rsi14"]) <= 40 else "rsi_midrange"),
        "volume_spike" if float(last.get("volume", 0.0)) > float(df["volume"].tail(20).mean() or 0.0) * 1.5 else "volume_normal",
        "trend_strong" if analysis["score_breakdown"].get("trend", 0.0) >= 0.6 else "trend_mixed",
    ]

    sentiment_tokens = news["tokens"]
    seasonality_tokens = seasonality["tokens"]
    token_bundle = sorted(set([*technical_tokens, *seasonality_tokens, *sentiment_tokens]))

    feature_row = {
        "ticker": ticker.upper(),
        "as_of_date": df.index[-1].to_pydatetime(),
        "period": period,
        "interval": interval,
        "price": float(last["adj close"]),
        "open": _safe_float(last.get("open")),
        "high": _safe_float(last.get("high")),
        "low": _safe_float(last.get("low")),
        "close": _safe_float(last.get("close")),
        "volume": _safe_float(last.get("volume")),
        "ret_1d": _safe_float(df["adj close"].pct_change(1).iloc[-1]),
        "ret_5d": _safe_float(df["adj close"].pct_change(5).iloc[-1]) if len(df) > 5 else None,
        "ret_21d": _safe_float(df["adj close"].pct_change(21).iloc[-1]) if len(df) > 21 else None,
        "ret_63d": _safe_float(df["adj close"].pct_change(63).iloc[-1]) if len(df) > 63 else None,
        "volume_5d_mean": _safe_float(df["volume"].tail(5).mean()),
        "volume_20d_mean": _safe_float(df["volume"].tail(20).mean()),
        "rsi14": _safe_float(last.get("rsi14")),
        "atr14": _safe_float(last.get("atr14")),
        "vol20": _safe_float(last.get("vol20")),
        "sma20": _safe_float(last.get("sma20")),
        "sma50": _safe_float(last.get("sma50")),
        "sma200": _safe_float(last.get("sma200")),
        "distance_sma20_pct": float((last["adj close"] / last["sma20"]) - 1.0) if not pd.isna(last.get("sma20")) and last.get("sma20") else None,
        "distance_sma50_pct": float((last["adj close"] / last["sma50"]) - 1.0) if not pd.isna(last.get("sma50")) and last.get("sma50") else None,
        "distance_sma200_pct": float((last["adj close"] / last["sma200"]) - 1.0) if not pd.isna(last.get("sma200")) and last.get("sma200") else None,
        "support_gap_pct": distribution.get("support_gap_pct"),
        "resistance_gap_pct": distribution.get("resistance_gap_pct"),
        "nearest_support": distribution.get("nearest_support"),
        "nearest_resistance": distribution.get("nearest_resistance"),
        "composite_score": analysis["composite_score"],
        "trend_score": analysis["score_breakdown"].get("trend"),
        "momentum_score": analysis["score_breakdown"].get("momentum"),
        "rsi_quality_score": analysis["score_breakdown"].get("rsi_quality"),
        "volatility_regime_score": analysis["score_breakdown"].get("volatility_regime"),
        "atr_efficiency_score": analysis["score_breakdown"].get("atr_efficiency"),
        "forecast_return_pct": forecast.get("forecast_return_pct"),
        "forecast_band_pct": forecast.get("confidence_band_pct"),
        "forecast_daily_trend_pct": forecast.get("daily_trend_pct"),
        "seasonality_dow": seasonality["day_of_week"],
        "seasonality_doy": seasonality["day_of_year"],
        "seasonality_dow_mean_return": seasonality["dow_mean_return"],
        "seasonality_dow_win_rate": seasonality["dow_win_rate"],
        "seasonality_doy_mean_return": seasonality["doy_mean_return"],
        "seasonality_doy_rank_pct": seasonality["doy_rank_pct"],
        "news_article_count": news["article_count"],
        "news_avg_sentiment": news["avg_sentiment"],
        "news_sentiment_std": news["sentiment_std"],
        "news_bullish_ratio": news["bullish_ratio"],
        "news_bearish_ratio": news["bearish_ratio"],
        "technical_tokens": technical_tokens,
        "seasonality_tokens": seasonality_tokens,
        "sentiment_tokens": sentiment_tokens,
        "token_bundle": token_bundle,
    }
    return feature_row


def build_training_frame(
    tickers: list[str],
    db_path: str = "sqlite:///quantflow.db",
    period: str = "2y",
    interval: str = "1d",
    lookback_days: int = 30,
    forecast_horizon: int = 20,
) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        try:
            rows.append(
                build_training_row(
                    ticker=ticker,
                    db_path=db_path,
                    period=period,
                    interval=interval,
                    lookback_days=lookback_days,
                    forecast_horizon=forecast_horizon,
                )
            )
        except Exception as exc:
            rows.append({"ticker": ticker.upper(), "error": str(exc), "period": period, "interval": interval})
    return pd.DataFrame(rows)