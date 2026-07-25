from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _clean_price_series(df: pd.DataFrame) -> pd.Series:
    prices = pd.to_numeric(df.get("adj close", df.get("close")), errors="coerce").dropna()
    if prices.empty:
        raise ValueError("No price history available for analytics")
    return prices


def _clean_volume_series(df: pd.DataFrame, index: pd.Index) -> pd.Series:
    if "volume" not in df.columns:
        return pd.Series(1.0, index=index)
    volume = pd.to_numeric(df["volume"], errors="coerce").reindex(index).fillna(0.0)
    return volume.clip(lower=0.0) + 1.0


def price_distribution_profile(df: pd.DataFrame, bins: int = 24) -> dict[str, Any]:
    prices = _clean_price_series(df)
    bins = max(8, min(int(bins), 80))
    volume = _clean_volume_series(df, prices.index)

    hist, edges = np.histogram(prices.to_numpy(), bins=bins)
    weighted_hist, _ = np.histogram(prices.to_numpy(), bins=edges, weights=volume.to_numpy())

    current_price = float(prices.iloc[-1])
    price_min = float(prices.min())
    price_max = float(prices.max())
    total = int(hist.sum())
    weighted_total = float(weighted_hist.sum())

    bins_out: list[dict[str, Any]] = []
    for idx, count in enumerate(hist):
        low = float(edges[idx])
        high = float(edges[idx + 1])
        midpoint = (low + high) / 2.0
        weighted_count = float(weighted_hist[idx])
        bins_out.append(
            {
                "price_low": low,
                "price_high": high,
                "midpoint": midpoint,
                "count": int(count),
                "weight": weighted_count,
                "density": float(count / total) if total else 0.0,
                "weighted_density": float(weighted_count / weighted_total) if weighted_total else 0.0,
                "distance_to_current_pct": float((midpoint / current_price) - 1.0) if current_price else 0.0,
            }
        )

    return {
        "current_price": current_price,
        "price_min": price_min,
        "price_max": price_max,
        "range_pct": float((price_max / price_min) - 1.0) if price_min else 0.0,
        "bins": bins_out,
    }


def support_resistance_from_distribution(df: pd.DataFrame, bins: int = 24, level_count: int = 3) -> dict[str, Any]:
    profile = price_distribution_profile(df, bins=bins)
    current_price = float(profile["current_price"])
    level_count = max(1, min(int(level_count), 6))

    support_bins = sorted(
        [b for b in profile["bins"] if b["midpoint"] <= current_price],
        key=lambda row: (row["weighted_density"], row["count"], row["midpoint"]),
        reverse=True,
    )[:level_count]
    resistance_bins = sorted(
        [b for b in profile["bins"] if b["midpoint"] >= current_price],
        key=lambda row: (row["weighted_density"], row["count"], -row["midpoint"]),
        reverse=True,
    )[:level_count]

    def as_level(row: dict[str, Any]) -> dict[str, Any]:
        level = float(row["midpoint"])
        return {
            "price": level,
            "band_low": float(row["price_low"]),
            "band_high": float(row["price_high"]),
            "weighted_density": float(row["weighted_density"]),
            "touch_count": int(row["count"]),
            "distance_pct": float((level / current_price) - 1.0) if current_price else 0.0,
        }

    support_levels = sorted([as_level(row) for row in support_bins], key=lambda row: row["price"], reverse=True)
    resistance_levels = sorted([as_level(row) for row in resistance_bins], key=lambda row: row["price"])
    nearest_support = support_levels[0]["price"] if support_levels else None
    nearest_resistance = resistance_levels[0]["price"] if resistance_levels else None

    return {
        **profile,
        "support_levels": support_levels,
        "resistance_levels": resistance_levels,
        "nearest_support": nearest_support,
        "nearest_resistance": nearest_resistance,
        "support_gap_pct": float((current_price / nearest_support) - 1.0) if nearest_support else None,
        "resistance_gap_pct": float((nearest_resistance / current_price) - 1.0) if nearest_resistance else None,
    }


def forecast_prices(df: pd.DataFrame, horizon: int = 30) -> dict[str, Any]:
    prices = _clean_price_series(df)
    horizon = max(5, min(int(horizon), 90))
    lookback = min(len(prices), max(60, horizon * 4))
    sample = prices.tail(lookback)
    x = np.arange(len(sample), dtype=float)
    y = np.log(sample.to_numpy(dtype=float))

    slope, intercept = np.polyfit(x, y, 1)
    trend = intercept + slope * x
    residuals = y - trend
    residual_std = float(np.std(residuals)) if len(residuals) > 1 else 0.0

    future_x = np.arange(len(sample), len(sample) + horizon, dtype=float)
    future_trend = intercept + slope * future_x
    seasonal_adjustment = float(np.nanmean(residuals[-min(20, len(residuals)):])) if len(residuals) else 0.0
    forecast_log = future_trend + seasonal_adjustment

    tail = sample.tail(min(40, len(sample)))
    history = [
        {"date": str(idx.date()), "price": float(value)}
        for idx, value in tail.items()
    ]

    last_date = sample.index[-1]
    forecast = []
    for step, pred in enumerate(forecast_log, start=1):
        center = float(np.exp(pred))
        upper = float(np.exp(pred + residual_std))
        lower = float(np.exp(pred - residual_std))
        forecast.append(
            {
                "date": str((last_date + pd.Timedelta(days=step)).date()),
                "price": center,
                "lower": lower,
                "upper": upper,
            }
        )

    return {
        "history": history,
        "forecast": forecast,
        "daily_trend_pct": float(np.exp(slope) - 1.0),
        "forecast_return_pct": float((forecast[-1]["price"] / float(sample.iloc[-1])) - 1.0) if forecast else 0.0,
        "confidence_band_pct": float(np.exp(residual_std) - 1.0) if residual_std else 0.0,
    }