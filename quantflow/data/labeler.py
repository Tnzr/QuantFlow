from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd

from ..features.indicators import fetch_ohlcv, compute_indicators
from ..features.analytics import forecast_prices, support_resistance_from_distribution
from ..data.universe import HIGH_INTEREST

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# Event-state definitions for ML/DL training
# ---------------------------------------------------------------------------
@dataclass
class EventLabels:
    """Labels derived from forward-looking returns for supervised training."""
    target_1d: Optional[float]      # 1-day forward return
    target_5d: Optional[float]      # 5-day forward return
    target_21d: Optional[float]    # 21-day forward return
    target_direction_5d: int        # -1=down, 0=flat, 1=up  (for classification)
    target_direction_21d: int       # -1=down, 0=flat, 1=up
    event_state: str                # "inter_event", "pre_event", "onset"  (per methodology §2.1)
    event_state_code: int           # 0=inter_event, 1=pre_event, 2=onset
    drawdown_5d_max: Optional[float]  # max drawdown over next 5 days
    drawdown_21d_max: Optional[float] # max drawdown over next 21 days
    is_volatile: bool               # whether the forward window was abnormally volatile


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


def compute_forward_labels(
    df: pd.DataFrame,
    idx: int,
    onset_drawdown_threshold: float = 0.05,
    pre_event_drawdown_threshold: float = 0.02,
    flat_threshold: float = 0.005,
    use_directional_labels: bool = True,
) -> EventLabels:
    """Compute forward-looking labels for a feature snapshot at position `idx`.

    Uses only data from idx+1 onwards to avoid lookahead bias.

    When use_directional_labels=True (default since 2026-07-19), class codes encode:
        0 = downtrend/sell  (forward 21d return < -flat_threshold)
        1 = hold/flat       (|forward 21d return| <= flat_threshold)
        2 = uptrend/buy      (forward 21d return > +flat_threshold)

    These are directly actionable trading signals: a predicted class 2 = BUY signal,
    class 0 = SELL signal, class 1 = HOLD. This replaces the prior non-directional
    drawdown-severity labeling (inter/pre/onset) which was adapted from an epilepsy
    seizure prediction methodology and could not produce directional trading signals.

    When use_directional_labels=False (legacy), the original drawdown-based scheme
    is used for backward-compatible dataset rebuilds.
    """
    price_col = "adj close" if "adj close" in df.columns else "close"
    if idx >= len(df) - 1:
        return EventLabels(None, None, None, 0, 0, "inter_event", 0, None, None, False)

    current_price = float(df.iloc[idx][price_col])

    def _forward_return(n: int) -> Optional[float]:
        target_idx = min(idx + n, len(df) - 1)
        if target_idx <= idx:
            return None
        future_price = float(df.iloc[target_idx][price_col])
        return (future_price / current_price) - 1.0 if current_price else None

    def _max_drawdown(n: int) -> Optional[float]:
        window = df.iloc[idx + 1 : min(idx + 1 + n, len(df))]
        if len(window) < 2:
            return None
        prices = window[price_col].astype(float)
        if prices.empty:
            return None
        peak = prices.expanding().max()
        dd = (prices / peak) - 1.0
        return float(dd.min())

    def _direction(ret: Optional[float]) -> int:
        if ret is None:
            return 0
        if ret > flat_threshold:
            return 1
        if ret < -flat_threshold:
            return -1
        return 0

    def _volatility_flag(n: int) -> bool:
        window = df.iloc[idx + 1 : min(idx + 1 + n, len(df))]
        if len(window) < 3:
            return False
        trailing = df.iloc[max(0, idx - 20) : idx + 1]
        if len(trailing) < 10:
            return False
        fwd_vol = float(window[price_col].pct_change().std() * np.sqrt(252))
        trail_vol = float(trailing[price_col].pct_change().std() * np.sqrt(252))
        return bool(fwd_vol > 2.0 * max(trail_vol, 0.01))

    ret_1d = _forward_return(1)
    ret_5d = _forward_return(5)
    ret_21d = _forward_return(21)
    dd5 = _max_drawdown(5)
    dd21 = _max_drawdown(21)

    if use_directional_labels:
        # Directional classification: uptrend / hold / downtrend
        # Maps directly to BUY / HOLD / SELL trading signals
        dir21 = _direction(ret_21d)
        if dir21 > 0:
            event_state = "uptrend"
            event_state_code = 2
        elif dir21 < 0:
            event_state = "downtrend"
            event_state_code = 0
        else:
            event_state = "hold"
            event_state_code = 1
    else:
        # Legacy drawdown-severity classification (non-directional)
        if dd5 is not None and dd5 <= -onset_drawdown_threshold:
            event_state = "onset"
            event_state_code = 2
        elif dd21 is not None and dd21 <= -pre_event_drawdown_threshold:
            event_state = "pre_event"
            event_state_code = 1
        elif ret_21d is not None and ret_21d < -flat_threshold:
            event_state = "pre_event"
            event_state_code = 1
        else:
            event_state = "inter_event"
            event_state_code = 0

    return EventLabels(
        target_1d=ret_1d,
        target_5d=ret_5d,
        target_21d=ret_21d,
        target_direction_5d=_direction(ret_5d),
        target_direction_21d=_direction(ret_21d),
        event_state=event_state,
        event_state_code=event_state_code,
        drawdown_5d_max=dd5,
        drawdown_21d_max=dd21,
        is_volatile=_volatility_flag(21),
    )


def _build_features_at(
    df: pd.DataFrame,
    idx: int,
    ticker: str,
    period: str = "2y",
    interval: str = "1d",
    forecast_horizon: int = 20,
) -> Dict[str, Any]:
    """Extract feature vector at a specific historical row `idx`.

    Reuses the same feature logic as build_training_row() but for an arbitrary
    point in the DataFrame rather than the last row.
    """
    if idx < 0 or idx >= len(df):
        raise ValueError(f"Index {idx} out of bounds for {ticker} (len={len(df)})")

    # Slice up to idx (inclusive) — causal: no future data
    df_slice = df.iloc[: idx + 1].copy()
    if len(df_slice) < 50:
        # Not enough history for reliable indicators
        return {"ticker": ticker.upper(), "as_of_date": df.index[idx].to_pydatetime(),
                "error": "insufficient_history", "period": period, "interval": interval}

    last = df_slice.iloc[-1]
    price_col = "adj close" if "adj close" in df_slice.columns else "close"

    # Compute forecast on the slice
    try:
        forecast = forecast_prices(df_slice, horizon=forecast_horizon)
    except Exception:
        forecast = {"forecast_return_pct": None, "confidence_band_pct": None, "daily_trend_pct": None}

    # Compute support/resistance on the slice
    try:
        distribution = support_resistance_from_distribution(df_slice, bins=24, level_count=4)
    except Exception:
        distribution = {
            "support_gap_pct": None,
            "resistance_gap_pct": None,
            "nearest_support": None,
            "nearest_resistance": None,
        }

    # Composite scoring via RuleEngine on the slice
    from ..recommend.engine import RuleEngine  # lazy import to avoid circular import
    try:
        engine = RuleEngine()
        analysis = engine._score_breakdown(df_slice)
        last_price = float(last[price_col])
        last_open = float(last.get("open", last_price))
        last_high = float(last.get("high", last_price))
        last_low = float(last.get("low", last_price))
        last_close = float(last.get("close", last_price))
        prev_row = df_slice.iloc[-2] if len(df_slice) > 1 else last
        prev_close = float(prev_row.get(price_col, last_price))
        nearest_support = df_slice.iloc[-1].get("nearest_support", distribution.get("nearest_support"))
        nearest_resistance = df_slice.iloc[-1].get("nearest_resistance", distribution.get("nearest_resistance"))
        base = (
            0.30 * analysis["trend"]
            + 0.25 * analysis["momentum"]
            + 0.15 * analysis["rsi_quality"]
            + 0.15 * analysis["volatility_regime"]
            + 0.15 * analysis["atr_efficiency"]
        )
        support_bonus = 0.0
        if nearest_support and last_price:
            support_gap = max(0.0, (last_price / nearest_support) - 1.0)
            support_bonus = max(0.0, 0.15 - support_gap)
        resistance_penalty = 0.0
        if nearest_resistance and last_price:
            resistance_gap = max(0.0, (nearest_resistance / last_price) - 1.0)
            resistance_penalty = max(0.0, 0.08 - resistance_gap)
        composite_score = max(0.0, min(1.0, base + support_bonus - resistance_penalty))
    except Exception:
        analysis = {
            "trend": None, "momentum": None, "rsi_quality": None,
            "volatility_regime": None, "atr_efficiency": None,
        }
        composite_score = None

    feature = {
        "ticker": ticker.upper(),
        "as_of_date": df.index[idx].to_pydatetime(),
        "period": period,
        "interval": interval,
        # Price / OHLCV
        "price": _safe_float(last.get(price_col)),
        "open": _safe_float(last.get("open")),
        "high": _safe_float(last.get("high")),
        "low": _safe_float(last.get("low")),
        "close": _safe_float(last.get("close")),
        "volume": _safe_float(last.get("volume")),
        # Returns
        "ret_1d": _safe_float(df_slice[price_col].pct_change(1).iloc[-1]),
        "ret_5d": _safe_float(df_slice[price_col].pct_change(5).iloc[-1]) if len(df_slice) > 5 else None,
        "ret_21d": _safe_float(df_slice[price_col].pct_change(21).iloc[-1]) if len(df_slice) > 21 else None,
        "ret_63d": _safe_float(df_slice[price_col].pct_change(63).iloc[-1]) if len(df_slice) > 63 else None,
        # Volume
        "volume_5d_mean": _safe_float(df_slice["volume"].tail(5).mean()) if "volume" in df_slice.columns else None,
        "volume_20d_mean": _safe_float(df_slice["volume"].tail(20).mean()) if "volume" in df_slice.columns else None,
        # Indicators
        "rsi14": _safe_float(last.get("rsi14")),
        "atr14": _safe_float(last.get("atr14")),
        "vol20": _safe_float(last.get("vol20")),
        "sma20": _safe_float(last.get("sma20")),
        "sma50": _safe_float(last.get("sma50")),
        "sma200": _safe_float(last.get("sma200")),
        # Distance to MAs
        "distance_sma20_pct": float((last[price_col] / last["sma20"]) - 1.0)
            if not pd.isna(last.get("sma20", np.nan)) and last.get("sma20", 0) != 0 else None,
        "distance_sma50_pct": float((last[price_col] / last["sma50"]) - 1.0)
            if not pd.isna(last.get("sma50", np.nan)) and last.get("sma50", 0) != 0 else None,
        "distance_sma200_pct": float((last[price_col] / last["sma200"]) - 1.0)
            if not pd.isna(last.get("sma200", np.nan)) and last.get("sma200", 0) != 0 else None,
        # Support / resistance
        "support_gap_pct": distribution.get("support_gap_pct"),
        "resistance_gap_pct": distribution.get("resistance_gap_pct"),
        "nearest_support": distribution.get("nearest_support"),
        "nearest_resistance": distribution.get("nearest_resistance"),
        # Scoring
        "composite_score": composite_score,
        "trend_score": analysis.get("trend"),
        "momentum_score": analysis.get("momentum"),
        "rsi_quality_score": analysis.get("rsi_quality"),
        "volatility_regime_score": analysis.get("volatility_regime"),
        "atr_efficiency_score": analysis.get("atr_efficiency"),
        # Forecast
        "forecast_return_pct": forecast.get("forecast_return_pct"),
        "forecast_band_pct": forecast.get("confidence_band_pct"),
        "forecast_daily_trend_pct": forecast.get("daily_trend_pct"),
        # OHLCV-derived candlestick features (relative to previous close)
        "open_log": _safe_float(np.log(last.get("open", 1.0) / prev_close)) if prev_close > 0 else None,
        "high_log": _safe_float(np.log(last.get("high", 1.0) / prev_close)) if prev_close > 0 else None,
        "low_log": _safe_float(np.log(last.get("low", 1.0) / prev_close)) if prev_close > 0 else None,
        "close_log": _safe_float(np.log(last_price / prev_close)) if prev_close > 0 else None,
        "volume_rel": _safe_float(last.get("volume", 0) / (last.get("volume_20d_mean", 1) or 1)),
        "range_pct": _safe_float((last_high - last_low) / last_price) if last_price > 0 else None,
        "gap_pct": _safe_float((last_open - prev_close) / prev_close) if prev_close > 0 else None,
        "body_pct": _safe_float(abs(last_close - last_open) / max(last_high - last_low, 0.0001)),
        "upper_wick_pct": _safe_float((last_high - max(last_open, last_close)) / max(last_high - last_low, 0.0001)),
        "lower_wick_pct": _safe_float((min(last_open, last_close) - last_low) / max(last_high - last_low, 0.0001)),
    }

    # Rolling volume and price-volume correlation
    if len(df_slice) >= 5:
        feature["volume_ratio_5d"] = _safe_float(
            last.get("volume", 0) / max(df_slice["volume"].tail(5).mean(), 1)
        )
        if len(df_slice) >= 5:
            rets = df_slice[price_col].pct_change().tail(5).dropna()
            vols = df_slice["volume"].tail(5).iloc[-len(rets):]
            if len(rets) >= 3 and vols.std() > 0:
                feature["price_vol_corr_5d"] = _safe_float(rets.corr(vols))
            else:
                feature["price_vol_corr_5d"] = 0.0
        else:
            feature["price_vol_corr_5d"] = 0.0
    else:
        feature["volume_ratio_5d"] = None
        feature["price_vol_corr_5d"] = None

    return feature


def build_labeled_dataset(
    tickers: Optional[List[str]] = None,
    period: str = "5y",
    interval: str = "1d",
    snapshot_step: int = 5,
    forecast_horizon: int = 20,
    onset_dd_threshold: float = 0.05,
    pre_event_dd_threshold: float = 0.02,
    min_history_bars: int = 100,
    max_rows: Optional[int] = None,
    export_path: Optional[str] = None,
    use_directional_labels: bool = True,
) -> pd.DataFrame:
    """Build a labeled training dataset for ML/DL handoff.

    For each ticker:
    1. Fetch OHLCV history via yfinance.
    2. Compute technical indicators.
    3. Iterate through historical dates at `snapshot_step` intervals.
    4. At each snapshot, extract features (causally: only data up to that date).
    5. Compute forward-looking labels (returns, drawdown, event state).
    6. Concatenate all rows into a single DataFrame.

    Args:
        tickers: List of ticker symbols. Uses HIGH_INTEREST (sector universe) if None.
        period: yfinance period string.
        interval: yfinance interval string.
        snapshot_step: Take a labeled row every N trading days.
        forecast_horizon: Horizon for forecast features.
        onset_dd_threshold: Drawdown threshold for onset state.
        pre_event_dd_threshold: Drawdown threshold for pre-event state.
        min_history_bars: Minimum bars before snapshots begin.
        max_rows: Cap total rows across all tickers.
        export_path: If set, also export to parquet.
    """
    if tickers is None:
        tickers = HIGH_INTEREST

    all_rows: List[Dict[str, Any]] = []
    tickers_failed: List[str] = []
    tickers_skipped: List[str] = []

    logger.info(f"Building labeled dataset for {len(tickers)} tickers...")

    for i, ticker in enumerate(tickers):
        if max_rows and len(all_rows) >= max_rows:
            logger.info(f"Reached max_rows={max_rows}, stopping at ticker {i+1}/{len(tickers)}")
            break

        try:
            # 1. Fetch and compute indicators
            df = fetch_ohlcv(ticker, period=period, interval=interval)
            if df.empty or len(df) < min_history_bars:
                tickers_skipped.append(ticker)
                logger.warning(f"  [{i+1}/{len(tickers)}] {ticker}: insufficient data ({len(df)} bars)")
                continue

            df = compute_indicators(df)

            # 2. Generate snapshots
            snapshot_indices = list(range(min_history_bars, len(df) - 21, snapshot_step))
            ticker_rows = 0

            for idx in snapshot_indices:
                # Causal feature extraction
                feature = _build_features_at(df, idx, ticker, period=period, interval=interval,
                                              forecast_horizon=forecast_horizon)
                if "error" in feature:
                    continue

                # Forward labels
                labels = compute_forward_labels(
                    df, idx,
                    onset_drawdown_threshold=onset_dd_threshold,
                    pre_event_drawdown_threshold=pre_event_dd_threshold,
                    use_directional_labels=use_directional_labels,
                )

                row = {
                    **feature,
                    "target_1d": labels.target_1d,
                    "target_5d": labels.target_5d,
                    "target_21d": labels.target_21d,
                    "target_direction_5d": labels.target_direction_5d,
                    "target_direction_21d": labels.target_direction_21d,
                    "event_state": labels.event_state,
                    "event_state_code": labels.event_state_code,
                    "drawdown_5d_max": labels.drawdown_5d_max,
                    "drawdown_21d_max": labels.drawdown_21d_max,
                    "is_volatile": labels.is_volatile,
                }
                all_rows.append(row)
                ticker_rows += 1

                if max_rows and len(all_rows) >= max_rows:
                    break

            logger.info(
                f"  [{i+1}/{len(tickers)}] {ticker}: {ticker_rows} snapshots | "
                f"total rows={len(all_rows)}"
            )

        except Exception as exc:
            tickers_failed.append(ticker)
            logger.error(f"  [{i+1}/{len(tickers)}] {ticker}: FAILED — {exc}")
            continue

    if not all_rows:
        raise RuntimeError("No labeled rows generated. Check ticker availability.")

    df_out = pd.DataFrame(all_rows)

    # Datetime conversion
    if "as_of_date" in df_out.columns:
        df_out["as_of_date"] = pd.to_datetime(df_out["as_of_date"], utc=True)

    # Log class distribution
    if "event_state_code" in df_out.columns:
        dist = df_out["event_state_code"].value_counts().to_dict()
        total = len(df_out)
        logger.info(
            f"Dataset complete: {len(df_out)} rows, {df_out['ticker'].nunique()} unique tickers"
        )
        logger.info(f"Class distribution: {dist}")
        for code, label in [(0, "inter_event"), (1, "pre_event"), (2, "onset")]:
            count = dist.get(code, 0)
            logger.info(f"  {label}: {count} ({100*count/total:.1f}%)")

    # Export
    if export_path:
        df_out.to_parquet(export_path, index=False)
        logger.info(f"Exported to {export_path}")

    return df_out