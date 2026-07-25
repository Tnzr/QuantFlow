#!/usr/bin/env python3
"""Build intraday 5-minute dataset with expanded features, sector encoding, and
volume/volatility/candlestick derivatives.

Features per bar:
    ret_1bar        - 1-bar return
    ret_12bar       - 12-bar (1h) cumulative return
    vol_norm        - volume / rolling 20-bar mean
    vol_delta       - signed volume (proxy for buy/sell pressure)
    cum_delta_12    - cumulative volume delta over 12 bars
    volatility_5    - rolling std of returns over 5 bars
    volatility_12   - rolling std over 12 bars (1h vol)
    spread_pct      - (high - low) / close
    body_pct        - |close - open| / (high - low + eps)
    upper_wick      - upper shadow ratio
    lower_wick      - lower shadow ratio
    vwap_gap        - close deviation from VWAP proxy ((H+L+C)/3)
    rel_vol_5       - volume ratio to 5-bar mean
    price_vol_corr  - rolling correlation of returns and volume_delta
    sector_*        - one-hot encoded sector

- 60-bar lookback = 5 hours of trading context
- 78-bar forecast = 1 trading day forward return
- 50 tickers from HIGH_INTEREST
"""
import argparse
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SECTORS = [
    "Technology", "Healthcare", "Financial Services", "Consumer Cyclical",
    "Industrials", "Communication Services", "Consumer Defensive",
    "Energy", "Basic Materials", "Real Estate", "Utilities", "Unknown",
]


def flatten_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c[0]).lower().strip() for c in df.columns]
    return df


def fetch_sectors(tickers: list) -> dict:
    sectors = {}
    for t in tqdm(tickers, desc="Fetching sector info"):
        try:
            info = yf.Ticker(t).info
            sectors[t] = info.get("sector", "Unknown")
        except Exception:
            sectors[t] = "Unknown"
        time.sleep(0.1)  # rate limit
    return sectors


def build_intraday_dataset(
    tickers: list,
    sectors: Optional[dict] = None,
    period: str = "60d",
    interval: str = "5m",
    lookback: int = 60,
    forecast_bars: int = 78,
    snapshot_step: int = 5,
    export_path: str = "data/intraday_5m.parquet",
):
    if sectors is None:
        sectors = {}

    rows = []
    failed = []

    for ticker in tqdm(tickers, desc="Fetching intraday 5m data"):
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False)
            if df is None or len(df) < lookback + forecast_bars + 20:
                logger.warning(f"Skip {ticker}: {len(df) if df is not None else 0} bars")
                failed.append(ticker)
                continue
            df = flatten_multiindex(df)
        except Exception:
            logger.warning(f"Skip {ticker}: fetch failed")
            failed.append(ticker)
            continue

        close = df["close"].astype(float).values
        open_ = df.get("open", df["close"]).astype(float).values
        high = df.get("high", df["close"]).astype(float).values
        low = df.get("low", df["close"]).astype(float).values
        volume = df.get("volume", pd.Series(1, index=df.index)).astype(float).values

        n = len(close)

        ret = np.zeros(n, dtype=np.float32)
        ret[1:] = (close[1:] - close[:-1]) / np.maximum(close[:-1], 0.01)

        ret_12 = np.zeros(n, dtype=np.float32)
        for i in range(12, n):
            ret_12[i] = (close[i] - close[i - 12]) / np.maximum(close[i - 12], 0.01)

        vol_mean_20 = pd.Series(volume).rolling(20, min_periods=1).mean().values
        vol_mean_5 = pd.Series(volume).rolling(5, min_periods=1).mean().values
        vol_norm = volume / np.maximum(vol_mean_20, 1.0)
        rel_vol_5 = volume / np.maximum(vol_mean_5, 1.0)

        vol_delta = np.sign(close - open_) * volume
        cum_delta_12 = pd.Series(vol_delta).rolling(12, min_periods=1).sum().values

        # Multi-horizon returns for pattern recognition at different timeframes
        ret_30 = np.zeros(n, dtype=np.float32)
        ret_60 = np.zeros(n, dtype=np.float32)
        ret_126 = np.zeros(n, dtype=np.float32)
        for i in range(30, n): ret_30[i] = (close[i] - close[i-30]) / np.maximum(close[i-30], 0.01)
        for i in range(60, n): ret_60[i] = (close[i] - close[i-60]) / np.maximum(close[i-60], 0.01)
        for i in range(126, n): ret_126[i] = (close[i] - close[i-126]) / np.maximum(close[i-126], 0.01)

        # Raw OHLCV context: normalized price relationships
        close_ma_20 = pd.Series(close).rolling(20, min_periods=1).mean().values
        close_norm = close / np.maximum(close_ma_20, 0.01) - 1.0  # deviation from 20-bar mean
        high_rel = high / np.maximum(close, 0.01) - 1.0           # high relative to close
        low_rel = low / np.maximum(close, 0.01) - 1.0             # low relative to close
        open_gap = np.zeros(n, dtype=np.float32)                   # gap from prev close
        open_gap[1:] = (open_[1:] - close[:-1]) / np.maximum(close[:-1], 0.01)
        vol_ma_60 = pd.Series(volume).rolling(60, min_periods=1).mean().values
        vol_profile = volume / np.maximum(vol_ma_60, 1.0)          # volume vs hour average

        volatility_5 = pd.Series(ret).rolling(5, min_periods=1).std().values
        volatility_12 = pd.Series(ret).rolling(12, min_periods=1).std().values

        hl_range = high - low
        spread_pct = np.where(close > 0, hl_range / close, 0.0).astype(np.float32)

        body = np.abs(close - open_)
        body_pct = np.where(hl_range > 1e-8, body / hl_range, 0.0).astype(np.float32)
        upper_wick = np.where(hl_range > 1e-8, (high - np.maximum(open_, close)) / hl_range, 0.0).astype(np.float32)
        lower_wick = np.where(hl_range > 1e-8, (np.minimum(open_, close) - low) / hl_range, 0.0).astype(np.float32)

        vwap_typical = (high + low + close) / 3.0
        vwap_gap = np.where(close > 0, (close - vwap_typical) / close, 0.0).astype(np.float32)

        price_vol_corr = pd.Series({
            i: float(pd.Series(ret[max(0,i-12):i+1]).corr(pd.Series(vol_delta[max(0,i-12):i+1])))
            if i >= 5 else 0.0
            for i in range(n)
        }).fillna(0.0).values

        sector = sectors.get(ticker, "Unknown")
        sector_idx = SECTORS.index(sector) if sector in SECTORS else len(SECTORS) - 1

        # Seasonality: day-of-week + hour-of-day + minutes-from-open
        ts = pd.DatetimeIndex(df.index)
        dow = ts.dayofweek.values
        hour = ts.hour.values + ts.minute.values / 60.0
        dow_sin = np.sin(2 * np.pi * dow / 5.0).astype(np.float32)
        dow_cos = np.cos(2 * np.pi * dow / 5.0).astype(np.float32)
        hour_sin = np.sin(2 * np.pi * hour / 6.5).astype(np.float32)
        hour_cos = np.cos(2 * np.pi * hour / 6.5).astype(np.float32)
        # Minutes since 9:30 AM market open (0-390, normalized)
        min_from_open = ((hour - 9.5) * 60).clip(0, 390).astype(np.float32) / 390.0

        all_features = np.column_stack([
            ret, ret_12, ret_30, ret_60, ret_126, vol_norm, vol_delta / 1e6, cum_delta_12 / 1e7,
            volatility_5, volatility_12, spread_pct, body_pct,
            upper_wick, lower_wick, vwap_gap, rel_vol_5, price_vol_corr,
            close_norm, high_rel, low_rel, open_gap, vol_profile,
            dow_sin, dow_cos, hour_sin, hour_cos, min_from_open,
        ]).astype(np.float32)

        feature_names = [
            "ret_1bar", "ret_12bar", "ret_30bar", "ret_60bar", "ret_126bar",
            "vol_norm", "vol_delta", "cum_delta",
            "volatility_5", "volatility_12", "spread_pct", "body_pct",
            "upper_wick", "lower_wick", "vwap_gap", "rel_vol_5", "price_vol_corr",
            "close_norm", "high_rel", "low_rel", "open_gap", "vol_profile",
            "dow_sin", "dow_cos", "hour_sin", "hour_cos", "min_from_open",
        ]

        for i in range(lookback, n - forecast_bars, snapshot_step):
            future_close = close[min(i + forecast_bars, n - 1)]
            target_1d = float((future_close / close[i]) - 1.0) if close[i] > 0 else 0.0

            row = {
                "ticker": ticker,
                "sector_idx": sector_idx,
                "target_return": target_1d,
                "target_21d": target_1d,
                "tau_forward": 21.0,
                "event_state_code": 2 if target_1d > 0.0005 else 0 if target_1d < -0.0005 else 1,
                "event_state": "buy" if target_1d > 0.0005 else "sell" if target_1d < -0.0005 else "hold",
                "drawdown_5d_max": 0.0,
                "drawdown_21d_max": 0.0,
                "adj_close": float(close[i]),
                "close": float(close[i]),
                "is_volatile": False,
                "as_of_date": str(df.index[i]),
            }
            for j, name in enumerate(feature_names):
                row[name] = float(all_features[i, j])

            rows.append(row)

        time.sleep(0.2)

    df_out = pd.DataFrame(rows)
    if export_path:
        df_out.to_parquet(export_path, index=False)
        logger.info(f"Saved {len(df_out)} rows to {export_path}")
        logger.info(f"Failed tickers: {len(failed)} {failed[:10]}")
        logger.info(f"Class dist: {df_out.event_state_code.value_counts().to_dict()}")

        tg = df_out.groupby("ticker")
        logger.info(f"Tickers: {tg.ngroups}, rows/ticker: mean={tg.size().mean():.0f}")

    return df_out


if __name__ == "__main__":
    from quantflow.data.universe import HIGH_INTEREST

    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--max-tickers", type=int, default=50)
    parser.add_argument("--output", default="data/intraday_5m.parquet")
    parser.add_argument("--no-sector", action="store_true")
    args = parser.parse_args()

    tickers = args.tickers if args.tickers else list(HIGH_INTEREST)[:args.max_tickers]
    logger.info(f"Building intraday dataset for {len(tickers)} tickers")

    sectors = None if args.no_sector else fetch_sectors(tickers)

    df = build_intraday_dataset(
        tickers, sectors=sectors,
        export_path=args.output,
    )
