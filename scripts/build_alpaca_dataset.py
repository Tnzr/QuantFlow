#!/usr/bin/env python3
"""Build intraday dataset from Alpaca historical bars (5+ years of 5-min data).

Drops into the existing feature computation pipeline — replaces yfinance fetch
with Alpaca REST API. Features and labels are identical to intraday_5m.parquet.

Usage:
    python scripts/build_alpaca_dataset.py --tickers 100 --years 5 --output data/alpaca_5m.parquet

Requires: ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables.
"""
import argparse, logging, time, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load .env from project root (for ALPACA_API_KEY / ALPACA_SECRET_KEY)
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass

from quantflow.data.alpaca_client import fetch_bars, is_configured

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Reuse existing feature names
SECTORS = [
    "Technology", "Healthcare", "Financial Services", "Consumer Cyclical",
    "Industrials", "Communication Services", "Consumer Defensive",
    "Energy", "Basic Materials", "Real Estate", "Utilities", "Unknown",
]


def fetch_alpaca_history(
    ticker: str, years: int = 5, timeframe: str = "5Min", feed: str = "iex",
    chunk_days: int = 365,
) -> pd.DataFrame:
    """Fetch multi-year history in chunks to respect pagination limits."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=years * 365)
    all_dfs = []

    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=chunk_days), end)
        try:
            df = fetch_bars(
                ticker, timeframe=timeframe, feed=feed,
                start=chunk_start.isoformat(), end=chunk_end.isoformat(),
                limit=10000,
            )
            if not df.empty:
                all_dfs.append(df)
        except Exception as exc:
            logger.debug(f"  Chunk {chunk_start.date()}→{chunk_end.date()} failed: {exc}")

        chunk_start = chunk_end
        time.sleep(0.15)  # rate limit

    if not all_dfs:
        return pd.DataFrame()

    result = pd.concat(all_dfs)
    # Flatten index: Alpaca returns DatetimeIndex named "date"
    result = result[~result.index.duplicated(keep="first")]
    result = result.sort_index()
    return result


def compute_features_from_ohlcv(df: pd.DataFrame, ticker: str,
                                 sector: str = "Unknown") -> pd.DataFrame:
    """Compute the same 27 features as build_intraday_dataset.py from raw OHLCV."""
    close = df["close"].astype(float).values
    open_ = df.get("open", df["close"]).astype(float).values
    high = df.get("high", df["close"]).astype(float).values
    low = df.get("low", df["close"]).astype(float).values
    volume = df.get("volume", pd.Series(1, index=df.index)).astype(float).values
    n = len(close)

    ret = np.zeros(n, dtype=np.float32)
    ret[1:] = (close[1:] - close[:-1]) / np.maximum(close[:-1], 0.01)
    ret_12 = np.zeros(n, dtype=np.float32)
    ret_30 = np.zeros(n, dtype=np.float32)
    ret_60 = np.zeros(n, dtype=np.float32)
    ret_126 = np.zeros(n, dtype=np.float32)
    for i in range(12, n): ret_12[i] = (close[i] - close[i-12]) / np.maximum(close[i-12], 0.01)
    for i in range(30, n): ret_30[i] = (close[i] - close[i-30]) / np.maximum(close[i-30], 0.01)
    for i in range(60, n): ret_60[i] = (close[i] - close[i-60]) / np.maximum(close[i-60], 0.01)
    for i in range(126, n): ret_126[i] = (close[i] - close[i-126]) / np.maximum(close[i-126], 0.01)

    vol_mean_20 = pd.Series(volume).rolling(20, min_periods=1).mean().values
    vol_mean_5 = pd.Series(volume).rolling(5, min_periods=1).mean().values
    vol_norm = volume / np.maximum(vol_mean_20, 1.0)
    rel_vol_5 = volume / np.maximum(vol_mean_5, 1.0)
    vol_delta = np.sign(close - open_) * volume
    cum_delta_12 = pd.Series(vol_delta).rolling(12, min_periods=1).sum().values

    volatility_5 = pd.Series(ret).rolling(5, min_periods=1).std().values
    volatility_12 = pd.Series(ret).rolling(12, min_periods=1).std().values

    hl_range = high - low
    spread_pct = np.where(close > 0, hl_range / close, 0.0).astype(np.float32)
    body = np.abs(close - open_)
    body_pct = np.where(hl_range > 1e-8, body / hl_range, 0.0).astype(np.float32)
    upper_wick = np.where(hl_range > 1e-8, (high - np.maximum(open_, close)) / hl_range, 0.0).astype(np.float32)
    lower_wick = np.where(hl_range > 1e-8, (np.minimum(open_, close) - low) / hl_range, 0.0).astype(np.float32)
    vwap_gap = np.where(close > 0, (close - (high + low + close) / 3.0) / close, 0.0).astype(np.float32)

    price_vol_corr = pd.Series({
        i: float(pd.Series(ret[max(0,i-12):i+1]).corr(pd.Series(vol_delta[max(0,i-12):i+1])))
        if i >= 5 else 0.0 for i in range(n)
    }).fillna(0.0).values

    close_ma_20 = pd.Series(close).rolling(20, min_periods=1).mean().values
    close_norm = close / np.maximum(close_ma_20, 0.01) - 1.0
    high_rel = high / np.maximum(close, 0.01) - 1.0
    low_rel = low / np.maximum(close, 0.01) - 1.0
    open_gap = np.zeros(n, dtype=np.float32)
    open_gap[1:] = (open_[1:] - close[:-1]) / np.maximum(close[:-1], 0.01)
    vol_ma_60 = pd.Series(volume).rolling(60, min_periods=1).mean().values
    vol_profile = volume / np.maximum(vol_ma_60, 1.0)

    ts = pd.DatetimeIndex(df.index)
    dow = ts.dayofweek.values
    hour = ts.hour.values + ts.minute.values / 60.0
    dow_sin = np.sin(2 * np.pi * dow / 5.0).astype(np.float32)
    dow_cos = np.cos(2 * np.pi * dow / 5.0).astype(np.float32)
    hour_sin = np.sin(2 * np.pi * hour / 6.5).astype(np.float32)
    hour_cos = np.cos(2 * np.pi * hour / 6.5).astype(np.float32)
    min_from_open = ((hour - 9.5) * 60).clip(0, 390).astype(np.float32) / 390.0

    # Sector one-hot
    sector_idx = SECTORS.index(sector) if sector in SECTORS else len(SECTORS) - 1

    # Build sample rows
    rows = []
    lookback = 60
    forecast_bars = 78  # 1 trading day
    snapshot_step = 5

    for i in range(lookback, n - forecast_bars, snapshot_step):
        future_78 = close[min(i + forecast_bars, n - 1)]
        target_1d = float((future_78 / close[i]) - 1.0) if close[i] > 0 else 0.0

        rows.append({
            "ticker": ticker,
            "sector_idx": sector_idx,
            "ret_1bar": float(ret[i]),
            "ret_12bar": float(ret_12[i]),
            "ret_30bar": float(ret_30[i]),
            "ret_60bar": float(ret_60[i]),
            "ret_126bar": float(ret_126[i]),
            "vol_norm": float(vol_norm[i]),
            "vol_delta": float(vol_delta[i]) / 1e6,
            "cum_delta": float(cum_delta_12[i]) / 1e7,
            "volatility_5": float(volatility_5[i]),
            "volatility_12": float(volatility_12[i]),
            "spread_pct": float(spread_pct[i]),
            "body_pct": float(body_pct[i]),
            "upper_wick": float(upper_wick[i]),
            "lower_wick": float(lower_wick[i]),
            "vwap_gap": float(vwap_gap[i]),
            "rel_vol_5": float(rel_vol_5[i]),
            "price_vol_corr": float(price_vol_corr[i]),
            "close_norm": float(close_norm[i]),
            "high_rel": float(high_rel[i]),
            "low_rel": float(low_rel[i]),
            "open_gap": float(open_gap[i]),
            "vol_profile": float(vol_profile[i]),
            "dow_sin": float(dow_sin[i]),
            "dow_cos": float(dow_cos[i]),
            "hour_sin": float(hour_sin[i]),
            "hour_cos": float(hour_cos[i]),
            "min_from_open": float(min_from_open[i]),
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
            "target_direction_5d": 0,
            "target_direction_21d": int(np.sign(target_1d)),
        })
    return pd.DataFrame(rows)


def build_alpaca_dataset(
    tickers: list,
    years: int = 5,
    timeframe: str = "5Min",
    feed: str = "iex",
    export_path: Optional[str] = None,
):
    if not is_configured():
        raise RuntimeError(
            "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set. "
            "Get a free paper-trading key at https://alpaca.markets."
        )

    all_rows = []
    failed = []
    sectors_cache = {}

    for ticker in tqdm(tickers, desc="Fetching Alpaca 5-min data"):
        try:
            df = fetch_alpaca_history(ticker, years=years, timeframe=timeframe, feed=feed)
            if df is None or len(df) < 1000:
                logger.warning(f"Skip {ticker}: {len(df) if df is not None else 0} bars")
                failed.append(ticker)
                continue
        except Exception as exc:
            logger.warning(f"Skip {ticker}: {exc}")
            failed.append(ticker)
            continue

        sector = sectors_cache.get(ticker, "Unknown")
        feat_df = compute_features_from_ohlcv(df, ticker, sector)
        all_rows.append(feat_df)
        logger.debug(f"  {ticker}: {len(feat_df)} samples from {len(df)} bars")
        time.sleep(0.3)

    if not all_rows:
        logger.error("No data fetched for any ticker")
        return None

    result = pd.concat(all_rows, ignore_index=True)
    if export_path:
        result.to_parquet(export_path, index=False)
        logger.info(f"Saved {len(result)} rows to {export_path}")
        tg = result.groupby("ticker")
        logger.info(f"Tickers: {tg.ngroups}, rows/ticker: mean={tg.size().mean():.0f}")
        logger.info(f"Class dist: {result.event_state_code.value_counts().to_dict()}")
        if failed:
            logger.info(f"Failed tickers: {len(failed)} {failed[:10]}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", type=int, default=50,
                        help="Number of tickers to fetch from SECTOR_UNIVERSE")
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--timeframe", default="5Min")
    parser.add_argument("--feed", default="iex", choices=["iex", "sip"])
    parser.add_argument("--output", default="data/alpaca_5m.parquet")
    parser.add_argument("--ticker-list", nargs="*", default=None,
                        help="Specific tickers (overrides --tickers count)")
    args = parser.parse_args()

    from quantflow.data.universe import HIGH_INTEREST

    if args.ticker_list:
        tickers = args.ticker_list
    else:
        tickers = list(HIGH_INTEREST)[:args.tickers]

    logger.info(f"Building Alpaca dataset: {len(tickers)} tickers, {args.years} years, {args.timeframe}")
    build_alpaca_dataset(tickers, years=args.years, timeframe=args.timeframe,
                         feed=args.feed, export_path=args.output)
