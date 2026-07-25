from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler, random_split

from ..data.labeler import build_labeled_dataset
from ..data.universe import SECTOR_UNIVERSE
from ..features.indicators import fetch_ohlcv, compute_indicators

logger = logging.getLogger(__name__)


FEATURE_COLUMNS = [
    "ret_1d", "ret_5d", "ret_21d", "ret_63d",
    "rsi14", "atr14", "vol20",
    "distance_sma20_pct", "distance_sma50_pct", "distance_sma200_pct",
    "support_gap_pct", "resistance_gap_pct",
    "composite_score", "trend_score", "momentum_score",
    "rsi_quality_score", "volatility_regime_score", "atr_efficiency_score",
    "forecast_return_pct", "forecast_band_pct", "forecast_daily_trend_pct",
    # Raw OHLCV context — essential for the BiLSTM to learn candlestick
    # patterns and price-action structure. Previously the encoder saw only
    # derived indicators, which discard the microstructure that LSTMs excel
    # at processing across temporal windows.
    "open_log",        # log(open / close_prev)
    "high_log",        # log(high / close_prev)
    "low_log",         # log(low / close_prev)
    "close_log",       # log(close / close_prev)
    "volume_rel",      # volume / volume_20d_mean
    "range_pct",       # (high - low) / close
    "gap_pct",         # (open - close_prev) / close_prev
    "body_pct",        # abs(close - open) / (high - low + eps)
    "upper_wick_pct",  # (high - max(open,close)) / (high - low + eps)
    "lower_wick_pct",  # (min(open,close) - low) / (high - low + eps)
    "volume_ratio_5d",  # vol_t / mean_vol_5d
    "price_vol_corr_5d",  # rolling 5d correlation of returns and volume
]

LABEL_COLUMNS = [
    "event_state_code",
    "target_1d", "target_5d", "target_21d",
    "target_direction_5d", "target_direction_21d",
    "drawdown_5d_max", "drawdown_21d_max",
    "tau_forward_synthetic",
]


def _build_tau_forward(row: pd.Series) -> float:
    """Synthesize tau (time-to-significant-move) from forward return magnitude.

    With directional labels (downtrend/hold/uptrend), tau represents the
    expected holding period for a trade to reach a meaningful return.
    Higher absolute forward return = shorter tau (signal is imminent).
    Flat returns = long tau (no urgency).
    """
    code = row.get("event_state_code", 1)
    target_21d = row.get("target_21d", 0.0) or 0.0

    # Scale: closer to 0 = further from decision point
    strength = abs(target_21d) * 252  # annualize
    strength = min(strength, 1.0)

    if code == 0:  # downtrend: urgency scales with return magnitude
        return max(3.0, min(21.0, 21.0 * (1.0 - strength)))
    elif code == 2:  # uptrend: urgency scales with return magnitude
        return max(3.0, min(21.0, 21.0 * (1.0 - strength)))
    else:  # hold: no urgency
        return 21.0


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select and impute feature columns for model input.
    Auto-detects when fewer than 5 FEATURE_COLUMNS match (intraday/custom)."""
    available = [c for c in FEATURE_COLUMNS if c in df.columns]
    if len(available) < 5:
        label_cols = {"ticker", "event_state", "event_state_code", "is_volatile",
                      "tau_forward", "target_return", "target_5d", "target_21d",
                      "target_1h", "target_4h", "target_return_path",
                      "drawdown_5d_max", "drawdown_21d_max", "as_of_date",
                      "adj_close", "close", "target_direction_5d", "target_direction_21d"}
        available = sorted([c for c in df.columns if c not in label_cols and df[c].dtype != 'object'])
        logger.info(f"Intraday/custom mode: {len(available)} features auto-detected")
    data = df[available].copy()
    data = data.fillna(0.0)
    data = data.replace([np.inf, -np.inf], 0.0)
    return data


class FinancialTimeSeriesDataset(Dataset):
    """Time-series dataset for stock market event-state prediction.

    Handles ticker-grouped, chronologically ordered sequences with lookback windows.
    Each sample is a window of consecutive feature rows with forward labels.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        lookback: int = 60,
        forecast_horizon: int = 21,
        feature_cols: Optional[List[str]] = None,
        label_cols: Optional[List[str]] = None,
        ticker_col: str = "ticker",
        date_col: str = "as_of_date",
    ):
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.feature_cols = feature_cols or FEATURE_COLUMNS
        self.label_cols = label_cols or LABEL_COLUMNS

        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values([ticker_col, date_col]).reset_index(drop=True)

        feature_df = _prepare_features(df)
        available_features = [c for c in self.feature_cols if c in feature_df.columns]
        if len(available_features) < 5:
            exclude = set(self.label_cols + [ticker_col, date_col])
            available_features = sorted([c for c in feature_df.columns
                                        if c not in exclude and feature_df[c].dtype != 'object'])
            logger.info(f"Auto-detected {len(available_features)} feature columns from {len(feature_df.columns)} candidates")
        self.feature_cols = available_features
        self._samples: List[Tuple[np.ndarray, Dict[str, np.ndarray], str, str]] = []
        # (ticker, start_idx, end_idx_exclusive) into self._samples, in insertion
        # order — used by TickerBatchSampler to guarantee no batch ever spans a
        # ticker boundary (methodology §4.3 point 3: sequence-level sampling).
        self.ticker_ranges: List[Tuple[str, int, int]] = []

        for ticker in df[ticker_col].unique():
            ticker_mask = df[ticker_col].values == ticker
            ticker_idx = np.where(ticker_mask)[0].tolist()
            if len(ticker_idx) <= self.lookback:
                continue
            dates = df[date_col].values[ticker_mask]
            ticker_features = feature_df.iloc[ticker_idx].values.astype(np.float32)
            ticker_labels = df.iloc[ticker_idx]

            range_start = len(self._samples)
            for i in range(self.lookback, len(ticker_idx) - 1):
                window = ticker_features[i - self.lookback:i]
                row_labels = ticker_labels.iloc[i]

                tau = _build_tau_forward(row_labels)
                dd5 = row_labels.get("drawdown_5d_max", 0.0) or 0.0
                dd21 = row_labels.get("drawdown_21d_max", 0.0) or 0.0
                date_str = str(dates[i])[:10] if i < len(dates) else ""
                # Ordinal day count survives the tensor-type filter in
                # __getitem__ (plain Python str does not) so real dates can be
                # reconstructed downstream for plotting/monitoring.
                try:
                    date_ordinal = np.int64(pd.Timestamp(date_str).toordinal()) if date_str else np.int64(0)
                except Exception:
                    date_ordinal = np.int64(0)

                label_dict = {
                    "event_state_code": np.array(int(row_labels.get("event_state_code", 0)), dtype=np.int64),
                    "tau_forward": np.array(tau, dtype=np.float32),
                    "target_return": np.array(row_labels.get("target_21d", 0.0) or 0.0, dtype=np.float32),
                    "target_5d": np.array(row_labels.get("target_5d", 0.0) or 0.0, dtype=np.float32),
                    "target_21d": np.array(row_labels.get("target_21d", 0.0) or 0.0, dtype=np.float32),
                    "target_direction_5d": np.array(int(row_labels.get("target_direction_5d", 0)), dtype=np.int64),
                    "target_direction_21d": np.array(int(row_labels.get("target_direction_21d", 0)), dtype=np.int64),
                    "drawdown_5d_max": np.array(dd5, dtype=np.float32),
                    "drawdown_21d_max": np.array(dd21, dtype=np.float32),
                    "adj_close": np.array(float(row_labels.get("close", 0.0) or 0.0), dtype=np.float32),
                    "date_ordinal": np.array(date_ordinal, dtype=np.int64),
                    "as_of_date": date_str,
                }
                self._samples.append((window, label_dict, ticker, date_str))
            if len(self._samples) > range_start:
                self.ticker_ranges.append((ticker, range_start, len(self._samples)))

        self.feature_dim = len(self.feature_cols)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], str]:
        features, labels, ticker, date = self._samples[idx]
        tensor_labels = {}
        for k, v in labels.items():
            if isinstance(v, np.ndarray) or isinstance(v, (int, float)):
                tensor_labels[k] = torch.tensor(v)
        return (
            torch.from_numpy(features),
            tensor_labels,
            ticker,
        )

    @property
    def class_distribution(self) -> Dict[str, int]:
        counts = {0: 0, 1: 0, 2: 0}
        for _, labels, _, _ in self._samples:
            code = int(labels["event_state_code"].item() if hasattr(labels["event_state_code"], 'item') else labels["event_state_code"])
            if code in counts:
                counts[code] += 1
        return counts

    def _get_metadata_by_idx(self, idx: int) -> dict:
        _, labels, ticker, date = self._samples[idx]
        return {
            "ticker": ticker,
            "date": date,
            "adj_close": float(labels.get("adj_close", 0.0)),
        }

    @property
    def sample_weights(self) -> np.ndarray:
        dist = self.class_distribution
        total = sum(dist.values())
        class_weight = {
            c: total / max(count, 1) for c, count in dist.items()
        }
        weights = np.zeros(len(self._samples), dtype=np.float64)
        for i, (_, labels, _, _) in enumerate(self._samples):
            code = int(labels["event_state_code"].item())
            weights[i] = class_weight.get(code, 1.0)
        return weights

    def get_balanced_sampler(self) -> WeightedRandomSampler:
        return WeightedRandomSampler(
            weights=torch.from_numpy(self.sample_weights).double(),
            num_samples=len(self._samples),
            replacement=True,
        )


class TickerBatchSampler(Sampler[List[int]]):
    """Yields batches that never span a ticker boundary.

    `FinancialTimeSeriesDataset` concatenates all tickers' chronological
    windows into one flat sample list. Plain sequential `batch_size` slicing
    can therefore produce a batch containing rows from two different tickers
    whenever a ticker's sample count isn't an exact multiple of `batch_size`.
    `Trainer.fit()` carries stateful hidden state keyed by a single ticker per
    batch — a boundary-spanning batch silently applies the wrong ticker's
    hidden state to part of the batch, violating methodology §4.3 point 3
    ("sequence-level, not sample-level, sampling"). This sampler guarantees
    every yielded batch is drawn from exactly one ticker's contiguous,
    chronologically-ordered range, with `drop_last` applied per-ticker.
    """

    def __init__(self, ticker_ranges: List[Tuple[str, int, int]], batch_size: int, drop_last: bool = True):
        self.ticker_ranges = ticker_ranges
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        for _ticker, start, end in self.ticker_ranges:
            idx = list(range(start, end))
            for i in range(0, len(idx), self.batch_size):
                chunk = idx[i:i + self.batch_size]
                if len(chunk) < self.batch_size and self.drop_last:
                    continue
                yield chunk

    def __len__(self) -> int:
        n = 0
        for _ticker, start, end in self.ticker_ranges:
            length = end - start
            n += (length // self.batch_size) if self.drop_last else -(-length // self.batch_size)
        return n


def build_dataset(
    tickers: Optional[List[str]] = None,
    period: str = "5y",
    interval: str = "1d",
    lookback: int = 60,
    forecast_horizon: int = 21,
    snapshot_step: int = 3,
    max_rows: Optional[int] = None,
    export_path: Optional[str] = None,
) -> pd.DataFrame:
    """Build a labeled dataset using the existing labeler pipeline and prepare for ML training."""
    if tickers is None:
        tickers = SECTOR_UNIVERSE.get("Financials", []) + SECTOR_UNIVERSE.get("Financials — Alternatives", [])
        logger.info(f"Using Financials sector tickers: {tickers}")

    logger.info(f"Building labeled dataset for {len(tickers)} tickers (period={period}, interval={interval})...")
    df = build_labeled_dataset(
        tickers=tickers,
        period=period,
        interval=interval,
        snapshot_step=snapshot_step,
        forecast_horizon=forecast_horizon,
        onset_dd_threshold=0.05,
        pre_event_dd_threshold=0.02,
        max_rows=max_rows,
        export_path=export_path,
    )
    df["tau_forward_synthetic"] = df.apply(_build_tau_forward, axis=1)
    return df


def prepare_dataloaders(
    df: pd.DataFrame,
    lookback: int = 60,
    forecast_horizon: int = 21,
    batch_size: int = 64,
    val_split: float = 0.15,
    test_split: float = 0.10,
    num_workers: int = 0,
    seed: int = 42,
    shuffle: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """Split dataset and return DataLoaders.

    Per §4.3 and §8.2 of the methodology:
    - Samples are grouped by ticker and presented chronologically within each ticker.
    - No random shuffling — hidden states carry across batches within a ticker.
    - Ticker order is deterministic (sorted by ticker symbol) for reproducibility.
    - Train/val/test split is done at ticker level to prevent data leakage.

    When shuffle=True (Phase A of two-phase curriculum), uses plain shuffled
    DataLoader instead of TickerBatchSampler. This breaks stateful hidden-state
    carry and ticker-boundary safety, but enables the model to learn separable
    features from diverse batch composition before fine-tuning chronologically.
    """
    tickers = df["ticker"].unique()
    np.random.seed(seed)
    np.random.shuffle(tickers)

    n_val = max(1, int(len(tickers) * val_split))
    n_test = max(1, int(len(tickers) * test_split))
    n_train = len(tickers) - n_val - n_test

    train_tickers = sorted(tickers[:n_train])
    val_tickers = sorted(tickers[n_train:n_train + n_val])
    test_tickers = sorted(tickers[n_train + n_val:])

    train_df = df[df["ticker"].isin(train_tickers)].copy()
    val_df = df[df["ticker"].isin(val_tickers)].copy()
    test_df = df[df["ticker"].isin(test_tickers)].copy()

    logger.info(f"Split: train={len(train_df)} rows ({n_train} tickers), "
                f"val={len(val_df)} rows ({n_val} tickers), "
                f"test={len(test_df)} rows ({n_test} tickers)")

    train_ds = FinancialTimeSeriesDataset(train_df, lookback=lookback, forecast_horizon=forecast_horizon)
    val_ds = FinancialTimeSeriesDataset(val_df, lookback=lookback, forecast_horizon=forecast_horizon)
    test_ds = FinancialTimeSeriesDataset(test_df, lookback=lookback, forecast_horizon=forecast_horizon)

    logger.info(f"Dataset created: feature_dim={train_ds.feature_dim}, "
                f"train_samples={len(train_ds)}, val_samples={len(val_ds)}, test_samples={len(test_ds)}")
    logger.info(f"Train class distribution: {train_ds.class_distribution}")

    if shuffle:
        logger.info(f"Training mode: SHUFFLED (Phase A curriculum — no ticker batching, no stateful carry)")
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    else:
        logger.info(f"Training mode: CHRONOLOGICAL per-ticker (stateful context preserved)")
        train_loader = DataLoader(
            train_ds, num_workers=num_workers,
            batch_sampler=TickerBatchSampler(train_ds.ticker_ranges, batch_size, drop_last=True),
        )

    val_loader = DataLoader(
        val_ds, num_workers=num_workers,
        batch_sampler=TickerBatchSampler(val_ds.ticker_ranges, batch_size, drop_last=False),
    )
    test_loader = DataLoader(
        test_ds, num_workers=num_workers,
        batch_sampler=TickerBatchSampler(test_ds.ticker_ranges, batch_size, drop_last=False),
    )

    return train_loader, val_loader, test_loader, train_ds.feature_dim
