"""Strictly causal multi-scale OHLCV features on wide panels.

Each feature at date ``t`` uses bars up to and including ``t`` only.
``tests/test_research_harness.py`` enforces this by truncating the input
and checking earlier values do not change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MOMENTUM_WINDOWS = (1, 5, 21, 63, 126, 252)


def rsi(close: pd.DataFrame, n: int) -> pd.DataFrame:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def multiscale_features(panel: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    c, h, lo, o, v = (panel[k] for k in ("close", "high", "low", "open", "volume"))
    r = c.pct_change(fill_method=None)
    rng = (h - lo).replace(0, np.nan)
    logv = np.log(v)
    f: dict[str, pd.DataFrame] = {}
    for n in MOMENTUM_WINDOWS:
        f[f"mom_{n}"] = c.pct_change(n, fill_method=None)
    f["mom_252_21"] = c.shift(21).pct_change(231, fill_method=None)
    for n in (2, 14, 50):
        f[f"rsi_{n}"] = rsi(c, n)
    for n in (20, 50, 200):
        f[f"dist_sma{n}"] = c / c.rolling(n).mean() - 1
    f["vol_z20"] = (logv - logv.rolling(20).mean()) / logv.rolling(20).std()
    f["vol_trend"] = v.rolling(5).mean() / v.rolling(63).mean() - 1
    f["clv"] = ((c - lo) - (h - c)) / rng
    f["clv_5"] = f["clv"].rolling(5).mean()
    f["gap"] = o / c.shift() - 1
    f["range_atr"] = rng / rng.rolling(20).mean()
    f["rv_ratio"] = r.rolling(5).std() / r.rolling(63).std()
    f["rv_63"] = r.rolling(63).std()
    f["obv_slope"] = (np.sign(r) * v).rolling(21).sum() / v.rolling(21).sum()
    lo52, hi52 = c.rolling(252).min(), c.rolling(252).max()
    f["pos_52w"] = (c - lo52) / (hi52 - lo52).replace(0, np.nan)
    f["skew_63"] = r.rolling(63).skew()
    return f


def forward_returns(close: pd.DataFrame, horizon: int) -> pd.DataFrame:
    return close.shift(-horizon) / close - 1


def vol_scaled_forward(close: pd.DataFrame, horizon: int, vol_window: int = 63) -> pd.DataFrame:
    """Forward return divided by trailing (causal) vol scaled to the horizon."""
    vol = close.pct_change(fill_method=None).rolling(vol_window).std() * np.sqrt(horizon)
    return forward_returns(close, horizon) / vol.replace(0, np.nan)


def xs_rank(df: pd.DataFrame) -> pd.DataFrame:
    return df.rank(axis=1, pct=True)


def build_dataset(
    panel: dict[str, pd.DataFrame],
    horizon: int = 5,
    step: int = 5,
    start: str | None = None,
    min_names: int = 10,
) -> pd.DataFrame:
    """Long (date, ticker) frame of rank-normalised features and labels.

    Columns: features..., ``y_rank`` (cross-sectional label rank),
    ``fwd_ret`` (raw forward return), ``fwd_vs`` (vol-scaled forward return),
    ``hold_ret`` (return to the next sample date, used for portfolio P&L).
    Samples every ``step`` bars to limit label overlap.
    """
    feats = multiscale_features(panel)
    close = panel["close"]
    dates = close.index if start is None else close.index[close.index >= pd.Timestamp(start)]
    dates = dates[::step]
    fwd = forward_returns(close, horizon).reindex(dates)
    cols = {k: xs_rank(v.reindex(dates)).stack() for k, v in feats.items()}
    cols["fwd_ret"] = fwd.stack()
    cols["hold_ret"] = forward_returns(close, step).reindex(dates).stack()
    cols["fwd_vs"] = vol_scaled_forward(close, horizon).reindex(dates).stack()
    cols["y_rank"] = xs_rank(fwd).stack()
    X = pd.concat(cols, axis=1)
    X.index.names = ["date", "ticker"]
    X = X.dropna(subset=["fwd_ret"])
    counts = X.groupby(level=0).size()
    X = X[X.index.get_level_values(0).isin(counts[counts >= min_names].index)]
    feat_cols = list(feats)
    X[feat_cols] = X[feat_cols].fillna(0.5)
    X.attrs["features"] = feat_cols
    return X
