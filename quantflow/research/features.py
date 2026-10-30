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
    out = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    out = out.mask((dn == 0) & (up > 0), 100.0)
    return out.mask((dn == 0) & (up == 0), 50.0).where(close.notna())


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


SWING_SCALES = (10, 21, 42, 63)
BAND_SCALES = (5, 10, 21, 42, 84)


def _efficiency_ratio(c: pd.DataFrame, n: int) -> pd.DataFrame:
    """Kaufman efficiency: net move / path length over ``n`` bars (1 = clean trend)."""
    return (c - c.shift(n)).abs() / c.diff().abs().rolling(n).sum().replace(0, np.nan)


def swing_features(panel: dict[str, pd.DataFrame], market: str = "SPY") -> dict[str, pd.DataFrame]:
    """Multi-scale pattern features aimed at 2-week to 2-month holds.

    * ``band_n``: Haar-style multi-resolution detail, (MA_n - MA_2n) / close,
      i.e. the price path decomposed into octave bands from 1 week to 4 months.
    * ``tmom_n``: volatility-normalised momentum (a t-stat of the n-day move).
    * ``er_n``: trend efficiency; ``dd_hi_n`` / ``up_lo_n``: position vs n-day
      high and low; ``macd_*``: EMA spreads at three scales.
    * ``trend_agree``: how many of the 10/21/42/63-day trends point the same way.
    * Interactions: pullback inside an uptrend, volatility squeeze, and a
      volume-confirmed 21-day breakout.
    * ``resid_mom_n``: momentum net of rolling beta to ``market`` (if present).
    """
    c, h, lo, v = (panel[k] for k in ("close", "high", "low", "volume"))
    r = c.pct_change(fill_method=None)
    rv63 = r.rolling(63).std()
    f: dict[str, pd.DataFrame] = {}
    for n in BAND_SCALES:
        f[f"band_{n}"] = (c.rolling(n).mean() - c.rolling(2 * n).mean()) / c
    for n in SWING_SCALES:
        f[f"tmom_{n}"] = c.pct_change(n, fill_method=None) / (rv63 * np.sqrt(n)).replace(0, np.nan)
        f[f"er_{n}"] = _efficiency_ratio(c, n)
    for n in (10, 21, 63):
        f[f"dd_hi_{n}"] = c / h.rolling(n).max() - 1
        f[f"up_lo_{n}"] = c / lo.rolling(n).min() - 1
    for fast, slow in ((5, 21), (10, 42), (21, 126)):
        f[f"macd_{fast}_{slow}"] = (c.ewm(span=fast, adjust=False).mean() - c.ewm(span=slow, adjust=False).mean()) / c
    signs = [np.sign(c.pct_change(n, fill_method=None)) for n in SWING_SCALES]
    f["trend_agree"] = sum(signs) / len(signs)
    up = (c > c.rolling(50).mean()).astype(float).where(c.notna())
    f["pullback_uptrend"] = up * -c.pct_change(5, fill_method=None) / rv63.replace(0, np.nan)
    f["squeeze"] = r.rolling(10).std() / rv63
    prior_hi = h.shift(1).rolling(21).max()
    f["breakout_vol"] = (c / prior_hi - 1) * (v.rolling(5).mean() / v.rolling(63).mean())
    if market in c.columns:
        m = r[market]
        var = m.rolling(126).var()
        beta = r.rolling(126).cov(m).div(var, axis=0)
        resid = r - beta.mul(m, axis=0)
        for n in (63, 126):
            f[f"resid_mom_{n}"] = resid.rolling(n).sum() / (resid.rolling(n).std() * np.sqrt(n)).replace(0, np.nan)
    return f


FEATURE_GROUPS = {
    "short (<=10d)": ["mom_1", "mom_5", "rsi_2", "clv", "clv_5", "gap", "range_atr", "rv_ratio",
                      "band_5", "band_10", "tmom_10", "er_10", "dd_hi_10", "up_lo_10", "macd_5_21", "squeeze"],
    "swing (21-63d)": ["mom_21", "mom_63", "rsi_14", "dist_sma20", "dist_sma50", "band_21", "band_42",
                       "tmom_21", "tmom_42", "tmom_63", "er_21", "er_42", "er_63", "dd_hi_21", "up_lo_21",
                       "dd_hi_63", "up_lo_63", "macd_10_42", "resid_mom_63", "skew_63", "rv_63"],
    "long (>=126d)": ["mom_126", "mom_252", "mom_252_21", "rsi_50", "dist_sma200", "pos_52w", "band_84",
                      "macd_21_126", "resid_mom_126"],
    "volume": ["vol_z20", "vol_trend", "obv_slope", "breakout_vol"],
    "cross-scale": ["trend_agree", "pullback_uptrend"],
}


def feature_group(name: str) -> str:
    return next((g for g, cols in FEATURE_GROUPS.items() if name in cols), "other")


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
    feature_set: str = "base",
) -> pd.DataFrame:
    """Long (date, ticker) frame of rank-normalised features and labels.

    Columns: features..., ``y_rank`` (cross-sectional label rank),
    ``fwd_ret`` (raw forward return), ``fwd_vs`` (vol-scaled forward return),
    ``hold_ret`` (return to the next sample date, used for portfolio P&L).
    Samples every ``step`` bars to limit label overlap. ``feature_set`` is
    ``"base"`` (24 features) or ``"swing"`` (base + :func:`swing_features`).
    """
    feats = multiscale_features(panel)
    base_cols = list(feats)
    if feature_set == "swing":
        feats.update(swing_features(panel))
    elif feature_set != "base":
        raise ValueError(f"unknown feature_set {feature_set!r}")
    close = panel["close"]
    dates = close.index if start is None else close.index[close.index >= pd.Timestamp(start)]
    dates = dates[::step]
    fwd = forward_returns(close, horizon).reindex(dates)
    cols = {k: xs_rank(v.reindex(dates)).stack() for k, v in feats.items()}
    cols["fwd_ret"] = fwd.stack()
    cols["hold_ret"] = forward_returns(close, step).reindex(dates).stack()
    cols["fwd_vs"] = vol_scaled_forward(close, horizon).reindex(dates).stack()
    cols["y_rank"] = xs_rank(fwd).stack()
    cols["y_vs_rank"] = xs_rank(vol_scaled_forward(close, horizon).reindex(dates)).stack()
    X = pd.concat(cols, axis=1)
    X.index.names = ["date", "ticker"]
    X = X.dropna(subset=["fwd_ret"])
    counts = X.groupby(level=0).size()
    X = X[X.index.get_level_values(0).isin(counts[counts >= min_names].index)]
    feat_cols = list(feats)
    X[feat_cols] = X[feat_cols].fillna(0.5)
    X.attrs["features"] = feat_cols
    X.attrs["base_features"] = base_cols
    return X
