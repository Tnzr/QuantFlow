"""Baseline ladder: every candidate model must beat these, net of costs, on
purged walk-forward out-of-sample years before it is considered."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from quantflow.research import metrics as M
from quantflow.research.costs import CostModel
from quantflow.research.features import build_dataset
from quantflow.research.portfolio import (
    DEFAULT_CAPITAL,
    backtest_scores,
    trend_long_flat,
)
from quantflow.research.registry import log_experiment, n_trials
from quantflow.research.splits import PurgedWalkForward

log = logging.getLogger(__name__)

# Current liquid names: survivorship-biased by construction (see registry notes).
DEFAULT_UNIVERSE = [
    "SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "EFA", "EEM",
    "TLT", "IEF", "GLD", "SLV", "USO", "AAPL", "MSFT", "AMZN", "GOOGL", "JPM", "BAC", "XOM", "CVX", "JNJ",
    "PFE", "UNH", "WMT", "HD", "KO", "PG", "INTC", "CSCO", "ORCL", "CAT", "GE", "MMM", "IBM", "MRK", "DIS",
    "MCD", "NKE", "T", "VZ", "C", "WFC",
]

# Pre-registered single-factor baselines: (feature, sign) fixed from the literature, never fitted.
FACTOR_BASELINES = {"momentum_12_1": ("mom_252_21", 1), "reversal_1w": ("mom_5", -1)}

ERAS = {"early": (2012, 2016), "mid": (2017, 2021), "late": (2022, 2030)}


@dataclass
class LadderConfig:
    horizon: int = 5
    step: int = 5
    start: str = "2007-01-01"
    oos_start: str = "2012-01-01"
    quantile: float = 0.2
    halflife: float = 0.0
    no_trade_band: float = 0.0
    capital: float = DEFAULT_CAPITAL
    models: list[str] = field(default_factory=lambda: ["ridge", "gbm"])
    shuffle_control: bool = True
    n_shuffles: int = 5
    seed: int = 7
    cost: dict = field(default_factory=lambda: asdict(CostModel()))


def _make_model(name: str, seed: int):
    if name == "ridge":
        from sklearn.linear_model import Ridge

        return Ridge(alpha=10.0)
    if name == "gbm":
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=200, l2_regularization=1.0, random_state=seed
        )
    raise ValueError(f"unknown model {name!r}")


def shuffle_within_date(y: pd.Series, rng: np.random.Generator) -> pd.Series:
    """Permute labels across tickers on each date; destroys signal, keeps marginals."""
    out = y.copy()
    for idx in y.groupby(level=0).indices.values():
        out.iloc[idx] = rng.permutation(y.iloc[idx].to_numpy())
    return out


def walk_forward_predict(
    X: pd.DataFrame,
    model: str | Callable[[], object],
    splitter: PurgedWalkForward,
    calendar: pd.DatetimeIndex,
    label: str = "y_rank",
    oos_start: str | None = None,
    shuffle_labels: bool = False,
    seed: int = 7,
) -> pd.Series:
    feats = X.attrs.get("features") or [c for c in X.columns if c not in ("y_rank", "fwd_ret", "fwd_vs", "hold_ret")]
    rng = np.random.default_rng(seed)
    y = X[label]
    preds = []
    for fold in splitter.split(X.index, calendar):
        if oos_start and fold.test_end < pd.Timestamp(oos_start):
            continue
        tr, te = X.iloc[fold.train], X.iloc[fold.test]
        ytr = y.iloc[fold.train].dropna()
        if shuffle_labels:
            ytr = shuffle_within_date(ytr, rng)
        m = _make_model(model, seed) if isinstance(model, str) else model()
        m.fit(tr.loc[ytr.index, feats].to_numpy(), ytr.to_numpy())
        preds.append(pd.Series(m.predict(te[feats].to_numpy()), index=te.index))
    if not preds:
        return pd.Series(dtype=float)
    s = pd.concat(preds)
    return s[s.index.get_level_values(0) >= pd.Timestamp(oos_start)] if oos_start else s


def evaluate_scores(scores: pd.Series, X: pd.DataFrame, cfg: LadderConfig, cost: CostModel) -> tuple[dict, pd.DataFrame]:
    ppy = max(1, round(252 / cfg.step))
    ic = M.rank_ic(scores, X.loc[scores.index, "fwd_ret"])
    pnl = backtest_scores(
        scores, X["hold_ret"], cost, q=cfg.quantile, halflife=cfg.halflife,
        no_trade_band=cfg.no_trade_band, capital=cfg.capital,
    )
    out = {
        **M.ic_summary(ic, max(1, cfg.horizon // cfg.step), ppy),
        "gross": M.summarize(pnl["gross"], periods_per_year=ppy),
        "net": M.summarize(pnl["net"], pnl["turnover"], ppy),
        "net_by_era": M.by_era(pnl["net"], ERAS, ppy),
    }
    return out, pnl


def run_ladder(panel: dict[str, pd.DataFrame], cfg: LadderConfig | None = None, family: str = "baseline_ladder", log_runs: bool = True, universe_note: str = "") -> dict:
    """Evaluate factor baselines, fitted models, a shuffled-label control and
    SPY trend sleeves on the same OOS window; log each to the registry."""
    cfg = cfg or LadderConfig()
    cost = CostModel(**{**cfg.cost, "periods_per_year": max(1, round(252 / cfg.step))})
    calendar = panel["close"].index
    X = build_dataset(panel, cfg.horizon, cfg.step, cfg.start)
    splitter = PurgedWalkForward(cfg.horizon, freq="Y", min_train_bars=756)
    oos = X[X.index.get_level_values(0) >= pd.Timestamp(cfg.oos_start)]
    results: dict[str, dict] = {}
    pnls: dict[str, pd.Series] = {}

    for name, (feat, sign) in FACTOR_BASELINES.items():
        results[f"factor:{name}"], p = evaluate_scores(sign * oos[feat], X, cfg, cost)
        pnls[f"factor:{name}"] = p["net"]
    for model in cfg.models:
        s = walk_forward_predict(X, model, splitter, calendar, oos_start=cfg.oos_start, seed=cfg.seed)
        if s.empty:
            continue
        results[f"model:{model}"], p = evaluate_scores(s, X, cfg, cost)
        pnls[f"model:{model}"] = p["net"]
    if cfg.shuffle_control and cfg.models:
        ctrl = cfg.models[-1]
        null_ic, null_sr = [], []
        for k in range(cfg.n_shuffles):
            s = walk_forward_predict(X, ctrl, splitter, calendar, oos_start=cfg.oos_start, shuffle_labels=True, seed=cfg.seed + k)
            if s.empty:
                continue
            r, _ = evaluate_scores(s, X, cfg, cost)
            null_ic.append(r["ic_mean"])
            null_sr.append(r["gross"]["sharpe"])
        if null_ic:
            results[f"control:shuffled_{ctrl}"] = {
                "ic_mean": float(np.mean(null_ic)),
                "ic_null_sd": float(np.std(null_ic)),
                "gross_sharpe_mean": float(np.mean(null_sr)),
                "n_shuffles": len(null_ic),
            }
            real = results.get(f"model:{ctrl}")
            if real and np.std(null_ic) > 0:
                real["ic_z_vs_shuffled"] = float((real["ic_mean"] - np.mean(null_ic)) / np.std(null_ic))

    if "SPY" in panel["close"].columns:
        spy = panel["close"]["SPY"]
        ppy = max(1, round(252 / cfg.step))
        for name, short in (("spy_trend_long_flat", False), ("spy_trend_long_short", True)):
            t = trend_long_flat(spy, 200, cfg.step, short, cost)
            t = t[t.index >= pd.Timestamp(cfg.oos_start)]
            results[f"sleeve:{name}"] = M.summarize(t["net"], t["turnover"], ppy)
            pnls[f"sleeve:{name}"] = t["net"]
        bh = spy.reindex(spy.index[:: cfg.step]).pct_change().shift(-1)
        results["benchmark:spy_buy_hold"] = M.summarize(bh[bh.index >= pd.Timestamp(cfg.oos_start)], periods_per_year=ppy)

    per_period = [M.sharpe(p, 1) for p in pnls.values()]
    prior = n_trials(family) if log_runs else 0
    trials = max(len(pnls) + prior, 2)
    for k, p in pnls.items():
        results[k]["dsr"] = M.deflated_sharpe(p, per_period, trials)
        results[k]["n_trials"] = trials
    if log_runs:
        conf = asdict(cfg)
        for k, v in results.items():
            log_experiment(family, k, {**conf, "strategy": k}, v, notes=universe_note)
    return results
