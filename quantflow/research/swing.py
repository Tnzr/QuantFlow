"""Swing-horizon study (2 weeks to 2 months).

For each forecast horizon H the study compares, on the same purged
walk-forward out-of-sample window and net of costs:

* pre-registered (never fitted) factors and an equal-weight multi-scale composite;
* fixed ridge / GBM on base vs multi-scale swing features;
* a GBM whose hyper-parameters and label are tuned by *nested* walk-forward
  (inner validation = last ``inner_years`` of each training window, purged);
* a shuffled-label null for the fixed swing GBM.

Every portfolio uses K = H / step staggered tranches so a slow signal is not
churned weekly. Grouped drop-one ablations measure which time scales carry
the signal. Every grid configuration counts as a trial for the deflated Sharpe.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from quantflow.research import metrics as M
from quantflow.research.costs import CostModel
from quantflow.research.features import FEATURE_GROUPS, build_dataset
from quantflow.research.portfolio import (
    DEFAULT_CAPITAL,
    backtest_scores,
    trend_long_flat,
)
from quantflow.research.registry import log_experiment, n_trials
from quantflow.research.runner import trailing_adv_dollars
from quantflow.research.splits import PurgedWalkForward

log = logging.getLogger(__name__)

SWING_FACTORS = {
    "momentum_12_1": ("mom_252_21", 1),
    "resid_momentum_126": ("resid_mom_126", 1),
    "near_52w_high": ("pos_52w", 1),
}
COMPOSITE = ["tmom_21", "tmom_63", "resid_mom_126", "trend_agree", "mom_252_21"]

DEFAULT_GRID = [
    {"model": "ridge", "alpha": 10.0, "label": "y_rank"},
    {"model": "ridge", "alpha": 100.0, "label": "y_vs_rank"},
    {"model": "gbm", "max_depth": 3, "learning_rate": 0.05, "max_iter": 200, "min_samples_leaf": 200, "label": "y_rank"},
    {"model": "gbm", "max_depth": 3, "learning_rate": 0.05, "max_iter": 200, "min_samples_leaf": 200, "label": "y_vs_rank"},
    {"model": "gbm", "max_depth": 2, "learning_rate": 0.03, "max_iter": 300, "min_samples_leaf": 400, "label": "y_rank"},
    {"model": "gbm", "max_depth": 4, "learning_rate": 0.05, "max_iter": 150, "min_samples_leaf": 300, "label": "y_vs_rank"},
]
FIXED_GBM = DEFAULT_GRID[2]
FIXED_RIDGE = DEFAULT_GRID[0]


@dataclass
class SwingConfig:
    horizons: tuple[int, ...] = (10, 21, 42)
    step: int = 5
    start: str = "2007-01-01"
    oos_start: str = "2012-01-01"
    quantile: float = 0.2
    inner_years: int = 3
    n_shuffles: int = 5
    ablation_horizon: int = 21
    capital: float = DEFAULT_CAPITAL
    seed: int = 7
    grid: list[dict] = field(default_factory=lambda: [dict(g) for g in DEFAULT_GRID])
    cost: dict = field(default_factory=lambda: asdict(CostModel()))


def make_model(spec: dict, seed: int = 7):
    if spec["model"] == "ridge":
        from sklearn.linear_model import Ridge

        return Ridge(alpha=spec.get("alpha", 10.0))
    if spec["model"] == "gbm":
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            max_depth=spec["max_depth"], learning_rate=spec["learning_rate"], max_iter=spec["max_iter"],
            min_samples_leaf=spec["min_samples_leaf"], l2_regularization=1.0, random_state=seed,
        )
    raise ValueError(f"unknown model {spec['model']!r}")


def _fit_predict(X, tr_idx, te_idx, feats, spec, seed, shuffle_rng=None):
    y = X[spec["label"]].iloc[tr_idx].dropna()
    if shuffle_rng is not None:
        from quantflow.research.runner import shuffle_within_date

        y = shuffle_within_date(y, shuffle_rng)
    m = make_model(spec, seed)
    m.fit(X.loc[y.index, feats].to_numpy(), y.to_numpy())
    te = X.iloc[te_idx]
    return pd.Series(m.predict(te[feats].to_numpy()), index=te.index)


def inner_split(X: pd.DataFrame, train_idx: np.ndarray, calendar: pd.DatetimeIndex, horizon: int, inner_years: int):
    """Split an outer training set into (inner_train, validation), purged by ``horizon`` bars."""
    dates = X.index.get_level_values(0)[train_idx]
    val_start = dates.max() - pd.DateOffset(years=inner_years)
    pos = calendar.searchsorted(dates)
    vpos = calendar.searchsorted(val_start)
    val = train_idx[dates >= val_start]
    inner = train_idx[(dates < val_start) & (pos + horizon < vpos)]
    return inner, val


def nested_predict(X, splitter, calendar, cfg: SwingConfig, horizon: int, feats: list[str]):
    """Outer walk-forward; each fold picks the grid spec with the best inner-validation rank IC."""
    preds, chosen = [], []
    for fold in splitter.split(X.index, calendar):
        if fold.test_end < pd.Timestamp(cfg.oos_start):
            continue
        inner, val = inner_split(X, fold.train, calendar, horizon, cfg.inner_years)
        scores = []
        for spec in cfg.grid:
            if len(inner) < 300 or len(val) < 100:
                scores.append(np.nan)
                continue
            p = _fit_predict(X, inner, val, feats, spec, cfg.seed)
            scores.append(M.rank_ic(p, X.loc[p.index, "fwd_ret"]).mean())
        best = int(np.nanargmax(scores)) if np.isfinite(scores).any() else 0
        chosen.append({"test_start": str(fold.test_start.date()), "spec": best, "inner_ic": [float(s) for s in scores]})
        preds.append(_fit_predict(X, fold.train, fold.test, feats, cfg.grid[best], cfg.seed))
    s = pd.concat(preds) if preds else pd.Series(dtype=float)
    return s[s.index.get_level_values(0) >= pd.Timestamp(cfg.oos_start)], chosen


def fixed_predict(X, splitter, calendar, cfg: SwingConfig, spec: dict, feats: list[str], shuffle_seed: int | None = None):
    rng = np.random.default_rng(shuffle_seed) if shuffle_seed is not None else None
    preds = [
        _fit_predict(X, f.train, f.test, feats, spec, cfg.seed, rng)
        for f in splitter.split(X.index, calendar)
        if f.test_end >= pd.Timestamp(cfg.oos_start)
    ]
    s = pd.concat(preds) if preds else pd.Series(dtype=float)
    return s[s.index.get_level_values(0) >= pd.Timestamp(cfg.oos_start)]


def run_swing_study(
    panel: dict[str, pd.DataFrame],
    cfg: SwingConfig | None = None,
    out_dir: Path | None = None,
    family: str = "swing_study",
    log_runs: bool = True,
    universe_note: str = "",
) -> dict:
    """Run the study; if ``out_dir`` is given, write per-strategy P&L, IC,
    scores and nulls as Parquet/JSON artifacts for :mod:`quantflow.research.plots`."""
    cfg = cfg or SwingConfig()
    ppy = max(1, round(252 / cfg.step))
    cost = CostModel(**{**cfg.cost, "periods_per_year": ppy})
    calendar = panel["close"].index
    adv = trailing_adv_dollars(panel)
    results: dict = {"config": asdict(cfg), "horizons": {}}
    pnl_frames, ic_frames, score_frames = {}, {}, {}
    nulls: dict = {}

    for h in cfg.horizons:
        X = build_dataset(panel, h, cfg.step, cfg.start, feature_set="swing")
        feats, base = X.attrs["features"], X.attrs["base_features"]
        splitter = PurgedWalkForward(h, freq="Y", min_train_bars=756)
        oos = X[X.index.get_level_values(0) >= pd.Timestamp(cfg.oos_start)]
        K = max(1, round(h / cfg.step))
        scores: dict[str, pd.Series] = {f"factor:{k}": sgn * oos[c] for k, (c, sgn) in SWING_FACTORS.items() if c in oos}
        comp = [c for c in COMPOSITE if c in oos]
        scores["composite:multiscale_ew"] = oos[comp].mean(axis=1)
        scores["model:ridge_base"] = fixed_predict(X, splitter, calendar, cfg, FIXED_RIDGE, base)
        scores["model:gbm_base"] = fixed_predict(X, splitter, calendar, cfg, FIXED_GBM, base)
        scores["model:gbm_swing"] = fixed_predict(X, splitter, calendar, cfg, FIXED_GBM, feats)
        scores["model:gbm_swing_tuned"], chosen = nested_predict(X, splitter, calendar, cfg, h, feats)

        hres: dict = {"tranches": K, "tuning": chosen}
        for name, s in scores.items():
            ic = M.rank_ic(s, X.loc[s.index, "fwd_ret"])
            p = backtest_scores(s, X["hold_ret"], cost, q=cfg.quantile, capital=cfg.capital, adv_dollars=adv, tranches=K)
            p1 = backtest_scores(s, X["hold_ret"], cost, q=cfg.quantile, capital=cfg.capital, adv_dollars=adv, tranches=1)
            hres[name] = {
                **M.ic_summary(ic, K, ppy),
                "gross": M.summarize(p["gross"], periods_per_year=ppy),
                "net": M.summarize(p["net"], p["turnover"], ppy),
                "net_untranched": M.summarize(p1["net"], p1["turnover"], ppy),
            }
            pnl_frames[(h, name)] = p
            ic_frames[(h, name)] = ic
            score_frames[(h, name)] = (s, X.loc[s.index, "fwd_ret"])

        null_ic, null_sr = [], []
        for k in range(cfg.n_shuffles):
            s = fixed_predict(X, splitter, calendar, cfg, FIXED_GBM, feats, shuffle_seed=cfg.seed + k)
            null_ic.append(float(M.rank_ic(s, X.loc[s.index, "fwd_ret"]).mean()))
            p = backtest_scores(s, X["hold_ret"], cost, q=cfg.quantile, capital=cfg.capital, adv_dollars=adv, tranches=K)
            null_sr.append(M.sharpe(p["gross"], ppy))
        nulls[h] = {"ic": null_ic, "gross_sharpe": null_sr}
        real = hres["model:gbm_swing"]
        if np.std(null_ic) > 0:
            real["ic_z_vs_shuffled"] = float((real["ic_mean"] - np.mean(null_ic)) / np.std(null_ic))

        if h == cfg.ablation_horizon:
            full_ic = ic_frames[(h, "model:gbm_swing")]
            abl = {}
            for g, cols in FEATURE_GROUPS.items():
                keep = [c for c in feats if c not in cols]
                s = fixed_predict(X, splitter, calendar, cfg, FIXED_GBM, keep)
                diff = (full_ic - M.rank_ic(s, X.loc[s.index, "fwd_ret"])).dropna()
                boot = M.block_bootstrap(diff, np.mean, n_boot=1000, block=K)
                abl[g] = {
                    "n_features": len(cols),
                    "delta_ic": float(diff.mean()),
                    "ci95": [float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))],
                }
            hres["ablation"] = abl
        results["horizons"][h] = hres

    if "SPY" in panel["close"].columns:
        spy = panel["close"]["SPY"]
        t = trend_long_flat(spy, 200, cfg.step, False, cost)
        t = t[t.index >= pd.Timestamp(cfg.oos_start)]
        bh = spy.reindex(spy.index[:: cfg.step]).pct_change().shift(-1)
        bh = bh[bh.index >= pd.Timestamp(cfg.oos_start)].dropna()
        results["benchmarks"] = {
            "spy_trend_long_flat": M.summarize(t["net"], t["turnover"], ppy),
            "spy_buy_hold": M.summarize(bh, periods_per_year=ppy),
        }
        pnl_frames[(0, "benchmark:spy_trend_long_flat")] = t[["gross", "net", "turnover"]]
        pnl_frames[(0, "benchmark:spy_buy_hold")] = pd.DataFrame({"gross": bh, "net": bh, "turnover": 0.0})

    strategies = [k for k in pnl_frames if k[0]]
    trials = len(cfg.grid) * len(cfg.horizons) + len(strategies) + (n_trials(family) if log_runs else 0)
    per_period = [M.sharpe(pnl_frames[k]["net"], 1) for k in strategies]
    for h, name in strategies:
        r = results["horizons"][h][name]
        r["dsr"] = M.deflated_sharpe(pnl_frames[(h, name)]["net"], per_period, trials)
        r["n_trials"] = trials
    results["nulls"] = nulls

    if log_runs:
        conf = asdict(cfg)
        for h, name in strategies:
            log_experiment(family, f"h{h}:{name}", {**conf, "horizon": h, "strategy": name},
                           results["horizons"][h][name], notes=universe_note)
    if out_dir is not None:
        save_artifacts(Path(out_dir), results, pnl_frames, ic_frames, score_frames)
    return results


def _key(h: int, name: str) -> str:
    return f"h{h}|{name}"


def save_artifacts(out: Path, results, pnl_frames, ic_frames, score_frames) -> None:
    out.mkdir(parents=True, exist_ok=True)
    pnl = pd.concat({_key(*k): v[["gross", "net", "turnover"]] for k, v in pnl_frames.items()}, names=["key", "date"])
    pnl.to_parquet(out / "pnl.parquet")
    pd.concat({_key(*k): v for k, v in ic_frames.items()}, names=["key", "date"]).rename("ic").to_frame().to_parquet(out / "ic.parquet")
    q = {_key(h, n): M.quantile_returns(s, y).mean() for (h, n), (s, y) in score_frames.items()}
    pd.DataFrame(q).T.to_parquet(out / "quantiles.parquet")
    (out / "results.json").write_text(json.dumps(results, indent=1, default=lambda x: float(x) if np.isscalar(x) else str(x)))
