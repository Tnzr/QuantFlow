import numpy as np
import pandas as pd
import pytest

from quantflow.research import metrics as M
from quantflow.research.costs import CostModel
from quantflow.research.features import build_dataset, multiscale_features
from quantflow.research.portfolio import (
    backtest_scores,
    quantile_weights,
    trend_long_flat,
)
from quantflow.research.registry import config_hash, log_experiment, n_trials
from quantflow.research.runner import LadderConfig, run_ladder, walk_forward_predict
from quantflow.research.splits import CombinatorialPurgedKFold, PurgedWalkForward


def synth_panel(n_days=1600, n_names=30, signal=0.0, seed=0):
    """Random-walk OHLCV; with ``signal`` > 0 next-week returns load on the
    sign of the trailing 5-day return (planted reversal)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=n_days)
    noise = rng.normal(0, 0.01, (n_days, n_names))
    r = noise.copy()
    for t in range(6, n_days):
        r[t] -= signal * np.sign(noise[t - 5:t].sum(0)) * 0.002
    close = pd.DataFrame(100 * np.exp(np.cumsum(r, 0)), idx, [f"T{i}" for i in range(n_names)])
    high = close * (1 + rng.uniform(0, 0.01, close.shape))
    low = close * (1 - rng.uniform(0, 0.01, close.shape))
    opn = close.shift().fillna(close) * (1 + rng.normal(0, 0.002, close.shape))
    vol = pd.DataFrame(rng.lognormal(13, 0.3, close.shape), idx, close.columns)
    return {"open": opn, "high": high, "low": low, "close": close, "volume": vol}


def test_walk_forward_purges_label_overlap_and_embargo():
    cal = pd.bdate_range("2010-01-01", periods=1500)
    idx = pd.MultiIndex.from_product([cal[::5], ["A", "B"]])
    h = 21
    folds = list(PurgedWalkForward(horizon=h, min_train_bars=500).split(idx, cal))
    assert folds
    dates = idx.get_level_values(0)
    pos = cal.searchsorted(dates)
    for f in folds:
        lo = pos[f.test].min()
        assert (pos[f.train] + h < lo).all(), "train label windows must end before test starts"
        assert set(f.train).isdisjoint(f.test)


def test_cpcv_split_count_and_embargo():
    cal = pd.bdate_range("2010-01-01", periods=600)
    idx = pd.Index(cal)
    cv = CombinatorialPurgedKFold(n_groups=6, n_test_groups=2, horizon=5, embargo=10)
    folds = list(cv.split(idx, cal))
    assert len(folds) == 15
    for f in folds:
        test = set(f.test)
        for i in f.train:
            assert not any(j in test for j in range(i, i + 6)), "label overlap with test"
            assert not any((i - k) in test and (i - k + 1) not in test for k in range(1, 11)), "embargo violated"


def test_features_are_causal_under_truncation():
    panel = synth_panel(700, 5)
    full = multiscale_features(panel)
    cut = 500
    trunc = multiscale_features({k: v.iloc[:cut] for k, v in panel.items()})
    for k in full:
        pd.testing.assert_frame_equal(full[k].iloc[:cut], trunc[k], check_freq=False, obj=k)


def test_dataset_labels_are_forward_and_rank_normalised():
    panel = synth_panel(600, 12)
    X = build_dataset(panel, horizon=5, step=5)
    d, t = X.index[100]
    c = panel["close"][t]
    i = c.index.get_loc(d)
    assert X.loc[(d, t), "fwd_ret"] == pytest.approx(c.iloc[i + 5] / c.iloc[i] - 1)
    feats = X.attrs["features"]
    assert X[feats].min().min() >= 0 and X[feats].max().max() <= 1


def test_planted_signal_detected_and_shuffled_control_is_null():
    panel = synth_panel(1600, 30, signal=1.0, seed=1)
    X = build_dataset(panel, horizon=5, step=5)
    cal = panel["close"].index
    split = PurgedWalkForward(horizon=5, min_train_bars=500)
    real = walk_forward_predict(X, "ridge", split, cal)
    ic_real = M.rank_ic(real, X.loc[real.index, "fwd_ret"]).mean()
    fakes = [walk_forward_predict(X, "ridge", split, cal, shuffle_labels=True, seed=k) for k in range(5)]
    ic_fake = np.mean([M.rank_ic(f, X.loc[f.index, "fwd_ret"]).mean() for f in fakes])
    assert ic_real > 0.05
    assert abs(ic_fake) < 0.03


def test_no_signal_gives_no_edge():
    panel = synth_panel(1600, 30, signal=0.0, seed=2)
    X = build_dataset(panel, horizon=5, step=5)
    s = walk_forward_predict(X, "ridge", PurgedWalkForward(5, min_train_bars=500), panel["close"].index)
    assert abs(M.rank_ic(s, X.loc[s.index, "fwd_ret"]).mean()) < 0.03


def test_quantile_weights_dollar_neutral():
    w = quantile_weights(pd.Series(np.arange(10.0), index=list("abcdefghij")), q=0.2)
    assert w.sum() == pytest.approx(0) and w.abs().sum() == pytest.approx(2)


def test_costs_charged_on_turnover_and_borrow():
    idx = pd.MultiIndex.from_product([pd.bdate_range("2020-01-01", periods=4, freq="W-FRI"), list("abcde")])
    rng = np.random.default_rng(0)
    scores = pd.Series(rng.normal(size=len(idx)), idx)
    zero = pd.Series(0.0, idx)
    pnl = backtest_scores(scores, zero, CostModel(), q=0.2)
    assert (pnl["gross"] == 0).all()
    assert (pnl["net"] < 0).all()
    free = backtest_scores(scores, zero, CostModel(0, 0, 0, 0, 0), q=0.2)
    assert (free["net"] == 0).all()


def test_trend_sleeve_uses_only_past_prices():
    close = pd.Series(np.r_[np.linspace(100, 200, 300), np.linspace(200, 100, 300)], pd.bdate_range("2015-01-01", periods=600))
    t = trend_long_flat(close, 50, 5, cost=CostModel(0, 0, 0, 0, 0))
    spiked = close.copy()
    spiked.iloc[-1] *= 3
    t2 = trend_long_flat(spiked, 50, 5, cost=CostModel(0, 0, 0, 0, 0))
    pd.testing.assert_series_equal(t["position"].iloc[:-2], t2["position"].iloc[:-2])


def test_deflated_sharpe_penalises_many_trials():
    r = pd.Series(np.random.default_rng(3).normal(0.002, 0.02, 500))
    trials = list(np.random.default_rng(4).normal(0, 0.05, 50))
    assert M.deflated_sharpe(r, trials, 50) < M.probabilistic_sharpe(r)


def test_diebold_mariano_detects_better_model():
    rng = np.random.default_rng(5)
    y = rng.normal(size=400)
    good, bad = (y - (y + rng.normal(0, 0.1, 400))) ** 2, (y - rng.normal(size=400)) ** 2
    res = M.diebold_mariano(pd.Series(good), pd.Series(bad))
    assert res["dm"] < 0 and res["p"] < 0.01


def test_registry_counts_distinct_configs(tmp_path):
    p = tmp_path / "exp.jsonl"
    log_experiment("fam", "a", {"x": 1}, {"sharpe": float("nan")}, path=p)
    log_experiment("fam", "a", {"x": 1}, {"sharpe": 1.0}, path=p)
    log_experiment("fam", "b", {"x": 2}, {"sharpe": 1.0}, path=p)
    assert n_trials("fam", p) == 2
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})


def test_run_ladder_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTFLOW_EXPERIMENT_REGISTRY", str(tmp_path / "exp.jsonl"))
    panel = synth_panel(1500, 20, signal=1.0, seed=6)
    panel = {k: v.rename(columns={"T0": "SPY"}) for k, v in panel.items()}
    cfg = LadderConfig(start="2010-06-01", oos_start="2013-01-01", models=["ridge"])
    res = run_ladder(panel, cfg)
    assert {"factor:momentum_12_1", "model:ridge", "control:shuffled_ridge", "sleeve:spy_trend_long_flat"} <= set(res)
    assert res["model:ridge"]["ic_mean"] > res["control:shuffled_ridge"]["ic_mean"]
    assert 0 <= res["model:ridge"]["dsr"] <= 1
    assert n_trials("baseline_ladder") >= 5
