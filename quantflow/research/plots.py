"""Scientific backtest figures from :func:`quantflow.research.swing.run_swing_study` artifacts.

Every chart shows uncertainty (CI bands, bootstrap distributions or a
shuffled-label null) next to the point estimate, and P&L is net of costs
unless labelled gross.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quantflow.research import metrics as M

PALETTE = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2", "#be185d", "#4b5563"]
PPY = 52

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 160, "savefig.bbox": "tight", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "axes.titleweight": "bold", "axes.titlesize": 10, "legend.frameon": False, "legend.fontsize": 8,
})


def load_artifacts(d: Path) -> dict:
    d = Path(d)
    return {
        "results": json.loads((d / "results.json").read_text()),
        "pnl": pd.read_parquet(d / "pnl.parquet"),
        "ic": pd.read_parquet(d / "ic.parquet")["ic"],
        "quantiles": pd.read_parquet(d / "quantiles.parquet"),
    }


def label(key: str) -> str:
    h, name = key.split("|")
    n = name.split(":", 1)[1]
    n = n.replace("_", " ")
    return n if h == "h0" else f"{n} ({h[1:]}d)"


def _net(pnl: pd.DataFrame, key: str, col: str = "net") -> pd.Series:
    s = pnl.xs(key, level=0)[col]
    s.index = pd.DatetimeIndex(s.index)
    return s


def _save(fig, path: Path) -> Path:
    fig.savefig(path)
    plt.close(fig)
    return path


def equity_drawdown(pnl, keys, path, title="Growth of $1, net of costs"):
    fig, (a, b) = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    for i, k in enumerate(keys):
        r = _net(pnl, k).fillna(0)
        eq = (1 + r).cumprod()
        a.plot(eq.index, eq, color=PALETTE[i % 8], lw=1.4, label=f"{label(k)}  SR {M.sharpe(r, PPY):.2f}")
        b.fill_between(eq.index, eq / eq.cummax() - 1, 0, color=PALETTE[i % 8], alpha=0.25, lw=0)
        b.plot(eq.index, eq / eq.cummax() - 1, color=PALETTE[i % 8], lw=0.8)
    a.set_yscale("log")
    a.set_ylabel("value of $1 (log)")
    a.set_title(title)
    a.legend(loc="upper left")
    b.set_ylabel("drawdown")
    b.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    return _save(fig, path)


def rolling_sharpe(pnl, keys, path, window=52):
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for i, k in enumerate(keys):
        rs = M.rolling_sharpe(_net(pnl, k), window, PPY)
        se = np.sqrt((1 + 0.5 * (rs / np.sqrt(PPY)) ** 2) / window) * np.sqrt(PPY)
        c = PALETTE[i % 8]
        ax.plot(rs.index, rs, color=c, lw=1.2, label=label(k))
        ax.fill_between(rs.index, rs - 1.96 * se, rs + 1.96 * se, color=c, alpha=0.12, lw=0)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title(f"Rolling {window}-week net Sharpe with 95% band")
    ax.legend(loc="lower left", ncol=2)
    return _save(fig, path)


def rolling_ic(ic, key, path, window=52, overlap=1):
    s = ic.xs(key, level=0)
    s.index = pd.DatetimeIndex(s.index)
    m = s.rolling(window).mean()
    se = s.rolling(window).std() / np.sqrt(window / max(1, overlap))
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.bar(s.index, s, width=5, color=np.where(s > 0, "#93c5fd", "#fca5a5"), lw=0)
    ax.plot(m.index, m, color=PALETTE[0], lw=1.6, label=f"{window}-week mean")
    ax.fill_between(m.index, m - 1.96 * se, m + 1.96 * se, color=PALETTE[0], alpha=0.2, lw=0, label="95% band")
    ax.axhline(0, color="k", lw=0.8)
    ax2 = ax.twinx()
    ax2.plot(s.index, s.cumsum(), color="#111827", lw=1, ls="--", label="cumulative IC")
    ax2.spines["right"].set_visible(True)
    ax2.grid(False)
    ax2.set_ylabel("cumulative IC")
    ax.set_ylabel("weekly rank IC")
    ax.set_title(f"Prediction skill over time: {label(key)}")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", ncol=3)
    return _save(fig, path)


def ic_term_structure(results, ic, names, path):
    """Mean OOS IC per horizon; CI uses non-overlapping dates (every K-th) so overlap does not shrink it."""
    hz = sorted(int(h) for h in results["horizons"])
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    w = 0.8 / len(names)
    for i, n in enumerate(names):
        mu, ci = [], []
        for h in hz:
            key = f"h{h}|{n}"
            s = ic.xs(key, level=0).dropna() if key in ic.index.get_level_values(0) else pd.Series(dtype=float)
            k = results["horizons"][str(h)]["tranches"]
            nono = s.iloc[::k]
            mu.append(s.mean())
            ci.append(1.96 * nono.std() / np.sqrt(len(nono)) if len(nono) > 2 else np.nan)
        x = np.arange(len(hz)) + (i - (len(names) - 1) / 2) * w
        ax.bar(x, mu, w * 0.9, yerr=ci, capsize=3, color=PALETTE[i % 8], label=n.split(":", 1)[1].replace("_", " "))
    for j, h in enumerate(hz):
        null = results["nulls"][str(h)]["ic"]
        ax.scatter(np.full(len(null), j + 0.45), null, marker="x", color="k", s=14, zorder=3,
                   label="shuffled-label null" if j == 0 else None)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(hz)), [f"{h} trading days" for h in hz])
    ax.set_ylabel("mean OOS rank IC (95% CI)")
    ax.set_title("Skill by forecast horizon vs shuffled-label null")
    ax.legend(ncol=2, loc="upper left")
    return _save(fig, path)


def quantile_bars(quantiles, keys, path):
    fig, axes = plt.subplots(1, len(keys), figsize=(2.6 * len(keys), 2.8), sharey=True)
    for ax, k, c in zip(np.atleast_1d(axes), keys, PALETTE):
        q = quantiles.loc[k].dropna()
        ax.bar(q.index.astype(int), q * 100, color=[c if v > q.mean() else "#d1d5db" for v in q])
        ax.set_title(label(k), fontsize=8)
        ax.set_xlabel("score quintile")
    np.atleast_1d(axes)[0].set_ylabel("mean forward return %")
    fig.suptitle("Monotonicity: forward return by score quintile (1 = lowest)", fontweight="bold", fontsize=10, y=1.06)
    return _save(fig, path)


def cost_sensitivity(pnl, keys, path, bps=tuple(range(0, 31, 2)), assumed=5.0):
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for i, k in enumerate(keys):
        d = pnl.xs(k, level=0)
        sr = [M.sharpe(d["gross"] - d["turnover"] * b / 1e4, PPY) for b in bps]
        ax.plot(bps, sr, marker="o", ms=3, color=PALETTE[i % 8], label=f"{label(k)} (turnover {d['turnover'].mean():.2f}/wk)")
    ax.axvline(assumed, color="k", ls=":", lw=1)
    ax.text(assumed + 0.4, ax.get_ylim()[1] * 0.9, "assumed cost", fontsize=7)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("one-way cost per unit traded (bps)")
    ax.set_ylabel("Sharpe")
    ax.set_title("Break-even: how much trading cost each strategy survives")
    ax.legend(loc="upper right")
    return _save(fig, path)


def bootstrap_vs_null(pnl, results, keys, path):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    data = [M.bootstrap_sharpe(_net(pnl, k), PPY, n_boot=1000, block=8) for k in keys]
    parts = ax.violinplot([d[np.isfinite(d)] for d in data], showmedians=True, widths=0.8)
    for b, c in zip(parts["bodies"], PALETTE):
        b.set_facecolor(c)
        b.set_alpha(0.35)
    for j, k in enumerate(keys, 1):
        h = k.split("|")[0][1:]
        null = results.get("nulls", {}).get(h, {}).get("gross_sharpe")
        if null:
            ax.scatter(np.full(len(null), j + 0.3), null, marker="x", color="k", s=14,
                       label="shuffled-label null (gross)" if j == 1 else None)
        lo, hi = np.nanpercentile(data[j - 1], [2.5, 97.5])
        ax.text(j, hi + 0.05, f"P(SR>0)={np.mean(data[j - 1] > 0):.0%}", ha="center", fontsize=7)
        ax.vlines(j, lo, hi, color="k", lw=0.6)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(1, len(keys) + 1), [label(k) for k in keys], rotation=20, ha="right")
    ax.set_ylabel("annualised net Sharpe")
    ax.set_title("Block-bootstrap distribution of net Sharpe (1,000 resamples)")
    ax.legend(loc="upper left")
    return _save(fig, path)


def annual_heatmap(pnl, keys, path):
    rows = {label(k): _net(pnl, k).groupby(lambda d: d.year).apply(lambda x: (1 + x).prod() - 1) for k in keys}
    df = pd.DataFrame(rows).T * 100
    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(keys) + 1.2))
    v = np.nanpercentile(np.abs(df.to_numpy()), 95)
    im = ax.imshow(df.to_numpy(), cmap="RdYlGn", vmin=-v, vmax=v, aspect="auto")
    ax.set_xticks(range(df.shape[1]), df.columns, rotation=0, fontsize=7)
    ax.set_yticks(range(df.shape[0]), df.index, fontsize=8)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            if np.isfinite(df.iat[i, j]):
                ax.text(j, i, f"{df.iat[i, j]:.0f}", ha="center", va="center", fontsize=6.5)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.025, label="net return %")
    ax.set_title("Calendar-year net return % (regime stability)")
    return _save(fig, path)


def ablation(results, path, horizon="21"):
    abl = results["horizons"][horizon].get("ablation") or {}
    if not abl:
        return None
    g = sorted(abl, key=lambda k: abl[k]["delta_ic"])
    mu = np.array([abl[k]["delta_ic"] for k in g])
    lo = mu - np.array([abl[k]["ci95"][0] for k in g])
    hi = np.array([abl[k]["ci95"][1] for k in g]) - mu
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.barh(g, mu, xerr=[lo, hi], capsize=3, color=[PALETTE[0] if m > 0 else "#9ca3af" for m in mu])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("IC lost when the group is removed (95% bootstrap CI)")
    ax.set_title(f"Which time scales carry the signal? ({horizon}-day GBM, drop-one-group)")
    return _save(fig, path)


def turnover_frontier(results, path, names):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for i, n in enumerate(names):
        for h, hr in results["horizons"].items():
            r = hr.get(n)
            if not r:
                continue
            x0, y0 = r["net_untranched"]["turnover"], r["net_untranched"]["sharpe"]
            x1, y1 = r["net"]["turnover"], r["net"]["sharpe"]
            ax.scatter([x0], [y0], facecolors="none", edgecolors=PALETTE[i % 8], s=18, alpha=0.6)
            ax.annotate("", (x1, y1), (x0, y0), arrowprops={"arrowstyle": "->", "color": PALETTE[i % 8], "alpha": 0.6})
            ax.scatter([x1], [y1], color=PALETTE[i % 8], s=24, label=n.split(":", 1)[1].replace("_", " ") if h == min(results["horizons"]) else None)
            ax.text(x1, y1, f" {h}d", fontsize=7)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("weekly turnover (log)")
    ax.set_ylabel("net Sharpe")
    ax.set_title("Turnover vs net Sharpe: hollow = weekly rebalance, filled = staggered tranches")
    ax.legend(loc="lower left", ncol=2)
    return _save(fig, path)


def tuning_choices(results, path):
    hz = sorted(results["horizons"], key=int)
    specs = results["config"]["grid"]
    fig, axes = plt.subplots(1, len(hz), figsize=(3 * len(hz), 2.6), sharey=True)
    for ax, h in zip(np.atleast_1d(axes), hz):
        t = results["horizons"][h]["tuning"]
        years = [x["test_start"][:4] for x in t]
        ax.scatter(years, [x["spec"] for x in t], color=PALETTE[0], s=22)
        ax.set_title(f"{h}-day horizon", fontsize=9)
        ax.tick_params(axis="x", rotation=90, labelsize=6)
    np.atleast_1d(axes)[0].set_yticks(range(len(specs)), [
        f"{s['model']} {s.get('max_depth', s.get('alpha'))} {s['label'].replace('_rank', '')}" for s in specs
    ], fontsize=7)
    fig.suptitle("Nested tuning: spec picked each year on inner validation", fontweight="bold", fontsize=10, y=1.06)
    return _save(fig, path)


def make_all(artifact_dir: Path, fig_dir: Path) -> dict[str, Path]:
    a = load_artifacts(artifact_dir)
    res, pnl = a["results"], a["pnl"]
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(pnl.index.get_level_values(0)))
    best = max(
        (k for k in keys if not k.startswith("h0")),
        key=lambda k: res["horizons"][k.split("|")[0][1:]][k.split("|")[1]]["net"]["sharpe"],
    )
    bh = [k for k in keys if k.startswith("h0")]
    ah = str(res["config"]["ablation_horizon"])
    core = [f"h{ah}|{n}" for n in ("factor:momentum_12_1", "composite:multiscale_ew", "model:gbm_swing", "model:gbm_swing_tuned")]
    core = [k for k in core if k in keys]
    names = ["factor:momentum_12_1", "composite:multiscale_ew", "model:gbm_base", "model:gbm_swing", "model:gbm_swing_tuned"]
    out = {
        "equity": equity_drawdown(pnl, list(dict.fromkeys([best, *core, *bh])), fig_dir / "equity.png"),
        "rolling_sharpe": rolling_sharpe(pnl, [best, *bh[-1:]], fig_dir / "rolling_sharpe.png"),
        "rolling_ic": rolling_ic(a["ic"], f"h{ah}|model:gbm_swing", fig_dir / "rolling_ic.png",
                                overlap=res["horizons"][ah]["tranches"]),
        "term_structure": ic_term_structure(res, a["ic"], names, fig_dir / "term_structure.png"),
        "quantiles": quantile_bars(a["quantiles"], core, fig_dir / "quantiles.png"),
        "cost": cost_sensitivity(pnl, core, fig_dir / "cost.png"),
        "bootstrap": bootstrap_vs_null(pnl, res, list(dict.fromkeys([*core, best, *bh])), fig_dir / "bootstrap.png"),
        "heatmap": annual_heatmap(pnl, list(dict.fromkeys([*core, best, *bh])), fig_dir / "heatmap.png"),
        "ablation": ablation(res, fig_dir / "ablation.png", ah),
        "turnover": turnover_frontier(res, fig_dir / "turnover.png", names),
        "tuning": tuning_choices(res, fig_dir / "tuning.png"),
    }
    return {k: v for k, v in out.items() if v is not None}
