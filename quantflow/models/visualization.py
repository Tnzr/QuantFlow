from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

_S = {
    "eq": "#2ecc71", "bm": "#3498db", "dd": "#e74c3c",
    "buy": "#27ae60", "sell": "#c0392b", "profit": "#2ecc71", "loss": "#e74c3c",
    "inter": "#27ae60", "pre": "#f39c12", "onset": "#e74c3c",
    "fcst": "#9b59b6", "unc": "#95a5a6", "cash": "#3498db",
    "spy": "#e67e22", "pos": "#1abc9c",
    "train": "#2ecc71", "val": "#e74c3c", "pred": "#9b59b6", "actual": "#2c3e50",
}

def _ax(ax, title="", xlabel="", ylabel=""):
    if title: ax.set_title(title, fontsize=10, fontweight="bold")
    if xlabel: ax.set_xlabel(xlabel, fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.tick_params(labelsize=7)
    for s in ax.spines.values(): s.set_visible(False)


# ── Dashboard (6-panel, 22×16) ─────────────────────────────────────────

def plot_backtest_dashboard(
    result, signals=None, df=None, output_path="reports/dashboard.png",
    ticker="", title="AI Backtest Dashboard", dpi=150,
) -> str:
    if not HAS_MPL: return ""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(22, 16), dpi=dpi)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.28)

    _plot_equity_curve(fig.add_subplot(gs[0, 0]), result, ticker)
    _plot_per_ticker_equity(fig.add_subplot(gs[0, 1]), result)
    _plot_drawdown(fig.add_subplot(gs[1, 0]), result)
    _plot_head_probabilities(fig.add_subplot(gs[1, 1]), result, signals)
    _plot_allocation_timeline(fig.add_subplot(gs[2, 0]), result)
    _plot_metrics_table(fig.add_subplot(gs[2, 1]), result)

    t = f"{title} — {ticker}" if ticker else title
    fig.suptitle(t, fontsize=14, fontweight="bold", y=0.99)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def _plot_equity_curve(ax, result, ticker=""):
    eq = result.equity_curve
    if eq is None or len(eq) < 2:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes); return

    vals = eq.values if hasattr(eq, "values") else np.array(eq)
    xs = np.arange(len(vals))

    if hasattr(eq, "index") and isinstance(eq.index, pd.DatetimeIndex):
        ax.plot(eq.index, vals, color=_S["eq"], linewidth=1.5)
        ax.fill_between(eq.index, vals[0], vals, alpha=0.12, color=_S["eq"])
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=7)
    else:
        ax.fill_between(xs, vals[0], vals, alpha=0.12, color=_S["eq"])
        ax.plot(xs, vals, color=_S["eq"], linewidth=1.5)

    ax.axhline(y=vals[0], color="gray", ls="--", lw=0.5, alpha=0.4)

    trades = result.trades if hasattr(result, "trades") else []
    if trades and hasattr(eq, "index") and isinstance(eq.index, pd.DatetimeIndex):
        for trade in trades:
            if trade.exit_date in eq.index:
                idx_loc = eq.index.get_loc(trade.exit_date)
                val_at = vals[idx_loc] if idx_loc < len(vals) else vals[-1]
                c = _S["buy"] if trade.ret > 0 else _S["sell"]
                m = "^" if trade.ret > 0 else "v"
                ax.scatter(trade.exit_date, val_at, color=c, marker=m, s=12, alpha=0.5, zorder=5)

    ax.text(0.02, 0.95, f"Trades: {result.n_trades} | CAGR: {result.cagr:.1%} | Sharpe: {result.sharpe_ratio:.2f}",
            transform=ax.transAxes, fontsize=7, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))
    _ax(ax, "Portfolio Equity Curve", xlabel="Date", ylabel="Portfolio Value")


def _plot_per_ticker_equity(ax, result):
    if not hasattr(result, "daily_signals") or not result.daily_signals:
        _plot_metrics_table(ax, result)
        return

    daily_alloc = result.daily_allocations if hasattr(result, "daily_allocations") and not result.daily_allocations.empty else None
    if daily_alloc is None or daily_alloc.empty:
        _plot_metrics_table(ax, result)
        return

    total_alloc = daily_alloc.sum().sum() if isinstance(daily_alloc.sum(), pd.Series) else float(daily_alloc.sum())
    if total_alloc > 0:
        top_tickers = daily_alloc.sum().nlargest(5).index.tolist()
    else:
        top_tickers = list(daily_alloc.columns[:5])

    if not top_tickers:
        ax.text(0.5, 0.5, "No per-ticker data", ha="center", va="center", transform=ax.transAxes); return

    for tk in top_tickers:
        tk_eq = daily_alloc[tk].fillna(0.0)
        tk_max = float(tk_eq.max()) if tk_eq.max() > 0 else 1.0
        tk_eq_norm = tk_eq / tk_max
        if isinstance(tk_eq_norm, pd.Series) and isinstance(tk_eq_norm.index, pd.DatetimeIndex):
            ax.plot(tk_eq_norm.index, tk_eq_norm.values, linewidth=1.2, alpha=0.85, label=tk)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.xaxis.set_major_locator(mdates.YearLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=7)

    _ax(ax, "Per-Ticker Norm. Equity", xlabel="Date", ylabel="Normalized Value")
    ax.legend(fontsize=6, loc="upper left", ncol=2)


def _plot_drawdown(ax, result):
    eq = result.equity_curve
    if eq is None or len(eq) < 2:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes); return

    vals = eq.values if hasattr(eq, "values") else np.array(eq)
    peak = np.maximum.accumulate(vals)
    dd = (vals - peak) / peak * 100

    if hasattr(eq, "index") and isinstance(eq.index, pd.DatetimeIndex):
        ax.fill_between(eq.index, 0, dd, alpha=0.25, color=_S["dd"])
        ax.plot(eq.index, dd, color=_S["dd"], linewidth=1.2)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=7)
    else:
        x = np.arange(len(dd))
        ax.fill_between(x, 0, dd, alpha=0.25, color=_S["dd"])
        ax.plot(x, dd, color=_S["dd"], linewidth=1.2)

    ax.axhline(y=0, color="gray", lw=0.5, alpha=0.3)
    ax.set_ylim(min(-100, float(np.min(dd)) * 1.3), 5)
    _ax(ax, "Drawdown %", xlabel="Date", ylabel="%")


def _plot_head_probabilities(ax, result, signals=None):
    sigs = signals or {}
    if isinstance(sigs, dict) and all(k in sigs for k in ("prob_inter_event", "prob_pre_event", "prob_onset")):
        pi, pp, po = sigs["prob_inter_event"], sigs["prob_pre_event"], sigs["prob_onset"]
        if len(pi) > 0:
            x = np.arange(len(pi))
            ax.fill_between(x, 0, po, alpha=0.25, color=_S["onset"], label="Onset")
            ax.fill_between(x, po, po + pp, alpha=0.25, color=_S["pre"], label="Pre-Event")
            ax.fill_between(x, po + pp, po + pp + pi, alpha=0.25, color=_S["inter"], label="Inter-Event")
            ax.set_ylim(0, max(1.0, float(np.max(pi + pp + po)) * 1.1))
            _ax(ax, "Aggregate State Probability Waterfall", xlabel="Trading Days", ylabel="Prob")
            ax.legend(fontsize=6, loc="upper right", ncol=3)
            return

    daily_sigs = getattr(result, "daily_signals", {})
    if daily_sigs:
        agg_int, agg_pre, agg_ons = [], [], []
        for tk_df in daily_sigs.values():
            if not tk_df.empty:
                agg_int.append(tk_df.get("prob_inter_event", pd.Series(dtype=float)))
                agg_pre.append(tk_df.get("prob_pre_event", pd.Series(dtype=float)))
                agg_ons.append(tk_df.get("prob_onset", pd.Series(dtype=float)))
        if agg_int:
            all_dates = sorted(set(d for df in agg_int for d in df.index))
            pi_agg = np.zeros(len(all_dates))
            pp_agg = np.zeros(len(all_dates))
            po_agg = np.zeros(len(all_dates))
            count = np.zeros(len(all_dates))
            for df_int, df_pre, df_ons in zip(agg_int, agg_pre, agg_ons):
                    for i, d in enumerate(all_dates):
                        try:
                            if d in df_int.index:
                                pi_agg[i] += float(df_int.loc[d]) if hasattr(df_int.loc[d], '__float__') else float(df_int.loc[d].iloc[0])
                                pp_agg[i] += float(df_pre.loc[d]) if hasattr(df_pre.loc[d], '__float__') else float(df_pre.loc[d].iloc[0])
                                po_agg[i] += float(df_ons.loc[d]) if hasattr(df_ons.loc[d], '__float__') else float(df_ons.loc[d].iloc[0])
                                count[i] += 1
                        except (ValueError, TypeError, IndexError, KeyError):
                            continue
            mask = count > 0
            pi_agg[mask] /= count[mask]
            pp_agg[mask] /= count[mask]
            po_agg[mask] /= count[mask]
            x = np.arange(len(all_dates))
            ax.fill_between(x, 0, po_agg, alpha=0.25, color=_S["onset"], label="Onset")
            ax.fill_between(x, po_agg, po_agg + pp_agg, alpha=0.25, color=_S["pre"], label="Pre-Event")
            ax.fill_between(x, po_agg + pp_agg, po_agg + pp_agg + pi_agg, alpha=0.25, color=_S["inter"], label="Inter-Event")
            ax.set_ylim(0, max(1.0, float(np.max(pi_agg + pp_agg + po_agg)) * 1.1))
            _ax(ax, "Portfolio-Agg. State Probs", xlabel="Trading Days", ylabel="Prob")
            ax.legend(fontsize=6, loc="upper right", ncol=3)
            return

    ax.text(0.5, 0.5, "No signal data", ha="center", va="center", transform=ax.transAxes)


def _plot_allocation_timeline(ax, result):
    daily_alloc = result.daily_allocations if hasattr(result, "daily_allocations") and not result.daily_allocations.empty else None
    if daily_alloc is None or daily_alloc.empty:
        ax.text(0.5, 0.5, "No allocation data", ha="center", va="center", transform=ax.transAxes); return

    top = daily_alloc.sum().nlargest(6).index.tolist()
    if not top:
        ax.text(0.5, 0.5, "No allocation data", ha="center", va="center", transform=ax.transAxes); return

    colors = plt.cm.tab10(np.linspace(0, 1, len(top)))
    has_date_idx = isinstance(daily_alloc.index, pd.DatetimeIndex)

    bottom = np.zeros(len(daily_alloc))
    for i, tk in enumerate(top):
        vals = daily_alloc[tk].fillna(0.0).values
        if has_date_idx:
            ax.fill_between(daily_alloc.index, bottom, bottom + vals, alpha=0.7, color=colors[i], label=tk)
        else:
            ax.fill_between(range(len(vals)), bottom, bottom + vals, alpha=0.7, color=colors[i], label=tk)
        bottom += vals

    other = daily_alloc.drop(columns=top, errors="ignore").sum(axis=1).fillna(0.0).values if len(daily_alloc.columns) > len(top) else np.zeros(len(daily_alloc))
    if np.any(other > 0):
        if has_date_idx:
            ax.fill_between(daily_alloc.index, bottom, bottom + other, alpha=0.5, color=_S["unc"], label="Other")
        else:
            ax.fill_between(range(len(other)), bottom, bottom + other, alpha=0.5, color=_S["unc"], label="Other")

    if has_date_idx:
        daily_cash = getattr(result, "daily_cash", None)
        if daily_cash is not None and len(daily_cash) > 0:
            ax.plot(daily_cash.index, daily_cash.values, color=_S["cash"], linewidth=1.5, linestyle="--", alpha=0.7, label="Cash")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=7)

    _ax(ax, "Portfolio Allocation Timeline", xlabel="Date", ylabel="Allocation ($)")
    ax.legend(fontsize=6, loc="upper left", ncol=3)


def _plot_metrics_table(ax, result):
    ax.axis("off")
    rows = [
        ("Trades", str(result.n_trades)),
        ("Win Rate", f"{result.win_rate:.1%}"),
        ("Avg Return", f"{result.avg_ret:.2%}"),
        ("Sharpe", f"{result.sharpe_ratio:.2f}"),
        ("Sortino", f"{result.sortino_ratio:.2f}"),
        ("CAGR", f"{result.cagr:.1%}"),
        ("Max DD", f"{result.max_drawdown:.1%}"),
        ("Profit Factor", f"{result.profit_factor:.2f}"),
        ("Total Return", f"{result.total_return:.1%}"),
        ("Avg Hold", f"{result.avg_hold_days:.0f}d"),
    ]
    t = ax.table(cellText=rows, colLabels=["Metric", "Value"], cellLoc="left", loc="center", colWidths=[0.5, 0.5])
    t.auto_set_font_size(False); t.set_fontsize(7); t.scale(1.0, 1.25)
    for i in range(len(rows)):
        t[i + 1, 0].set_facecolor("#f8f9fa" if i % 2 == 0 else "white")
        t[i + 1, 1].set_facecolor("#f8f9fa" if i % 2 == 0 else "white")
    for j in range(2):
        t[0, j].set_facecolor("#2c3e50"); t[0, j].set_text_props(color="white", fontweight="bold")
    ax.set_title("Performance Metrics", fontsize=10, fontweight="bold", pad=6)


# ── Allocation Monitor (standalone, 18×10) ─────────────────────────────

def plot_allocation_monitor(
    result, output_path="reports/allocation_monitor.png", dpi=150,
) -> str:
    if not HAS_MPL: return ""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(18, 10), dpi=dpi)
    gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.35)

    _plot_allocation_stacked(fig.add_subplot(gs[0]), result)
    _plot_per_ticker_pnl(fig.add_subplot(gs[1]), result)
    _plot_rolling_sharpe(fig.add_subplot(gs[2]), result)

    fig.suptitle("Portfolio Allocation Monitor", fontsize=14, fontweight="bold", y=0.99)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def _plot_allocation_stacked(ax, result):
    daily_alloc = result.daily_allocations if hasattr(result, "daily_allocations") and not result.daily_allocations.empty else None
    if daily_alloc is None or daily_alloc.empty:
        ax.text(0.5, 0.5, "No allocation data", ha="center", va="center", transform=ax.transAxes); return

    top = daily_alloc.sum().nlargest(5).index.tolist()
    colors = plt.cm.tab10(np.linspace(0, 1, len(top)))
    bottom = np.zeros(len(daily_alloc))
    for i, tk in enumerate(top):
        vals = daily_alloc[tk].fillna(0.0).values
        ax.fill_between(range(len(vals)), bottom, bottom + vals, alpha=0.7, color=colors[i], label=tk)
        bottom += vals

    daily_cash = getattr(result, "daily_cash", None)
    if daily_cash is not None and len(daily_cash) > 0 and hasattr(daily_cash, "values"):
        ax.plot(range(len(daily_cash)), daily_cash.values, color=_S["cash"], linewidth=1.5, linestyle="--", alpha=0.7, label="Cash")

    _ax(ax, "Allocation Stacked Area", xlabel="Trading Days", ylabel="Allocation ($)")
    ax.legend(fontsize=6, loc="upper left", ncol=6)


def _plot_per_ticker_pnl(ax, result):
    if not hasattr(result, "trades") or not result.trades:
        ax.text(0.5, 0.5, "No trade data", ha="center", va="center", transform=ax.transAxes); return

    ticker_pnl: Dict[str, float] = {}
    for trade in result.trades:
        ticker_pnl[trade.ticker] = ticker_pnl.get(trade.ticker, 0.0) + trade.ret

    sorted_items = sorted(ticker_pnl.items(), key=lambda x: x[1])
    labels = [item[0] for item in sorted_items]
    vals = [item[1] * 100 for item in sorted_items]
    colors_bar = [_S["profit"] if v > 0 else _S["loss"] for v in vals]

    ax.barh(range(len(labels)), vals, color=colors_bar, alpha=0.7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(x=0, color="gray", lw=0.5)
    _ax(ax, "Per-Ticker P&L", xlabel="P&L %", ylabel="")


def _plot_rolling_sharpe(ax, result):
    eq = result.equity_curve
    if eq is None or len(eq) < 63:
        ax.text(0.5, 0.5, "Insufficient data for rolling Sharpe", ha="center", va="center", transform=ax.transAxes); return

    vals = eq.values if hasattr(eq, "values") else np.array(eq)
    daily_ret = np.diff(vals) / vals[:-1]
    daily_ret = np.insert(daily_ret, 0, 0.0)

    window = 63
    rolling_sharpe = []
    for i in range(window, len(daily_ret)):
        w = daily_ret[i - window:i]
        excess = w - (0.02 / 252)
        if excess.std() > 0:
            rolling_sharpe.append(np.sqrt(252) * excess.mean() / excess.std())
        else:
            rolling_sharpe.append(0.0)

    if rolling_sharpe:
        ax.plot(range(window, len(daily_ret)), rolling_sharpe, color=_S["eq"], linewidth=1.5)
        ax.axhline(y=0, color="gray", lw=0.5, alpha=0.3)
        _ax(ax, "Rolling Sharpe (63-day)", xlabel="Trading Days", ylabel="Sharpe")
    else:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)


# ── Training Monitor (8-panel, 20×12) ─────────────────────────────────

def plot_training_monitor(
    history, predictions=None, output_path="reports/training_monitor.png",
    dpi=150,
) -> str:
    if not HAS_MPL: return ""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(20, 12), dpi=dpi)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.40, wspace=0.35)

    tr = history.get("train_history", [])
    vl = history.get("val_history", [])

    if tr:
        _plot_loss_curves(fig.add_subplot(gs[0, 0]), tr, vl)
        _plot_accuracy_curves(fig.add_subplot(gs[0, 1]), tr, vl)
        _plot_lr_schedule(fig.add_subplot(gs[0, 2]), tr)

    if tr:
        _plot_loss_breakdown(fig.add_subplot(gs[1, 0]), tr, vl)

    if predictions is not None and not predictions.empty:
        _plot_classification_confusion(fig.add_subplot(gs[1, 1]), predictions)
        _plot_prediction_vs_actual(fig.add_subplot(gs[1, 2]), predictions)
        _plot_state_transitions(fig.add_subplot(gs[2, :2]), predictions)
        _plot_forecast_error(fig.add_subplot(gs[2, 2]), predictions)
    else:
        _plot_state_transitions_empty(fig.add_subplot(gs[2, :]))

    fig.suptitle("Training Performance Monitor", fontsize=14, fontweight="bold", y=0.99)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def _plot_loss_curves(ax, tr, vl):
    if not tr: return
    epochs = [e.get("epoch", i) for i, e in enumerate(tr)]
    ax.plot(epochs, [e.get("total", 0) for e in tr], color=_S["train"], linewidth=1.5, label="Train")
    if vl:
        ax.plot([e.get("epoch", i) for i, e in enumerate(vl)], [e.get("total", 0) for e in vl],
                color=_S["val"], linewidth=1.5, label="Val")
    _ax(ax, "Loss", xlabel="Epoch", ylabel="Loss"); ax.legend(fontsize=6)
    ax.set_yscale("log")


def _plot_accuracy_curves(ax, tr, vl):
    if not tr: return
    epochs = [e.get("epoch", i) for i, e in enumerate(tr)]
    ax.plot(epochs, [e.get("accuracy", 0) * 100 for e in tr], color=_S["train"], linewidth=1.5, label="Train Acc%")
    if vl:
        ax.plot([e.get("epoch", i) for i, e in enumerate(vl)], [e.get("accuracy", 0) * 100 for e in vl],
                color=_S["val"], linewidth=1.5, label="Val Acc%")
    ax.set_ylim(0, 100)
    _ax(ax, "Accuracy", xlabel="Epoch", ylabel="%"); ax.legend(fontsize=6)


def _plot_lr_schedule(ax, tr):
    if not tr: return
    epochs = [e.get("epoch", i) for i, e in enumerate(tr)]
    ax.plot(epochs, [e.get("lr", 0) for e in tr], color=_S["fcst"], linewidth=1.5)
    _ax(ax, "LR Schedule", xlabel="Epoch", ylabel="LR")
    ax.set_yscale("log")


def _plot_loss_breakdown(ax, tr, vl):
    if not tr: return
    last = tr[-1]
    components = {k: v for k, v in last.items() if k not in ("epoch", "lr", "total", "accuracy") and not k.startswith("_")}
    components.pop("total", None)
    labels = list(components.keys())
    vals = list(components.values())
    if not labels: return
    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
    ax.barh(labels, vals, color=colors, alpha=0.7)
    _ax(ax, "Loss Components (Final Epoch)", xlabel="Value")


def _plot_classification_confusion(ax, preds):
    if "event_state_pred" not in preds.columns or "true_state" not in preds.columns: return
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(preds["true_state"], preds["event_state_pred"], labels=[0, 1, 2])
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    labels = ["Inter", "Pre", "Onset"]
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(labels, fontsize=7)
    ax.set_yticks([0, 1, 2]); ax.set_yticklabels(labels, fontsize=7)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8, fontweight="bold")
    _ax(ax, "Confusion Matrix", xlabel="Predicted", ylabel="Actual")
    plt.colorbar(im, ax=ax, shrink=0.8)


def _plot_prediction_vs_actual(ax, preds):
    if "forecast_tau" not in preds.columns: return
    sample = preds.head(200) if len(preds) > 200 else preds
    ax.scatter(sample["true_state"], sample["forecast_tau"], c=sample["event_state_pred"],
              cmap="RdYlGn_r", alpha=0.4, s=10, edgecolors="none")
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["Inter", "Pre", "Onset"], fontsize=7)
    _ax(ax, "Forecast Tau by True State", xlabel="True State", ylabel="Predicted Tau (days)")


def _plot_state_transitions(ax, preds):
    if "event_state_pred" not in preds.columns: return

    n_show = min(500, len(preds))
    x_pos = range(n_show)

    if "true_state" in preds.columns:
        true_states = preds["true_state"].values[:n_show]
        true_colors = [_S["inter"], _S["pre"], _S["onset"]]
        for s in [0, 1, 2]:
            mask = true_states == s
            ax.scatter(np.array(x_pos)[mask], np.full(mask.sum(), s), c=true_colors[s],
                       s=12, alpha=0.3, marker="o", label=f"True {['Inter','Pre','Onset'][s]}")

    pred_states = preds["event_state_pred"].values[:n_show]
    pred_colors = [_S["inter"], _S["pre"], _S["onset"]]
    ax.scatter(x_pos, pred_states, c=[pred_colors[s] for s in pred_states], s=3, alpha=0.6, marker="s")

    ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["Inter", "Pre", "Onset"], fontsize=7)
    ax.set_ylim(-0.5, 2.5)
    _ax(ax, "State Transitions (first 500 samples)", xlabel="Sample Index", ylabel="State")
    if "true_state" in preds.columns:
        ax.legend(fontsize=6, loc="upper right", ncol=3)


def _plot_state_transitions_empty(ax):
    ax.text(0.5, 0.5, "Run training to generate prediction data", ha="center", va="center",
            transform=ax.transAxes, fontsize=10, color="gray")
    ax.set_xticks([]); ax.set_yticks([])


def _plot_forecast_error(ax, preds):
    if "forecast_tau" not in preds.columns: return
    true_tau = preds.get("true_tau", None)
    if true_tau is None:
        true_tau = pd.Series(np.full(len(preds), 21.0), index=preds.index)
    errors = np.abs(preds["forecast_tau"] - true_tau)
    sample = errors.head(500) if len(errors) > 500 else errors
    ax.hist(sample, bins=30, color=_S["fcst"], alpha=0.6)
    ax.axvline(x=np.mean(sample), color="black", lw=1.5, label=f"MAE: {np.mean(sample):.1f}d")
    _ax(ax, "Forecast Error Distribution", xlabel="Absolute Error (days)", ylabel="Count")
    ax.legend(fontsize=6)


# ── Forecasting Monitor (standalone, 12×8) ─────────────────────────────

def plot_forecast_horizon(
    predictions, output_path="reports/forecast_horizon.png", dpi=150,
) -> str:
    if not HAS_MPL: return ""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(16, 6), dpi=dpi)

    if predictions.empty or "forecast_tau" not in predictions.columns:
        ax.text(0.5, 0.5, "No forecast data", ha="center", va="center", transform=ax.transAxes)
        fig.savefig(output_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return output_path

    sample = predictions.head(200) if len(predictions) > 200 else predictions
    x = np.arange(len(sample))
    ax.fill_between(x, 0, sample["forecast_tau"], alpha=0.15, color=_S["fcst"])
    ax.plot(x, sample["forecast_tau"], color=_S["fcst"], linewidth=1.5, label="Predicted Tau")
    if "uncertainty" in sample.columns:
        upper = sample["forecast_tau"] + sample["uncertainty"]
        lower = np.maximum(0, sample["forecast_tau"] - sample["uncertainty"])
        ax.fill_between(x, lower, upper, alpha=0.08, color=_S["unc"], label="+/-Uncertainty")
    ax.axhline(y=21, color="gray", ls="--", lw=0.5, alpha=0.4, label="Horizon")
    ax.axhline(y=5, color=_S["onset"], ls="--", lw=0.5, alpha=0.4, label="Onset")
    _ax(ax, "Forecast Horizon Visualization", xlabel="Sample", ylabel="Days to Event")
    ax.legend(fontsize=7)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_inference_monitor(
    result, output_path="reports/inference_monitor.png", ticker="", dpi=150,
) -> str:
    if not HAS_MPL: return ""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    daily_signals = getattr(result, "daily_signals", {})
    if not daily_signals:
        fig, ax = plt.subplots(figsize=(12, 8), dpi=dpi)
        ax.text(0.5, 0.5, "No inference data", ha="center", va="center", transform=ax.transAxes)
        fig.savefig(output_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return output_path

    tickers_to_plot = [ticker] if ticker and ticker in daily_signals else sorted(daily_signals.keys())[:4]
    n_plots = len(tickers_to_plot)
    if n_plots == 0:
        return ""

    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3 * n_plots), dpi=dpi, squeeze=False)
    axes = axes.flatten()

    for idx, tk in enumerate(tickers_to_plot):
        ax = axes[idx]
        sig_df = daily_signals[tk]
        if sig_df.empty:
            ax.text(0.5, 0.5, f"No data for {tk}", ha="center", va="center", transform=ax.transAxes)
            continue

        x = np.arange(len(sig_df))
        pi = sig_df.get("prob_inter_event", pd.Series(np.zeros(len(sig_df))))
        pp = sig_df.get("prob_pre_event", pd.Series(np.zeros(len(sig_df))))
        po = sig_df.get("prob_onset", pd.Series(np.zeros(len(sig_df))))

        ax.fill_between(x, 0, po, alpha=0.25, color=_S["onset"], label="Onset")
        ax.fill_between(x, po, po + pp, alpha=0.25, color=_S["pre"], label="Pre-Event")
        ax.fill_between(x, po + pp, 1.0, alpha=0.25, color=_S["inter"], label="Inter-Event")

        if "forecast_tau" in sig_df.columns:
            ax2 = ax.twinx()
            ax2.plot(x, sig_df["forecast_tau"], color=_S["fcst"], linewidth=1.5, alpha=0.6, label="Tau (days)")
            ax2.set_ylabel("Days", fontsize=8)
            ax2.tick_params(labelsize=7)
            ax2.set_ylim(0, max(25, float(sig_df["forecast_tau"].max()) * 1.15))

        ax.set_ylim(0, 1.05)
        _ax(ax, f"{tk} — Signal Waterfall", xlabel="Trading Days", ylabel="Prob")

        has_forecast = "forecast_tau" in sig_df.columns
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = (ax2.get_legend_handles_labels() if has_forecast else ([], []))
        ax.legend(handles1 + handles2, labels1 + labels2, fontsize=6, loc="upper right", ncol=4)

    for extra in range(n_plots, len(axes)):
        axes[extra].set_visible(False)

    fig.suptitle("Inference Signal Monitor", fontsize=14, fontweight="bold", y=0.99)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


# ── Portfolio Allocation (pie + win rate leaders, 18×7) ────────────────

def plot_portfolio_allocation(
    result, output_path="reports/portfolio_allocation.png", dpi=150,
) -> str:
    if not HAS_MPL: return ""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), dpi=dpi)

    by_ticker = result.by_ticker if hasattr(result, "by_ticker") else None
    if by_ticker is None or by_ticker.empty:
        for a in [ax1, ax2]: a.text(0.5, 0.5, "No data", ha="center", va="center", transform=a.transAxes)
        fig.savefig(output_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return output_path

    top_n = min(10, len(by_ticker))
    top = by_ticker.nlargest(top_n, "n_trades")

    labels = list(top["ticker"].values) + ["Cash"]
    total_trades = int(by_ticker["n_trades"].sum()) if "n_trades" in by_ticker.columns else 0
    cash_pct = 0.20
    trade_pcts = list((top["n_trades"].values / max(1, top["n_trades"].sum())) * (1 - cash_pct)) + [cash_pct]
    colors = list(plt.cm.tab10(np.linspace(0, 1, len(labels) - 1))) + [_S["cash"]]
    ax1.pie(trade_pcts, labels=labels, colors=colors, autopct="%1.0f%%",
            startangle=90, textprops={"fontsize": 7}, pctdistance=0.82)
    ax1.set_title("Trade Activity Allocation", fontsize=10, fontweight="bold")

    top_wins = by_ticker.nlargest(min(10, len(by_ticker)), "win_rate")
    win_tks = top_wins["ticker"].values
    win_vals = top_wins["win_rate"].values * 100
    bars = ax2.barh(range(len(win_tks)), win_vals, color=_S["eq"], alpha=0.7)
    ax2.set_yticks(range(len(win_tks)))
    ax2.set_yticklabels(win_tks, fontsize=7)
    for bar, val in zip(bars, win_vals):
        ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2, f"{val:.0f}%", va="center", fontsize=7)
    ax2.set_xlim(0, 100)
    _ax(ax2, "Win Rate Leaders", xlabel="Win Rate %")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


# ── Generators ─────────────────────────────────────────────────────────

def generate_backtest_visualizations(
    result, signals=None, predictions=None,
    output_dir="reports", ticker="", prefix="",
) -> Dict[str, str]:
    paths = {}
    name = f"{prefix}_{ticker}" if ticker and prefix else ticker or "portfolio"
    if not ticker and prefix:
        name = prefix
    elif not name:
        name = "portfolio"

    paths["dashboard"] = plot_backtest_dashboard(
        result, signals, output_path=f"{output_dir}/{name}_dashboard.png", ticker=ticker,
    )
    paths["allocation_monitor"] = plot_allocation_monitor(
        result, output_path=f"{output_dir}/{name}_allocation_monitor.png",
    )
    paths["portfolio_allocation"] = plot_portfolio_allocation(
        result, output_path=f"{output_dir}/{name}_portfolio_alloc.png",
    )
    paths["inference"] = plot_inference_monitor(
        result, output_path=f"{output_dir}/{name}_inference.png", ticker=ticker,
    )

    return paths


def generate_training_visualizations(
    history, predictions=None, output_dir="reports", prefix="training",
) -> Dict[str, str]:
    paths = {}
    paths["monitor"] = plot_training_monitor(
        history, predictions,
        output_path=f"{output_dir}/{prefix}_monitor.png",
    )
    return paths
