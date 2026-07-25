"""Backtest visualization — equity curve, drawdown, sigma buckets, monthly heatmap."""

from __future__ import annotations

import io
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_backtest_report(report) -> object:
    """Generate comprehensive 6-panel backtest visualization.

    Returns a wandb.Image or PIL Image suitable for Streamlit/wandb.
    Panels:
      1. Equity curve + drawdown overlay
      2. Rolling Sharpe (60-bar window)
      3. Trade PnL distribution
      4. Sigma-bucketed win rate and avg return
      5. Cumulative PnL by exit reason
      6. Monthly returns heatmap
    """
    fig = plt.figure(figsize=(20, 16), dpi=100)
    gs = fig.add_gridspec(3, 2, hspace=0.30, wspace=0.25)

    # ── Panel 1: Equity curve + drawdown ──────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    eq = np.array(report.equity_curve)
    ax1.fill_between(range(len(eq)), report.capital, eq,
                     where=eq >= report.capital, color="#27ae60", alpha=0.3)
    ax1.fill_between(range(len(eq)), report.capital, eq,
                     where=eq < report.capital, color="#e74c3c", alpha=0.3)
    ax1.plot(eq, color="#2c3e50", linewidth=1.5)
    ax1.axhline(report.capital, color="gray", ls="--", alpha=0.5)
    ax1.set_title(f"Equity Curve — {report.total_return*100:.1f}% return, "
                  f"Sharpe={report.sharpe:.2f}, MaxDD={report.max_drawdown*100:.1f}%")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.grid(True, alpha=0.15, linestyle="--")

    # ── Drawdown subplot ──────────────────────────────────────────────
    ax1b = ax1.twinx()
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak * 100
    ax1b.fill_between(range(len(eq)), 0, dd, color="#e74c3c", alpha=0.2)
    ax1b.set_ylabel("Drawdown %", color="#e74c3c")
    ax1b.set_ylim(dd.min() * 1.2, 0)

    # ── Panel 2: Rolling Sharpe ───────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    returns = np.diff(eq) / (eq[:-1] + 1e-10)
    window = min(60, len(returns) // 3)
    if window > 5:
        rolling_sharpe = pd.Series(returns).rolling(window).apply(
            lambda x: np.mean(x) / max(np.std(x), 1e-10) * np.sqrt(252)
        ).values
        ax2.plot(rolling_sharpe, color="#8e44ad", linewidth=1.0)
        ax2.axhline(0, color="gray", ls="--", alpha=0.5)
        ax2.axhline(1.0, color="#27ae60", ls=":", alpha=0.5, label="Sharpe=1")
        ax2.axhline(2.0, color="#2ecc71", ls=":", alpha=0.5, label="Sharpe=2")
        ax2.legend(fontsize=6)
    ax2.set_title(f"Rolling Sharpe ({window}-bar)")
    ax2.set_ylabel("Sharpe Ratio")
    ax2.grid(True, alpha=0.15, linestyle="--")

    # ── Panel 3: Trade PnL distribution ───────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    if report.trades:
        pnls = [t.return_pct for t in report.trades]
        ax3.hist(pnls, bins=30, color="#8e44ad", alpha=0.7, edgecolor="white")
        ax3.axvline(0, color="black", linewidth=1.5)
        ax3.axvline(np.mean(pnls), color="#e74c3c", ls="--",
                    label=f"Mean: {np.mean(pnls):.1f}%")
        ax3.legend(fontsize=7)
    ax3.set_title(f"Trade PnL Distribution — {report.num_trades} trades, "
                  f"Win={report.win_rate*100:.0f}%, Avg Win={report.avg_win:.1f}%, "
                  f"Avg Loss={report.avg_loss:.1f}%")
    ax3.set_xlabel("Return per Trade (%)")
    ax3.set_ylabel("Count")
    ax3.grid(True, alpha=0.15, linestyle="--")

    # ── Panel 4: Sigma-bucketed performance ───────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    if report.sigma_buckets:
        buckets = report.sigma_buckets
        labels = list(buckets.keys())
        win_rates = [buckets[b]["win_rate"] * 100 for b in labels]
        avg_rets = [buckets[b]["avg_return"] for b in labels]
        counts = [buckets[b]["count"] for b in labels]

        x = np.arange(len(labels))
        w = 0.35
        bars1 = ax4.bar(x - w/2, win_rates, w, color="#9b59b6", alpha=0.8, label="Win Rate %")
        ax4.set_ylabel("Win Rate (%)", color="#9b59b6")
        ax4.tick_params(axis="y", labelcolor="#9b59b6")

        ax4b = ax4.twinx()
        bars2 = ax4b.bar(x + w/2, avg_rets, w, color="#2ecc71", alpha=0.8, label="Avg Return %")
        ax4b.set_ylabel("Avg Return (%)", color="#2ecc71")
        ax4b.tick_params(axis="y", labelcolor="#2ecc71")

        for i, (wr, ar, c) in enumerate(zip(win_rates, avg_rets, counts)):
            ax4.text(i, wr + 1, f"n={c}", ha="center", fontsize=7)

        ax4.set_xticks(x)
        ax4.set_xticklabels([f"{l}\n(σ<{'3' if l=='low' else '6' if l=='mid' else '∞'}%)" for l in labels], fontsize=7)
        ax4.set_title("Performance by Sigma Bucket (✓ low-σ should outperform high-σ)")
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4b.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, fontsize=6)

    # ── Panel 5: Exit reason analysis ─────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    if report.trades:
        reasons = {}
        for t in report.trades:
            reasons[t.exit_reason] = reasons.get(t.exit_reason, {"count": 0, "pnl": 0})
            reasons[t.exit_reason]["count"] += 1
            reasons[t.exit_reason]["pnl"] += t.return_pct

        r_names = list(reasons.keys())
        r_counts = [reasons[r]["count"] for r in r_names]
        r_pnls = [reasons[r]["pnl"] for r in r_names]
        colors = ["#2ecc71" if p > 0 else "#e74c3c" for p in r_pnls]

        bars = ax5.barh(r_names, r_pnls, color=colors, alpha=0.8)
        for i, (c, p) in enumerate(zip(r_counts, r_pnls)):
            ax5.text(p + (0.5 if p >= 0 else -0.5), i, f"{c} trades", va="center",
                    fontsize=7, ha="left" if p >= 0 else "right")
    ax5.set_title("Cumulative PnL by Exit Reason")
    ax5.set_xlabel("Total PnL (%)")
    ax5.axvline(0, color="gray", ls="--", alpha=0.5)
    ax5.grid(True, alpha=0.15, linestyle="--", axis="x")

    # ── Panel 6: Per-ticker summary ───────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    if not report.per_ticker.empty:
        ticker_summary = report.per_ticker.groupby("ticker").agg(
            total_pnl=("return_pct", "sum"),
            trades=("return_pct", "count"),
            win_rate=("return_pct", lambda x: (x > 0).mean()),
        ).sort_values("total_pnl", ascending=False)

        colors_t = ["#27ae60" if p > 0 else "#e74c3c" for p in ticker_summary["total_pnl"]]
        ax6.barh(ticker_summary.index[:15], ticker_summary["total_pnl"][:15],
                color=colors_t[:15], alpha=0.8)
    ax6.set_title("PnL by Ticker (top 15)")
    ax6.set_xlabel("Total PnL (%)")
    ax6.axvline(0, color="gray", ls="--", alpha=0.5)
    ax6.grid(True, alpha=0.15, linestyle="--", axis="x")

    fig.suptitle("CascadeANP Phase 2 Backtest Report", fontsize=13, fontweight="bold", y=0.98)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white", dpi=120)
    plt.close(fig)
    buf.seek(0)
    from PIL import Image
    return Image.open(buf)
