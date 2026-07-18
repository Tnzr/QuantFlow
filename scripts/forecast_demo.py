#!/usr/bin/env python3
"""Forecasting demo — per-ticker ML forecast with confidence bands and price overlay."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantflow.models.architectures import create_model
from quantflow.models.trainer import TrainingConfig
from quantflow.models.dataset import FEATURE_COLUMNS
from quantflow.features.indicators import fetch_ohlcv, compute_indicators
from quantflow.data.labeler import _build_features_at


def build_features_for_inference(df: pd.DataFrame, lookback: int = 60) -> np.ndarray:
    from quantflow.models.dataset import _prepare_features

    n = len(df)
    feature_rows = []
    for i in range(lookback, n):
        feat = _build_features_at(df, i, ticker="ticker", forecast_horizon=21)
        feature_rows.append(feat)

    feat_df = pd.DataFrame(feature_rows)
    feat_df = _prepare_features(feat_df)
    expected = [c for c in FEATURE_COLUMNS if c in feat_df.columns]
    arr = feat_df[expected].fillna(0.0).values.astype(np.float32)
    return arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--model", default="checkpoints/bilstm_full_universe_v3.pt")
    parser.add_argument("--arch", default="bilstm")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--period", default="5y")
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()

    import torch
    ticker = args.ticker.upper()

    config = TrainingConfig(
        architecture=args.arch, lookback=args.lookback, hidden_dim=args.hidden_dim,
    )
    model = create_model(
        architecture=config.architecture, input_dim=len(FEATURE_COLUMNS),
        hidden_dim=config.hidden_dim, forecast_horizon=21,
    )
    model.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=True))
    model.eval()

    print(f"\n[cyan]Forecasting {ticker}...[/cyan]")

    df = fetch_ohlcv(ticker, period=args.period)
    df = compute_indicators(df)
    print(f"  OHLCV: {len(df)} rows, {df.index[0].date()} to {df.index[-1].date()}")

    features = build_features_for_inference(df, args.lookback)

    probs_all, forecasts_all, sigmas_all = [], [], []
    batch_size = 64
    with torch.no_grad():
        for start in range(args.lookback, len(features), batch_size):
            end = min(start + batch_size, len(features))
            windows = []
            for i in range(start, end):
                w = features[i - args.lookback:i]
                windows.append(w)
            if not windows:
                continue
            batch = torch.from_numpy(np.stack(windows))
            out = model(batch)
            probs_all.append(out["past_state_probs"].numpy())
            forecasts_all.append(out["future_forecast"].numpy().mean(axis=1))
            sigmas_all.append(out["uncertainty_sigma"].numpy().mean(axis=1))

    pa = np.concatenate(probs_all)
    fa = np.concatenate(forecasts_all)
    sa = np.concatenate(sigmas_all)

    idx_start = args.lookback

    print(f"\n[yellow]Latest Signal ({df.index[-1].date()}):[/yellow]")
    print(f"  Price: ${df['adj close'].iloc[-1]:.2f}")
    print(f"  State: {['inter_event','pre_event','onset'][int(pa[-1].argmax())]}")
    print(f"  Probs: inter={pa[-1][0]:.3f}, pre={pa[-1][1]:.3f}, onset={pa[-1][2]:.3f}")
    print(f"  Forecast tau: {fa[-1]:.1f} days (±{sa[-1]:.1f})")
    print(f"  Confidence band: [{max(0, fa[-1]-2*sa[-1]):.1f}, {fa[-1]+2*sa[-1]:.1f}] days")

    print(f"\n[yellow]Recent Forecast History (last 20 days):[/yellow]")
    print(f"  {'Date':>12} {'Price':>8} {'State':>12} {'Tau(d)':>8} {'±σ':>6} {'Inter':>6} {'Pre':>6} {'Onset':>6}")
    for i in range(max(0, len(pa)-20), len(pa)):
        d = df.index[i + idx_start]
        state = int(pa[i].argmax())
        print(f"  {str(d.date()):>12} ${df['adj close'].iloc[i+idx_start]:>7.2f} "
              f"{['inter','pre','onset'][state]:>12} {fa[i]:>8.1f} {sa[i]:>6.1f} "
              f"{pa[i][0]:>6.3f} {pa[i][1]:>6.3f} {pa[i][2]:>6.3f}")

    onset_count = int((pa.argmax(axis=1) == 2).sum())
    pre_count = int((pa.argmax(axis=1) == 1).sum())
    inter_count = len(pa) - onset_count - pre_count
    total = len(pa)
    print(f"\n[yellow]Summary stats:[/yellow]")
    print(f"  inter={inter_count} ({100*inter_count/total:.1f}%), "
          f"pre={pre_count} ({100*pre_count/total:.1f}%), "
          f"onset={onset_count} ({100*onset_count/total:.1f}%)")
    print(f"  Mean tau: {fa.mean():.1f}d, Median tau: {np.median(fa):.1f}d")
    print(f"  Mean sigma: {sa.mean():.2f}")

    if args.plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

        ax1 = axes[0]
        dates = df.index[idx_start:idx_start+len(pa)]
        prices = df['adj close'].values[idx_start:idx_start+len(pa)]
        ax1.plot(dates, prices, color="#2ecc71", linewidth=1.5, label=f"{ticker} Price")
        ax1.set_ylabel("Price ($)")
        ax1.legend(loc="upper left")
        ax1.grid(alpha=0.2)

        ax2 = axes[1]
        ax2.fill_between(range(len(pa)), 0, pa[:, 0], alpha=0.3, color="#27ae60", label="Inter")
        ax2.fill_between(range(len(pa)), pa[:, 0], pa[:, 0]+pa[:, 1], alpha=0.3, color="#f39c12", label="Pre")
        ax2.fill_between(range(len(pa)), pa[:, 0]+pa[:, 1], 1.0, alpha=0.3, color="#e74c3c", label="Onset")
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("State Probability")
        ax2.legend(loc="upper left", ncol=3, fontsize=8)
        ax2.grid(alpha=0.2)

        ax3 = axes[2]
        ax3.fill_between(range(len(pa)), np.maximum(0, fa-2*sa), fa+2*sa, alpha=0.15, color="#9b59b6")
        ax3.plot(range(len(pa)), fa, color="#9b59b6", linewidth=1.5, label="Forecast tau")
        ax3.axhline(21, color="gray", ls="--", alpha=0.4)
        ax3.axhline(5, color="#e74c3c", ls="--", alpha=0.4)
        ax3.set_ylabel("Days to Event")
        ax3.legend(loc="upper left")
        ax3.grid(alpha=0.2)

        out_path = f"reports/forecast_{ticker}.png"
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", facecolor="white", dpi=120)
        plt.close(fig)
        print(f"\n[green]Saved: {out_path}[/green]")


if __name__ == "__main__":
    main()
