#!/usr/bin/env python3
"""Run portfolio backtest with trained model on full universe."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantflow.models.trainer import TrainingConfig, Trainer
from quantflow.models.architectures import create_model
from quantflow.models.dataset import FEATURE_COLUMNS
from quantflow.models.backtest_engine import AIBacktestEngine
from quantflow.data.universe import SECTOR_UNIVERSE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="checkpoints/bilstm_full_universe.pt")
    parser.add_argument("--arch", default="bilstm")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--forecast-horizon", type=int, default=21)
    parser.add_argument("--tickers", default="", help="Comma-separated, empty=ALL from SECTOR_UNIVERSE")
    parser.add_argument("--max-tickers", type=int, default=30, help="Cap tickers for yfinance rate limits")
    parser.add_argument("--period", default="5y")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--entry-threshold", type=float, default=0.60)
    parser.add_argument("--exit-threshold", type=float, default=0.40)
    parser.add_argument("--max-hold-days", type=int, default=21)
    parser.add_argument("--stop-loss-pct", type=float, default=0.05)
    parser.add_argument("--take-profit-pct", type=float, default=0.0)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--device", default="")
    parser.add_argument("--viz", action="store_true")
    parser.add_argument("--viz-dir", default="reports")
    args = parser.parse_args()

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = sorted(set(t for lst in SECTOR_UNIVERSE.values() for t in lst))

    if args.max_tickers:
        tickers = tickers[:args.max_tickers]

    print(f"Portfolio backtest: {len(tickers)} tickers")
    print(f"Tickers: {tickers}")

    config = TrainingConfig(
        architecture=args.arch,
        lookback=args.lookback,
        forecast_horizon=args.forecast_horizon,
        hidden_dim=args.hidden_dim,
    )

    model = create_model(
        architecture=config.architecture,
        input_dim=len(FEATURE_COLUMNS),
        hidden_dim=config.hidden_dim,
        forecast_horizon=config.forecast_horizon,
    )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(model, config=config, device=device)
    trainer.load_model(args.model)

    engine = AIBacktestEngine(
        trainer=trainer,
        entry_threshold=args.entry_threshold,
        exit_threshold=args.exit_threshold,
        max_hold_days=args.max_hold_days,
        stop_loss_pct=args.stop_loss_pct,
        take_profit_pct=args.take_profit_pct,
        top_n=args.top_n,
    )

    result = engine.run(
        tickers=tickers,
        period=args.period,
        interval=args.interval,
        start_date=args.start,
    )

    print(f"\n===== Backtest Results =====")
    print(f"Trades: {result.n_trades}")
    print(f"Win Rate: {result.win_rate:.2%}")
    print(f"Avg Return: {result.avg_ret:.2%}")
    print(f"Max Drawdown: {result.max_drawdown:.2%}")
    print(f"Sharpe: {result.sharpe_ratio:.2f}")
    print(f"Sortino: {result.sortino_ratio:.2f}")
    print(f"CAGR: {result.cagr:.2%}")
    print(f"Profit Factor: {result.profit_factor:.2f}")
    print(f"Total Return: {result.total_return:.2%}")

    if hasattr(result, "daily_equity") and len(result.daily_equity) > 0:
        print(f"\nEquity Curve: start=${result.daily_equity.iloc[0]:.4f} end=${result.daily_equity.iloc[-1]:.4f}")
        print(f"Trading days: {len(result.daily_equity)}")

    if not result.by_ticker.empty:
        print(f"\n===== By Ticker =====")
        for row in result.by_ticker.to_dict(orient="records"):
            print(f"  {row['ticker']:>8}: trades={row['n_trades']} win={row['win_rate']:.2%} avg_ret={row['avg_ret']:.2%}")

    if result.trade_log:
        exit_reasons = {}
        for t in result.trade_log:
            r = t.get("exit_reason", "unknown")
            exit_reasons[r] = exit_reasons.get(r, 0) + 1
        print(f"\n===== Exit Reasons =====")
        for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")

    if args.viz:
        try:
            from quantflow.models.visualization import generate_backtest_visualizations
            import pandas as pd

            signals = {}
            daily_signals = getattr(result, "daily_signals", {})
            if daily_signals:
                first_tk = next(iter(daily_signals.keys()), None)
                if first_tk and not daily_signals[first_tk].empty:
                    df_sig = daily_signals[first_tk]
                    signals = {
                        "prob_inter_event": df_sig.get("prob_inter_event", pd.Series(dtype=float)).values,
                        "prob_pre_event": df_sig.get("prob_pre_event", pd.Series(dtype=float)).values,
                        "prob_onset": df_sig.get("prob_onset", pd.Series(dtype=float)).values,
                        "forecast_tau": df_sig.get("forecast_tau", pd.Series(dtype=float)).values,
                    }

            paths = generate_backtest_visualizations(
                result, signals,
                output_dir=args.viz_dir,
                ticker="portfolio",
                prefix="portfolio_backtest",
            )
            for name, path in paths.items():
                if path:
                    print(f"  viz/{name}: {path}")
        except Exception as e:
            print(f"Viz failed: {e}")


if __name__ == "__main__":
    main()
