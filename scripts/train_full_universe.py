#!/usr/bin/env python3
"""Full-universe training with wandb logging and visualization output."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import wandb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantflow.models.trainer import TrainingConfig, train_model
from quantflow.models.losses import LossConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/full_universe_5y.parquet")
    parser.add_argument("--arch", default="bilstm", choices=["bilstm", "transformer", "tcn"])
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--forecast-horizon", type=int, default=21)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--save", default="")
    parser.add_argument("--wandb-project", default="quantflow")
    parser.add_argument("--wandb-name", default="")
    parser.add_argument("--device", default="")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--cls-weight", type=float, default=0.30)
    parser.add_argument("--reg-weight", type=float, default=0.20)
    parser.add_argument("--coh-weight", type=float, default=0.10)
    parser.add_argument("--dir-weight", type=float, default=0.10)
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    df = pd.read_parquet(args.data)
    if args.max_rows and len(df) > args.max_rows:
        df = df.sample(n=args.max_rows, random_state=42)
        print(f"Sampled {args.max_rows} rows")

    print(f"Dataset: {len(df)} rows, {df['ticker'].nunique()} tickers")
    print(f"Class distribution: \n{df['event_state_code'].value_counts().to_dict()}")

    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lookback=args.lookback,
        forecast_horizon=args.forecast_horizon,
        architecture=args.arch,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        checkpoint_dir=args.checkpoint_dir,
    )

    loss_config = LossConfig(
        classification_weight=args.cls_weight,
        regression_weight=args.reg_weight,
        coherence_weight=args.coh_weight,
        direction_weight=args.dir_weight,
    )

    if not args.no_wandb:
        wandb_name = args.wandb_name or f"{args.arch}-h{args.hidden_dim}-e{args.epochs}"
        wandb.init(
            project=args.wandb_project,
            name=wandb_name,
            config={
                "architecture": args.arch,
                "hidden_dim": args.hidden_dim,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "lookback": args.lookback,
                "forecast_horizon": args.forecast_horizon,
                "dataset_rows": len(df),
                "dataset_tickers": df['ticker'].nunique(),
                "class_distribution": df['event_state_code'].value_counts().to_dict(),
            },
        )

    print(f"Training {args.arch} (hidden_dim={args.hidden_dim}, epochs={args.epochs})...")
    trainer, history, predictions = train_model(
        df, config=config, loss_config=loss_config, device=args.device,
    )

    print(f"Training complete.")
    print(f"Best val loss: {trainer.best_val_loss:.4f}")
    print(f"Epochs run: {len(history['train_history'])}")

    if predictions is not None and not predictions.empty:
        acc = (predictions["true_state"] == predictions["event_state_pred"]).mean()
        print(f"Test accuracy: {acc:.3%}")
        print(f"Predictions: {len(predictions)} samples")

    if args.save:
        trainer.save_model(args.save)
        if not args.no_wandb and wandb.run:
            wandb.save(args.save)

    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
