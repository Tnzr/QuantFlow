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
    # NOTE: defaults below are None-by-default and only override LossConfig's
    # own dataclass defaults when explicitly passed, so this script's defaults
    # can never silently diverge from LossConfig's defaults again (previously
    # --cls-weight defaulted to 0.30 here vs 0.50 in LossConfig, so running
    # without the flag used a different value than a direct LossConfig()).
    parser.add_argument("--cls-weight", type=float, default=None)
    parser.add_argument("--reg-weight", type=float, default=None)
    parser.add_argument("--coh-weight", type=float, default=None)
    parser.add_argument("--dir-weight", type=float, default=None)
    parser.add_argument("--focal-gamma", type=float, default=None,
                         help="Focal loss gamma. Standard/stable value is 2.0 (LossConfig default). "
                              "Values >3 combined with class weighting can cause classification collapse.")
    parser.add_argument("--focal-alpha", type=float, default=None)
    parser.add_argument("--class-weight-mode", choices=["dynamic", "static"], default=None,
                         help="'dynamic' (default, recommended) computes class weights per-batch from "
                              "that batch's own class frequency per methodology §6.2.1/§8.2. 'static' uses "
                              "the hardcoded --static-class-weights tuple, which is prone to collapse if "
                              "hand-tuned too aggressively.")
    parser.add_argument("--static-class-weights", type=float, nargs=3, default=None,
                         metavar=("INTER", "PRE", "ONSET"),
                         help="Only used when --class-weight-mode static.")
    parser.add_argument("--no-coherent-heads", action="store_true",
                         help="Disable use_coherent_heads (decouples classification logits from the "
                              "forecast head's input) for ablation testing collapse root cause.")
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
        use_coherent_heads=not args.no_coherent_heads,
    )

    loss_overrides = {}
    if args.cls_weight is not None:
        loss_overrides["classification_weight"] = args.cls_weight
    if args.reg_weight is not None:
        loss_overrides["regression_weight"] = args.reg_weight
    if args.coh_weight is not None:
        loss_overrides["coherence_weight"] = args.coh_weight
    if args.dir_weight is not None:
        loss_overrides["direction_weight"] = args.dir_weight
    if args.focal_gamma is not None:
        loss_overrides["focal_gamma"] = args.focal_gamma
    if args.focal_alpha is not None:
        loss_overrides["focal_alpha"] = args.focal_alpha
    if args.class_weight_mode is not None:
        loss_overrides["class_weight_mode"] = args.class_weight_mode
    if args.static_class_weights is not None:
        loss_overrides["class_weights"] = tuple(args.static_class_weights)

    loss_config = LossConfig(**loss_overrides)
    print(f"Loss config: {loss_config}")

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
