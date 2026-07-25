#!/usr/bin/env python3
"""Shuffle ablation: test whether chronological ordering causes classification collapse.

Per the remediation plan's handoff section (§NEXT STEPS, §1): run a 10-epoch
ablation with shuffle=True, stateful=False, h_prev=None to see if classification
recall for inter/onset becomes non-zero (matching the RF result of ~93%/81%).

If this ablation shows non-zero minority-class recall → hypothesis confirmed →
implement two-phase curriculum.
If it ALSO collapses → hypothesis wrong → investigate model capacity/LR/optimizer.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantflow.models.dataset import FinancialTimeSeriesDataset
from quantflow.models.architectures import create_model
from quantflow.models.losses import CompositeLoss, LossConfig


@dataclass
class AblationConfig:
    epochs: int = 10
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 2.0
    lookback: int = 60
    forecast_horizon: int = 21
    arch: str = "bilstm"
    hidden_dim: int = 128
    dropout: float = 0.2
    # Shuffle vs chronological — the independent variable
    shuffle: bool = True


def make_shuffled_loaders(
    df: pd.DataFrame,
    config: AblationConfig,
    val_split: float = 0.15,
    test_split: float = 0.10,
    seed: int = 42,
):
    """Build shuffled DataLoaders with NO TickerBatchSampler."""
    tickers = sorted(df["ticker"].unique())
    np.random.seed(seed)
    np.random.shuffle(tickers)

    n_val = max(1, int(len(tickers) * val_split))
    n_test = max(1, int(len(tickers) * test_split))
    n_train = len(tickers) - n_val - n_test

    train_tickers = tickers[:n_train]
    val_tickers = tickers[n_train:n_train + n_val]
    test_tickers = tickers[n_train + n_val:]

    train_df = df[df["ticker"].isin(train_tickers)].copy()
    val_df = df[df["ticker"].isin(val_tickers)].copy()
    test_df = df[df["ticker"].isin(test_tickers)].copy()

    train_ds = FinancialTimeSeriesDataset(train_df, lookback=config.lookback, forecast_horizon=config.forecast_horizon)
    val_ds = FinancialTimeSeriesDataset(val_df, lookback=config.lookback, forecast_horizon=config.forecast_horizon)
    test_ds = FinancialTimeSeriesDataset(test_df, lookback=config.lookback, forecast_horizon=config.forecast_horizon)

    print(f"Split: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    print(f"Train class dist: {train_ds.class_distribution}")

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=config.shuffle, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader, train_ds.feature_dim


def evaluate(model, criterion, loader, device):
    model.eval()
    total_losses = {}
    total_correct = 0
    total_samples = 0
    class_correct = {0: 0, 1: 0, 2: 0}
    class_count = {0: 0, 1: 0, 2: 0}
    pred_count = {0: 0, 1: 0, 2: 0}
    confusion = np.zeros((3, 3), dtype=np.int64)

    with torch.no_grad():
        for features_batch, labels_batch, _ in loader:
            features_batch = features_batch.to(device)
            labels_dev = {k: v.to(device) for k, v in labels_batch.items() if isinstance(v, torch.Tensor)}
            outputs = model(features_batch, h_prev=None)
            losses = criterion(outputs, labels_dev)

            for k, v in losses.items():
                total_losses[k] = total_losses.get(k, 0.0) + v.item()

            preds = outputs["past_state_logits"].argmax(dim=-1)
            labels_tensor = labels_dev["event_state_code"]
            total_correct += (preds == labels_tensor).sum().item()
            total_samples += labels_tensor.size(0)

            for c in range(3):
                mask = labels_tensor == c
                class_count[c] += mask.sum().item()
                class_correct[c] += (preds[mask] == c).sum().item()
                pred_count[c] += (preds == c).sum().item()

            for t, p in zip(labels_tensor.cpu().numpy(), preds.cpu().numpy()):
                confusion[int(t), int(p)] += 1

    n_batches = max(len(loader), 1)
    metrics = {k: v / n_batches for k, v in total_losses.items()}
    metrics["accuracy"] = total_correct / max(total_samples, 1)
    metrics["accuracy_inter"] = class_correct[0] / max(class_count[0], 1)
    metrics["accuracy_pre"] = class_correct[1] / max(class_count[1], 1)
    metrics["accuracy_onset"] = class_correct[2] / max(class_count[2], 1)
    # Recall = row-normalized: fraction of actual class i that was predicted i
    for c in range(3):
        row_sum = max(class_count[c], 1)
        cm_norm = confusion[c, c] / row_sum
        metrics[f"recall_inter" if c == 0 else f"recall_pre" if c == 1 else f"recall_onset"] = cm_norm

    total_preds = max(sum(pred_count.values()), 1)
    metrics["pred_class_dominance"] = max(pred_count.values()) / total_preds
    return metrics


def run_ablation(df: pd.DataFrame, config: AblationConfig, device: str):
    train_loader, val_loader, test_loader, feature_dim = make_shuffled_loaders(df, config)

    model = create_model(
        architecture=config.arch,
        input_dim=feature_dim,
        hidden_dim=config.hidden_dim,
        forecast_horizon=config.forecast_horizon,
        dropout=config.dropout,
    ).to(device)

    loss_cfg = LossConfig()
    criterion = CompositeLoss(loss_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs * len(train_loader), eta_min=1e-5)

    print(f"\n{'='*60}")
    print(f"ABLATION: shuffle={config.shuffle}, stateful=False, arch={config.arch}")
    print(f"Epochs={config.epochs}, LR={config.lr}, params={sum(p.numel() for p in model.parameters()):,}")
    print(f"Loss: class_weight_mode={loss_cfg.class_weight_mode}, focal_gamma={loss_cfg.focal_gamma}")
    print(f"{'='*60}\n")

    for epoch in range(config.epochs):
        model.train()
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}", leave=False)
        for features_batch, labels_batch, _ in pbar:
            features_batch = features_batch.to(device)
            labels_dev = {k: v.to(device) for k, v in labels_batch.items() if isinstance(v, torch.Tensor)}

            optimizer.zero_grad()
            outputs = model(features_batch, h_prev=None)
            losses = criterion(outputs, labels_dev)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            scheduler.step()

            epoch_loss += losses["total"].item()
            preds = outputs["past_state_logits"].argmax(dim=-1)
            epoch_correct += (preds == labels_dev["event_state_code"]).sum().item()
            epoch_samples += labels_dev["event_state_code"].size(0)
            pbar.set_postfix(loss=f"{losses['total'].item():.3f}")

        elapsed = time.time() - epoch_start
        train_acc = epoch_correct / max(epoch_samples, 1)
        val_metrics = evaluate(model, criterion, val_loader, device)

        # Print key metrics: recall is what we care about
        print(
            f"Epoch {epoch+1:2d} ({elapsed:4.1f}s) | loss={epoch_loss/len(train_loader):.3f} "
            f"| train_acc={train_acc:.3f} "
            f"| val_loss={val_metrics.get('total',0):.3f} val_acc={val_metrics['accuracy']:.3f} "
            f"| RECALL: i={val_metrics['recall_inter']:.2f} p={val_metrics['recall_pre']:.2f} o={val_metrics['recall_onset']:.2f} "
            f"| dom={val_metrics['pred_class_dominance']:.2f}"
        )

    # Final test evaluation
    print(f"\n{'='*60}")
    print("FINAL TEST EVALUATION")
    print(f"{'='*60}")
    test_metrics = evaluate(model, criterion, test_loader, device)
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test accuracy_inter: {test_metrics['accuracy_inter']:.4f}")
    print(f"Test accuracy_pre:   {test_metrics['accuracy_pre']:.4f}")
    print(f"Test accuracy_onset: {test_metrics['accuracy_onset']:.4f}")
    print(f"Test RECALL inter:  {test_metrics['recall_inter']:.4f}")
    print(f"Test RECALL pre:    {test_metrics['recall_pre']:.4f}")
    print(f"Test RECALL onset:  {test_metrics['recall_onset']:.4f}")
    print(f"Test class dominance: {test_metrics['pred_class_dominance']:.4f}")

    # Diagnosis
    minority_recall = (test_metrics["recall_inter"] + test_metrics["recall_onset"]) / 2.0
    if minority_recall > 0.1:
        print(f"\n>>> HYPOTHESIS CONFIRMED: minority-class recall = {minority_recall:.3f} (non-zero)")
        print("    Shuffle breaks the collapse. Two-phase curriculum is the right fix.")
    else:
        print(f"\n>>> HYPOTHESIS REJECTED: minority-class recall = {minority_recall:.3f} (still near-zero)")
        print("    Collapse is NOT caused by chronological ordering alone.")
        print("    Next: check model capacity, LR, optimizer, weight init.")

    return test_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/daily_20_tickers.parquet")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--arch", default="bilstm", choices=["bilstm", "transformer", "tcn"])
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--no-shuffle", action="store_true", help="Run chronological baseline instead of shuffle")
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    df = pd.read_parquet(args.data)
    if args.max_rows and len(df) > args.max_rows:
        df = df.sample(n=args.max_rows, random_state=42)
    print(f"Data: {len(df)} rows, {df['ticker'].nunique()} tickers")
    print(f"Class dist: {df['event_state_code'].value_counts().to_dict()}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    config = AblationConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        arch=args.arch,
        hidden_dim=args.hidden_dim,
        shuffle=not args.no_shuffle,
    )

    run_ablation(df, config, device)


if __name__ == "__main__":
    main()
