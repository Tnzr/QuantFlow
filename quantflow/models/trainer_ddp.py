"""Multi-GPU distributed training for the Temporal State Model."""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quantflow.models.trainer import TrainingConfig, Trainer
from quantflow.models.architectures import create_model
from quantflow.models.losses import LossConfig, CompositeLoss
from quantflow.models.dataset import prepare_dataloaders, FinancialTimeSeriesDataset, FEATURE_COLUMNS

logger = logging.getLogger(__name__)


def setup(rank: int, world_size: int):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def train_ddp(rank: int, world_size: int, df: pd.DataFrame, config: TrainingConfig, loss_config: LossConfig):
    setup(rank, world_size)

    feature_dim = len(FEATURE_COLUMNS)
    model = create_model(
        architecture=config.architecture,
        input_dim=feature_dim,
        hidden_dim=config.hidden_dim,
        forecast_horizon=config.forecast_horizon,
        dropout=config.dropout,
        use_multi_scale=config.use_multi_scale,
        use_coherent_heads=config.use_coherent_heads,
    ).to(rank)
    model = DDP(model, device_ids=[rank])

    criterion = CompositeLoss(loss_config).to(rank)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    train_loader, val_loader, test_loader, _ = prepare_dataloaders(
        df,
        lookback=config.lookback,
        forecast_horizon=config.forecast_horizon,
        batch_size=config.batch_size,
    )

    if rank == 0:
        logger.info(f"DDP training: {world_size} GPUs, {config.epochs} epochs")

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0

        pbar = tqdm(train_loader, desc=f"[GPU{rank}] Epoch {epoch+1}", disable=(rank != 0), leave=False)
        for features, labels_dict, _ in pbar:
            features = features.to(rank)
            labels_dict = {k: v.to(rank) for k, v in labels_dict.items()}

            optimizer.zero_grad()
            outputs = model(features)
            losses = criterion(outputs, labels_dict)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()

            epoch_loss += losses["total"].item()
            preds = outputs["past_state_logits"].argmax(dim=-1)
            epoch_correct += (preds == labels_dict["event_state_code"]).sum().item()
            epoch_samples += labels_dict["event_state_code"].size(0)
            pbar.set_postfix(loss=losses["total"].item())

        epoch_loss_tensor = torch.tensor(epoch_loss, device=rank)
        epoch_correct_tensor = torch.tensor(epoch_correct, device=rank)
        epoch_samples_tensor = torch.tensor(epoch_samples, device=rank)

        dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(epoch_correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(epoch_samples_tensor, op=dist.ReduceOp.SUM)

        if rank == 0:
            avg_loss = epoch_loss_tensor.item() / (world_size * max(len(train_loader), 1))
            accuracy = epoch_correct_tensor.item() / max(epoch_samples_tensor.item(), 1)
            logger.info(f"Epoch {epoch+1}/{config.epochs} | train_loss={avg_loss:.4f} | acc={accuracy:.3f}")

            best_path = Path(config.checkpoint_dir) / "best_model_ddp.pt"
            best_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.module.state_dict(), best_path)
            logger.info(f"Saved checkpoint to {best_path}")

    cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/full_universe_5y.parquet")
    parser.add_argument("--arch", default="bilstm")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--forecast-horizon", type=int, default=21)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--save", default="checkpoints/model_ddp.pt")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available. Use single-GPU training instead.")
        sys.exit(1)

    world_size = torch.cuda.device_count()
    print(f"Starting DDP training on {world_size} GPUs")

    df = pd.read_parquet(args.data)
    print(f"Dataset: {len(df)} rows, {df['ticker'].nunique()} tickers")

    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lookback=args.lookback,
        forecast_horizon=args.forecast_horizon,
        architecture=args.arch,
        hidden_dim=args.hidden_dim,
        checkpoint_dir=args.checkpoint_dir,
    )

    loss_config = LossConfig()

    mp.spawn(
        train_ddp,
        args=(world_size, df, config, loss_config),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
