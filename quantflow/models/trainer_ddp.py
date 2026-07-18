"""Multi-GPU distributed training preserving chronological stateful methodology."""
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
from quantflow.models.dataset import FinancialTimeSeriesDataset, FEATURE_COLUMNS

logger = logging.getLogger(__name__)


def setup(rank: int, world_size: int):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def _prepare_dataloaders_ddp(
    df: pd.DataFrame,
    lookback: int = 60,
    forecast_horizon: int = 21,
    batch_size: int = 64,
    val_split: float = 0.15,
    test_split: float = 0.10,
    num_workers: int = 0,
    seed: int = 42,
    rank: int = 0,
    world_size: int = 1,
):
    """Split dataset chronologically and return DDP-safe DataLoaders.

    Preserves per-ticker chronological ordering. Tickers are partitioned across
    GPUs deterministically — each GPU owns a non-overlapping subset of tickers.
    Hidden states are per-GPU, per-ticker — no cross-GPU state sync needed.
    """
    tickers = df["ticker"].unique()
    np.random.seed(seed)
    np.random.shuffle(tickers)

    n_val = max(1, int(len(tickers) * val_split))
    n_test = max(1, int(len(tickers) * test_split))
    n_train = len(tickers) - n_val - n_test

    train_tickers = sorted(tickers[:n_train])
    val_tickers = sorted(tickers[n_train:n_train + n_val])
    test_tickers = sorted(tickers[n_train + n_val:])

    train_df = df[df["ticker"].isin(train_tickers)].copy()
    val_df = df[df["ticker"].isin(val_tickers)].copy()
    test_df = df[df["ticker"].isin(test_tickers)].copy()

    if rank == 0:
        logger.info(f"Split: train={len(train_df)} rows ({n_train} tickers), "
                    f"val={len(val_df)} rows ({n_val} tickers), "
                    f"test={len(test_df)} rows ({n_test} tickers)")

    train_ds = FinancialTimeSeriesDataset(train_df, lookback=lookback, forecast_horizon=forecast_horizon)
    val_ds = FinancialTimeSeriesDataset(val_df, lookback=lookback, forecast_horizon=forecast_horizon)
    test_ds = FinancialTimeSeriesDataset(test_df, lookback=lookback, forecast_horizon=forecast_horizon)

    if rank == 0:
        logger.info(f"Dataset: train_samples={len(train_ds)}, val_samples={len(val_ds)}, test_samples={len(test_ds)}")
        logger.info(f"Train class distribution: {train_ds.class_distribution}")
        logger.info(f"Training mode: CHRONOLOGICAL per-ticker DDP ({world_size} GPUs)")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

    return train_loader, val_loader, test_loader, train_ds.feature_dim


def train_ddp(rank: int, world_size: int, df: pd.DataFrame, config: TrainingConfig, loss_config: LossConfig, args):
    setup(rank, world_size)

    train_loader, val_loader, test_loader, feature_dim = _prepare_dataloaders_ddp(
        df,
        lookback=config.lookback,
        forecast_horizon=config.forecast_horizon,
        batch_size=config.batch_size,
        rank=rank,
        world_size=world_size,
    )

    model = create_model(
        architecture=config.architecture,
        input_dim=feature_dim,
        hidden_dim=config.hidden_dim,
        forecast_horizon=config.forecast_horizon,
        dropout=config.dropout,
        use_multi_scale=config.use_multi_scale,
        use_coherent_heads=config.use_coherent_heads,
    ).to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    criterion = CompositeLoss(loss_config).to(rank)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs * len(train_loader), eta_min=config.min_lr,
    )

    hidden_states = {}
    prev_ticker = None

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0
        epoch_class_correct = {0: 0, 1: 0, 2: 0}
        epoch_class_count = {0: 0, 1: 0, 2: 0}

        pbar = tqdm(train_loader, desc=f"[GPU{rank}] Epoch {epoch+1}/{config.epochs}", disable=(rank != 0), leave=False)

        for features_raw, labels_dict, tickers in pbar:
            ticker = tickers[0] if isinstance(tickers, (list, tuple)) else tickers
            if ticker != prev_ticker:
                prev_ticker = ticker

            features = features_raw.to(rank)
            labels = {k: v.to(rank) for k, v in labels_dict.items() if isinstance(v, torch.Tensor)}

            h_prev = hidden_states.get(ticker, None)
            if h_prev is not None:
                h_prev = [h.detach() for h in h_prev]

            optimizer.zero_grad()
            outputs = model(features, h_prev=h_prev)
            losses = criterion(outputs, labels)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            scheduler.step()

            h_new = outputs.get("hidden_state", None)
            if h_new is not None:
                hidden_states[ticker] = [h.detach() for h in h_new]

            batch_loss = losses["total"].item()
            epoch_loss += batch_loss

            preds = outputs["past_state_logits"].argmax(dim=-1)
            ldev = labels["event_state_code"]
            epoch_correct += (preds == ldev).sum().item()
            epoch_samples += ldev.size(0)
            for c in range(3):
                mask = ldev == c
                epoch_class_count[c] += mask.sum().item()
                epoch_class_correct[c] += (preds[mask] == c).sum().item()

            pbar.set_postfix(loss=f"{batch_loss:.2f}")

        epoch_loss_tensor = torch.tensor(epoch_loss, device=rank)
        epoch_correct_tensor = torch.tensor(epoch_correct, device=rank)
        epoch_samples_tensor = torch.tensor(epoch_samples, device=rank)

        dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(epoch_correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(epoch_samples_tensor, op=dist.ReduceOp.SUM)

        if rank == 0:
            avg_loss = epoch_loss_tensor.item() / (world_size * max(len(train_loader), 1))
            accuracy = epoch_correct_tensor.item() / max(epoch_samples_tensor.item(), 1)
            logger.info(f"Epoch {epoch+1}/{config.epochs} | train_loss={avg_loss:.4f} | acc={accuracy:.3f} | "
                       f"i={epoch_class_correct[0]/max(epoch_class_count[0],1):.2f} "
                       f"p={epoch_class_correct[1]/max(epoch_class_count[1],1):.2f} "
                       f"o={epoch_class_correct[2]/max(epoch_class_count[2],1):.2f}")

            best_path = Path(config.checkpoint_dir) / "best_model_ddp.pt"
            best_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.module.state_dict(), best_path)

        if not config.keep_hidden_across_epochs:
            hidden_states = {}
            prev_ticker = None

    if rank == 0 and args.save:
        torch.save(model.module.state_dict(), args.save)
        logger.info(f"Model saved to {args.save}")

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
    parser.add_argument("--cls-weight", type=float, default=0.40)
    parser.add_argument("--reg-weight", type=float, default=0.25)
    parser.add_argument("--coh-weight", type=float, default=0.05)
    parser.add_argument("--dir-weight", type=float, default=0.05)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available. Use single-GPU training instead.")
        sys.exit(1)

    world_size = torch.cuda.device_count()
    print(f"Starting DDP training on {world_size} GPUs | Chronological stateful mode | 100 epochs")

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
        stateful=True,
        keep_hidden_across_epochs=True,
    )

    loss_config = LossConfig(
        classification_weight=args.cls_weight,
        regression_weight=args.reg_weight,
        coherence_weight=args.coh_weight,
        direction_weight=args.dir_weight,
    )

    mp.spawn(
        train_ddp,
        args=(world_size, df, config, loss_config, args),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
