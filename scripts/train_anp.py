#!/usr/bin/env python3
"""Train Attention Neural Process (AttnNP) on intraday 5-min data.

Uses random-sampling: each batch randomly samples context/target bars
per ticker, breaking chronological correlation and providing natural
batch diversity.

Loss: Gaussian NLL + β·KL + direction penalty
"""

import argparse
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import wandb
from tqdm import tqdm

from quantflow.models.anp import AttnNP, ANPLoss
from quantflow.models.anp_dataset import create_anp_dataloader


def train_anp(
    data_path: str,
    n_context: int = 80,
    n_target: int = 15,
    forecast_horizon: int = 78,
    hidden_dim: int = 128,
    latent_dim: int = 64,
    enc_layers: int = 3,
    heads: int = 4,
    dropout: float = 0.1,
    beta: float = 0.1,
    batch_size: int = 32,
    epochs: int = 30,
    lr: float = 2e-4,
    weight_decay: float = 1e-5,
    device: str = "cuda",
    wandb_project: str = "quantflow",
    wandb_name: Optional[str] = None,
    save_path: Optional[str] = None,
    no_wandb: bool = False,
):
    df = pd.read_parquet(data_path)
    print(f"Data: {len(df)} rows, {df['ticker'].nunique()} tickers")

    # Split tickers: 80% train, 15% val, 5% test
    all_tickers = sorted(df["ticker"].unique())
    np.random.seed(42)
    np.random.shuffle(all_tickers)
    n_train = int(len(all_tickers) * 0.80)
    n_val = int(len(all_tickers) * 0.15)

    train_tickers = all_tickers[:n_train]
    val_tickers = all_tickers[n_train:n_train + n_val]
    test_tickers = all_tickers[n_train + n_val:]

    train_df = df[df["ticker"].isin(train_tickers)]
    val_df = df[df["ticker"].isin(val_tickers)]
    test_df = df[df["ticker"].isin(test_tickers)]
    print(f"Split: train={len(train_tickers)} val={len(val_tickers)} test={len(test_tickers)} tickers")

    train_loader, n_features = create_anp_dataloader(
        train_df, n_context, n_target, forecast_horizon, batch_size,
        episodes_per_ticker=20)
    val_loader, _ = create_anp_dataloader(
        val_df, n_context, n_target, forecast_horizon, batch_size,
        episodes_per_ticker=5)

    print(f"Features: {n_features}, context bars: {n_context}, target bars: {n_target}")

    model = AttnNP(x_dim=n_features, hidden_dim=hidden_dim, latent_dim=latent_dim,
                   enc_layers=enc_layers, heads=heads, dropout=dropout).to(device)
    criterion = ANPLoss(beta=beta)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader), eta_min=1e-5)

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    if not no_wandb:
        name = wandb_name or f"anp-h{hidden_dim}-c{n_context}-t{n_target}"
        wandb.init(project=wandb_project, name=name, config={
            "model": "AttnNP",
            "hidden_dim": hidden_dim,
            "latent_dim": latent_dim,
            "n_context": n_context,
            "n_target": n_target,
            "n_features": n_features,
            "epochs": epochs,
            "lr": lr,
            "beta": beta,
        }, settings=wandb.Settings(init_timeout=180))

    best_val_loss = float("inf")
    epoch = 0  # Track for scheduler

    for epoch_num in range(epochs):
        model.train()
        train_losses = {"total": 0, "regression": 0, "sigma_reg": 0, "kl": 0, "direction": 0}
        train_dir, train_total = 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch_num + 1}/{epochs}", leave=False)
        for x_c, y_c, x_t, y_t, tickers_list in pbar:
            x_c = x_c.to(device)
            y_c = y_c.to(device)
            x_t = x_t.to(device)
            y_t_dev = y_t.to(device)

            optimizer.zero_grad()
            outputs = model(x_c, y_c, x_t)
            loss = criterion(outputs, {"target_return": y_t_dev})
            loss["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            scheduler.step()

            for k in train_losses:
                train_losses[k] += loss[k].item()

            pred = outputs["predicted_return"]
            dir_correct = (torch.sign(pred) == torch.sign(y_t_dev)).float().sum().item()
            train_dir += dir_correct
            train_total += len(pred)

            pbar.set_postfix(loss=f"{loss['total'].item():.3f}", reg=f"{loss['regression'].item():.3f}")

        # Validation
        model.eval()
        val_losses = {"total": 0, "regression": 0, "sigma_reg": 0, "kl": 0, "direction": 0}
        val_dir, val_total = 0, 0
        all_val_preds = []
        all_val_targets = []

        with torch.no_grad():
            for x_c, y_c, x_t, y_t, tickers_list in val_loader:
                x_c = x_c.to(device)
                y_c = y_c.to(device)
                x_t = x_t.to(device)
                y_t_dev = y_t.to(device)

                outputs = model(x_c, y_c, x_t)
                loss = criterion(outputs, {"target_return": y_t_dev})

                for k in val_losses:
                    val_losses[k] += loss[k].item()

                pred = outputs["predicted_return"]
                dir_correct = (torch.sign(pred) == torch.sign(y_t_dev)).float().sum().item()
                val_dir += dir_correct
                val_total += len(pred)

                all_val_preds.append(pred.cpu().numpy())
                all_val_targets.append(y_t_dev.cpu().numpy())

        n_train = len(train_loader)
        n_val = len(val_loader)
        dir_acc_val = val_dir / max(val_total, 1)

        val_loss = val_losses["total"] / max(n_val, 1)
        print(f"Epoch {epoch_num + 1:2d}/{epochs} | "
              f"tl={train_losses['total']/n_train:.4f} | "
              f"vl={val_loss:.4f} | "
              f"reg={val_losses['regression']/n_val:.4f} | "
              f"kl={val_losses['kl']/n_val:.4f} | "
              f"σreg={val_losses.get('sigma_reg', 0)/max(n_val,1):.4f} | "
              f"dir_acc={dir_acc_val:.1%}")

        if not no_wandb:
            wandb.log({
                "epoch": epoch_num + 1,
                "train/loss": train_losses["total"] / n_train,
                "train/regression": train_losses["regression"] / n_train,
                "train/kl": train_losses["kl"] / n_train,
                "val/loss": val_loss,
                "val/regression": val_losses["regression"] / n_val,
                "val/kl": val_losses["kl"] / n_val,
                "val/dir_acc": dir_acc_val,
                "val/pred_buy": float((all_val_preds[-1] > 0.002).mean()) if all_val_preds else 0,
                "val/pred_sell": float((all_val_preds[-1] < -0.002).mean()) if all_val_preds else 0,
            })

            # Comprehensive ANP visualization every 5 epochs
            if (epoch_num + 1) % 5 == 0:
                try:
                    from quantflow.models.anp_viz import generate_anp_viz
                    viz_img = generate_anp_viz(model, df, device,
                                                n_context=n_context, n_target=n_target,
                                                epoch=epoch_num + 1)
                    if viz_img is not None:
                        wandb.log({"viz/anp_forecast": viz_img})
                except Exception as exc:
                    print(f"  Viz generation failed: {exc}")

        # Best model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_path:
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config": {"n_features": n_features, "hidden_dim": hidden_dim,
                               "latent_dim": latent_dim, "n_context": n_context,
                               "n_target": n_target, "enc_layers": enc_layers, "heads": heads},
                    "epoch": epoch_num,
                    "val_loss": val_loss,
                }, save_path)
                print(f"  Saved best model (val_loss={val_loss:.4f})")

    print(f"\nBest val loss: {best_val_loss:.4f}")

    if not no_wandb:
        wandb.finish()

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/intraday_5m.parquet")
    parser.add_argument("--n-context", type=int, default=80)
    parser.add_argument("--n-target", type=int, default=15)
    parser.add_argument("--forecast-horizon", type=int, default=78)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--enc-layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--save", default="checkpoints/anp_model.pt")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    train_anp(
        data_path=args.data,
        n_context=args.n_context,
        n_target=args.n_target,
        forecast_horizon=args.forecast_horizon,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        enc_layers=args.enc_layers,
        heads=args.heads,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        beta=args.beta,
        save_path=args.save,
        wandb_name=args.wandb_name,
        no_wandb=args.no_wandb,
    )
