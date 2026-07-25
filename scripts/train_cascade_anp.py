#!/usr/bin/env python3
"""Train CascadeANP: MultiScale GRU cascade encoder + ANP decoder with random
context/target sampling from encoded sequences.

Training flow:
  1. Sample a ticker trajectory → encode through CascadeANP.encode()
  2. Randomly select N_c context tokens + N_t target tokens
  3. Decoder cross-attends to context, conditioned on latent z
  4. Predict returns at target positions, compute MSE + KL + direction loss
"""

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import wandb
from tqdm import tqdm

from quantflow.models.cascade_anp import CascadeANP, CascadeANPLoss


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    label_cols = {"ticker", "event_state", "event_state_code", "is_volatile",
                  "tau_forward", "target_return", "target_5d", "target_21d",
                  "target_1h", "target_4h", "target_return_path",
                  "drawdown_5d_max", "drawdown_21d_max", "as_of_date",
                  "adj_close", "close", "target_direction_5d", "target_direction_21d",
                  "sector_idx"}
    return sorted([c for c in df.columns if c not in label_cols and df[c].dtype != 'object'])


class CascadeANPDataLoader:
    """Per-epoch, randomly samples trajectories from tickers, encodes full
    sequence through the cascade, then serves random context/target tokens."""

    def __init__(self, df: pd.DataFrame, tickers: List[str],
                 n_context: int = 80, n_target: int = 21,
                 encode_len: int = 300, min_bars: int = 300,
                 episodes_per_ticker: int = 15, augment_noise: float = 0.0):
        self.tickers = tickers
        self.n_context = n_context
        self.n_target = n_target
        self.encode_len = encode_len
        self.episodes_per_ticker = episodes_per_ticker
        self.augment_noise = augment_noise
        self.feature_cols = get_feature_cols(df)
        self.n_features = len(self.feature_cols)

        self.ticker_data = {}
        for ticker in tickers:
            g = df[df["ticker"] == ticker].sort_values("as_of_date")
            if len(g) >= min_bars:
                self.ticker_data[ticker] = {
                    "feats": g[self.feature_cols].fillna(0.0).values.astype(np.float32),
                    "returns": g["target_return"].fillna(0.0).values.astype(np.float32),
                    "prices": g["adj_close"].fillna(0.0).values.astype(np.float32),
                    "dates": g["as_of_date"].values,
                }

    def __len__(self):
        return len(self.ticker_data) * self.episodes_per_ticker

    def sample_batch(self, batch_size: int, device: str, model: CascadeANP) -> Tuple:
        tickers = list(self.ticker_data.keys())
        bs = min(batch_size, len(tickers))
        chosen = np.random.choice(tickers, bs, replace=True)

        all_x_c, all_y_c, all_x_t, all_y_t, all_target_paths = [], [], [], [], []
        all_raw_ctx, all_raw_feats = [], []

        for ticker in chosen:
            data = self.ticker_data[ticker]
            feats = data["feats"]
            returns = data["returns"]
            T = len(feats)

            # Take a random sub-window of encode_len bars
            max_start = max(0, T - self.encode_len - self.n_target)
            if max_start == 0:
                start = 0
            else:
                start = np.random.randint(0, max_start)
            end = min(start + self.encode_len, T)

            window_feats = torch.from_numpy(feats[start:end]).unsqueeze(0).to(device)
            window_rets = returns[start:end]

            # Data augmentation: add small Gaussian noise to features
            if self.augment_noise > 0:
                window_feats = window_feats + torch.randn_like(window_feats) * self.augment_noise

            # Encode through cascade — returns (fused, raw_bilstm, scale_acts)
            fused, raw_bilstm, scale_acts = model.encode(window_feats)
            fused = fused[0].detach()  # [L, H]
            raw_context = raw_bilstm[0].detach()  # [L, H] — BiLSTM before cascade smoothing

            L = fused.shape[0]
            if L < self.n_context + self.n_target:
                continue

            # Randomly sample context and target indices (fixed counts)
            max_ctx = L - self.n_target
            n_c = min(self.n_context, max_ctx)
            ctx_indices = np.sort(np.random.choice(max_ctx, n_c, replace=True))
            ctx_indices = np.unique(ctx_indices)
            if len(ctx_indices) < n_c:
                # Pad with last valid index
                pad_len = n_c - len(ctx_indices)
                ctx_indices = np.concatenate([ctx_indices, np.full(pad_len, ctx_indices[-1])])
            ctx_indices = ctx_indices[:n_c]

            tgt_start = max(ctx_indices[-1] + 1, L - self.n_target)
            n_t = min(self.n_target, L - tgt_start)
            tgt_indices = np.arange(tgt_start, min(tgt_start + n_t, L))
            if len(tgt_indices) < n_t:
                pad_len = n_t - len(tgt_indices)
                tgt_indices = np.concatenate([tgt_indices, np.full(pad_len, L - 1)])
            tgt_indices = tgt_indices[:n_t]

            # Raw features for conv encoder: STRICTLY causal window ending at
            # tgt_start (exclusive of target/future bars). Previously this
            # passed the *entire* window_feats (400 bars including the target
            # itself), which both leaked the answer and, worse, caused
            # AdaptiveAvgPool1d to average over the whole window's macro drift
            # instead of the local pre-forecast signal — collapsing every
            # snapshot toward the same population-average trend.
            x_c = fused[ctx_indices].unsqueeze(0)
            causal_start = max(0, tgt_start - self.n_context)
            raw_feat_window = torch.from_numpy(
                feats[start + causal_start:start + tgt_start]
            ).unsqueeze(0).to(device)  # [1, <=n_context, F], never touches target bars
            y_c = torch.from_numpy(window_rets[ctx_indices, None]).unsqueeze(0).to(device)
            x_t = fused[tgt_indices].unsqueeze(0)

            # Target return path: actual forward returns at each target step
            mid_idx_c = ctx_indices[-1]
            ref_price = data["prices"][start + mid_idx_c]
            target_path = np.zeros(n_t, dtype=np.float32)
            for ti in range(n_t):
                price_at_target = data["prices"][start + tgt_indices[ti]]
                if ref_price > 0:
                    target_path[ti] = (price_at_target / ref_price) - 1.0

            # Scalar target for backward compats
            mid_idx = tgt_indices[n_t // 2] if n_t > 0 else tgt_indices[0]
            y_t = torch.tensor(window_rets[mid_idx], device=device).float()

            all_x_c.append(x_c)
            all_y_c.append(y_c)
            all_x_t.append(x_t)
            all_y_t.append(y_t)
            all_target_paths.append(torch.from_numpy(target_path))
            # Spread-sample raw BiLSTM context for skip connection (30 tokens)
            raw_L = raw_context.shape[0]
            raw_stride = max(1, raw_L // 30)
            raw_indices = np.sort(np.arange(0, raw_L, raw_stride)[:30])
            all_raw_ctx.append(raw_context[raw_indices].unsqueeze(0))
            # Pad causal raw feature window to fixed n_context length (batching requires equal shapes)
            actual_len = raw_feat_window.shape[1]
            if actual_len < self.n_context:
                pad = raw_feat_window[:, :1, :].expand(-1, self.n_context - actual_len, -1)
                raw_feat_window = torch.cat([pad, raw_feat_window], dim=1)
            all_raw_feats.append(raw_feat_window)

        if not all_x_c:
            return None

        # Stack to batch — ensure all same size
        if len(all_x_c) < 2:
            return None
        x_c_batch = torch.cat(all_x_c, dim=0)
        y_c_batch = torch.cat(all_y_c, dim=0)
        x_t_batch = torch.cat(all_x_t, dim=0)
        y_t_batch = torch.stack(all_y_t)
        target_path_batch = torch.stack([t.to(device) for t in all_target_paths])
        raw_ctx_batch = torch.cat([r.to(device) for r in all_raw_ctx], dim=0)
        raw_feats_batch = torch.cat(all_raw_feats, dim=0)

        return x_c_batch, y_c_batch, x_t_batch, y_t_batch, target_path_batch, raw_ctx_batch, raw_feats_batch


def train_cascade_anp(
    data_path: str,
    n_context: int = 80,
    n_target: int = 21,
    encode_len: int = 300,
    hidden_dim: int = 128,
    latent_dim: int = 64,
    cascade_layers: int = 4,
    heads: int = 4,
    dropout: float = 0.1,
    augment_noise: float = 0.001,
    beta: float = 0.2,
    batch_size: int = 16,
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

    all_tickers = sorted(df["ticker"].unique())
    np.random.seed(42)
    np.random.shuffle(all_tickers)
    n_train = int(len(all_tickers) * 0.80)
    n_val = int(len(all_tickers) * 0.10)

    train_tickers = all_tickers[:n_train]
    val_tickers = all_tickers[n_train:n_train + n_val]
    print(f"Split: train={len(train_tickers)} val={len(val_tickers)} tickers")

    n_features = len(get_feature_cols(df))
    train_loader = CascadeANPDataLoader(df, train_tickers, n_context, n_target, encode_len,
                                         augment_noise=augment_noise)
    val_loader = CascadeANPDataLoader(df, val_tickers, n_context, n_target, encode_len,
                                       episodes_per_ticker=5)

    model = CascadeANP(n_features, hidden_dim, latent_dim, cascade_layers, heads, dropout).to(device)
    criterion = CascadeANPLoss(beta=beta)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * 100, eta_min=1e-5)

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Features: {n_features}, context: {n_context}, target: {n_target}, encode_len: {encode_len}")

    if not no_wandb:
        name = wandb_name or f"cascade-anp-h{hidden_dim}-c{n_context}"
        wandb.init(project=wandb_project, name=name, config={
            "model": "CascadeANP",
            "hidden_dim": hidden_dim, "latent_dim": latent_dim,
            "n_context": n_context, "n_target": n_target,
            "encode_len": encode_len, "epochs": epochs, "lr": lr, "beta": beta,
        }, settings=wandb.Settings(init_timeout=180))

    best_val_loss = float("inf")
    steps_per_epoch = 50
    mae_history = {}  # epoch -> [mae_step_0, ..., mae_step_N]

    for epoch_num in range(epochs):
        model.train()
        train_losses = {"total": 0, "temporal_mae": 0, "path_loss": 0, "kl": 0, "direction": 0}
        train_dir, train_total = 0, 0
        direction_near, direction_far = 0, 0
        n_near, n_far = 0, 0
        step_errors = torch.zeros(n_target, device=device)
        step_counts = torch.zeros(n_target, device=device)

        for step in range(steps_per_epoch):
            batch = train_loader.sample_batch(batch_size, device, model)
            if batch is None:
                continue
            x_c, y_c, x_t, y_t_dev, target_path, raw_ctx, raw_feats = batch

            optimizer.zero_grad()
            outputs = model(x_c, y_c, n_steps=n_target, teacher_forcing=True,
                           target_returns=target_path,
                           raw_features=raw_feats)
            loss = criterion(outputs, {"target_return": y_t_dev,
                                        "target_return_path": target_path})
            loss["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            scheduler.step()

            for k in train_losses:
                train_losses[k] += loss[k].item()

            # Per-step error tracking
            mu_y = outputs["mu_y"]
            B, S = mu_y.shape
            abs_err_step = torch.abs(mu_y - target_path)  # [B, S]
            for s in range(S):
                step_errors[s] += abs_err_step[:, s].sum()
                step_counts[s] += B

            # Near-term (t=1-5) vs far-term (t=16-21) direction
            near_start, near_end = 0, min(5, S)
            far_start, far_end = max(0, S - 6), S
            if near_end > near_start:
                pred_near = mu_y[:, near_start:near_end].mean(dim=-1)
                true_near = target_path[:, near_start:near_end].mean(dim=-1)
                direction_near += (torch.sign(pred_near) == torch.sign(true_near)).float().sum().item()
                n_near += len(pred_near)
            if far_end > far_start:
                pred_far = mu_y[:, far_start:far_end].mean(dim=-1)
                true_far = target_path[:, far_start:far_end].mean(dim=-1)
                direction_far += (torch.sign(pred_far) == torch.sign(true_far)).float().sum().item()
                n_far += len(pred_far)

            pred = outputs["predicted_return"]
            dir_correct = (torch.sign(pred) == torch.sign(y_t_dev)).float().sum().item()
            train_dir += dir_correct
            train_total += len(pred)

            # Batch-level logging every 10 steps
            if not no_wandb and step % 10 == 0:
                mae_per_step = (step_errors / step_counts.clamp(min=1)).tolist()
                wandb.log({
                    f"batch/loss": loss["total"].item(),
                    f"batch/temporal_mae": loss["temporal_mae"].item(),
                    f"batch/mae_step_0": mae_per_step[0],
                    f"batch/mae_step_mid": mae_per_step[min(10, S - 1)],
                    f"batch/mae_step_last": mae_per_step[-1],
                    f"batch/sigma_mean": outputs["sigma_y"].mean().item(),
                    f"batch/kl": loss["kl"].item(),
                }, commit=False)

        # Per-step RMSE (averaged over epoch)
        mae_per_step = (step_errors / step_counts.clamp(min=1)).cpu().tolist() if step_counts.sum() > 0 else [0] * n_target
        mae_history[epoch_num + 1] = mae_per_step  # store for evolution chart

        # Validation
        model.eval()
        val_losses = {"total": 0, "temporal_mae": 0, "path_loss": 0, "kl": 0, "direction": 0}
        val_dir, val_total = 0, 0
        val_dir_near, val_dir_far, val_n_near, val_n_far = 0, 0, 0, 0

        with torch.no_grad():
            for step in range(20):
                batch = val_loader.sample_batch(batch_size, device, model)
                if batch is None:
                    continue
                x_c, y_c, x_t, y_t_dev, target_path, raw_ctx, raw_feats = batch
                outputs = model(x_c, y_c, n_steps=n_target, teacher_forcing=False,
                               raw_features=raw_feats)
                loss = criterion(outputs, {"target_return": y_t_dev,
                                            "target_return_path": target_path})

                for k in val_losses:
                    val_losses[k] += loss[k].item()

                pred = outputs["predicted_return"]
                dir_correct = (torch.sign(pred) == torch.sign(y_t_dev)).float().sum().item()
                val_dir += dir_correct
                val_total += len(pred)

                mu_y = outputs["mu_y"]
                S = mu_y.shape[1]
                near_s, near_e = 0, min(5, S)
                far_s, far_e = max(0, S - 6), S
                if near_e > near_s:
                    pn = mu_y[:, near_s:near_e].mean(dim=-1)
                    tn = target_path[:, near_s:near_e].mean(dim=-1)
                    val_dir_near += (torch.sign(pn) == torch.sign(tn)).float().sum().item()
                    val_n_near += len(pn)
                if far_e > far_s:
                    pf = mu_y[:, far_s:far_e].mean(dim=-1)
                    tf = target_path[:, far_s:far_e].mean(dim=-1)
                    val_dir_far += (torch.sign(pf) == torch.sign(tf)).float().sum().item()
                    val_n_far += len(pf)

        n_s = steps_per_epoch; n_v = 20
        dir_acc_val = val_dir / max(val_total, 1)
        dir_near_val = val_dir_near / max(val_n_near, 1)
        dir_far_val = val_dir_far / max(val_n_far, 1)
        train_dir_acc = train_dir / max(train_total, 1)

        print(f"Epoch {epoch_num + 1:2d}/{epochs} | "
              f"tl={train_losses['total']/n_s:.4f} vl={val_losses['total']/n_v:.4f} | "
              f"mae={val_losses['temporal_mae']/n_v:.4f} | "
              f"dir={dir_acc_val:.1%} near={dir_near_val:.1%} far={dir_far_val:.1%} | "
              f"step_mae[0]={mae_per_step[0]:.3f} [{len(mae_per_step)//2}]={mae_per_step[len(mae_per_step)//2]:.3f} [-1]={mae_per_step[-1]:.3f}")

        val_loss = val_losses["total"] / n_v

        if not no_wandb:
            epoch_log = {
                "epoch": epoch_num + 1,
                "train/loss": train_losses["total"] / n_s,
                "train/mae": train_losses["temporal_mae"] / n_s,
                "train/dir_acc": train_dir_acc,
                "train/dir_near": direction_near / max(n_near, 1),
                "train/dir_far": direction_far / max(n_far, 1),
                "val/loss": val_loss,
                "val/mae": val_losses["temporal_mae"] / n_v,
                "val/dir_acc": dir_acc_val,
                "val/dir_near": dir_near_val,
                "val/dir_far": dir_far_val,
                "val/sigma_mean": outputs["sigma_y"].mean().item() if "sigma_y" in outputs else 0,
                "val/mae_step_0": mae_per_step[0],
                "val/mae_step_mid": mae_per_step[len(mae_per_step) // 2],
                "val/mae_step_last": mae_per_step[-1],
            }
            # Add per-step MAE as a wandb table
            step_table = wandb.Table(data=[[i, m] for i, m in enumerate(mae_per_step)],
                                     columns=["step", "mae"])
            epoch_log["val/mae_by_step"] = wandb.plot.line(step_table, "step", "mae",
                                                            title="MAE per forecast step")

            # MAE evolution chart (all epochs so far)
            if len(mae_history) >= 2:
                try:
                    from quantflow.models.mae_evolution_viz import build_mae_evolution_figure
                    mae_evol = build_mae_evolution_figure(mae_history, n_target)
                    if mae_evol is not None:
                        epoch_log["viz/mae_evolution"] = mae_evol
                except Exception as exc:
                    print(f"  MAE evol viz failed: {exc}")

            # Price forecast viz every epoch
            try:
                from quantflow.models.cascade_anp_viz import generate_cascade_anp_viz
                viz_img = generate_cascade_anp_viz(
                    model, df, device, train_loader.feature_cols,
                    n_context=n_context, n_target=n_target,
                    epoch=epoch_num + 1,
                )
                if viz_img is not None:
                    epoch_log["viz/cascade_forecast"] = viz_img
                else:
                    print(f"  Viz returned None (no candidates)")
            except Exception as exc:
                import traceback
                print(f"  CascadeANP viz failed: {exc}")
                print(f"  Traceback: {traceback.format_exc()[:200]}")

            wandb.log(epoch_log)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_path:
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config": {"n_features": n_features, "hidden_dim": hidden_dim,
                               "latent_dim": latent_dim, "cascade_layers": cascade_layers,
                               "n_context": n_context, "n_target": n_target},
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
    parser.add_argument("--n-target", type=int, default=21)
    parser.add_argument("--encode-len", type=int, default=300)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--cascade-layers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--augment-noise", type=float, default=0.001)
    parser.add_argument("--save", default="checkpoints/cascade_anp.pt")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    train_cascade_anp(
        data_path=args.data,
        n_context=args.n_context, n_target=args.n_target,
        encode_len=args.encode_len,
        hidden_dim=args.hidden_dim, latent_dim=args.latent_dim,
        cascade_layers=args.cascade_layers,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        beta=args.beta, save_path=args.save,
        augment_noise=args.augment_noise,
        wandb_name=args.wandb_name, no_wandb=args.no_wandb,
    )
