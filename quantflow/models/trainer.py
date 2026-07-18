from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from .architectures import TemporalStateModel, create_model
from .losses import CompositeLoss, LossConfig
from .dataset import prepare_dataloaders

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 2.0
    gradient_accumulation_steps: int = 1
    gradient_noise_eta: float = 0.005

    lr_scheduler: str = "cosine"
    warmup_epochs: int = 0
    min_lr: float = 1e-5

    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 0.001

    use_amp: bool = False
    stateful: bool = False
    keep_hidden_across_epochs: bool = True

    save_best_only: bool = True
    save_frequency_epochs: int = 5
    checkpoint_dir: str = "checkpoints"

    lookback: int = 60
    forecast_horizon: int = 21

    architecture: str = "bilstm"
    hidden_dim: int = 128
    dropout: float = 0.2
    use_multi_scale: bool = True
    use_coherent_heads: bool = True


class Trainer:
    """Training engine for the Temporal State Model per methodology §8."""

    def __init__(
        self,
        model: TemporalStateModel,
        config: Optional[TrainingConfig] = None,
        loss_config: Optional[LossConfig] = None,
        device: Optional[str] = None,
    ):
        self.config = config or TrainingConfig()
        self.loss_config = loss_config or LossConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = CompositeLoss(self.loss_config).to(self.device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.scaler = torch.amp.GradScaler("cuda") if self.config.use_amp and self.device == "cuda" else None

        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.train_history: List[Dict[str, float]] = []
        self.val_history: List[Dict[str, float]] = []

        checkpoint_path = Path(self.config.checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

    def _create_scheduler(self, steps_per_epoch: int) -> torch.optim.lr_scheduler.LRScheduler:
        total_steps = self.config.epochs * steps_per_epoch

        if self.config.warmup_epochs > 0:
            warmup_steps = self.config.warmup_epochs * steps_per_epoch
            warmup = LinearLR(self.optimizer, start_factor=0.1, total_iters=warmup_steps)
            cosine = CosineAnnealingLR(self.optimizer, T_max=total_steps - warmup_steps, eta_min=self.config.min_lr)
            return SequentialLR(self.optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

        return CosineAnnealingLR(self.optimizer, T_max=total_steps, eta_min=self.config.min_lr)

    def _to_device(self, batch: Tuple[torch.Tensor, Dict[str, torch.Tensor], str]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        features, labels, _ = batch
        features = features.to(self.device)
        labels = {k: v.to(self.device) for k, v in labels.items()}
        return features, labels

    def _step(self, batch: Tuple[torch.Tensor, Dict[str, torch.Tensor]]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        self.optimizer.zero_grad()

        features, labels = self._to_device(batch)
        outputs = self.model(features)
        losses = self.criterion(outputs, labels)

        if self.scaler:
            self.scaler.scale(losses["total"]).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)

            if self.config.gradient_noise_eta > 0 and self.model.training:
                with torch.no_grad():
                    for p in self.model.parameters():
                        if p.grad is not None:
                            noise = torch.randn_like(p.grad) * self.config.gradient_noise_eta * p.grad.norm(2).clamp(max=1.0)
                            p.grad.add_(noise)

            self.optimizer.step()

        return outputs, losses, labels

    def _evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_losses: Dict[str, float] = {}
        total_correct = 0
        total_samples = 0
        class_correct = {0: 0, 1: 0, 2: 0}
        class_count = {0: 0, 1: 0, 2: 0}
        forecast_errors = []
        forecast_errors_weighted = []
        forecast_direction_correct = 0
        forecast_direction_total = 0

        with torch.no_grad():
            for batch in loader:
                features, labels = self._to_device(batch)
                outputs = self.model(features)
                losses = self.criterion(outputs, labels)

                for k, v in losses.items():
                    total_losses[k] = total_losses.get(k, 0.0) + v.item()

                preds = outputs["past_state_logits"].argmax(dim=-1)
                labels_dev = labels["event_state_code"]
                total_correct += (preds == labels_dev).sum().item()
                total_samples += labels_dev.size(0)
                for c in range(3):
                    mask = labels_dev == c
                    class_count[c] += mask.sum().item()
                    class_correct[c] += (preds[mask] == c).sum().item()

                fcast = outputs["future_forecast"].squeeze(-1)
                if fcast.dim() > 1:
                    fcast = fcast.mean(dim=-1)
                tgt_tau = labels["tau_forward"]
                tau_clamped = tgt_tau.clamp(1.0, 21.0)
                abs_err = (fcast - tau_clamped).abs()
                forecast_errors.append(float(abs_err.mean()))

                temporal_weight = torch.exp(-tau_clamped / 5.0)
                w_err = (abs_err * temporal_weight).mean()
                forecast_errors_weighted.append(float(w_err))

                forecast_direction_correct += ((fcast.sign() == tau_clamped.sign()) | (fcast > 0)).sum().item()
                forecast_direction_total += fcast.size(0)

        avg_losses = {k: v / max(len(loader), 1) for k, v in total_losses.items()}
        avg_losses["accuracy"] = total_correct / max(total_samples, 1)
        avg_losses["accuracy_inter"] = class_correct[0] / max(class_count[0], 1)
        avg_losses["accuracy_pre"] = class_correct[1] / max(class_count[1], 1)
        avg_losses["accuracy_onset"] = class_correct[2] / max(class_count[2], 1)
        avg_losses["forecast_mae"] = float(np.mean(forecast_errors)) if forecast_errors else 0.0
        avg_losses["forecast_mae_temporal"] = float(np.mean(forecast_errors_weighted)) if forecast_errors_weighted else 0.0
        avg_losses["forecast_dir_acc"] = forecast_direction_correct / max(forecast_direction_total, 1)
        return avg_losses

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:
        scheduler = self._create_scheduler(len(train_loader))
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Starting training: {self.config.epochs} epochs, device={self.device}")
        logger.info(f"Model: {total_params:,} parameters | LR warmup: {self.config.warmup_epochs} epochs")

        try:
            import wandb
            _wandb_available = True
        except ImportError:
            _wandb_available = False

        for epoch in range(self.current_epoch, self.config.epochs):
            self.model.train()
            epoch_start = time.time()
            epoch_losses: Dict[str, float] = {}
            epoch_correct = 0
            epoch_samples = 0
            epoch_class_correct = {0: 0, 1: 0, 2: 0}
            epoch_class_count = {0: 0, 1: 0, 2: 0}
            batch_losses = []
            batch_grad_norms = []

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.config.epochs}", leave=False)

            accumulation_counter = 0
            for batch_idx, batch in enumerate(pbar):
                outputs, losses_dict, labels = self._step(batch)
                accumulation_counter += 1

                batch_loss = losses_dict["total"].item()
                batch_losses.append(batch_loss)

                grad_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.data.norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5
                batch_grad_norms.append(grad_norm)

                for k, v in losses_dict.items():
                    epoch_losses[k] = epoch_losses.get(k, 0.0) + v.item()

                preds = outputs["past_state_logits"].argmax(dim=-1)
                labels_dev = labels["event_state_code"]
                epoch_correct += (preds == labels_dev).sum().item()
                epoch_samples += labels_dev.size(0)
                for c in range(3):
                    mask = labels_dev == c
                    epoch_class_count[c] += mask.sum().item()
                    epoch_class_correct[c] += (preds[mask] == c).sum().item()

                current_lr = self.optimizer.param_groups[0]["lr"]
                pbar.set_postfix(loss=f"{batch_loss:.2f}", lr=f"{current_lr:.2e}")

                global_step = epoch * len(train_loader) + batch_idx
                if _wandb_available and wandb.run and batch_idx % 10 == 0:
                    wandb.log({
                        "batch/loss": batch_loss,
                        "batch/grad_norm": grad_norm,
                        "batch/lr": current_lr,
                    }, step=global_step, commit=False)

                scheduler.step()

            n_batches = max(len(train_loader), 1)
            train_metrics = {k: v / n_batches for k, v in epoch_losses.items()}
            train_metrics["accuracy"] = epoch_correct / max(epoch_samples, 1)
            train_metrics["accuracy_inter"] = epoch_class_correct[0] / max(epoch_class_count[0], 1)
            train_metrics["accuracy_pre"] = epoch_class_correct[1] / max(epoch_class_count[1], 1)
            train_metrics["accuracy_onset"] = epoch_class_correct[2] / max(epoch_class_count[2], 1)
            train_metrics["lr"] = self.optimizer.param_groups[0]["lr"]
            train_metrics["epoch"] = epoch + 1
            train_metrics["grad_norm_mean"] = float(np.mean(batch_grad_norms)) if batch_grad_norms else 0.0
            self.train_history.append(train_metrics)

            val_metrics = self._evaluate(val_loader)
            val_metrics["epoch"] = epoch + 1
            self.val_history.append(val_metrics)

            elapsed = time.time() - epoch_start
            logger.info(
                f"Epoch {epoch + 1}/{self.config.epochs} ({elapsed:.1f}s) | "
                f"tl={train_metrics.get('total', 0):.3f} | "
                f"vl={val_metrics.get('total', 0):.3f} | "
                f"acc={val_metrics.get('accuracy', 0):.3f} | "
                f"i={train_metrics.get('accuracy_inter', 0):.2f} "
                f"p={train_metrics.get('accuracy_pre', 0):.2f} "
                f"o={train_metrics.get('accuracy_onset', 0):.2f} | "
                f"fcst_mae={val_metrics.get('forecast_mae', 0):.1f}d | "
                f"grad={train_metrics['grad_norm_mean']:.3f}"
            )

            if _wandb_available and wandb.run:
                epoch_last_step = epoch * len(train_loader) + max(batch_idx, 0)
                wandb.log({
                    "epoch": epoch + 1,
                    "train/loss": train_metrics.get("total", 0),
                    "train/accuracy": train_metrics.get("accuracy", 0),
                    "train/accuracy_inter": train_metrics.get("accuracy_inter", 0),
                    "train/accuracy_pre": train_metrics.get("accuracy_pre", 0),
                    "train/accuracy_onset": train_metrics.get("accuracy_onset", 0),
                    "train/lr": train_metrics.get("lr", 0),
                    "train/grad_norm": train_metrics["grad_norm_mean"],
                    "val/loss": val_metrics.get("total", 0),
                    "val/accuracy": val_metrics.get("accuracy", 0),
                    "val/accuracy_inter": val_metrics.get("accuracy_inter", 0),
                    "val/accuracy_pre": val_metrics.get("accuracy_pre", 0),
                    "val/accuracy_onset": val_metrics.get("accuracy_onset", 0),
                    "val/forecast_mae": val_metrics.get("forecast_mae", 0),
                    "val/forecast_mae_temporal": val_metrics.get("forecast_mae_temporal", 0),
                    "val/forecast_dir_acc": val_metrics.get("forecast_dir_acc", 0),
                }, step=epoch_last_step, commit=True)
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    viz_img = self._generate_epoch_inference_viz(val_loader, epoch + 1)
                    if viz_img is not None:
                        wandb.log({"viz/epoch_forecast_overview": viz_img}, step=epoch_last_step, commit=False)

            val_total = val_metrics.get("total", float("inf"))
            if val_total < self.best_val_loss - self.config.early_stopping_min_delta:
                self.best_val_loss = val_total
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_metrics, is_best=True)
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

            if not self.config.save_best_only and (epoch + 1) % self.config.save_frequency_epochs == 0:
                self._save_checkpoint(epoch, val_metrics, is_best=False)

        if _wandb_available and wandb.run:
            wandb.run.summary["best_val_loss"] = self.best_val_loss
            wandb.run.summary["final_train_accuracy"] = self.train_history[-1].get("accuracy", 0) if self.train_history else 0
            wandb.run.summary["final_val_accuracy"] = self.val_history[-1].get("accuracy", 0) if self.val_history else 0
            wandb.run.summary["epochs_run"] = len(self.train_history)
            wandb.run.summary["model_params"] = total_params

        return {"train_history": self.train_history, "val_history": self.val_history}

    def _generate_epoch_inference_viz(self, loader: DataLoader, epoch: int):
        """Generate epoch-level inference visualization for wandb.

        Produces a 4-panel figure:
        1. State probability ribbon with buy/sell/hold overlay
        2. Forecast tau vs actual forward return error
        3. Latent token activation heatmap (scale S0→S3)
        4. Price with drawdown-based trend regions
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import io

            self.model.eval()
            all_probs, all_forecasts, all_sigmas, all_targets = [], [], [], []
            all_scale_acts = [[] for _ in range(4)]
            max_samples = 300

            with torch.no_grad():
                for batch in loader:
                    if len(all_probs) > 0 and sum(arr.shape[0] for arr in all_probs) >= max_samples:
                        break
                    features, labels = self._to_device(batch)
                    outputs = self.model(features)

                    probs = outputs["past_state_probs"].cpu().numpy()
                    fcast = outputs["future_forecast"].squeeze(-1).cpu().numpy()
                    if fcast.ndim > 1:
                        fcast = fcast.mean(axis=-1)
                    sigma = outputs["uncertainty_sigma"].squeeze(-1).cpu().numpy()
                    if sigma.ndim > 1:
                        sigma = sigma.mean(axis=-1)
                    tau = labels["tau_forward"].cpu().numpy()

                    all_probs.append(probs)
                    all_forecasts.append(fcast)
                    all_sigmas.append(sigma)
                    all_targets.append(tau)

                    skips = outputs.get("scale_activations", [])
                    for s in range(min(4, len(skips))):
                        act = skips[s].cpu().numpy()
                        all_scale_acts[s].append(act.mean(axis=1))

            if not all_probs:
                return None

            try:
                probs = np.concatenate([p[:max_samples]] if len(all_probs) == 1 else all_probs, axis=0)[:max_samples]
                forecasts = np.concatenate([f[:max_samples]] if len(all_forecasts) == 1 else all_forecasts, axis=0)[:max_samples]
                sigmas = np.concatenate([s[:max_samples]] if len(all_sigmas) == 1 else all_sigmas, axis=0)[:max_samples]
                targets = np.concatenate([t[:max_samples]] if len(all_targets) == 1 else all_targets, axis=0)[:max_samples] if all_targets else np.zeros(probs.shape[0])
            except ValueError:
                bsz = min(a.shape[0] for a in all_probs)
                probs = np.concatenate([a[:bsz] for a in all_probs], axis=0)[:max_samples]
                forecasts = np.concatenate([a[:bsz] for a in all_forecasts], axis=0)[:max_samples]
                sigmas = np.concatenate([a[:bsz] for a in all_sigmas], axis=0)[:max_samples]
                targets = np.concatenate([a[:bsz] for a in all_targets], axis=0)[:max_samples] if all_targets else np.zeros(probs.shape[0])

            for s in range(4):
                if all_scale_acts[s]:
                    try:
                        all_scale_acts[s] = np.concatenate(all_scale_acts[s], axis=0)[:probs.shape[0]]
                    except ValueError:
                        bsz = min(a.shape[0] for a in all_scale_acts[s])
                        all_scale_acts[s] = np.concatenate([a[:bsz] for a in all_scale_acts[s]], axis=0)[:probs.shape[0]]

            fig, axes = plt.subplots(4, 1, figsize=(18, 16), dpi=100)

            # Panel 1: State probability ribbon + buy/sell regions
            ax1 = axes[0]
            x = np.arange(len(probs))
            ax1.fill_between(x, 0, probs[:, 0], alpha=0.3, color="#27ae60", label="Inter")
            ax1.fill_between(x, probs[:, 0], probs[:, 0] + probs[:, 1], alpha=0.3, color="#f39c12", label="Pre")
            ax1.fill_between(x, probs[:, 0] + probs[:, 1], 1.0, alpha=0.3, color="#e74c3c", label="Onset")

            risk = probs[:, 1] + probs[:, 2] * 1.5
            ax1.plot(x, risk, color="#1abc9c", linewidth=1.0, alpha=0.7, label="Risk Score")
            buy_mask = risk >= 0.60
            sell_mask = probs[:, 0] >= 0.40
            for i in range(len(x)):
                if buy_mask[i]:
                    ax1.axvspan(i - 0.5, i + 0.5, alpha=0.12, color="#27ae60")
            ax1.set_ylim(0, 1.05)
            ax1.set_ylabel("Probability"); ax1.set_title(f"Epoch {epoch} — Head Probabilities & Signal Regions")
            ax1.legend(loc="upper left", ncol=4, fontsize=7)

            # Panel 2: Forecast tau with error vs actual
            ax2 = axes[1]
            ax2.plot(x, forecasts, color="#9b59b6", linewidth=1.5, label="Forecast tau")
            ax2.fill_between(x, np.maximum(0, forecasts - 2 * sigmas), forecasts + 2 * sigmas,
                             alpha=0.12, color="#9b59b6")
            ax2.plot(x, targets, color="#e67e22", linewidth=1.0, alpha=0.6, label="Target tau")
            ax2.axhline(21, color="gray", ls="--", alpha=0.4)
            ax2.axhline(5, color="#e74c3c", ls="--", alpha=0.4)
            mae = np.abs(forecasts - targets).mean()
            ax2.text(0.02, 0.95, f"MAE: {mae:.1f}d", transform=ax2.transAxes, fontsize=7,
                     va="top", bbox=dict(boxstyle="round", fc="white", alpha=0.8))
            ax2.set_ylabel("Days"); ax2.set_title("Forecast Tau vs Target with ±2σ")
            ax2.legend(loc="upper right", fontsize=7)

            # Panel 3: Latent token activations
            ax3 = axes[2]
            has_scale_acts = any(len(a) > 0 for a in all_scale_acts)
            if has_scale_acts:
                import matplotlib.gridspec as gs_inner
                inner_gs = gs_inner.GridSpecFromSubplotSpec(4, 1, subplot_spec=ax3.get_subplotspec(),
                                                            hspace=0.05)
                inner_axes = [fig.add_subplot(inner_gs[i]) for i in range(4)]
                for s in range(4):
                    ax_inner = inner_axes[s]
                    if s < len(all_scale_acts) and len(all_scale_acts[s]) > 0:
                        act = all_scale_acts[s]
                        n_dims = min(8, act.shape[0])
                        im = ax_inner.imshow(act[:n_dims], aspect="auto", cmap="RdYlBu_r", interpolation="nearest")
                        ax_inner.set_ylabel(f"S{s}", fontsize=8, rotation=0, labelpad=10)
                        ax_inner.tick_params(labelsize=6)
                        ax_inner.set_yticklabels([])
                    else:
                        ax_inner.text(0.5, 0.5, f"S{s}: no data", ha="center", va="center",
                                     transform=ax_inner.transAxes, fontsize=7)
                    if s < 3:
                        ax_inner.set_xticklabels([])
                ax3.set_title("Latent Token Activations (4-scale cascade, top-8 dims)", fontsize=10)
            else:
                ax3.text(0.5, 0.5, "No scale activations available (TCN/Transformer pipeline)",
                         ha="center", va="center", transform=ax3.transAxes, fontsize=10)

            # Panel 4: Drawdown-based trend regions (cumulative forward returns)
            ax4 = axes[3]
            if len(targets) > 0:
                target_returns = np.clip(targets / 252.0, -0.05, 0.05)
                price_from_returns = 100.0 * np.cumprod(1 + np.concatenate([[0], target_returns[:-1]]))
                peak = np.maximum.accumulate(price_from_returns)
                dd = (price_from_returns - peak) / peak * 100
                ax4.plot(x, price_from_returns, color="#2ecc71", linewidth=1.5, label="Forward Return Index")

                dd_threshold = 3.0
                in_drawdown = dd < -dd_threshold

                exit_drawdown_signal = np.zeros(len(dd), dtype=bool)
                for i in range(1, len(dd)):
                    if dd[i] >= -dd_threshold * 0.5 and in_drawdown[i - 1]:
                        exit_drawdown_signal[i] = True

                for i in range(len(x)):
                    if in_drawdown[i]:
                        ax4.axvspan(i - 0.5, i + 0.5, alpha=0.10, color="#e74c3c")
                    elif i > 0 and not in_drawdown[i]:
                        ax4.axvspan(i - 0.5, i + 0.5, alpha=0.06, color="#27ae60")

                ax4_twin = ax4.twinx()
                ax4_twin.fill_between(x, 0, dd, alpha=0.15, color="#e74c3c")
                ax4_twin.axhline(y=-dd_threshold, color="#e74c3c", ls="--", lw=0.5, alpha=0.5)
                ax4_twin.set_ylim(-30, 5)
                ax4_twin.set_ylabel("Drawdown %", fontsize=8, color="#e74c3c")
                ax4_twin.tick_params(colors="#e74c3c", labelsize=7)

                buy_confirms = buy_mask & ~in_drawdown
                buy_confirmed_count = int(buy_confirms.sum())
                ax4.set_title(f"Forward Return Index & Drawdown (≥{dd_threshold}%% dd = red, buy={buy_confirmed_count}/{int(buy_mask.sum())} confirmed)", fontsize=10)
                ax4.legend(loc="upper left", fontsize=7)
                ax4.set_ylabel("Return Index")

            for ax in axes:
                ax.grid(True, alpha=0.15, linestyle="--")
                ax.tick_params(labelsize=7)
            fig.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white", dpi=80)
            plt.close(fig)
            buf.seek(0)
            import wandb
            from PIL import Image
            img = Image.open(buf)
            return wandb.Image(img, caption=f"Epoch {epoch} Forecast")
        except Exception as exc:
            logger.warning(f"Epoch inference viz failed: {exc}")
            return None

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
            "metrics": metrics,
        }
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if is_best:
            path = checkpoint_dir / "best_model.pt"
        else:
            path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
        torch.save(checkpoint, path)

    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        logger.info(f"Model loaded from {path}")

    def predict(self, loader: DataLoader) -> pd.DataFrame:
        self.model.eval()
        all_outputs = []
        with torch.no_grad():
            for batch in loader:
                features, labels = self._to_device(batch)
                outputs = self.model(features)

                probs = outputs["past_state_probs"].cpu().numpy()
                forecast = outputs["future_forecast"].cpu().numpy()
                sigma = outputs["uncertainty_sigma"].cpu().numpy()
                _, _, tickers = batch
                _, labels_cpu, _ = batch

                for i in range(len(tickers)):
                    all_outputs.append({
                        "ticker": tickers[i],
                        "event_state_pred": int(np.argmax(probs[i])),
                        "prob_inter_event": float(probs[i][0]),
                        "prob_pre_event": float(probs[i][1]),
                        "prob_onset": float(probs[i][2]),
                        "forecast_tau": float(forecast[i].mean()),
                        "uncertainty": float(sigma[i].mean()),
                        "true_state": int(labels_cpu["event_state_code"][i].item()),
                    })
        return pd.DataFrame(all_outputs)


def train_model(
    df: pd.DataFrame,
    config: Optional[TrainingConfig] = None,
    loss_config: Optional[LossConfig] = None,
    device: Optional[str] = None,
) -> Tuple[Trainer, Dict[str, Any]]:
    config = config or TrainingConfig()

    train_loader, val_loader, test_loader, feature_dim = prepare_dataloaders(
        df,
        lookback=config.lookback,
        forecast_horizon=config.forecast_horizon,
        batch_size=config.batch_size,
    )

    model = create_model(
        architecture=config.architecture,
        input_dim=feature_dim,
        hidden_dim=config.hidden_dim,
        forecast_horizon=config.forecast_horizon,
        dropout=config.dropout,
        use_multi_scale=config.use_multi_scale,
        use_coherent_heads=config.use_coherent_heads,
    )

    trainer = Trainer(model, config, loss_config, device)
    history = trainer.fit(train_loader, val_loader)

    test_metrics = trainer._evaluate(test_loader)
    logger.info(f"Test metrics: {test_metrics}")

    predictions = trainer.predict(test_loader)

    try:
        import wandb
        if wandb.run:
            wandb.run.summary["test/loss"] = test_metrics.get("total", 0)
            wandb.run.summary["test/accuracy"] = test_metrics.get("accuracy", 0)
            wandb.run.summary["test/accuracy_inter"] = test_metrics.get("accuracy_inter", 0)
            wandb.run.summary["test/accuracy_pre"] = test_metrics.get("accuracy_pre", 0)
            wandb.run.summary["test/accuracy_onset"] = test_metrics.get("accuracy_onset", 0)

            if "event_state_pred" in predictions.columns and "true_state" in predictions.columns:
                from sklearn.metrics import confusion_matrix, classification_report
                cm = confusion_matrix(predictions["true_state"], predictions["event_state_pred"], labels=[0, 1, 2])
                report = classification_report(
                    predictions["true_state"], predictions["event_state_pred"],
                    target_names=["inter", "pre", "onset"], output_dict=True, zero_division=0,
                )
                logger.info(f"Per-class metrics: {report}")

                import io
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                im = ax1.imshow(cm, cmap="Blues", aspect="auto")
                ax1.set_xticks([0, 1, 2]); ax1.set_xticklabels(["Inter", "Pre", "Onset"])
                ax1.set_yticks([0, 1, 2]); ax1.set_yticklabels(["Inter", "Pre", "Onset"])
                for i in range(3):
                    for j in range(3):
                        text_color = "white" if cm[i, j] > cm.max() / 2 else "black"
                        ax1.text(j, i, str(cm[i, j]), ha="center", va="center", fontweight="bold", color=text_color)
                ax1.set_xlabel("Predicted"); ax1.set_ylabel("Actual"); ax1.set_title("Confusion Matrix")
                plt.colorbar(im, ax=ax1, shrink=0.8)

                pred_counts = predictions["event_state_pred"].value_counts().reindex([0, 1, 2], fill_value=0)
                true_counts = predictions["true_state"].value_counts().reindex([0, 1, 2], fill_value=0)
                x_pos = np.arange(3)
                w = 0.35
                ax2.bar(x_pos - w/2, pred_counts.values, w, label="Predicted", alpha=0.8, color="#9b59b6")
                ax2.bar(x_pos + w/2, true_counts.values, w, label="Actual", alpha=0.8, color="#2ecc71")
                ax2.set_xticks(x_pos); ax2.set_xticklabels(["Inter", "Pre", "Onset"])
                ax2.set_ylabel("Count"); ax2.set_title("Predicted vs Actual Distribution")
                ax2.legend(fontsize=8)

                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
                plt.close(fig)
                buf.seek(0)
                from PIL import Image
                wandb.log({"test/confusion_and_distribution": wandb.Image(Image.open(buf))})
    except Exception as exc:
        logger.warning(f"Wandb logging failed: {exc}")

    try:
        from .visualization import generate_training_visualizations
        viz_paths = generate_training_visualizations(
            history, predictions,
            output_dir=config.checkpoint_dir,
            prefix="training",
        )
        logger.info(f"Training visualizations saved: {viz_paths}")
        try:
            import wandb
            if wandb.run:
                for name, path in viz_paths.items():
                    if path and Path(path).exists():
                        wandb.log({f"viz/{name}": wandb.Image(str(path))})
        except Exception:
            pass
    except Exception:
        pass

    return trainer, history, predictions
