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

    lr_scheduler: str = "cosine"
    warmup_epochs: int = 5
    min_lr: float = 1e-6

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
        warmup_steps = self.config.warmup_epochs * steps_per_epoch

        warmup = LinearLR(self.optimizer, start_factor=1e-3, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(self.optimizer, T_max=total_steps - warmup_steps, eta_min=self.config.min_lr)
        return SequentialLR(self.optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

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
            self.optimizer.step()

        return outputs, losses, labels

    def _evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_losses: Dict[str, float] = {}
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in loader:
                features, labels = self._to_device(batch)
                outputs = self.model(features)
                losses = self.criterion(outputs, labels)

                for k, v in losses.items():
                    total_losses[k] = total_losses.get(k, 0.0) + v.item()

                preds = outputs["past_state_logits"].argmax(dim=-1)
                total_correct += (preds == labels["event_state_code"]).sum().item()
                total_samples += labels["event_state_code"].size(0)

        avg_losses = {k: v / max(len(loader), 1) for k, v in total_losses.items()}
        avg_losses["accuracy"] = total_correct / max(total_samples, 1)
        return avg_losses

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:
        scheduler = self._create_scheduler(len(train_loader))
        logger.info(f"Starting training: {self.config.epochs} epochs, device={self.device}")
        logger.info(f"Model: {sum(p.numel() for p in self.model.parameters()):,} parameters")

        for epoch in range(self.current_epoch, self.config.epochs):
            self.model.train()
            epoch_start = time.time()
            epoch_losses: Dict[str, float] = {}
            epoch_correct = 0
            epoch_samples = 0

            accumulation_counter = 0
            for batch_idx, batch in enumerate(train_loader):
                outputs, losses_dict, labels = self._step(batch)
                accumulation_counter += 1

                for k, v in losses_dict.items():
                    epoch_losses[k] = epoch_losses.get(k, 0.0) + v.item()

                preds = outputs["past_state_logits"].argmax(dim=-1)
                labels_dev = labels["event_state_code"]
                epoch_correct += (preds == labels_dev).sum().item()
                epoch_samples += labels_dev.size(0)
                scheduler.step()

            n_batches = max(len(train_loader), 1)
            train_metrics = {k: v / n_batches for k, v in epoch_losses.items()}
            train_metrics["accuracy"] = epoch_correct / max(epoch_samples, 1)
            train_metrics["lr"] = self.optimizer.param_groups[0]["lr"]
            train_metrics["epoch"] = epoch + 1
            self.train_history.append(train_metrics)

            val_metrics = self._evaluate(val_loader)
            val_metrics["epoch"] = epoch + 1
            self.val_history.append(val_metrics)

            elapsed = time.time() - epoch_start
            logger.info(
                f"Epoch {epoch + 1}/{self.config.epochs} ({elapsed:.1f}s) | "
                f"train_loss={train_metrics.get('total', 0):.4f} | "
                f"val_loss={val_metrics.get('total', 0):.4f} | "
                f"acc={val_metrics.get('accuracy', 0):.3f}"
            )

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

        return {"train_history": self.train_history, "val_history": self.val_history}

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
        from .visualization import generate_training_visualizations
        viz_paths = generate_training_visualizations(
            history, predictions,
            output_dir=config.checkpoint_dir,
            prefix="training",
        )
        logger.info(f"Training visualizations saved: {viz_paths}")
    except Exception:
        pass

    return trainer, history, predictions
