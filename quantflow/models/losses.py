from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossConfig:
    regression_weight: float = 0.40
    direction_weight: float = 0.15
    diversity_weight: float = 0.05
    temporal_weight: float = 0.25
    dynamics_weight: float = 0.05
    sigma_reg_weight: float = 0.10

    regression_loss: str = "smoothl1"
    label_smoothing: float = 0.05
    sigma_target: float = 0.05  # target uncertainty: models should aim for ~5% sigma

    # Legacy fields
    classification_weight: float = 0.0
    distance_weight: float = 0.0
    coherence_weight: float = 0.0
    ranking_weight: float = 0.0
    uncertainty_weight: float = 0.0
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    classification_temporal_boost: float = 0.5
    regression_temporal_boost: float = 1.0
    temporal_focus_tau_min: float = 5.0
    classification_temporal_max_multiplier: float = 4.0
    regression_temporal_max_multiplier: float = 3.0
    class_weight_mode: str = "dynamic"
    class_weights: tuple = (1.5, 0.7, 1.5)
    dynamic_class_weight_min: float = 0.1
    dynamic_class_weight_max: float = 12.0
    sharpness_weight: float = 0.0
    sharpness_weight_legacy: float = 0.0
    dir_coherence_weight: float = 0.0
    anti_collapse_weight: float = 0.0

    def normalize_weights(self) -> Dict[str, float]:
        raw = {
            "reg": self.regression_weight,
            "dir": self.direction_weight,
            "div": self.diversity_weight,
            "tmp": self.temporal_weight,
            "dyn": self.dynamics_weight,
            "sig": self.sigma_reg_weight,
        }
        total = sum(raw.values())
        if total == 0:
            return {k: 0.0 for k in raw}
        return {k: v / total for k, v in raw.items()}


class CompositeLoss(nn.Module):
    """Return-forecasting loss — smooth L1 + direction penalty + diversity.

    No NLL (Gaussian log-likelihood) since it drives loss negative when the
    model predicts very small sigma — this incentivizes overconfident near-zero
    predictions (collapse to safe output). Instead, use a simple sigma
    regularization that pulls sigma toward a target value.
    """

    def __init__(self, config: Optional[LossConfig] = None):
        super().__init__()
        self.config = config or LossConfig()

    def _regression_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.config.regression_loss == "smoothl1":
            return F.smooth_l1_loss(pred, target)
        elif self.config.regression_loss == "huber":
            return F.huber_loss(pred, target, delta=0.02)
        return F.smooth_l1_loss(pred, target)

    def _direction_penalty(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_sign = torch.sign(pred)
        true_sign = torch.sign(target)
        mismatch = (pred_sign != true_sign).float()
        return mismatch.mean()

    def _diversity_loss(self, pred: torch.Tensor) -> torch.Tensor:
        """Penalize zero-variance predictions — forces the model to produce
        varied forecasts across samples rather than collapsing to a single value."""
        if len(pred) < 2:
            return torch.tensor(0.0, device=pred.device)
        std = pred.std()
        return F.relu(0.03 - std) * 10.0

    def _hold_penalty(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        in_hold = (pred.abs() < 0.01).float()
        target_directional = (target.abs() >= 0.01).float()
        return (in_hold * target_directional).mean()

    def _temporal_loss(self, pred: torch.Tensor, target: torch.Tensor,
                       forecast_path: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Inverse temporal distance loss: near-term forecast errors weighted
        more heavily than far-term. This forces the model to be accurate at
        short horizons (where prediction is feasible) while allowing flexibility
        at longer horizons (where macro trends dominate)."""
        if forecast_path is None or forecast_path.dim() < 2:
            return torch.tensor(0.0, device=pred.device)
        B, H = forecast_path.shape
        # Linear decay: step 0 weight = 1.0, step H-1 weight = 0.2
        weights = torch.linspace(1.0, 0.2, H, device=pred.device)
        weights = weights / weights.mean()  # normalize so mean=1.0
        # Compare each step to the scalar target (approximate)
        # For a real multi-step loss, compare path[t] to the t-step forward return
        diff = F.smooth_l1_loss(forecast_path, target.unsqueeze(-1).expand_as(forecast_path), reduction="none")
        weighted = diff * weights
        return weighted.mean()

    def _sigma_regularization(self, sigma: torch.Tensor) -> torch.Tensor:
        """Pull sigma toward a target value — prevents overconfidence (sigma→0)
        and excessive uncertainty (sigma→∞)."""
        target = self.config.sigma_target
        return F.smooth_l1_loss(sigma, torch.full_like(sigma, target))

    def _dynamics_loss(self, residual: torch.Tensor) -> torch.Tensor:
        return residual.pow(2).mean()

    def forward(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        weights = self.config.normalize_weights()

        predicted_return = outputs["predicted_return"]
        aleatoric_sigma = outputs["aleatoric_sigma"]
        dynamics_residual = outputs.get("dynamics_residual", torch.zeros(1, device=predicted_return.device))
        target_return = targets.get("target_return", targets.get("target_21d", torch.zeros_like(predicted_return)))

        losses = {}
        losses["regression"] = weights["reg"] * self._regression_loss(predicted_return, target_return)
        losses["direction"] = weights["dir"] * self._direction_penalty(predicted_return, target_return)
        losses["diversity"] = weights["div"] * self._diversity_loss(predicted_return)
        losses["temporal"] = weights["tmp"] * self._temporal_loss(
            predicted_return, target_return, outputs.get("forecast_path"))
        losses["sigma_reg"] = weights["sig"] * self._sigma_regularization(aleatoric_sigma)
        if weights["dyn"] > 0:
            losses["dynamics"] = weights["dyn"] * self._dynamics_loss(dynamics_residual)

        losses["total"] = sum(losses.values())
        return losses
