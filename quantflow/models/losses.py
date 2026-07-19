from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossConfig:
    classification_weight: float = 0.50
    regression_weight: float = 0.20
    distance_weight: float = 0.10
    coherence_weight: float = 0.05
    dynamics_weight: float = 0.02
    uncertainty_weight: float = 0.05
    direction_weight: float = 0.05
    ranking_weight: float = 0.03

    classification_loss_type: str = "focal"
    regression_loss: str = "smoothl1"
    focal_alpha: float = 0.25
    focal_gamma: float = 5.0
    label_smoothing: float = 0.05

    classification_temporal_boost: float = 0.5
    regression_temporal_boost: float = 1.0
    temporal_focus_tau_min: float = 5.0
    classification_temporal_max_multiplier: float = 4.0
    regression_temporal_max_multiplier: float = 3.0

    class_weights: tuple = (5.0, 0.25, 3.0)

    def normalize_weights(self) -> Dict[str, float]:
        raw = {
            "cls": self.classification_weight,
            "fut": self.regression_weight,
            "dist": self.distance_weight,
            "coh": self.coherence_weight,
            "dyn": self.dynamics_weight,
            "unc": self.uncertainty_weight,
            "dir": self.direction_weight,
            "rank": self.ranking_weight,
        }
        total = sum(raw.values())
        if total == 0:
            return {k: 0.0 for k in raw}
        return {k: v / total for k, v in raw.items()}


class CompositeLoss(nn.Module):
    """Complex Systems Engineering Loss per methodology §6.

    Enhanced with:
    - Class-weighted focal loss for imbalance handling
    - Direction-biased return prediction loss
    - Temporal distance weighting (closer events weighted higher)
    - Coherence between state probabilities and forecast horizon
    """

    def __init__(self, config: Optional[LossConfig] = None):
        super().__init__()
        self.config = config or LossConfig()

    def _focal_loss(self, logits: torch.Tensor, targets: torch.Tensor, alpha: float = None, gamma: float = None) -> torch.Tensor:
        alpha = alpha or self.config.focal_alpha
        gamma = gamma or self.config.focal_gamma
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        focal = alpha * (1 - pt) ** gamma * ce
        return focal.mean()

    def _classification_loss(self, logits: torch.Tensor, targets: torch.Tensor, tau_target: torch.Tensor) -> torch.Tensor:
        epsilon = self.config.label_smoothing
        n_classes = logits.size(-1)
        smooth_targets = torch.full_like(logits, epsilon / (n_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - epsilon)
        neg_log = -(smooth_targets * F.log_softmax(logits, dim=-1))
        base_loss = neg_log.sum(dim=-1)

        alpha = self.config.focal_alpha
        gamma = self.config.focal_gamma
        prob_true = F.softmax(logits, dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_factor = alpha * (1.0 - prob_true) ** gamma

        cw = torch.tensor(self.config.class_weights, device=logits.device, dtype=torch.float32)
        class_weight_per_sample = cw[targets]

        tau_clamped = tau_target.clamp(1.0, 21.0)
        temporal_weight = 1.0 + self.config.classification_temporal_boost * torch.exp(
            -tau_clamped / max(self.config.temporal_focus_tau_min, 1e-6)
        )
        temporal_weight = temporal_weight.clamp(1.0, self.config.classification_temporal_max_multiplier)

        loss = focal_factor * base_loss * class_weight_per_sample * temporal_weight
        return loss.mean()

    def _regression_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_sq = pred.squeeze(-1) if pred.dim() > 1 else pred
        if self.config.regression_loss == "smoothl1":
            loss = F.smooth_l1_loss(pred_sq, target, reduction="none")
        elif self.config.regression_loss == "weighted_mse":
            loss = (pred_sq - target) ** 2
        else:
            loss = (pred_sq - target) ** 2

        tau_clamped = target.clamp(1.0, 21.0)
        temporal_weight = 1.0 + self.config.regression_temporal_boost * torch.exp(
            -tau_clamped / max(self.config.temporal_focus_tau_min, 1e-6)
        )
        temporal_weight = temporal_weight.clamp(1.0, self.config.regression_temporal_max_multiplier)

        return (loss * temporal_weight).mean()

    def _distance_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_sq = pred.squeeze(-1) if pred.dim() > 1 else pred
        diff = F.smooth_l1_loss(pred_sq, target, reduction="none")
        tau_clamped = target.clamp(1.0, 21.0)
        weight = 1.0 + torch.exp(-tau_clamped / max(self.config.temporal_focus_tau_min, 1e-6))
        return (diff * weight).mean()

    def _coherence_loss(self, past_probs: torch.Tensor, forecast: torch.Tensor, event_targets: torch.Tensor) -> torch.Tensor:
        inter_event_mask = event_targets == 0
        pre_event_mask = (event_targets == 1) | (event_targets == 2)
        forecast_sq = forecast.squeeze(-1) if forecast.dim() > 1 else forecast

        loss = torch.tensor(0.0, device=past_probs.device)
        if inter_event_mask.any():
            forecast_norm = torch.sigmoid(forecast_sq[inter_event_mask] / 21.0)
            past_inter_conf = past_probs[inter_event_mask, 0]
            loss = loss + (forecast_norm.squeeze() - 1.0 + past_inter_conf).pow(2).mean()
        if pre_event_mask.any():
            forecast_norm = torch.sigmoid(forecast_sq[pre_event_mask] / 21.0)
            past_pre_conf = past_probs[pre_event_mask, 1] + past_probs[pre_event_mask, 2]
            loss = loss + (forecast_norm.squeeze() - past_pre_conf).pow(2).mean()
        return loss * 0.5 if (inter_event_mask.any() or pre_event_mask.any()) else loss

    def _direction_loss(self, forecast: torch.Tensor, target_ret: torch.Tensor) -> torch.Tensor:
        """Direction-biased percent change loss.

        Penalizes forecast direction when it differs from actual price change.
        Target should be forward returns (e.g., target_21d).
        """
        forecast_sq = forecast.squeeze(-1) if forecast.dim() > 1 else forecast
        direction_pred = torch.sign(forecast_sq - 21.0)
        direction_true = torch.sign(target_ret)
        mismatch = (direction_pred != direction_true).float()
        mag = target_ret.abs()
        return (mismatch * mag.abs()).mean() * 0.2 + F.smooth_l1_loss(
            forecast_sq, 21.0 * (1.0 - torch.tanh(target_ret * 5.0)),
            reduction="mean",
        ) * 0.8

    def _dynamics_loss(self, residual: torch.Tensor, h_current: torch.Tensor, h_prev: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual_norm = residual.pow(2).mean()
        smooth_penalty = torch.tensor(0.0, device=residual.device)
        if h_prev is not None and isinstance(h_current, torch.Tensor):
            if h_current.shape == h_prev.shape:
                smooth_penalty = (h_current - h_prev).pow(2).mean()
        return residual_norm + 0.1 * smooth_penalty

    def _uncertainty_loss(self, mu: torch.Tensor, sigma: torch.Tensor, forecast: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mu_a = mu.squeeze(-1) if mu.dim() > 1 else mu
        sigma_a = sigma.squeeze(-1) if sigma.dim() > 1 else sigma
        forecast_a = forecast.squeeze(-1) if forecast.dim() > 1 else forecast
        nll = 0.5 * (torch.log(sigma_a.pow(2) + 1e-6) + (forecast_a - target).pow(2) / (sigma_a.pow(2) + 1e-6))
        return nll.mean()

    def _ranking_loss(self, forecast: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        forecast_sq = forecast.squeeze(-1) if forecast.dim() > 1 else forecast
        if len(forecast_sq) < 2:
            return torch.tensor(0.0, device=forecast.device)
        f_prev = forecast_sq[:-1]
        f_curr = forecast_sq[1:]
        t_prev = target[:-1]
        t_curr = target[1:]
        rank_pred = (f_curr - f_prev)
        rank_true = (t_curr - t_prev)
        violations = F.relu(torch.sign(rank_true) * (-rank_pred))
        return violations.mean()

    def forward(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        weights = self.config.normalize_weights()

        past_logits = outputs["past_state_logits"]
        event_targets = targets["event_state_code"].long()
        tau_target = targets["tau_forward"]
        future_forecast = outputs["future_forecast"].squeeze(-1)
        past_probs = outputs["past_state_probs"]
        dynamics_residual = outputs["dynamics_residual"]
        unc_mu = outputs["uncertainty_mu"].squeeze(-1)
        unc_sigma = outputs["uncertainty_sigma"].squeeze(-1)

        losses = {}
        losses["classification"] = weights["cls"] * self._classification_loss(past_logits, event_targets, tau_target)
        losses["regression"] = weights["fut"] * self._regression_loss(future_forecast, tau_target)
        losses["distance"] = weights["dist"] * self._distance_loss(future_forecast, tau_target)
        losses["coherence"] = weights["coh"] * self._coherence_loss(past_probs, future_forecast, event_targets)
        losses["dynamics"] = weights["dyn"] * self._dynamics_loss(dynamics_residual, torch.zeros(1, device=dynamics_residual.device))

        target_ret = targets.get("target_21d", torch.zeros_like(tau_target))
        losses["direction"] = weights["dir"] * self._direction_loss(future_forecast, target_ret)

        losses["uncertainty"] = weights["unc"] * self._uncertainty_loss(unc_mu, unc_sigma, future_forecast, tau_target)
        losses["ranking"] = weights["rank"] * self._ranking_loss(future_forecast, tau_target)
        losses["total"] = sum(losses.values())

        return losses
