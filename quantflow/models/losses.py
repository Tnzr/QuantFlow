from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossConfig:
    classification_weight: float = 0.35
    regression_weight: float = 0.25
    distance_weight: float = 0.10
    coherence_weight: float = 0.10
    dynamics_weight: float = 0.10
    uncertainty_weight: float = 0.10
    ranking_weight: float = 0.05

    classification_loss_type: str = "focal"
    regression_loss: str = "smoothl1"
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    label_smoothing: float = 0.05

    classification_temporal_boost: float = 0.5
    regression_temporal_boost: float = 1.0
    temporal_focus_tau_min: float = 5.0
    classification_temporal_max_multiplier: float = 4.0
    regression_temporal_max_multiplier: float = 3.0

    def normalize_weights(self) -> Dict[str, float]:
        raw = {
            "cls": self.classification_weight,
            "fut": self.regression_weight,
            "dist": self.distance_weight,
            "coh": self.coherence_weight,
            "dyn": self.dynamics_weight,
            "unc": self.uncertainty_weight,
            "rank": self.ranking_weight,
        }
        total = sum(raw.values())
        if total == 0:
            return {k: 0.0 for k in raw}
        return {k: v / total for k, v in raw.items()}


class CompositeLoss(nn.Module):
    """Complex Systems Engineering Loss per methodology §6."""

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

    def _classification_loss(self, logits: torch.Tensor, targets: torch.Tensor, sample_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        loss = self._focal_loss(logits, targets)
        if sample_weights is not None:
            loss = loss * self._focal_loss(logits, targets, alpha=1.0, gamma=0.0).detach().mean()
            ce = F.cross_entropy(logits, targets, reduction="none")
            pt = torch.exp(-ce)
            alpha = self.config.focal_alpha
            gamma = self.config.focal_gamma
            focal = alpha * (1 - pt) ** gamma * ce
            loss = (focal * sample_weights).mean()
        return loss

    def _regression_loss(self, pred: torch.Tensor, target: torch.Tensor, active_mask: torch.Tensor, temporal_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not active_mask.any():
            return torch.tensor(0.0, device=pred.device)

        pred_active = pred[active_mask]
        target_active = target[active_mask]

        if self.config.regression_loss == "smoothl1":
            loss = F.smooth_l1_loss(pred_active, target_active, reduction="none")
        elif self.config.regression_loss == "weighted_mse":
            loss = (pred_active - target_active) ** 2
        else:
            loss = (pred_active - target_active) ** 2

        if temporal_weights is not None:
            tw = temporal_weights[active_mask]
            loss = loss * tw

        return loss.mean()

    def _distance_loss(self, pred: torch.Tensor, target: torch.Tensor, tau_weights: torch.Tensor) -> torch.Tensor:
        diff = F.smooth_l1_loss(pred, target, reduction="none")
        return (diff * tau_weights).mean()

    def _coherence_loss(self, past_probs: torch.Tensor, forecast: torch.Tensor, event_targets: torch.Tensor) -> torch.Tensor:
        inter_event_mask = event_targets == 0
        pre_event_mask = (event_targets == 1) | (event_targets == 2)

        loss = torch.tensor(0.0, device=past_probs.device)
        if inter_event_mask.any():
            forecast_inter = forecast[inter_event_mask].sigmoid()
            past_inter_conf = past_probs[inter_event_mask, 0]
            loss = loss + (forecast_inter.squeeze() - past_inter_conf).pow(2).mean()
        if pre_event_mask.any():
            forecast_pre = forecast[pre_event_mask].sigmoid()
            past_pre_conf = past_probs[pre_event_mask, 1] + past_probs[pre_event_mask, 2]
            loss = loss + (forecast_pre.squeeze() - past_pre_conf).pow(2).mean()
        return loss * 0.5 if (inter_event_mask.any() or pre_event_mask.any()) else loss

    def _dynamics_loss(self, residual: torch.Tensor, h_current: torch.Tensor, h_prev: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual_norm = residual.pow(2).mean()
        smooth_penalty = torch.tensor(0.0, device=residual.device)
        if h_prev is not None and isinstance(h_current, torch.Tensor):
            smooth_penalty = (h_current - h_prev).pow(2).mean()
        return residual_norm + 0.1 * smooth_penalty

    def _uncertainty_loss(self, mu: torch.Tensor, sigma: torch.Tensor, forecast: torch.Tensor, target: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
        if not active_mask.any():
            return torch.tensor(0.0, device=mu.device)
        mu_a = mu[active_mask]
        sigma_a = sigma[active_mask]
        forecast_a = forecast[active_mask]
        target_a = target[active_mask]
        nll = 0.5 * (torch.log(sigma_a.pow(2)) + (forecast_a - target_a).pow(2) / sigma_a.pow(2))
        return nll.mean()

    def _ranking_loss(self, forecast: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
        if active_mask.sum() < 2:
            return torch.tensor(0.0, device=forecast.device)
        mask_shift = active_mask[:-1] & active_mask[1:]
        if not mask_shift.any():
            return torch.tensor(0.0, device=forecast.device)
        f_prev = forecast[:-1][mask_shift]
        f_curr = forecast[1:][mask_shift]
        violations = F.relu(f_curr - f_prev)
        return violations.mean()

    def _compute_temporal_weights(self, tau: torch.Tensor, active_mask: torch.Tensor, boost: float, max_multiplier: float = 3.0) -> torch.Tensor:
        weights = torch.ones_like(tau)
        if not active_mask.any():
            return weights
        active_tau = tau[active_mask]
        tau_min = self.config.temporal_focus_tau_min
        boost_weights = 1.0 + boost * torch.exp(-active_tau / max(tau_min, 1e-6))
        boost_weights = boost_weights.clamp(1.0, max_multiplier)
        weights[active_mask] = boost_weights
        return weights

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

        active_mask = event_targets > 0
        temporal_cls_weights = self._compute_temporal_weights(tau_target, active_mask, self.config.classification_temporal_boost, self.config.classification_temporal_max_multiplier)
        temporal_reg_weights = self._compute_temporal_weights(tau_target, active_mask, self.config.regression_temporal_boost, self.config.regression_temporal_max_multiplier)

        losses = {}
        losses["classification"] = weights["cls"] * self._classification_loss(past_logits, event_targets, temporal_cls_weights)
        losses["regression"] = weights["fut"] * self._regression_loss(future_forecast, tau_target, active_mask, temporal_reg_weights)
        losses["distance"] = weights["dist"] * self._distance_loss(future_forecast, tau_target, temporal_reg_weights)
        losses["coherence"] = weights["coh"] * self._coherence_loss(past_probs, future_forecast, event_targets)
        losses["dynamics"] = weights["dyn"] * self._dynamics_loss(dynamics_residual, torch.zeros(1))
        losses["uncertainty"] = weights["unc"] * self._uncertainty_loss(unc_mu, unc_sigma, future_forecast, tau_target, active_mask)
        losses["ranking"] = weights["rank"] * self._ranking_loss(future_forecast, active_mask)
        losses["total"] = sum(losses.values())

        return losses
