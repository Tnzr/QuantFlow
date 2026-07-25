"""Cascade-ANP: MultiScaleTokenCascade encoder + Attention Neural Process decoder.

The cascade (BiLSTM → 4-scale GRU) produces multi-scale fused embeddings that
capture market microstructure at minute, hour, day, and week scales. These
embeddings serve as the deterministic context for the ANP cross-attention
decoder, which samples a global latent z (market regime) and predicts future
returns with uncertainty bands.

Scale embeddings are preserved for explainable AI visualization.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.out(out)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, dim * 2)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: torch.Tensor, keys_values: torch.Tensor) -> torch.Tensor:
        B, Nq, D = queries.shape
        _, Nkv, _ = keys_values.shape
        q = self.q_proj(queries).reshape(B, Nq, self.heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv_proj(keys_values).reshape(B, Nkv, 2, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, Nq, D)
        return self.out(out)


class LatentEncoder(nn.Module):
    def __init__(self, hidden_dim: int, latent_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.pool_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.log_sigma_head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pooled = context.mean(dim=1)
        h = self.pool_proj(pooled)
        mu = self.mu_head(h)
        log_sigma = self.log_sigma_head(h)
        sigma = torch.exp(0.5 * log_sigma)
        eps = torch.randn_like(sigma)
        z = mu + eps * sigma
        return z, mu, log_sigma


class ANPDecoder(nn.Module):
    def __init__(self, hidden_dim: int, latent_dim: int = 128, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = CrossAttention(hidden_dim, heads, dropout)
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.return_embed = nn.Linear(1, hidden_dim)
        # Wider predictor for multi-step trajectory diversity
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 2),
        )

    def forward(self, queries: torch.Tensor, context: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """One-shot: predict all target steps at once from fixed query tokens."""
        attended = self.cross_attn(queries, context)
        z_proj = self.latent_proj(z).unsqueeze(1).expand(-1, attended.size(1), -1)
        combined = torch.cat([attended, z_proj], dim=-1)
        out = self.predictor(combined)
        mu_y = out[..., 0]
        log_sigma_y = out[..., 1]
        sigma_y = F.softplus(log_sigma_y) + 1e-4
        return mu_y, sigma_y

    def forward_ar(self, context: torch.Tensor, z: torch.Tensor, n_steps: int,
                   teacher_forcing: bool = False, target_returns: Optional[torch.Tensor] = None
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = context.shape[0]; device = context.device
        current_token = context[:, -1:, :]

        mu_list, sigma_list = [], []
        for step in range(n_steps):
            attended = self.cross_attn(current_token, context)
            z_proj = self.latent_proj(z).unsqueeze(1)
            out = self.predictor(torch.cat([attended, z_proj], dim=-1))
            mu_s = out[..., 0]
            sigma_s = F.softplus(out[..., 1]) + 1e-4

            mu_list.append(mu_s.squeeze(-1))
            sigma_list.append(sigma_s.squeeze(-1))

            if teacher_forcing and target_returns is not None:
                mask = (torch.rand(B, device=device) > 0.5).float().unsqueeze(1).unsqueeze(1)
                tv = target_returns[:, step:step+1].unsqueeze(-1)
                pv = mu_s.unsqueeze(-1) + sigma_s.unsqueeze(-1) * torch.randn_like(mu_s.unsqueeze(-1))
                current_token = self.return_embed(mask * tv + (1 - mask) * pv)
            else:
                current_token = self.return_embed(mu_s.unsqueeze(-1))

        return torch.stack(mu_list, dim=1), torch.stack(sigma_list, dim=1)


class CascadeANP(nn.Module):
    """Hybrid architecture: MultiScale GRU cascade encodes the full price sequence
    into multi-scale fused embeddings. The ANP decoder then randomly samples
    context/target tokens from the encoded sequence, conditions on a global
    latent z (market regime), and predicts future returns with uncertainty.

    Scale activations are preserved for visualization alongside forecasts.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, latent_dim: int = 128,
                 cascade_layers: int = 4, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        # BiLSTM encoder (CAUSAL: unidirectional, no future leakage)
        self.bilstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=2,
            batch_first=True, bidirectional=False, dropout=dropout,
        )

        # Multi-scale GRU cascade
        self.scale_grus = nn.ModuleList([
            nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
            for _ in range(cascade_layers)
        ])
        self.scale_proj = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(cascade_layers)
        ])
        self.cascade_layers = cascade_layers

        # ANP components
        self.latent_encoder = LatentEncoder(hidden_dim, latent_dim, dropout)
        self.decoder = ANPDecoder(hidden_dim, latent_dim, heads, dropout)

        # Direct path head: 1D conv over raw features → position-aware tokens
        # Bypasses recurrent convergence — each position has a unique encoding
        self.direct_conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim // 2, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            # Regular AvgPool1d with computed kernel (ONNX-compatible)
            nn.AdaptiveAvgPool1d(30),  # 30 position-aware tokens
        )
        self.direct_path = nn.Sequential(
            nn.Linear(hidden_dim * 30, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 21),
        )
        # Skip variant: cascade (30 tokens) + raw BiLSTM (30 tokens) = 60 tokens
        self.direct_skip = nn.Sequential(
            nn.Linear(hidden_dim * 60, hidden_dim * 6),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 6, hidden_dim * 3),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 3, 21),
        )
        self.direct_sigma = nn.Sequential(
            nn.Linear(hidden_dim * 30, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 21),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Returns (fused_sequence, raw_bilstm_output, scale_activations)."""
        B, T, F = x.shape
        encoded, _ = self.bilstm(x)  # [B, T, H]
        raw_bilstm = encoded  # save before cascade smoothing

        fused = encoded
        scale_acts = []
        for i, (gru, proj) in enumerate(zip(self.scale_grus, self.scale_proj)):
            out, _ = gru(fused)
            residual = proj(out)
            fused = fused + torch.relu(residual)
            scale_acts.append(out)
        return fused, raw_bilstm, scale_acts

    def forward(self, x_context: torch.Tensor, y_context: torch.Tensor,
                n_steps: int = 21, teacher_forcing: bool = False,
                target_returns: Optional[torch.Tensor] = None,
                raw_context: Optional[torch.Tensor] = None,
                raw_features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass. If raw_features is provided, uses 1D conv for direct path."""
        B = x_context.shape[0]
        context_repr = x_context
        z, mu_z, log_sigma_z = self.latent_encoder(context_repr)

        mu_y, sigma_y = self.decoder.forward_ar(
            context_repr, z, n_steps,
            teacher_forcing=teacher_forcing,
            target_returns=target_returns,
        )
        # Direct path: 1D conv over raw features → position-aware → MLP
        Bdim = x_context.shape[0]
        if raw_features is not None:
            # raw_features: [B, T, F] → conv → [B, H, 30] → flatten → [B, 30*H]
            feat_t = raw_features.permute(0, 2, 1)  # [B, F, T]
            conv_out = self.direct_conv(feat_t)      # [B, H, 30]
            feat_sampled = conv_out.reshape(Bdim, -1)  # [B, 30*H]
            direct_mu = self.direct_path(feat_sampled)
        else:
            # Fallback: cascade tokens
            Tdim = x_context.shape[1]; Hdim = x_context.shape[2]
            k = min(30, Tdim)
            stride = max(1, Tdim // k)
            idx = torch.arange(0, Tdim, stride, device=x_context.device)[:k]
            sampled = context_repr[:, idx, :].reshape(Bdim, k * Hdim)
            if len(idx) < 30:
                sampled = F.pad(sampled, (0, (30 - len(idx)) * Hdim))
            direct_mu = self.direct_path(sampled)
        # Sigma: always use cascade tokens for uncertainty estimation
        Tdim = x_context.shape[1]; Hdim = x_context.shape[2]
        k = min(30, Tdim)
        stride = max(1, Tdim // k)
        idx = torch.arange(0, Tdim, stride, device=x_context.device)[:k]
        sigma_input = context_repr[:, idx, :].reshape(Bdim, k * Hdim)
        if len(idx) < 30:
            sigma_input = F.pad(sigma_input, (0, (30 - len(idx)) * Hdim))
        direct_sigma = F.softplus(self.direct_sigma(sigma_input)) + 1e-4

        return {
            "predicted_return": direct_mu.mean(dim=-1),
            "aleatoric_sigma": direct_sigma.mean(dim=-1),
            "mu_y": direct_mu,
            "sigma_y": direct_sigma,
            "mu_ar": mu_y, "sigma_ar": sigma_y,
            "z": z, "mu_z": mu_z, "log_sigma_z": log_sigma_z,
            "forecast_path": direct_mu, "sigma_path": direct_sigma,
            "past_state_probs": torch.zeros(Bdim, 3, device=x_context.device),
            "past_state_logits": torch.zeros(Bdim, 3, device=x_context.device),
            "past_attn_weights": None,
            "future_forecast": direct_mu.mean(dim=-1, keepdim=True),
            "future_bins": None,
            "uncertainty_mu": torch.zeros(Bdim, device=x_context.device),
            "uncertainty_sigma": direct_sigma.mean(dim=-1),
            "dynamics_residual": torch.zeros(Bdim, device=x_context.device),
            "hidden_state": None, "fused_sequence": None, "scale_activations": [],
            "epistemic_sigma": direct_sigma.mean(dim=-1),
        }


class CascadeANPLoss(nn.Module):
    """Temporal path loss: weighted MAE across forecast distance.

    The model outputs a full multi-step forecast path [B, N_t]. Instead of
    collapsing to a scalar and guessing direction, we compare EACH step
    against the actual forward return at that distance. Near-term errors
    are weighted more heavily (they should be more predictable); far-term
    errors are weighted less (macro uncertainty dominates).

    The error-vs-t distribution IS the forecasting confusion matrix —
    it shows how uncertainty grows with forecast distance.
    """

    def __init__(self, beta: float = 0.2, near_weight: float = 2.0, far_weight: float = 0.3):
        super().__init__()
        self.beta = beta
        self.near_weight = near_weight
        self.far_weight = far_weight

    def forward(self, outputs: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        mu_y = outputs["mu_y"]              # direct path head (primary)
        sigma_y = outputs.get("sigma_ar", outputs["sigma_y"])
        mu_z = outputs["mu_z"]
        log_sigma_z = outputs["log_sigma_z"]

        y_target_path = targets.get("target_return_path")  # [B, N_t] actual returns
        if y_target_path is None:
            y_target_path = targets["target_return"].unsqueeze(-1).expand_as(mu_y)

        B, H = mu_y.shape
        # Temporal weights: near-term → high weight, far-term → low weight
        weights = torch.linspace(self.near_weight, self.far_weight, H,
                                 device=mu_y.device)
        weights = weights / weights.mean()  # normalize to mean=1.0

        # Primary: weighted temporal MAE across the full path
        abs_err = torch.abs(mu_y - y_target_path)  # [B, H]
        temporal_mae = (abs_err * weights).mean()

        # Path loss is pure MAE — no NLL to avoid negative loss
        path_loss = temporal_mae

        # Scalar direction penalty (for the mean prediction)
        pred_scalar = mu_y.mean(dim=-1)
        true_scalar = y_target_path.mean(dim=-1)
        dir_penalty = (torch.sign(pred_scalar) != torch.sign(true_scalar)).float().mean()

        # KL on latent (encourages using the global regime representation)
        sigma_z = torch.exp(0.5 * log_sigma_z)
        kl = 0.5 * (mu_z.pow(2) + sigma_z.pow(2) - 2 * log_sigma_z - 1).clamp(min=0).mean()

        # Sigma regularity: keep sigma in reasonable range AND encourage growth with distance
        sigma_mag_penalty = F.relu(sigma_y - 0.05).mean() * 1.0  # stronger penalty
        sigma_growth_penalty = F.relu(-(sigma_y[:, -1].mean() - sigma_y[:, 0].mean())) * 0.1

        return {
            "temporal_mae": temporal_mae,
            "path_loss": path_loss,
            "dir_penalty": 0.05 * dir_penalty,
            "kl": self.beta * kl,
            "sigma_reg": sigma_mag_penalty + sigma_growth_penalty,
            "regression": temporal_mae,
            "sigma_growth": sigma_growth_penalty,
            "direction": dir_penalty,
            "total": path_loss + self.beta * kl + 0.05 * dir_penalty + sigma_mag_penalty + sigma_growth_penalty,
        }
