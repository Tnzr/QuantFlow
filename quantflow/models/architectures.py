from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """1D convolution with causal padding — no future information leakage."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.padding, 0)))


class TCNBlock(nn.Module):
    """Temporal convolutional block with residual connection."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.activation(self.conv1(x))
        out = self.dropout(out)
        out = self.activation(self.conv2(out))
        out = self.dropout(out)
        return out + residual


class TCNEncoder(nn.Module):
    """TCN encoder per methodology §3.2.1 — dilated causal convolutions for long sequences."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 5, kernel_size: int = 7, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, 1)
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i
            self.blocks.append(TCNBlock(hidden_dim, kernel_size, dilation, dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return x.transpose(1, 2)


class MultiScaleTokenCascade(nn.Module):
    """Multi-scale temporal tokenization per methodology §4.

    Processes fused tokens through 4 cascading scales with UNet-like skip connections."""
    def __init__(self, token_dim: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.scales = nn.ModuleList([
            nn.GRU(token_dim if s == 0 else hidden_dim, hidden_dim, num_layers=2,
                   batch_first=True, bidirectional=False, dropout=dropout)
            for s in range(4)
        ])
        self.down_projections = nn.ModuleList([
            nn.Linear(token_dim if s == 0 else hidden_dim, hidden_dim)
            for s in range(4)
        ])
        self.skip_projections = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(4)
        ])
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, u_t: torch.Tensor, h_prev: Optional[list[torch.Tensor]] = None) -> Tuple[torch.Tensor, list[torch.Tensor]]:
        h_new = []
        x = u_t
        skips = []

        for s in range(4):
            x_proj = self.down_projections[s](x)
            if h_prev is not None and s < len(h_prev) and h_prev[s] is not None:
                h = h_prev[s]
                if h.dim() == 3 and h.size(1) != x_proj.size(0):
                    h = h.transpose(0, 1).contiguous()
                out, h = self.scales[s](x_proj, h)
            else:
                out, h = self.scales[s](x_proj)
            h_new.append(h)
            skips.append(out)
            x = out

        skip_sum = sum(self.skip_projections[i](skips[i]) for i in range(4))
        fused = self.fusion(torch.cat([x, skip_sum], dim=-1))
        return fused, h_new


class MultiScaleTokenSCascade(nn.Module):
    """Alternative multi-scale cascade using dilated 1D convolutions.

    For architectures where RNN-based cascade is inappropriate (TCN/Transformer pipeline)."""
    def __init__(self, token_dim: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        scales_config = [
            (3, 1),
            (5, 2),
            (7, 4),
            (11, 8),
        ]
        self.scale_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        for ks, dilation in scales_config:
            self.scale_convs.append(
                nn.Sequential(
                    CausalConv1d(token_dim if not self.scale_convs else hidden_dim, hidden_dim, ks, dilation),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    CausalConv1d(hidden_dim, hidden_dim, ks, dilation),
                )
            )
            self.skip_convs.append(nn.Conv1d(hidden_dim, hidden_dim, 1))
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, u_t: torch.Tensor, _h: Optional[list] = None) -> Tuple[torch.Tensor, list]:
        x = u_t.transpose(1, 2)
        skips = []
        for s in range(4):
            x = self.scale_convs[s](x)
            skips.append(x)
        skip_sum = sum(self.skip_convs[i](skips[i]) for i in range(4))
        x = x.mean(dim=-1)
        skip_sum = skip_sum.mean(dim=-1)
        fused = self.fusion(torch.cat([x, skip_sum], dim=-1))
        return fused, []


class ModalityFusionGate(nn.Module):
    """Attention-gated modality fusion per methodology §3.2.2."""

    def __init__(self, hidden_dim: int, num_modalities: int = 1):
        super().__init__()
        self.num_modalities = num_modalities
        self.hidden = hidden_dim
        if num_modalities > 1:
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim * num_modalities, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, num_modalities),
            )

    def forward(self, modality_tokens: list[torch.Tensor]) -> torch.Tensor:
        if len(modality_tokens) == 1:
            return modality_tokens[0]
        concat = torch.cat(modality_tokens, dim=-1)
        alpha = F.softmax(self.attention(concat), dim=-1)
        return sum(alpha[..., i:i + 1] * modality_tokens[i] for i in range(len(modality_tokens)))


class PastStateHead(nn.Module):
    """Certainty generator — classifies current state relative to critical event per §5.2."""

    def __init__(self, hidden_dim: int, num_classes: int = 3, dropout: float = 0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.mlp(x)
        probs = F.softmax(logits, dim=-1)
        return probs, logits


class FutureForecastHead(nn.Module):
    """Anticipation trajectory — forecasts time-to-event distribution per §5.3."""

    def __init__(self, hidden_dim: int, forecast_horizon: int = 21, dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.horizon = forecast_horizon
        self.coherent_proj = nn.Linear(hidden_dim + 3, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.regression = nn.Linear(hidden_dim // 4, 1)
        self.bin_classifier = nn.Linear(hidden_dim // 4, forecast_horizon)

    def forward(self, x: torch.Tensor, class_logits: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if class_logits is not None:
            x = torch.cat([x, class_logits], dim=-1)
            x = self.coherent_proj(x)
        features = self.mlp(x)
        forecast_value = self.regression(features)
        bin_logits = self.bin_classifier(features)
        return {"forecast": forecast_value, "bins": bin_logits}


class UncertaintyHead(nn.Module):
    """Bayesian-style uncertainty estimation per §3.2.5 and §6.2.6."""

    def __init__(self, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 2),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.mlp(x)
        mu = out[..., 0:1]
        log_var = out[..., 1:2]
        sigma = torch.exp(0.5 * log_var).clamp(min=1e-6)
        return mu, sigma


class DynamicsResidualHead(nn.Module):
    """Models innovation residual between predicted and observed state transitions per §3.2.5."""

    def __init__(self, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class TemporalStateModel(nn.Module):
    """Base class for all Temporal State Models per methodology §3.

    Subclassed by architecture-specific implementations (BiLSTM, TCN, Transformer)."""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_classes: int = 3,
        forecast_horizon: int = 21,
        num_modalities: int = 1,
        dropout: float = 0.2,
        use_multi_scale: bool = True,
        use_coherent_heads: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.forecast_horizon = forecast_horizon
        self.use_coherent_heads = use_coherent_heads
        self.use_multi_scale = use_multi_scale

        self.fusion_gate = ModalityFusionGate(hidden_dim, num_modalities)
        self.multi_scale = MultiScaleTokenCascade(hidden_dim, hidden_dim, dropout) if use_multi_scale else nn.Identity()
        self.past_head = PastStateHead(hidden_dim, num_classes, dropout)
        self.future_head = FutureForecastHead(hidden_dim, forecast_horizon, dropout)
        self.uncertainty_head = UncertaintyHead(hidden_dim, dropout)
        self.dynamics_head = DynamicsResidualHead(hidden_dim, dropout)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor, h_prev: Optional[list] = None) -> Dict[str, torch.Tensor]:
        encoded = self._encode(x)
        fused_seq, h_new = self.multi_scale(encoded, h_prev) if self.use_multi_scale else (encoded, None)

        fused = fused_seq[:, -1, :]

        past_probs, past_logits = self.past_head(fused)
        class_logits = past_logits if self.use_coherent_heads else None
        future = self.future_head(fused, class_logits)

        unc_mu, unc_sigma = self.uncertainty_head(fused)
        dynamics = self.dynamics_head(fused)

        return {
            "past_state_probs": past_probs,
            "past_state_logits": past_logits,
            "future_forecast": future["forecast"],
            "future_bins": future["bins"],
            "uncertainty_mu": unc_mu,
            "uncertainty_sigma": unc_sigma,
            "dynamics_residual": dynamics,
            "hidden_state": h_new,
        }


class BiLSTMDualHead(TemporalStateModel):
    """BiLSTM-based encoder per methodology §3.2.1 — proven baseline for medium-length sequences."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_classes: int = 3,
        forecast_horizon: int = 21,
        num_lstm_layers: int = 2,
        dropout: float = 0.2,
        use_multi_scale: bool = True,
        use_coherent_heads: bool = True,
    ):
        super().__init__(input_dim, hidden_dim, num_classes, forecast_horizon, 1, dropout, use_multi_scale, use_coherent_heads)
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_lstm_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout_layer = nn.Dropout(dropout)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.bilstm(x)
        out = self.dropout_layer(out)
        return self.proj(out)


class TemporalTransformer(TemporalStateModel):
    """Transformer encoder per methodology §3.2.1 and §7.1."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_classes: int = 3,
        forecast_horizon: int = 21,
        num_heads: int = 8,
        num_layers: int = 4,
        max_len: int = 1024,
        dropout: float = 0.2,
        use_multi_scale: bool = True,
        use_coherent_heads: bool = True,
    ):
        super().__init__(input_dim, hidden_dim, num_classes, forecast_horizon, 1, dropout, use_multi_scale, use_coherent_heads)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            hidden_dim, num_heads, hidden_dim * 4, dropout, batch_first=True, activation="gelu", norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self._causal_mask_cache = None
        self._causal_mask_len = -1

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if self._causal_mask_len != seq_len:
            self._causal_mask_cache = torch.triu(
                torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1
            )
            self._causal_mask_len = seq_len
        return self._causal_mask_cache

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]
        mask = self._get_causal_mask(seq_len, x.device)
        return self.encoder(x, mask=mask, is_causal=True)


class MultiScaleTCNEncoder(TemporalStateModel):
    """TCN-based encoder per methodology §7.1 — dilated causal convolutions, fully parallelizable."""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_classes: int = 3,
        forecast_horizon: int = 21,
        num_layers: int = 5,
        kernel_size: int = 7,
        dropout: float = 0.2,
        use_coherent_heads: bool = True,
    ):
        super().__init__(input_dim, hidden_dim, num_classes, forecast_horizon, 1, dropout, False, use_coherent_heads)
        self.tcn = TCNEncoder(input_dim, hidden_dim, num_layers, kernel_size, dropout)
        self.multi_scale = MultiScaleTokenSCascade(hidden_dim, hidden_dim, dropout)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.tcn(x)


MODELS = {
    "bilstm": BiLSTMDualHead,
    "transformer": TemporalTransformer,
    "tcn": MultiScaleTCNEncoder,
}


def create_model(architecture: str, input_dim: int, **kwargs) -> TemporalStateModel:
    if architecture not in MODELS:
        raise ValueError(f"Unknown architecture: {architecture}. Choose from {list(MODELS.keys())}")
    return MODELS[architecture](input_dim=input_dim, **kwargs)
