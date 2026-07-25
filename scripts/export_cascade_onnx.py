#!/usr/bin/env python3
"""Export CascadeANP direct-path sub-models to ONNX for portable inference.

The full CascadeANP model includes recurrent (LSTM/GRU) and autoregressive
decoder components that are not directly ONNX-exportable due to dynamic loops
and variable sequence lengths. Instead, this script exports:

1. direct_path   — 1D conv positional encoder + MLP → 21-step return forecast
2. direct_sigma  — MLP over cascade token samples → 21-step uncertainty

Combined, these provide the predicted_return and aleatoric_sigma that the
inference API already returns. Export targets ONNX opset 17 for broad
compatibility with tract-rs, ONNX Runtime, and other runtimes.

Usage:
    python scripts/export_cascade_onnx.py
    python scripts/export_cascade_onnx.py --checkpoint checkpoints/cascade_anp.pt
    python scripts/export_cascade_onnx.py --input-width 27 --output checkpoints/cascade_anp.onnx
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantflow.models.cascade_anp import CascadeANP  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
REGISTRY_PATH = CHECKPOINTS_DIR / "model_registry.json"


def export_direct_path(model: CascadeANP, hidden_dim: int, input_width: int, output_path: Path):
    """Export the direct-path head: raw features → 1D conv → 30 position tokens → MLP → 21 returns."""
    class DirectPathWrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.direct_conv = model.direct_conv
            self.direct_path = model.direct_path

        def forward(self, x: torch.Tensor):
            x = x.permute(0, 2, 1)
            conv_out = self.direct_conv(x)
            feat_sampled = conv_out.reshape(x.shape[0], -1)
            return self.direct_path(feat_sampled)

    wrapper = DirectPathWrapper().eval()
    dummy = torch.randn(1, 80, input_width)

    torch.onnx.export(
        wrapper, dummy, output_path,
        input_names=["raw_features"],
        output_names=["predicted_returns_21"],
        dynamic_axes={"raw_features": {0: "batch", 1: "sequence"}},
        opset_version=17,
        do_constant_folding=True,
        verbose=False,
    )
    print(f"  direct_path exported → {output_path}")


def export_direct_sigma(model: CascadeANP, hidden_dim: int, output_path: Path):
    """Export the direct-sigma head: cascade token samples → MLP → 21 sigmas."""
    class DirectSigmaWrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.direct_sigma = model.direct_sigma

        def forward(self, x: torch.Tensor):
            sigma_raw = self.direct_sigma(x)
            return torch.nn.functional.softplus(sigma_raw) + 1e-4

    wrapper = DirectSigmaWrapper().eval()
    dummy = torch.randn(1, hidden_dim * 30)

    onnx_path = output_path.with_name(output_path.stem + "_sigma" + output_path.suffix)
    torch.onnx.export(
        wrapper, dummy, onnx_path,
        input_names=["cascade_tokens_30"],
        output_names=["aleatoric_sigma_21"],
        dynamic_axes={"cascade_tokens_30": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
        verbose=False,
    )
    print(f"  direct_sigma exported → {onnx_path}")


def create_model_registry_entry(
    checkpoint_path: str,
    val_loss: float,
    epoch: int,
    dataset_hash: str,
    training_date: str,
    hidden_dim: int,
    input_width: int,
    onnx_paths: list[str],
) -> dict:
    model_id = f"cascade_anp_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    return {
        "model_id": model_id,
        "checkpoint": str(checkpoint_path),
        "epoch": epoch,
        "val_loss": round(val_loss, 6),
        "training_date": training_date,
        "dataset_hash": dataset_hash,
        "architecture": {
            "name": "CascadeANP",
            "hidden_dim": hidden_dim,
            "latent_dim": 128,
            "cascade_layers": 4,
            "input_width": input_width,
            "output_steps": 21,
        },
        "onnx_exports": onnx_paths,
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }


def load_registry() -> dict:
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    return {"models": [], "updated_at": None}


def save_registry(registry: dict):
    registry["updated_at"] = datetime.now(timezone.utc).isoformat()
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"  registry updated → {REGISTRY_PATH}")


def compute_dataset_hash(data_path: str) -> str:
    import hashlib
    path = Path(data_path)
    if not path.exists():
        return "unknown"
    h = hashlib.sha256()
    h.update(str(path.stat().st_size).encode())
    h.update(str(path.stat().st_mtime).encode())
    return h.hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser(description="Export CascadeANP to ONNX")
    parser.add_argument("--checkpoint", default="checkpoints/cascade_anp.pt",
                        help="Path to the CascadeANP PyTorch checkpoint")
    parser.add_argument("--data", default="data/alpaca_5m.parquet",
                        help="Path to the training dataset (for hash)")
    parser.add_argument("--input-width", type=int, default=27,
                        help="Number of input features")
    parser.add_argument("--hidden-dim", type=int, default=128,
                        help="Hidden dimension")
    parser.add_argument("--output", default="checkpoints/cascade_anp.onnx",
                        help="Output ONNX file path")
    parser.add_argument("--registry-only", action="store_true",
                        help="Only update registry, skip ONNX export")
    args = parser.parse_args()

    checkpoint_path = PROJECT_ROOT / args.checkpoint

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})

    hidden_dim = args.hidden_dim or cfg.get("hidden_dim", 128)
    input_width = args.input_width or cfg.get("input_dim", 27)
    dataset_hash = compute_dataset_hash(str(PROJECT_ROOT / args.data))

    print(f"Config: hidden_dim={hidden_dim}, input_width={input_width}")
    print(f"Epoch: {ckpt.get('epoch', '?')}, val_loss: {ckpt.get('val_loss', '?'):.6f}")

    model = CascadeANP(input_width, hidden_dim,
                       latent_dim=cfg.get("latent_dim", 128),
                       cascade_layers=cfg.get("cascade_layers", 4))
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    onnx_paths = []
    if not args.registry_only:
        output_path = PROJECT_ROOT / args.output
        try:
            export_direct_path(model, hidden_dim, input_width, output_path)
            onnx_paths.append(str(output_path))
        except Exception as e:
            print(f"  WARNING: direct_path export failed: {e}")

        sigma_path = output_path.with_name(output_path.stem + "_sigma" + output_path.suffix)
        try:
            export_direct_sigma(model, hidden_dim, sigma_path)
            onnx_paths.append(str(sigma_path))
        except Exception as e:
            print(f"  WARNING: direct_sigma export failed: {e}")

    entry = create_model_registry_entry(
        checkpoint_path=str(checkpoint_path),
        val_loss=float(ckpt.get("val_loss", 0.0)),
        epoch=int(ckpt.get("epoch", 0)),
        dataset_hash=dataset_hash,
        training_date=ckpt.get("training_date", "unknown"),
        hidden_dim=hidden_dim,
        input_width=input_width,
        onnx_paths=onnx_paths,
    )

    registry = load_registry()
    registry.setdefault("models", [])
    existing = [m for m in registry["models"] if m.get("model_id") != entry["model_id"]]
    existing.append(entry)
    registry["models"] = existing[-10:]
    save_registry(registry)

    print("\nDone.")


if __name__ == "__main__":
    main()
