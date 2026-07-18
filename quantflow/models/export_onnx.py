"""Export trained PyTorch model to ONNX format for production inference."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quantflow.models.architectures import create_model
from quantflow.models.dataset import FEATURE_COLUMNS


def export_to_onnx(
    checkpoint_path: str,
    output_path: str,
    architecture: str = "bilstm",
    hidden_dim: int = 128,
    lookback: int = 60,
    feature_dim: int = 21,
    forecast_horizon: int = 21,
    opset_version: int = 17,
):
    model = create_model(
        architecture=architecture,
        input_dim=feature_dim,
        hidden_dim=hidden_dim,
        forecast_horizon=forecast_horizon,
    )
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval()

    dummy_input = torch.randn(1, lookback, feature_dim)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["features"],
        output_names=["past_state_probs", "future_forecast", "uncertainty_sigma"],
        dynamic_axes={
            "features": {0: "batch_size"},
            "past_state_probs": {0: "batch_size"},
            "future_forecast": {0: "batch_size"},
            "uncertainty_sigma": {0: "batch_size"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )

    print(f"Exported ONNX model to {output_path}")
    print(f"  Architecture: {architecture}")
    print(f"  Input shape:  (batch, {lookback}, {feature_dim})")
    print(f"  Outputs:      past_state_probs (batch, 3), future_forecast (batch, 1), uncertainty_sigma (batch, 1)")

    import onnxruntime as ort
    session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    outputs = session.run(None, {"features": dummy_input.numpy().astype("float32")})
    print(f"  ONNX test:    passthrough OK ({len(outputs)} outputs)")
    for name, out in zip(["past_state_probs", "future_forecast", "uncertainty_sigma"], outputs):
        print(f"    {name}: shape {out.shape}")


def main():
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--output", required=True, help="Output .onnx path")
    parser.add_argument("--arch", default="bilstm", choices=["bilstm", "transformer", "tcn"])
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--forecast-horizon", type=int, default=21)
    args = parser.parse_args()

    export_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        architecture=args.arch,
        hidden_dim=args.hidden_dim,
        lookback=args.lookback,
        feature_dim=len(FEATURE_COLUMNS),
        forecast_horizon=args.forecast_horizon,
    )


if __name__ == "__main__":
    main()
