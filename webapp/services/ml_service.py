"""ML inference service using ONNX Runtime for production predictions."""
from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import onnxruntime as ort

from webapp.config import Config

logger = logging.getLogger(__name__)


class ONNXInferenceService:
    """Production ML inference using ONNX Runtime.

    Provides event-state classification and time-to-event forecasting
    with uncertainty estimates. Supports batched inference for multi-
    ticker portfolio analysis.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or Config.ONNX_MODEL_PATH
        self.session: Optional[ort.InferenceSession] = None
        self._init_session()

    def _init_session(self) -> None:
        if Path(self.model_path).exists():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if ort.get_device() == "GPU" else ["CPUExecutionProvider"]
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            logger.info(f"ONNX session initialized: {self.model_path}")
        else:
            logger.warning(f"ONNX model not found at {self.model_path}, using mock predictions")

    @property
    def is_ready(self) -> bool:
        return self.session is not None

    def predict(self, features_window: np.ndarray) -> Dict[str, np.ndarray]:
        """Run inference on a pre-built features window.

        Args:
            features_window: (batch_size, lookback, feature_dim) float32 array

        Returns:
            dict with keys: past_state_probs, future_forecast, uncertainty_sigma
        """
        if not self.is_ready:
            return self._mock_predict(features_window.shape[0])

        features = features_window.astype(np.float32)
        outputs = self.session.run(None, {"features": features})
        return {
            "past_state_probs": outputs[0],
            "future_forecast": outputs[1],
            "uncertainty_sigma": outputs[2],
        }

    def predict_ticker(self, features_window: np.ndarray) -> dict:
        """Run single-ticker inference and return human-readable results.

        Args:
            features_window: (1, lookback, feature_dim) float32 array

        Returns:
            dict with ticker-level forecast details
        """
        result = self.predict(features_window)
        probs = result["past_state_probs"][0]
        forecast = float(result["future_forecast"][0].mean())
        sigma = float(result["uncertainty_sigma"][0].mean())

        state_idx = int(np.argmax(probs))
        state_labels = {0: "inter_event", 1: "pre_event", 2: "onset"}

        return {
            "state": state_idx,
            "state_label": state_labels.get(state_idx, "unknown"),
            "prob_inter_event": float(probs[0]),
            "prob_pre_event": float(probs[1]),
            "prob_onset": float(probs[2]),
            "forecast_tau_days": round(forecast, 1),
            "uncertainty_sigma": round(sigma, 2),
            "confidence_band_low": round(max(0, forecast - 2 * sigma), 1),
            "confidence_band_high": round(forecast + 2 * sigma, 1),
        }

    def predict_portfolio(self, tickers: List[str], features_batch: np.ndarray) -> List[dict]:
        """Batch predict for multiple tickers simultaneously.

        Args:
            tickers: list of ticker symbols
            features_batch: (n_tickers, lookback, feature_dim) float32

        Returns:
            list of per-ticker prediction dicts
        """
        results = self.predict(features_batch)
        probs_all = results["past_state_probs"]
        forecasts = results["future_forecast"].mean(axis=1)
        sigmas = results["uncertainty_sigma"].mean(axis=1)

        outputs = []
        for i, ticker in enumerate(tickers):
            probs = probs_all[i]
            state_idx = int(np.argmax(probs))
            state_labels = {0: "inter_event", 1: "pre_event", 2: "onset"}
            risk_score = float(probs[1] + probs[2] * 1.5)

            outputs.append({
                "ticker": ticker,
                "state": state_idx,
                "state_label": state_labels.get(state_idx, "unknown"),
                "prob_inter_event": float(probs[0]),
                "prob_pre_event": float(probs[1]),
                "prob_onset": float(probs[2]),
                "forecast_tau_days": round(float(forecasts[i]), 1),
                "uncertainty_sigma": round(float(sigmas[i]), 2),
                "risk_score": round(risk_score, 3),
            })

        return sorted(outputs, key=lambda x: x["risk_score"], reverse=True)

    def _mock_predict(self, batch_size: int = 1) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(42)
        probs = rng.dirichlet([6.5, 2.5, 1.0], size=batch_size).astype(np.float32)
        return {
            "past_state_probs": probs,
            "future_forecast": rng.normal(18, 4, (batch_size, 1)).clip(1, 21).astype(np.float32),
            "uncertainty_sigma": rng.uniform(1.5, 5.0, (batch_size, 1)).astype(np.float32),
        }

    def cache_prediction(self, ticker: str, prediction: dict) -> None:
        """Store ML prediction in database cache."""
        try:
            import sqlite3
            conn = sqlite3.connect(Config.DATABASE_PATH)
            conn.execute("""INSERT OR REPLACE INTO ml_predictions
                (ticker, model_version_id, prediction_date, prob_inter_event,
                 prob_pre_event, prob_onset, forecast_tau_days, uncertainty_sigma)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    ticker, 1, date.today().isoformat(),
                    prediction["prob_inter_event"],
                    prediction["prob_pre_event"],
                    prediction["prob_onset"],
                    prediction["forecast_tau_days"],
                    prediction["uncertainty_sigma"],
                ),
            )
            conn.commit()
            conn.close()
        except Exception as exc:
            logger.error(f"Prediction cache failed: {exc}")


_ml_service: Optional[ONNXInferenceService] = None


def get_ml_service() -> ONNXInferenceService:
    global _ml_service
    if _ml_service is None:
        _ml_service = ONNXInferenceService()
    return _ml_service
