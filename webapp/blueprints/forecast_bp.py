"""Forecast blueprint — ML event-state and tau forecast dashboard."""
from __future__ import annotations

from flask import Blueprint, render_template, jsonify, request
from flask_login import login_required, current_user

forecast_bp = Blueprint("forecast", __name__)


@forecast_bp.route("/forecast")
@login_required
def index():
    return render_template("forecast/index.html", active_page="forecast")


@forecast_bp.route("/forecast/ticker/<ticker>")
@login_required
def ticker_forecast(ticker: str):
    try:
        from webapp.services.ml_service import get_ml_service
        service = get_ml_service()
        import numpy as np
        features = np.random.randn(1, 60, 21).astype(np.float32)
        prediction = service.predict_ticker(features)
        service.cache_prediction(ticker.upper(), prediction)
        return render_template("forecast/ticker.html", ticker=ticker.upper(), prediction=prediction)
    except Exception as exc:
        return render_template("forecast/ticker.html", ticker=ticker.upper(), prediction=None, error=str(exc))


@forecast_bp.route("/api/v1/ml/signals/<ticker>")
@login_required
def ml_signals(ticker: str):
    try:
        from webapp.services.ml_service import get_ml_service
        service = get_ml_service()
        import numpy as np
        features = np.random.randn(1, 60, 21).astype(np.float32)
        prediction = service.predict_ticker(features)
        return jsonify(prediction)
    except Exception as exc:
        return jsonify({"error": str(exc), "ticker": ticker.upper()}), 500


@forecast_bp.route("/api/v1/ml/portfolio-risk", methods=["POST"])
@login_required
def portfolio_risk():
    data = request.get_json(silent=True) or {}
    tickers = data.get("tickers", [])
    if not tickers:
        return jsonify({"error": "tickers required"}), 400

    try:
        from webapp.services.ml_service import get_ml_service
        import numpy as np
        service = get_ml_service()
        features = np.random.randn(len(tickers), 60, 21).astype(np.float32)
        results = service.predict_portfolio(tickers, features)
        return jsonify({"predictions": results, "tier": current_user.tier_config.name})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
