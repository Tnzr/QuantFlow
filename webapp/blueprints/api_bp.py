"""API blueprint — JSON endpoints for all core services."""
from __future__ import annotations

from flask import Blueprint, jsonify, request
from flask_login import login_required, current_user

api_bp = Blueprint("api", __name__, url_prefix="/api/v1")


@api_bp.route("/health")
def health():
    return jsonify({"status": "healthy", "version": "1.0.0-beta"})


@api_bp.route("/price/<ticker>")
@login_required
def price(ticker: str):
    return jsonify({"ticker": ticker.upper(), "price": 0.0, "timestamp": ""})


@api_bp.route("/data/universe")
def universe():
    try:
        from quantflow.data.universe import HIGH_INTEREST
        return jsonify({"tickers": sorted(list(HIGH_INTEREST)), "count": len(HIGH_INTEREST)})
    except Exception:
        return jsonify({"tickers": [], "count": 0})


@api_bp.route("/data/timeseries")
@login_required
def timeseries():
    return jsonify({"data": [], "ticker": ""})


@api_bp.route("/scan/results")
@login_required
def scan_results():
    return jsonify({"results": [], "preset": ""})


@api_bp.route("/recommendations")
@login_required
def recommendations():
    return jsonify({"recommendations": []})
