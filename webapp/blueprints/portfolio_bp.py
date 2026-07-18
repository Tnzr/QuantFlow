"""Portfolio blueprint — positions, allocation, optimization."""
from __future__ import annotations

from flask import Blueprint, render_template, request, jsonify
from flask_login import login_required, current_user

portfolio_bp = Blueprint("portfolio", __name__)


@portfolio_bp.route("/portfolio")
@login_required
def index():
    return render_template("portfolio/index.html", active_page="portfolio")


@portfolio_bp.route("/portfolio/allocate")
@login_required
def allocate():
    return render_template("portfolio/allocate.html")


@portfolio_bp.route("/portfolio/rebalance", methods=["POST"])
@login_required
def rebalance():
    return jsonify({"status": "ok", "message": "Rebalance queued"})
