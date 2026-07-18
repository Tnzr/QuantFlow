"""Backtest blueprint — config, run, results."""
from __future__ import annotations

from flask import Blueprint, render_template, request, jsonify
from flask_login import login_required, current_user

backtest_bp = Blueprint("backtest", __name__)


@backtest_bp.route("/backtest")
@login_required
def index():
    return render_template("backtest/index.html", active_page="backtest")


@backtest_bp.route("/backtest/run", methods=["POST"])
@login_required
def run():
    return jsonify({"status": "queued", "job_id": "backtest_001"})
