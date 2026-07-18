"""Research blueprint — scanner, recommendations, ticker deep-dive, landing page."""
from __future__ import annotations

from flask import Blueprint, render_template, request, jsonify, redirect, url_for
from flask_login import login_required, current_user

research_bp = Blueprint("research", __name__)


@research_bp.route("/")
def landing():
    if current_user and current_user.is_authenticated:
        return redirect(url_for("dashboard.index"))
    return render_template("landing.html")


@research_bp.route("/research")
@login_required
def index():
    return render_template("research/index.html", active_page="research")


@research_bp.route("/research/ticker/<ticker>")
@login_required
def ticker_detail(ticker: str):
    return render_template("research/ticker.html", ticker=ticker.upper())


@research_bp.route("/research/scan")
@login_required
def scan():
    return render_template("research/scanner.html")


@research_bp.route("/research/recommendations")
@login_required
def recommendations():
    return render_template("research/recommendations.html")


@research_bp.route("/pricing")
def pricing():
    return render_template("pricing.html")


@research_bp.route("/docs")
def docs():
    return render_template("docs.html")
