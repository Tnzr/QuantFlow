"""Dashboard blueprint — main overview page."""
from __future__ import annotations

from flask import Blueprint, render_template
from flask_login import login_required, current_user

dashboard_bp = Blueprint("dashboard", __name__)


@dashboard_bp.route("/dashboard")
@login_required
def index():
    return render_template("dashboard.html", active_page="dashboard")
