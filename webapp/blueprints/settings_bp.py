"""Settings blueprint — profile, billing, broker config."""
from __future__ import annotations

from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_required, current_user

from webapp.services.stripe_service import create_customer_portal_session

settings_bp = Blueprint("settings", __name__)


@settings_bp.route("/settings")
@login_required
def index():
    return render_template("settings/profile.html", active_page="settings")


@settings_bp.route("/settings", methods=["POST"])
@login_required
def update():
    return redirect(url_for("settings.index"))


@settings_bp.route("/settings/billing")
@login_required
def billing():
    portal_url = None
    if current_user.stripe_customer_id:
        portal_url = create_customer_portal_session(current_user.stripe_customer_id)

    sub_status = {"status": "inactive"}
    if current_user.stripe_customer_id:
        from webapp.services.stripe_service import get_subscription_status
        sub_status = get_subscription_status(current_user.stripe_customer_id)

    return render_template(
        "settings/billing.html",
        active_page="settings",
        portal_url=portal_url,
        sub_status=sub_status,
    )
