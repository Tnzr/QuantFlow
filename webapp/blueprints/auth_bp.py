"""Auth blueprint — login, register, logout, Firebase integration."""
from __future__ import annotations

from flask import Blueprint, redirect, render_template, request, url_for, flash, jsonify
from flask_login import login_user, logout_user, login_required, current_user

from webapp.config import Config
from webapp.services.auth_service import User

auth_bp = Blueprint("auth", __name__)


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard.index"))

    firebase = request.args.get("firebase") == "1" or request.is_json

    if firebase and request.method == "POST":
        data = request.get_json(silent=True) or {}
        id_token = data.get("id_token", "")
        decoded = User.verify_firebase_token(id_token)
        if decoded:
            user = User.create_or_update(
                decoded.get("uid", ""),
                decoded.get("email", ""),
                decoded.get("name", "") or decoded.get("email", ""),
            )
            if user:
                login_user(user, remember=True)
                return jsonify({"status": "ok", "redirect": url_for("dashboard.index")})
        return jsonify({"error": "authentication failed"}), 401

    if request.method == "POST":
        email = request.form.get("email", "")
        password = request.form.get("password", "")
        if email == "demo@quantflow.ai" and password == "demo123":
            user = User.create_or_update("demo_user", email, "Demo User")
            if user:
                login_user(user, remember=True)
                flash("Welcome back!", "success")
                return redirect(url_for("dashboard.index"))
        flash("Invalid credentials. Try demo@quantflow.ai / demo123", "error")

    return render_template("auth/login.html")


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard.index"))

    selected_tier = request.args.get("tier", "free")

    firebase = request.args.get("firebase") == "1" or request.is_json
    if firebase and request.method == "POST":
        data = request.get_json(silent=True) or {}
        id_token = data.get("id_token", "")
        tier = data.get("tier", "free")
        decoded = User.verify_firebase_token(id_token)
        if decoded:
            user = User.create_or_update(
                decoded.get("uid", ""),
                decoded.get("email", ""),
                decoded.get("name", "") or decoded.get("email", ""),
            )
            if user:
                if tier != "free":
                    user.subscription_tier = tier
                login_user(user, remember=True)
                if tier != "free":
                    from webapp.services.stripe_service import create_checkout_session
                    checkout_url = create_checkout_session(
                        user.email,
                        Config.SUBSCRIPTION_TIERS[tier].get("price_id", ""),
                        user.id,
                    )
                    if checkout_url:
                        return jsonify({"status": "ok", "redirect": checkout_url})
                return jsonify({"status": "ok", "redirect": url_for("dashboard.index")})
        return jsonify({"error": "registration failed"}), 400

    if request.method == "POST":
        email = request.form.get("email", "")
        display_name = request.form.get("display_name", email)
        tier = request.form.get("tier", "free")
        user = User.create_or_update(f"dev_{email}", email, display_name)
        if user:
            user.subscription_tier = tier
            login_user(user, remember=True)
            flash("Account created!", "success")
            if tier != "free":
                redirect_url = url_for("settings.billing")
                return redirect(redirect_url)
            return redirect(url_for("dashboard.index"))
        flash("Registration failed", "error")

    return render_template("auth/register.html", selected_tier=selected_tier)


@auth_bp.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Signed out successfully.", "info")
    return redirect(url_for("auth.login"))


@auth_bp.route("/firebase/auth", methods=["POST"])
def firebase_auth_token():
    data = request.get_json(silent=True) or {}
    id_token = data.get("id_token", "")
    decoded = User.verify_firebase_token(id_token)
    if decoded:
        user = User.create_or_update(
            decoded.get("uid", ""),
            decoded.get("email", ""),
            decoded.get("name", "") or decoded.get("email", ""),
        )
        if user:
            login_user(user, remember=True)
            return jsonify({"status": "ok", "user_id": user.id})
        return jsonify({"error": "user creation failed"}), 500
    return jsonify({"error": "invalid token"}), 401
