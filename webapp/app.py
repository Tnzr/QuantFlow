"""QuantFlow WebApp — Flask application factory."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import firebase_admin
from firebase_admin import credentials as firebase_creds
from flask import Flask
from flask_login import LoginManager

_proj_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_proj_root))

from webapp.config import Config

logger = logging.getLogger(__name__)
login_manager = LoginManager()


def _init_firebase(app: Flask) -> None:
    if not Config.FIREBASE_AUTH_ENABLED:
        return
    creds_path = Config.FIREBASE_CREDENTIALS_PATH
    if not creds_path or not Path(creds_path).exists():
        logger.warning("Firebase auth enabled but credentials not found")
        return
    try:
        cred = firebase_creds.Certificate(creds_path)
        firebase_admin.initialize_app(cred)
        logger.info("Firebase initialized")
    except Exception as exc:
        logger.error(f"Firebase init failed: {exc}")


def create_app(config_class=Config) -> Flask:
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
        static_url_path="/static",
    )
    app.config.from_object(config_class)
    app.secret_key = Config.SECRET_KEY

    login_manager.init_app(app)
    login_manager.login_view = "auth.login"
    login_manager.login_message_category = "info"
    login_manager.session_protection = "strong"

    _init_firebase(app)

    from webapp.services.auth_service import User
    @login_manager.user_loader
    def load_user(user_id: str):
        return User.get(user_id)

    from webapp.blueprints.auth_bp import auth_bp
    from webapp.blueprints.dashboard_bp import dashboard_bp
    from webapp.blueprints.research_bp import research_bp
    from webapp.blueprints.portfolio_bp import portfolio_bp
    from webapp.blueprints.backtest_bp import backtest_bp
    from webapp.blueprints.forecast_bp import forecast_bp
    from webapp.blueprints.agent_bp import agent_bp
    from webapp.blueprints.settings_bp import settings_bp
    from webapp.blueprints.api_bp import api_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(research_bp)
    app.register_blueprint(portfolio_bp)
    app.register_blueprint(backtest_bp)
    app.register_blueprint(forecast_bp)
    app.register_blueprint(agent_bp)
    app.register_blueprint(settings_bp)
    app.register_blueprint(api_bp)

    from webapp.services.stripe_service import stripe_bp_for_webhook
    app.register_blueprint(stripe_bp_for_webhook)

    @app.context_processor
    def inject_globals():
        from flask_login import current_user
        return {
            "app_name": Config.APP_NAME,
            "app_version": Config.APP_VERSION,
            "current_user": current_user,
            "stripe_publishable_key": Config.STRIPE_PUBLISHABLE_KEY,
            "firebase_api_key": Config.FIREBASE_API_KEY,
            "firebase_auth_domain": Config.FIREBASE_AUTH_DOMAIN,
            "firebase_project_id": Config.FIREBASE_PROJECT_ID,
            "tiers": Config.SUBSCRIPTION_TIERS,
        }

    @app.errorhandler(404)
    def not_found(e):
        from flask import render_template
        return render_template("errors/404.html"), 404

    @app.errorhandler(500)
    def server_error(e):
        from flask import render_template
        return render_template("errors/500.html"), 500

    return app
