"""Stripe subscription service with webhook handling."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Optional

import stripe
from flask import Blueprint, jsonify, request

from webapp.config import Config

logger = logging.getLogger(__name__)

stripe.api_key = Config.STRIPE_SECRET_KEY
stripe_bp_for_webhook = Blueprint("stripe_webhook", __name__)


def create_checkout_session(
    customer_email: str,
    price_id: str,
    user_id: str,
    success_url: str = "",
    cancel_url: str = "",
) -> Optional[str]:
    try:
        session_data = {
            "payment_method_types": ["card"],
            "line_items": [{"price": price_id, "quantity": 1}],
            "mode": "subscription",
            "success_url": success_url or f"{Config.API_BASE_URL}/settings/billing?session_id={{CHECKOUT_SESSION_ID}}",
            "cancel_url": cancel_url or f"{Config.API_BASE_URL}/pricing",
            "client_reference_id": user_id,
            "customer_email": customer_email,
        }
        checkout = stripe.checkout.Session.create(**session_data)
        return checkout.url
    except Exception as exc:
        logger.error(f"Stripe checkout session failed: {exc}")
        return None


def create_customer_portal_session(customer_id: str) -> Optional[str]:
    try:
        portal = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=f"{Config.API_BASE_URL}/settings/billing",
        )
        return portal.url
    except Exception as exc:
        logger.error(f"Stripe portal session failed: {exc}")
        return None


def get_subscription_status(customer_id: str) -> dict:
    try:
        subs = stripe.Subscription.list(customer=customer_id, limit=1, status="all")
        if subs.data:
            sub = subs.data[0]
            return {
                "status": sub.status,
                "current_period_end": datetime.fromtimestamp(sub.current_period_end, tz=timezone.utc).isoformat(),
                "plan": sub.items.data[0].price.nickname if sub.items.data else "unknown",
                "cancel_at_period_end": sub.cancel_at_period_end,
            }
        return {"status": "inactive"}
    except Exception as exc:
        logger.error(f"Stripe subscription status failed: {exc}")
        return {"status": "error", "error": str(exc)}


@stripe_bp_for_webhook.route("/api/stripe/webhook", methods=["POST"])
def stripe_webhook():
    payload = request.get_data(as_text=True)
    sig_header = request.headers.get("Stripe-Signature", "")
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, Config.STRIPE_WEBHOOK_SECRET,
        )
    except ValueError:
        return jsonify({"error": "invalid payload"}), 400
    except stripe.error.SignatureVerificationError:
        return jsonify({"error": "invalid signature"}), 400

    if event["type"] == "checkout.session.completed":
        _handle_checkout_completed(event["data"]["object"])
    elif event["type"] == "customer.subscription.updated":
        _handle_subscription_updated(event["data"]["object"])
    elif event["type"] == "customer.subscription.deleted":
        _handle_subscription_deleted(event["data"]["object"])

    return jsonify({"status": "ok"})


def _handle_checkout_completed(session: dict) -> None:
    user_id = session.get("client_reference_id")
    customer_id = session.get("customer")
    subscription_id = session.get("subscription")
    if not user_id:
        return

    try:
        line_items = stripe.checkout.Session.list_line_items(session["id"])
        price_id = line_items.data[0].price.id if line_items.data else ""
        tier = "free"
        if price_id == Config.STRIPE_PRO_PRICE_ID:
            tier = "pro"
        elif price_id == Config.STRIPE_ENTERPRISE_PRICE_ID:
            tier = "enterprise"

        from webapp.services.auth_service import User
        User.update_tier(user_id, tier, customer_id, subscription_id)
        logger.info(f"User {user_id} upgraded to {tier}")
    except Exception as exc:
        logger.error(f"Checkout handler failed: {exc}")


def _handle_subscription_updated(subscription: dict) -> None:
    customer_id = subscription.get("customer")
    status = subscription.get("status", "")
    if status == "active":
        logger.info(f"Subscription active for customer {customer_id}")
    elif status == "past_due":
        logger.warning(f"Subscription past due for customer {customer_id}")


def _handle_subscription_deleted(subscription: dict) -> None:
    customer_id = subscription.get("customer")
    try:
        from webapp.services.auth_service import User
        import sqlite3
        conn = sqlite3.connect(Config.DATABASE_PATH)
        conn.execute(
            "UPDATE users SET subscription_tier='free', stripe_subscription_id=NULL WHERE stripe_customer_id=?",
            (customer_id,),
        )
        conn.commit()
        conn.close()
        logger.info(f"Subscription deleted, downgraded customer {customer_id} to free")
    except Exception as exc:
        logger.error(f"Subscription deletion handler failed: {exc}")
