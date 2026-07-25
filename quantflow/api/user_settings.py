from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any


class UserSettingsStoreError(RuntimeError):
    pass


def _get_firestore_client():
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore
    except Exception as exc:
        raise UserSettingsStoreError("firebase-admin is not installed") from exc

    try:
        if not firebase_admin._apps:
            cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
            if cred_path:
                firebase_admin.initialize_app(credentials.Certificate(cred_path))
            else:
                firebase_admin.initialize_app()
        return firestore.client()
    except Exception as exc:
        raise UserSettingsStoreError(f"Failed to initialize Firestore client: {exc}") from exc


def load_user_settings(uid: str) -> dict[str, Any]:
    db = _get_firestore_client()
    try:
        doc = db.collection("quantflow_user_settings").document(uid).get()
        if not doc.exists:
            return {
                "uid": uid,
                "settings": {},
                "created_at": None,
                "updated_at": None,
            }
        payload = doc.to_dict() or {}
        settings = payload.get("settings") or {}
        if not isinstance(settings, dict):
            settings = {}
        return {
            "uid": uid,
            "settings": settings,
            "created_at": payload.get("created_at"),
            "updated_at": payload.get("updated_at"),
        }
    except Exception as exc:
        raise UserSettingsStoreError(f"Failed to load user settings: {exc}") from exc


def save_user_settings(uid: str, settings: dict[str, Any]) -> dict[str, Any]:
    db = _get_firestore_client()
    now = datetime.now(timezone.utc).isoformat()
    try:
        ref = db.collection("quantflow_user_settings").document(uid)
        existing = ref.get()
        created_at = now
        if existing.exists:
            payload = existing.to_dict() or {}
            created_at = str(payload.get("created_at") or now)

        payload = {
            "uid": uid,
            "settings": settings,
            "created_at": created_at,
            "updated_at": now,
        }
        ref.set(payload, merge=True)
        return payload
    except Exception as exc:
        raise UserSettingsStoreError(f"Failed to save user settings: {exc}") from exc
