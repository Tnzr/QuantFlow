from __future__ import annotations

import os
from typing import Optional, Any

from fastapi import Header, HTTPException


def _auth_disabled() -> bool:
    return os.getenv("QF_API_AUTH_ENABLED", "false").lower() not in {"1", "true", "yes"}


def _token_check(authorization: Optional[str], x_api_key: Optional[str]) -> bool:
    expected = os.getenv("QF_API_TOKEN", "").strip()
    if not expected:
        return False
    if x_api_key and x_api_key.strip() == expected:
        return True
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
        return token == expected
    return False


def _firebase_enabled() -> bool:
    return os.getenv("QF_FIREBASE_AUTH_ENABLED", "false").lower() in {"1", "true", "yes"}


def _extract_bearer(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    if not authorization.lower().startswith("bearer "):
        return None
    token = authorization[7:].strip()
    return token or None


def _firebase_claims(authorization: Optional[str]) -> Optional[dict[str, Any]]:
    # Optional Firebase check. Requires firebase-admin to be installed/configured.
    if not _firebase_enabled():
        return None
    token = _extract_bearer(authorization)
    if not token:
        return None

    try:
        import firebase_admin
        from firebase_admin import auth, credentials

        if not firebase_admin._apps:
            cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
            if cred_path:
                firebase_admin.initialize_app(credentials.Certificate(cred_path))
            else:
                firebase_admin.initialize_app()

        claims = auth.verify_id_token(token)
        return claims if isinstance(claims, dict) else None
    except Exception:
        return None


def _has_firebase_write_access(claims: dict[str, Any]) -> bool:
    required_claim_key = os.getenv("QF_FIREBASE_WRITE_CLAIM", "quantflow_write").strip()
    claim_val = claims.get(required_claim_key)
    if claim_val is True:
        return True
    if isinstance(claim_val, str) and claim_val.lower() in {"1", "true", "yes", "writer", "admin"}:
        return True

    roles_csv = os.getenv("QF_FIREBASE_WRITE_ROLES", "writer,admin")
    allowed_roles = {r.strip().lower() for r in roles_csv.split(",") if r.strip()}
    roles = claims.get("roles")
    if isinstance(roles, str):
        role_values = {roles.lower()}
    elif isinstance(roles, list):
        role_values = {str(r).lower() for r in roles}
    else:
        role_values = set()

    return bool(allowed_roles.intersection(role_values))


def require_write_auth(
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    """Protect mutating endpoints when auth is enabled.

    Modes:
    - disabled: allow all (default for local dev)
    - token mode: set QF_API_AUTH_ENABLED=true and QF_API_TOKEN
    - firebase mode: set QF_API_AUTH_ENABLED=true and QF_FIREBASE_AUTH_ENABLED=true
    """
    if _auth_disabled():
        return {"mode": "disabled"}

    if _token_check(authorization=authorization, x_api_key=x_api_key):
        return {"mode": "token"}

    claims = _firebase_claims(authorization=authorization)
    if claims is not None:
        if _has_firebase_write_access(claims):
            return {"mode": "firebase", "claims": claims}
        raise HTTPException(status_code=403, detail="Forbidden: missing write role/claim")

    raise HTTPException(status_code=401, detail="Unauthorized")


def require_firebase_user(
    authorization: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    """Require a valid Firebase bearer token and return user claims.

    This dependency intentionally enforces Google/Firebase auth only and does not
    allow API key fallback, making it suitable for per-user settings endpoints.
    """
    if not _firebase_enabled():
        raise HTTPException(status_code=503, detail="Firebase auth is disabled on this server")

    claims = _firebase_claims(authorization=authorization)
    if claims is None:
        raise HTTPException(status_code=401, detail="Unauthorized: valid Firebase bearer token required")

    uid = str(claims.get("uid") or claims.get("user_id") or claims.get("sub") or "").strip()
    if not uid:
        raise HTTPException(status_code=401, detail="Unauthorized: Firebase token is missing user identity")

    return {
        "mode": "firebase",
        "uid": uid,
        "claims": claims,
    }
