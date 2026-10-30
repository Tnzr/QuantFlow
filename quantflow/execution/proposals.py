"""Ed25519-signed allocation proposals for downstream consumers (SolidarityFlow).

Signature covers the canonical JSON (sorted keys, compact separators) of every
field except ``signature`` and ``public_key``.
"""
from __future__ import annotations

import base64
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from .risk_guards import MAX_RISK_PCT, MAX_SLIPPAGE_PCT

DEFAULT_KEY_FILE = Path(os.getenv("QUANTFLOW_ED25519_KEY_FILE", str(Path.home() / ".quantflow" / "ed25519.key")))
SCHEMA_VERSION = "quantflow.allocation_proposal.v1"
UNSIGNED_FIELDS = ("signature", "public_key")


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode()


def load_signing_key(key_file: Optional[Path] = None, create: bool = True) -> Ed25519PrivateKey:
    """Key from ``QUANTFLOW_ED25519_PRIVATE_KEY`` (base64 raw 32 bytes) or a local file."""
    raw = os.getenv("QUANTFLOW_ED25519_PRIVATE_KEY")
    if raw:
        return Ed25519PrivateKey.from_private_bytes(base64.b64decode(raw))
    path = Path(key_file or DEFAULT_KEY_FILE)
    if path.exists():
        return Ed25519PrivateKey.from_private_bytes(base64.b64decode(path.read_text().strip()))
    if not create:
        raise FileNotFoundError(f"no Ed25519 key at {path}")
    key = Ed25519PrivateKey.generate()
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_bytes = key.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption())
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(_b64(raw_bytes))
    try:
        os.link(tmp, path)  # atomic publish of a complete file; fails if another worker won
    except FileExistsError:
        return Ed25519PrivateKey.from_private_bytes(base64.b64decode(path.read_text().strip()))
    finally:
        tmp.unlink(missing_ok=True)
    return key


def public_key_b64(key: Ed25519PrivateKey) -> str:
    return _b64(key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw))


def canonical_bytes(proposal: dict) -> bytes:
    body = {k: v for k, v in proposal.items() if k not in UNSIGNED_FIELDS}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), default=str).encode()


def build_proposal(
    ticker: str,
    structure: str,
    legs: list[dict],
    *,
    max_risk_pct: float = MAX_RISK_PCT,
    slippage_cap_pct: float = MAX_SLIPPAGE_PCT,
    confidence: Optional[float] = None,
    thesis: str = "",
    scenario: Optional[str] = None,
    max_loss: Optional[float] = None,
    expires_at: Optional[str] = None,
    evidence: Optional[dict] = None,
) -> dict:
    if max_risk_pct > MAX_RISK_PCT or slippage_cap_pct > MAX_SLIPPAGE_PCT:
        raise ValueError("proposal caps exceed configured risk guards")
    return {
        "schema": SCHEMA_VERSION,
        "proposal_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ticker": ticker.upper(),
        "structure": structure,
        "legs": legs,
        "max_risk_pct": max_risk_pct,
        "slippage_cap_pct": slippage_cap_pct,
        "max_loss": max_loss,
        "confidence": confidence,
        "thesis": thesis,
        "scenario": scenario,
        "expires_at": expires_at,
        "broker_mode": "alpaca_paper",
        "evidence": evidence or {},
    }


def sign_proposal(proposal: dict, key: Optional[Ed25519PrivateKey] = None) -> dict:
    key = key or load_signing_key()
    signed = dict(proposal)
    signed["signature"] = _b64(key.sign(canonical_bytes(proposal)))
    signed["public_key"] = public_key_b64(key)
    return signed


def trusted_public_keys() -> list[str]:
    """Keys from ``QUANTFLOW_TRUSTED_ED25519_PUBKEYS`` (comma-separated base64) plus this host's key."""
    keys = [k.strip() for k in os.getenv("QUANTFLOW_TRUSTED_ED25519_PUBKEYS", "").split(",") if k.strip()]
    try:
        keys.append(public_key_b64(load_signing_key(create=False)))
    except (FileNotFoundError, ValueError):
        pass
    return keys


def verify_proposal(proposal: dict, public_key: Optional[str] = None) -> bool:
    """Verify against ``public_key`` or the trusted keys; the embedded ``public_key`` is ignored."""
    if not proposal.get("signature"):
        return False
    msg = canonical_bytes(proposal)
    for pk in [public_key] if public_key else trusted_public_keys():
        try:
            Ed25519PublicKey.from_public_bytes(base64.b64decode(pk)).verify(base64.b64decode(proposal["signature"]), msg)
            return True
        except (InvalidSignature, ValueError):
            continue
    return False
