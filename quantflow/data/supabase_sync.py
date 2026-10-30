"""Best-effort mirror of election/agent/proposal rows into Supabase (PostgREST).

Configure ``SUPABASE_URL`` and ``SUPABASE_SERVICE_ROLE_KEY``; schema lives in
``supabase/migrations``. Local SQLite stays the source of truth, so a missing or
failing Supabase never blocks the pipeline.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Iterable

import requests

TABLES = {"election_events", "sentiment_snapshots", "agent_state_snapshots", "allocation_proposals"}


def supabase_enabled() -> bool:
    return bool(os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SERVICE_ROLE_KEY"))


def _jsonable(row: dict) -> dict:
    return {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in row.items()}


def upsert_rows(table: str, rows: Iterable[dict], on_conflict: str | None = None, timeout: float = 15.0) -> tuple[bool, str]:
    if table not in TABLES:
        raise ValueError(f"unknown table {table}")
    rows = [_jsonable(r) for r in rows]
    if not rows:
        return True, "nothing to send"
    if not supabase_enabled():
        return False, "supabase not configured"
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    url = f"{os.environ['SUPABASE_URL'].rstrip('/')}/rest/v1/{table}"
    params = {"on_conflict": on_conflict} if on_conflict else None
    try:
        r = requests.post(
            url,
            params=params,
            headers={
                "apikey": key,
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "Prefer": "resolution=ignore-duplicates,return=minimal",
            },
            data=json.dumps(rows, default=str),
            timeout=timeout,
        )
        if r.status_code >= 300:
            return False, f"{r.status_code}: {r.text[:200]}"
        return True, f"sent {len(rows)}"
    except Exception as exc:
        return False, str(exc)[:200]


def publish_proposal(signed: dict) -> tuple[bool, str]:
    row = {
        "proposal_id": signed["proposal_id"],
        "created_at": signed["timestamp"],
        "ticker": signed["ticker"],
        "structure": signed["structure"],
        "payload": {k: v for k, v in signed.items() if k not in ("signature", "public_key")},
        "signature": signed["signature"],
        "public_key": signed["public_key"],
    }
    return upsert_rows("allocation_proposals", [row], on_conflict="proposal_id")
