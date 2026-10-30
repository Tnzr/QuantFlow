"""Optional DeepSeek reviewer. Bounded, advisory, and never used inside backtests
(LLM output is not point-in-time reproducible)."""
from __future__ import annotations

import json
import os
from typing import Optional

import requests

DEEPSEEK_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com") + "/chat/completions"
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
MAX_ADJUSTMENT = 0.10

SYSTEM_PROMPT = (
    "You review a deterministic quant signal. Reply with JSON only: "
    '{"adjustment": float in [-0.1, 0.1], "risks": [str], "rationale": str}. '
    "Positive adjustment raises confidence. Be conservative; prefer 0 when evidence is thin."
)


def deepseek_enabled() -> bool:
    return bool(os.getenv("DEEPSEEK_API_KEY"))


def review_signal(state: dict, timeout: float = 20.0) -> Optional[dict]:
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        return None
    summary = {k: state.get(k) for k in ("ticker", "technical", "seasonal", "sentiment", "gate", "volatility")}
    try:
        r = requests.post(
            DEEPSEEK_URL,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": DEEPSEEK_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(summary, default=str)},
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0,
            },
            timeout=timeout,
        )
        r.raise_for_status()
        content = json.loads(r.json()["choices"][0]["message"]["content"])
        adj = float(content.get("adjustment", 0.0))
        content["adjustment"] = max(-MAX_ADJUSTMENT, min(MAX_ADJUSTMENT, adj))
        content["model"] = DEEPSEEK_MODEL
        return content
    except Exception as exc:
        return {"adjustment": 0.0, "error": str(exc)[:200], "model": DEEPSEEK_MODEL}
