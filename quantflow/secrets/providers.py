from __future__ import annotations

from typing import Dict
import json

from ..broker.secret_manager import resolve_secret, SecretResolutionError


class SecretProviderError(RuntimeError):
    pass


def resolve_secret_refs(
    provider_name: str,
    refs: Dict[str, str],
    provider_options: Dict[str, str] | None = None,
) -> Dict[str, str]:
    del provider_options  # Reserved for future provider-specific runtime options.

    out: Dict[str, str] = {}
    for key, ref in (refs or {}).items():
        k = str(key)
        r = str(ref)
        if not r:
            continue
        try:
            out[k] = resolve_secret(provider_name, r)
        except SecretResolutionError as e:
            raise SecretProviderError(f"{provider_name} secret resolution failed for {k}: {e}") from e

    # Normalize JSON-string headers payload if provided as a dict-encoded secret.
    if "custom_headers" in out:
        payload = out.get("custom_headers", "")
        try:
            parsed = json.loads(str(payload))
            if isinstance(parsed, dict):
                out["custom_headers"] = json.dumps({str(k): str(v) for k, v in parsed.items()})
        except Exception:
            pass

    return out
