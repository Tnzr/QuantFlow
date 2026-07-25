from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import json
import os
import uuid
from threading import Lock
from urllib import request, error

from ..secrets.providers import SecretProviderError, resolve_secret_refs


_RUNTIME_AUTH_CONFIG: Dict[str, Any] = {}
_RUNTIME_AUTH_LOCK = Lock()


def set_runtime_mcp_auth_config(config: Dict[str, Any]) -> None:
    clean: Dict[str, Any] = {}
    if "auth_header" in config and isinstance(config.get("auth_header"), str):
        clean["auth_header"] = config["auth_header"].strip() or "Authorization"
    if "bearer_token" in config and isinstance(config.get("bearer_token"), str):
        clean["bearer_token"] = config["bearer_token"].strip()
    if "api_key_header" in config and isinstance(config.get("api_key_header"), str):
        clean["api_key_header"] = config["api_key_header"].strip() or "X-API-Key"
    if "api_key" in config and isinstance(config.get("api_key"), str):
        clean["api_key"] = config["api_key"].strip()
    if "cookie" in config and isinstance(config.get("cookie"), str):
        clean["cookie"] = config["cookie"].strip()
    if "custom_headers" in config and isinstance(config.get("custom_headers"), dict):
        clean["custom_headers"] = {str(k): str(v) for k, v in dict(config["custom_headers"]).items()}
    if "secret_provider" in config and isinstance(config.get("secret_provider"), str):
        clean["secret_provider"] = config["secret_provider"].strip().lower()
    if "secret_refs" in config and isinstance(config.get("secret_refs"), dict):
        clean["secret_refs"] = {str(k): str(v) for k, v in dict(config["secret_refs"]).items()}
    if "provider_options" in config and isinstance(config.get("provider_options"), dict):
        clean["provider_options"] = {str(k): str(v) for k, v in dict(config["provider_options"]).items()}

    with _RUNTIME_AUTH_LOCK:
        _RUNTIME_AUTH_CONFIG.update(clean)


def clear_runtime_mcp_auth_config() -> None:
    with _RUNTIME_AUTH_LOCK:
        _RUNTIME_AUTH_CONFIG.clear()


def _runtime_mcp_auth_config() -> Dict[str, Any]:
    with _RUNTIME_AUTH_LOCK:
        return dict(_RUNTIME_AUTH_CONFIG)


def get_runtime_mcp_auth_config_summary() -> Dict[str, Any]:
    cfg = _runtime_mcp_auth_config()
    return {
        "configured": bool(
            cfg.get("bearer_token")
            or cfg.get("api_key")
            or cfg.get("cookie")
            or cfg.get("custom_headers")
            or cfg.get("secret_provider")
            or cfg.get("secret_refs")
        ),
        "auth_header": cfg.get("auth_header") or "Authorization",
        "has_bearer": bool(cfg.get("bearer_token")),
        "has_api_key": bool(cfg.get("api_key")),
        "api_key_header": cfg.get("api_key_header") or "X-API-Key",
        "has_cookie": bool(cfg.get("cookie")),
        "custom_header_keys": sorted(list((cfg.get("custom_headers") or {}).keys())),
        "secret_provider": cfg.get("secret_provider") or None,
        "secret_ref_keys": sorted(list((cfg.get("secret_refs") or {}).keys())),
    }


def _parse_custom_headers_json(payload: str) -> Dict[str, str]:
    try:
        data = json.loads(payload)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except Exception:
        return {}
    return {}


def _env_secret_refs() -> Dict[str, str]:
    refs: Dict[str, str] = {}
    direct_map = {
        "bearer_token": "QF_MCP_BEARER_TOKEN_SECRET_REF",
        "api_key": "QF_MCP_API_KEY_SECRET_REF",
        "cookie": "QF_MCP_COOKIE_SECRET_REF",
        "custom_headers": "QF_MCP_HEADERS_SECRET_REF",
    }
    for key, env_key in direct_map.items():
        val = os.getenv(env_key, "").strip()
        if val:
            refs[key] = val

    refs_json = os.getenv("QF_MCP_SECRET_REFS_JSON", "").strip()
    if refs_json:
        try:
            data = json.loads(refs_json)
            if isinstance(data, dict):
                for key, value in data.items():
                    refs[str(key)] = str(value)
        except Exception:
            pass

    return refs


@dataclass
class MCPClientConfig:
    endpoint: str = "https://agent.robinhood.com/mcp/trading"
    transport: str = "streamable_http"
    timeout_seconds: int = 10
    payload_mode: str = "jsonrpc"
    auth_header: str = "Authorization"
    bearer_token: str = ""
    api_key_header: str = "X-API-Key"
    api_key: str = ""
    cookie: str = ""
    custom_headers: Dict[str, str] = field(default_factory=dict)
    secret_provider: str = ""
    secret_refs: Dict[str, str] = field(default_factory=dict)
    secret_resolution_error: str = ""
    runtime_override_active: bool = False

    @classmethod
    def from_env(cls) -> "MCPClientConfig":
        endpoint = os.getenv("QF_MCP_ENDPOINT", "https://agent.robinhood.com/mcp/trading").strip()
        payload_mode = os.getenv("QF_MCP_PAYLOAD_MODE", "jsonrpc").strip().lower() or "jsonrpc"
        auth_header = os.getenv("QF_MCP_AUTH_HEADER", "Authorization").strip() or "Authorization"
        bearer_token = (
            os.getenv("QF_MCP_BEARER_TOKEN")
            or os.getenv("ROBINHOOD_MCP_BEARER_TOKEN")
            or os.getenv("MCP_BEARER_TOKEN")
            or ""
        ).strip()
        api_key_header = os.getenv("QF_MCP_API_KEY_HEADER", "X-API-Key").strip() or "X-API-Key"
        api_key = (
            os.getenv("QF_MCP_API_KEY")
            or os.getenv("ROBINHOOD_MCP_API_KEY")
            or os.getenv("MCP_API_KEY")
            or ""
        ).strip()
        cookie = (
            os.getenv("QF_MCP_COOKIE")
            or os.getenv("ROBINHOOD_MCP_COOKIE")
            or os.getenv("MCP_COOKIE")
            or ""
        ).strip()
        timeout_val = os.getenv("QF_MCP_TIMEOUT_SECONDS", "10").strip()
        try:
            timeout_seconds = max(1, int(timeout_val))
        except Exception:
            timeout_seconds = 10

        custom_headers: Dict[str, str] = _parse_custom_headers_json(os.getenv("QF_MCP_HEADERS_JSON", "").strip())

        secret_provider = os.getenv("QF_SECRET_PROVIDER", "").strip().lower()
        secret_refs = _env_secret_refs()
        provider_options = {
            "addr": os.getenv("QF_VAULT_ADDR", "").strip(),
            "token": os.getenv("QF_VAULT_TOKEN", "").strip(),
            "namespace": os.getenv("QF_VAULT_NAMESPACE", "").strip(),
            "mount_point": os.getenv("QF_VAULT_MOUNT_POINT", "").strip(),
            "region": os.getenv("QF_AWS_REGION", "").strip(),
            "project_id": os.getenv("QF_GCP_PROJECT_ID", "").strip(),
            "vault_url": os.getenv("QF_AZURE_KEYVAULT_URL", "").strip(),
        }

        runtime_cfg = _runtime_mcp_auth_config()
        runtime_override_active = bool(runtime_cfg)
        if runtime_cfg.get("auth_header"):
            auth_header = str(runtime_cfg.get("auth_header"))
        if runtime_cfg.get("bearer_token"):
            bearer_token = str(runtime_cfg.get("bearer_token"))
        if runtime_cfg.get("api_key_header"):
            api_key_header = str(runtime_cfg.get("api_key_header"))
        if runtime_cfg.get("api_key"):
            api_key = str(runtime_cfg.get("api_key"))
        if runtime_cfg.get("cookie"):
            cookie = str(runtime_cfg.get("cookie"))
        if isinstance(runtime_cfg.get("custom_headers"), dict):
            custom_headers = {str(k): str(v) for k, v in dict(runtime_cfg.get("custom_headers") or {}).items()}
        if runtime_cfg.get("secret_provider"):
            secret_provider = str(runtime_cfg.get("secret_provider", "")).strip().lower()
        if isinstance(runtime_cfg.get("secret_refs"), dict):
            secret_refs = {str(k): str(v) for k, v in dict(runtime_cfg.get("secret_refs") or {}).items()}
        if isinstance(runtime_cfg.get("provider_options"), dict):
            provider_options.update({str(k): str(v) for k, v in dict(runtime_cfg.get("provider_options") or {}).items()})

        secret_resolution_error = ""
        if secret_provider and secret_refs:
            try:
                resolved = resolve_secret_refs(secret_provider, secret_refs, provider_options=provider_options)
                if resolved.get("bearer_token"):
                    bearer_token = str(resolved.get("bearer_token"))
                if resolved.get("api_key"):
                    api_key = str(resolved.get("api_key"))
                if resolved.get("cookie"):
                    cookie = str(resolved.get("cookie"))
                if resolved.get("custom_headers"):
                    secret_headers = _parse_custom_headers_json(str(resolved.get("custom_headers")))
                    if secret_headers:
                        custom_headers = secret_headers
            except SecretProviderError as e:
                secret_resolution_error = str(e)

        return cls(
            endpoint=endpoint,
            transport="streamable_http",
            timeout_seconds=timeout_seconds,
            payload_mode=payload_mode,
            auth_header=auth_header,
            bearer_token=bearer_token,
            api_key_header=api_key_header,
            api_key=api_key,
            cookie=cookie,
            custom_headers=custom_headers,
            secret_provider=secret_provider,
            secret_refs=secret_refs,
            secret_resolution_error=secret_resolution_error,
            runtime_override_active=runtime_override_active,
        )


class MCPClientError(RuntimeError):
    pass


class RobinhoodMCPClient:
    """Lightweight MCP client scaffold.

    This class intentionally avoids coupling QuantFlow to a single MCP SDK.
    It provides a stable abstraction point while MCP transport details are finalized.
    """

    def __init__(self, config: Optional[MCPClientConfig] = None):
        self.config = config or MCPClientConfig.from_env()
        self._connected = False
        self._last_error: Optional[str] = None

    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        # Connection/auth happens in the user's AI platform (Codex/Cursor/Claude/etc).
        # QuantFlow keeps this as a soft-connect marker for now.
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def auth_summary(self) -> Dict[str, Any]:
        return {
            "configured": bool(self.config.bearer_token or self.config.api_key or self.config.cookie or self.config.custom_headers),
            "auth_header": self.config.auth_header,
            "has_bearer": bool(self.config.bearer_token),
            "has_api_key": bool(self.config.api_key),
            "api_key_header": self.config.api_key_header,
            "has_cookie": bool(self.config.cookie),
            "custom_header_keys": sorted(list(self.config.custom_headers.keys())),
            "payload_mode": self.config.payload_mode,
            "secret_provider": self.config.secret_provider or None,
            "secret_ref_keys": sorted(list(self.config.secret_refs.keys())),
            "secret_resolution_error": self.config.secret_resolution_error or None,
            "runtime_override_active": self.config.runtime_override_active,
        }

    def _headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.config.bearer_token:
            token = self.config.bearer_token
            if not token.lower().startswith("bearer "):
                token = f"Bearer {token}"
            headers[self.config.auth_header] = token
        if self.config.api_key:
            headers[self.config.api_key_header] = self.config.api_key
        if self.config.cookie:
            headers["Cookie"] = self.config.cookie
        for key, value in (self.config.custom_headers or {}).items():
            headers[str(key)] = str(value)
        return headers

    def _payload(self, method: str) -> Dict[str, Any]:
        if self.config.payload_mode == "raw":
            return {"method": method, "params": {}}
        return {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": method,
            "params": {},
        }

    def _load_fixture_json(self, env_key: str) -> Optional[Dict[str, Any]]:
        payload = os.getenv(env_key, "").strip()
        if not payload:
            return None
        try:
            data = json.loads(payload)
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    def _mcp_read_call(self, method: str) -> Dict[str, Any]:
        body = json.dumps(self._payload(method)).encode("utf-8")
        req = request.Request(
            self.config.endpoint,
            data=body,
            headers=self._headers(),
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
                raw = resp.read().decode("utf-8")
                data = json.loads(raw) if raw else {}
                if isinstance(data, dict) and isinstance(data.get("error"), dict):
                    err = data.get("error") or {}
                    raise MCPClientError(f"MCP error {err.get('code', '?')}: {err.get('message', 'unknown error')}")
                if isinstance(data, dict) and "result" in data:
                    result = data.get("result")
                    if isinstance(result, dict):
                        return result
                    raise MCPClientError("MCP JSON-RPC result is not an object")
                if isinstance(data, dict):
                    return data
                raise MCPClientError("unexpected MCP response shape")
        except error.HTTPError as e:
            detail = ""
            try:
                body = e.read().decode("utf-8", errors="ignore")
                detail = body[:280]
            except Exception:
                detail = ""
            msg = f"MCP HTTP error {e.code}: {e.reason}"
            if detail:
                msg = f"{msg} | {detail}"
            self._last_error = msg
            raise MCPClientError(msg) from e
        except error.URLError as e:
            msg = f"MCP transport error: {e.reason}"
            self._last_error = msg
            raise MCPClientError(msg) from e
        except json.JSONDecodeError as e:
            msg = "MCP returned non-JSON response"
            self._last_error = msg
            raise MCPClientError(msg) from e

    def account(self) -> Dict[str, Any]:
        fixture = self._load_fixture_json("QF_MCP_ACCOUNT_JSON")
        if fixture is not None:
            return fixture

        data = self._mcp_read_call("account")
        if "account" in data and isinstance(data["account"], dict):
            return data["account"]
        if all(k in data for k in ("equity", "cash", "buying_power")):
            return data
        raise MCPClientError("MCP account response missing required fields")

    def positions(self) -> Dict[str, Any]:
        fixture = self._load_fixture_json("QF_MCP_POSITIONS_JSON")
        if fixture is not None:
            return fixture

        data = self._mcp_read_call("positions")
        if "positions" in data and isinstance(data["positions"], list):
            return data
        if isinstance(data, list):
            return {"positions": data}
        raise MCPClientError("MCP positions response missing 'positions' list")
