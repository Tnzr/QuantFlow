from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import json
import os
from urllib import request, error


@dataclass
class MCPClientConfig:
    endpoint: str = "https://agent.robinhood.com/mcp/trading"
    transport: str = "streamable_http"
    timeout_seconds: int = 10


class MCPClientError(RuntimeError):
    pass


class RobinhoodMCPClient:
    """Lightweight MCP client scaffold.

    This class intentionally avoids coupling QuantFlow to a single MCP SDK.
    It provides a stable abstraction point while MCP transport details are finalized.
    """

    def __init__(self, config: Optional[MCPClientConfig] = None):
        self.config = config or MCPClientConfig()
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
        # The public endpoint may require AI-platform level auth/session. We still
        # attempt a plain JSON call and then surface actionable errors.
        body = json.dumps({"method": method, "params": {}}).encode("utf-8")
        req = request.Request(
            self.config.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
                raw = resp.read().decode("utf-8")
                data = json.loads(raw) if raw else {}
                if isinstance(data, dict):
                    return data
                raise MCPClientError("unexpected MCP response shape")
        except error.HTTPError as e:
            msg = f"MCP HTTP error {e.code}: {e.reason}"
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
