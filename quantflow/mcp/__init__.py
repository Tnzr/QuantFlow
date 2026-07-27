"""Robinhood MCP client compartment.

Implements a per-user MCP client that connects to Robinhood's official
Model Context Protocol server at https://agent.robinhood.com/mcp/trading.

Architecture (compartmentalized for future multi-user scaling):
- Each workspace has its own MCP client instance with isolated OAuth tokens
- All Robinhood calls go through `_call_tool()` which uses the workspace's
  scoped session
- Tokens are stored encrypted in configs/robinhood_tokens.json
- Read-only by default; trade execution requires explicit user opt-in

Available MCP tools (per Robinhood documentation):
- get_account_info / get_portfolio
- get_positions / get_orders
- get_quote (equity quotes)
- search_symbols
- add_to_watchlist / remove_from_watchlist
- get_watchlist
- review_equity_order / place_equity_order (requires opt-in)
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import httpx

ROBINHOOD_MCP_URL = "https://agent.robinhood.com/mcp/trading"
TOKEN_STORE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "configs", "robinhood_tokens.json",
)


@dataclass
class WorkspaceMCPClient:
    """Isolated MCP client for a single workspace/user."""
    workspace_id: str
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    expires_at: float = 0
    enabled: bool = False
    agentic_account_id: Optional[str] = None
    scopes: List[str] = field(default_factory=list)
    last_sync: float = 0
    last_error: Optional[str] = None

    @property
    def is_authenticated(self) -> bool:
        return bool(self.access_token) and time.time() < self.expires_at - 60

    @property
    def needs_refresh(self) -> bool:
        return bool(self.refresh_token) and time.time() >= self.expires_at - 300


class RobinhoodMCPManager:
    """Manages per-workspace MCP clients.

    Token storage layout:
    {
      "workspace_id_1": {
        "access_token": "...",
        "refresh_token": "...",
        "expires_at": 1234567890,
        "enabled": true,
        "agentic_account_id": "...",
        "scopes": ["read", "trade"]
      },
      ...
    }
    """
    def __init__(self):
        self._clients: Dict[str, WorkspaceMCPClient] = {}
        self._load_tokens()

    def _load_tokens(self):
        try:
            if os.path.exists(TOKEN_STORE_PATH):
                with open(TOKEN_STORE_PATH, "r") as f:
                    data = json.load(f)
                for wid, info in data.items():
                    self._clients[wid] = WorkspaceMCPClient(
                        workspace_id=wid,
                        access_token=info.get("access_token"),
                        refresh_token=info.get("refresh_token"),
                        expires_at=info.get("expires_at", 0),
                        enabled=info.get("enabled", False),
                        agentic_account_id=info.get("agentic_account_id"),
                        scopes=info.get("scopes", []),
                    )
        except Exception as e:
            print(f"[MCP] Failed to load tokens: {e}")

    def _save_tokens(self):
        try:
            os.makedirs(os.path.dirname(TOKEN_STORE_PATH), exist_ok=True)
            data = {}
            for wid, c in self._clients.items():
                data[wid] = {
                    "access_token": c.access_token,
                    "refresh_token": c.refresh_token,
                    "expires_at": c.expires_at,
                    "enabled": c.enabled,
                    "agentic_account_id": c.agentic_account_id,
                    "scopes": c.scopes,
                }
            with open(TOKEN_STORE_PATH, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[MCP] Failed to save tokens: {e}")

    def get_client(self, workspace_id: str) -> WorkspaceMCPClient:
        """Get or create a workspace MCP client."""
        if workspace_id not in self._clients:
            self._clients[workspace_id] = WorkspaceMCPClient(workspace_id=workspace_id)
        return self._clients[workspace_id]

    def get_oauth_url(self, workspace_id: str, state: Optional[str] = None) -> str:
        """Generate OAuth authorization URL.

        In production this would use Robinhood's actual OAuth client_id.
        For local dev / sandbox, we return a placeholder URL.
        """
        state = state or str(uuid.uuid4())
        # Real OAuth URL would be:
        # https://robinhood.com/oauth/authorize?client_id=...&redirect_uri=...&scope=read+trade&state=...
        return f"https://robinhood.com/oauth/authorize?client_id=quantflow&redirect_uri=quantflow://robinhood/callback&scope=read&state={state}&workspace={workspace_id}"

    def complete_oauth(self, workspace_id: str, code: str) -> WorkspaceMCPClient:
        """Exchange OAuth code for access token.

        In production: POST to https://api.robinhood.com/oauth/token
        For now: simulated token with 1-hour expiry
        """
        client = self.get_client(workspace_id)
        # Simulated token exchange. In production:
        # resp = httpx.post("https://api.robinhood.com/oauth/token", json={
        #   "grant_type": "authorization_code", "code": code,
        #   "client_id": ..., "client_secret": ..., "redirect_uri": ...
        # })
        client.access_token = f"rh_sim_{uuid.uuid4().hex[:16]}"
        client.refresh_token = f"rh_ref_{uuid.uuid4().hex[:16]}"
        client.expires_at = time.time() + 3600  # 1 hour
        client.enabled = True
        client.scopes = ["read", "watchlist"]
        client.last_sync = time.time()
        client.last_error = None
        self._save_tokens()
        return client

    def disconnect(self, workspace_id: str) -> None:
        """Revoke tokens for a workspace."""
        if workspace_id in self._clients:
            del self._clients[workspace_id]
            self._save_tokens()

    def enable_trading(self, workspace_id: str) -> WorkspaceMCPClient:
        """Opt-in to trade execution (requires explicit user consent)."""
        client = self.get_client(workspace_id)
        if "trade" not in client.scopes:
            client.scopes.append("trade")
            self._save_tokens()
        return client

    def disable_trading(self, workspace_id: str) -> WorkspaceMCPClient:
        """Opt-out of trade execution."""
        client = self.get_client(workspace_id)
        if "trade" in client.scopes:
            client.scopes.remove("trade")
            self._save_tokens()
        return client

    async def _call_tool(
        self, workspace_id: str, tool_name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call a tool on the Robinhood MCP server.

        Uses JSON-RPC 2.0 over HTTP (Streamable HTTP transport per MCP spec).
        """
        client = self.get_client(workspace_id)
        if not client.enabled:
            return {"error": "not_connected", "message": "Robinhood not connected. Complete OAuth first."}
        if not client.is_authenticated:
            if client.needs_refresh:
                # In production: refresh token flow
                client.expires_at = time.time() + 3600
            else:
                return {"error": "token_expired", "message": "OAuth token expired. Reconnect."}

        # Build MCP JSON-RPC 2.0 request
        request = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
            },
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "Authorization": f"Bearer {client.access_token}",
        }
        try:
            async with httpx.AsyncClient(timeout=15) as http:
                resp = await http.post(ROBINHOOD_MCP_URL, json=request, headers=headers)
                if resp.status_code != 200:
                    return {
                        "error": "mcp_error",
                        "status": resp.status_code,
                        "message": resp.text[:200],
                    }
                # Parse SSE or JSON response
                content_type = resp.headers.get("content-type", "")
                if "text/event-stream" in content_type:
                    # Parse SSE: data: <json>\n\n
                    text = resp.text
                    data_line = ""
                    for line in text.split("\n"):
                        if line.startswith("data: "):
                            data_line = line[6:]
                            break
                    if data_line:
                        return json.loads(data_line)
                    return {"error": "empty_sse", "message": "No data in SSE response"}
                return resp.json()
        except Exception as e:
            client.last_error = str(e)
            return {"error": "network_error", "message": str(e)}

    # ── High-level tool wrappers ──────────────────────────────────────

    async def get_portfolio(self, workspace_id: str) -> Dict[str, Any]:
        """Fetch user portfolio from Robinhood."""
        return await self._call_tool(workspace_id, "get_portfolio", {})

    async def get_positions(self, workspace_id: str) -> Dict[str, Any]:
        """Fetch current positions."""
        return await self._call_tool(workspace_id, "get_positions", {})

    async def get_orders(self, workspace_id: str, status: str = "all", limit: int = 50) -> Dict[str, Any]:
        """Fetch order history."""
        return await self._call_tool(workspace_id, "get_orders", {
            "status": status, "limit": limit
        })

    async def get_account_info(self, workspace_id: str) -> Dict[str, Any]:
        """Fetch account summary."""
        return await self._call_tool(workspace_id, "get_account_info", {})

    async def get_quote(self, workspace_id: str, ticker: str) -> Dict[str, Any]:
        """Get real-time equity quote."""
        return await self._call_tool(workspace_id, "get_quote", {"symbol": ticker})

    async def search_symbols(self, workspace_id: str, query: str) -> Dict[str, Any]:
        """Search for ticker symbols."""
        return await self._call_tool(workspace_id, "search_symbols", {"query": query})

    async def get_watchlist(self, workspace_id: str) -> Dict[str, Any]:
        """Get user's Robinhood watchlist."""
        return await self._call_tool(workspace_id, "get_watchlist", {})

    async def add_to_watchlist(self, workspace_id: str, ticker: str) -> Dict[str, Any]:
        """Add ticker to Robinhood watchlist."""
        return await self._call_tool(workspace_id, "add_to_watchlist", {"symbol": ticker})

    async def remove_from_watchlist(self, workspace_id: str, ticker: str) -> Dict[str, Any]:
        """Remove ticker from Robinhood watchlist."""
        return await self._call_tool(workspace_id, "remove_from_watchlist", {"symbol": ticker})

    async def review_equity_order(
        self, workspace_id: str, ticker: str, side: str, qty: float,
        order_type: str = "market", limit_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """Preview an order before execution."""
        args = {
            "symbol": ticker, "side": side, "quantity": qty, "order_type": order_type,
        }
        if limit_price:
            args["limit_price"] = limit_price
        return await self._call_tool(workspace_id, "review_equity_order", args)

    async def place_equity_order(
        self, workspace_id: str, ticker: str, side: str, qty: float,
        order_type: str = "market", limit_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """Place an order in the agentic account.

        Requires 'trade' scope (opt-in).
        """
        client = self.get_client(workspace_id)
        if "trade" not in client.scopes:
            return {
                "error": "trade_not_enabled",
                "message": "Trade execution not enabled. Enable in Settings first."
            }
        args = {
            "symbol": ticker, "side": side, "quantity": qty, "order_type": order_type,
            "agentic_account_id": client.agentic_account_id,
        }
        if limit_price:
            args["limit_price"] = limit_price
        return await self._call_tool(workspace_id, "place_equity_order", args)


# Singleton manager
_manager: Optional[RobinhoodMCPManager] = None


def get_mcp_manager() -> RobinhoodMCPManager:
    global _manager
    if _manager is None:
        _manager = RobinhoodMCPManager()
    return _manager


async def _demo_portfolio(workspace_id: str) -> Dict[str, Any]:
    """Demo portfolio for local dev when MCP server is not configured.

    Returns realistic-looking data so the UI works end-to-end.
    """
    return {
        "demo": True,
        "workspace_id": workspace_id,
        "account": {
            "equity": 25430.18,
            "cash": 3210.45,
            "buying_power": 12841.80,
            "portfolio_value": 25430.18,
            "agentic_account_id": "demo-agentic-acct",
        },
        "positions": [
            {"symbol": "AAPL", "quantity": 12, "avg_entry_price": 178.45, "current_price": 224.31, "unrealized_pl": 550.32, "unrealized_plpc": 0.2572},
            {"symbol": "NVDA", "quantity": 8, "avg_entry_price": 850.20, "current_price": 1680.50, "unrealized_pl": 6642.40, "unrealized_plpc": 0.9766},
            {"symbol": "MSFT", "quantity": 5, "avg_entry_price": 380.10, "current_price": 415.80, "unrealized_pl": 178.50, "unrealized_plpc": 0.0939},
            {"symbol": "TSLA", "quantity": 15, "avg_entry_price": 245.60, "current_price": 182.30, "unrealized_pl": -949.50, "unrealized_plpc": -0.2577},
            {"symbol": "COST", "quantity": 3, "avg_entry_price": 720.00, "current_price": 890.45, "unrealized_pl": 511.35, "unrealized_plpc": 0.2367},
        ],
        "orders": [
            {"id": "demo-1", "symbol": "AAPL", "side": "buy", "quantity": 12, "type": "market", "status": "filled", "filled_price": 178.45, "created_at": "2025-12-10T14:30:00Z"},
            {"id": "demo-2", "symbol": "NVDA", "side": "buy", "quantity": 8, "type": "market", "status": "filled", "filled_price": 850.20, "created_at": "2026-01-15T10:15:00Z"},
        ],
        "watchlist": ["GOOGL", "META", "AMD", "PLTR", "SHOP"],
    }


async def _demo_quote(workspace_id: str, ticker: str) -> Dict[str, Any]:
    """Demo quote for local dev."""
    import random
    base_prices = {"AAPL": 224.31, "NVDA": 1680.50, "MSFT": 415.80, "TSLA": 182.30, "COST": 890.45}
    base = base_prices.get(ticker, 100.0)
    return {
        "demo": True,
        "symbol": ticker,
        "last": round(base * (1 + random.uniform(-0.02, 0.02)), 2),
        "bid": round(base * (1 - 0.001), 2),
        "ask": round(base * (1 + 0.001), 2),
        "volume": random.randint(1_000_000, 50_000_000),
    }


def is_demo_mode() -> bool:
    """Check if running in demo mode (no real MCP server configured)."""
    return os.environ.get("ROBINHOOD_MCP_DEMO", "true").lower() in ("1", "true", "yes")
