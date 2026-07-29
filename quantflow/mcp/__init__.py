"""Robinhood MCP client compartment.

Implements a per-user MCP client that connects to Robinhood's official
Model Context Protocol server at https://agent.robinhood.com/mcp/trading.

Architecture (compartmentalized for future multi-user scaling):
- Each workspace has its own MCP client instance with isolated OAuth tokens
- All Robinhood calls go through `_call_tool()` which uses the workspace's
  scoped session
- Tokens are stored encrypted in configs/robinhood_tokens.json
- Read-only by default; trade execution requires explicit user opt-in

OAuth Flow (Robinhood's standard OAuth 2.0 + PKCE):
1. Register as public client with Robinhood (no client_secret needed)
2. Construct authorize URL with PKCE
3. Open browser to robinhood.com/oauth for user authentication
4. Server redirects back with code
5. Exchange code for token at api.robinhood.com/oauth2/token/
6. Store token for future requests

Available MCP tools (per Robinhood documentation):

Account & Portfolio:
- get_accounts, get_portfolio, get_equity_positions, get_equity_orders
- get_realized_pnl, get_pnl_trade_history

Market Data:
- get_equity_historicals, get_equity_fundamentals, get_financials
- get_equity_quotes, get_equity_price_book, get_equity_technical_indicators
- get_earnings_results, get_earnings_calendar
- get_indexes, get_index_quotes
- search

Watchlist:
- get_watchlists, get_watchlist_items, get_option_watchlist
- get_popular_watchlists, create_watchlist, update_watchlist
- follow_watchlist, unfollow_watchlist
- add_to_watchlist, remove_from_watchlist
- add_option_to_watchlist, remove_option_from_watchlist

Trading (requires opt-in):
- review_equity_order, place_equity_order, cancel_equity_order
- get_equity_tradability
- get_option_chains, get_option_instruments, get_option_quotes
- get_option_positions, get_option_orders
- review_option_order, place_option_order, cancel_option_order

Scanner:
- get_scans, get_scanner_filter_specs
- create_scan, run_scan
- update_scan_filters, update_scan_config
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import secrets
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode, urlparse, parse_qs

import httpx

ROBINHOOD_MCP_URL = "https://agent.robinhood.com/mcp/trading"
ROBINHOOD_OAUTH_AUTHORIZE_URL = "https://robinhood.com/oauth"
ROBINHOOD_OAUTH_TOKEN_URL = "https://api.robinhood.com/oauth2/token/"
ROBINHOOD_OAUTH_REGISTER_URL = "https://agent.robinhood.com/oauth/trading/register"
ROBINHOOD_REDIRECT_URI = "http://localhost:3000/robinhood/oauth/callback"

TOKEN_STORE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "configs", "robinhood_tokens.json",
)

CLIENT_ID_STORE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "configs", "robinhood_client_id.json",
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
        self._cached_client_id: Optional[str] = None
        self._pkce_store: Dict[str, Dict[str, Any]] = {}
        self._load_tokens()
        self._load_client_id()

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

    def _load_client_id(self):
        """Load cached OAuth client_id from disk."""
        try:
            if os.path.exists(CLIENT_ID_STORE_PATH):
                with open(CLIENT_ID_STORE_PATH, "r") as f:
                    data = json.load(f)
                self._cached_client_id = data.get("client_id")
        except Exception as e:
            print(f"[MCP] Failed to load client_id: {e}")

    def _save_client_id(self, client_id: str):
        """Save OAuth client_id to disk for reuse."""
        try:
            os.makedirs(os.path.dirname(CLIENT_ID_STORE_PATH), exist_ok=True)
            with open(CLIENT_ID_STORE_PATH, "w") as f:
                json.dump({"client_id": client_id}, f, indent=2)
            self._cached_client_id = client_id
        except Exception as e:
            print(f"[MCP] Failed to save client_id: {e}")

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

    async def _register_client(self) -> Optional[str]:
        """Register as a public OAuth client with Robinhood.

        Returns client_id on success, None on failure.
        """
        # Return cached client_id if available
        if self._cached_client_id:
            return self._cached_client_id

        # Check environment first
        env_client_id = os.environ.get("ROBINHOOD_CLIENT_ID")
        if env_client_id:
            self._save_client_id(env_client_id)
            return env_client_id

        # Register new client
        try:
            async with httpx.AsyncClient(timeout=15) as http:
                resp = await http.post(
                    ROBINHOOD_OAUTH_REGISTER_URL,
                    json={
                        "client_name": "QuantFlow",
                        "redirect_uris": [ROBINHOOD_REDIRECT_URI],
                        "grant_types": ["authorization_code", "refresh_token"],
                        "response_types": ["code"],
                        "scope": "internal",
                        "token_endpoint_auth_method": "none",
                    },
                    headers={"Content-Type": "application/json"},
                )
                if resp.status_code in (200, 201):
                    data = resp.json()
                    client_id = data.get("client_id")
                    if client_id:
                        self._save_client_id(client_id)
                        print(f"[MCP] Registered OAuth client: {client_id}")
                        return client_id
                    else:
                        print(f"[MCP] Client registration returned no client_id: {resp.text[:200]}")
                        return None
                else:
                    print(f"[MCP] Client registration failed: {resp.status_code} {resp.text[:200]}")
                    return None
        except Exception as e:
            print(f"[MCP] Client registration error: {e}")
            return None

    async def get_oauth_challenge(self, workspace_id: str) -> Dict[str, Any]:
        """Get OAuth challenge from Robinhood MCP server.

        Uses Robinhood's actual OAuth 2.0 + PKCE flow:
        1. Register as public client (if not already done)
        2. Construct authorize URL with PKCE
        3. Return URL for user to open

        Returns dict with:
        - authorize_url: URL to open in browser
        - state: state parameter for CSRF protection
        - code_verifier: PKCE code verifier (for token exchange)
        - challenge_type: "oauth" or "none" (if already authenticated)
        """
        client = self.get_client(workspace_id)

        # If already authenticated, no challenge needed
        if client.is_authenticated:
            return {"challenge_type": "none", "already_authenticated": True}

        # Register as public client first
        client_id = await self._register_client()
        if not client_id:
            return {
                "challenge_type": "error",
                "error": "Failed to register OAuth client with Robinhood. Set ROBINHOOD_CLIENT_ID manually if you have one.",
            }

        # Generate PKCE parameters
        state = secrets.token_urlsafe(32)
        code_verifier = secrets.token_urlsafe(64)
        code_challenge = base64.urlsafe_b64encode(
            hashlib.sha256(code_verifier.encode()).digest()
        ).rstrip(b"=").decode("utf-8")

        # Store code_verifier keyed by state for later retrieval in callback
        self._pkce_store[state] = {
            "code_verifier": code_verifier,
            "workspace_id": workspace_id,
            "created_at": time.time(),
        }

        # Construct authorize URL using Robinhood's actual endpoint
        params = {
            "client_id": client_id,
            "redirect_uri": ROBINHOOD_REDIRECT_URI,
            "response_type": "code",
            "scope": "internal",
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }
        authorize_url = f"{ROBINHOOD_OAUTH_AUTHORIZE_URL}?{urlencode(params)}"

        return {
            "challenge_type": "oauth",
            "authorize_url": authorize_url,
            "state": state,
            "code_verifier": code_verifier,
            "workspace_id": workspace_id,
            "client_id": client_id,
        }

    async def complete_oauth(self, workspace_id: str, code: str, state: str, code_verifier: str) -> WorkspaceMCPClient:
        """Exchange OAuth code for access token.

        This is called from the OAuth callback after the user authenticates
        with Robinhood in their browser.
        """
        client = self.get_client(workspace_id)
        client_id = await self._register_client()

        # Retrieve code_verifier from store if not provided
        if not code_verifier and state:
            pkce_data = self._pkce_store.get(state)
            if pkce_data:
                code_verifier = pkce_data.get("code_verifier")
                # Clean up after use
                del self._pkce_store[state]

        try:
            async with httpx.AsyncClient(timeout=15) as http:
                # Exchange code for token using Robinhood's actual token endpoint
                token_request = {
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": ROBINHOOD_REDIRECT_URI,
                    "code_verifier": code_verifier,
                }
                if client_id:
                    token_request["client_id"] = client_id

                resp = await http.post(
                    ROBINHOOD_OAUTH_TOKEN_URL,
                    json=token_request,
                    headers={"Content-Type": "application/json"},
                )

                if resp.status_code == 200:
                    token_data = resp.json()
                    client.access_token = token_data.get("access_token")
                    client.refresh_token = token_data.get("refresh_token")
                    client.expires_at = time.time() + token_data.get("expires_in", 3600)
                    client.enabled = True
                    client.scopes = ["read", "watchlist"]
                    client.last_sync = time.time()
                    client.last_error = None
                    self._save_tokens()
                    return client
                else:
                    client.last_error = f"Token exchange failed: {resp.status_code} {resp.text[:200]}"
                    return client

        except Exception as e:
            client.last_error = f"Token exchange error: {str(e)}"
            return client

    def set_manual_token(self, workspace_id: str, access_token: str, refresh_token: Optional[str] = None, expires_in: int = 3600) -> WorkspaceMCPClient:
        """Set a manually provided OAuth token.

        Used when the user completes OAuth externally (e.g., via browser)
        and pastes the token into the app.
        """
        client = self.get_client(workspace_id)
        client.access_token = access_token
        client.refresh_token = refresh_token
        client.expires_at = time.time() + expires_in
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

    # ── Account & Portfolio Tools ─────────────────────────────────────

    async def get_accounts(self, workspace_id: str) -> Dict[str, Any]:
        """View all Robinhood accounts."""
        return await self._call_tool(workspace_id, "get_accounts", {})

    async def get_portfolio(self, workspace_id: str) -> Dict[str, Any]:
        """Get snapshot of portfolio with total value and buying power."""
        return await self._call_tool(workspace_id, "get_portfolio", {})

    async def get_equity_positions(self, workspace_id: str, account_number: Optional[str] = None) -> Dict[str, Any]:
        """View open equity positions with quantity and cost basis.

        If account_number is provided, returns positions for that specific account.
        Otherwise returns positions across all accounts.
        """
        args = {}
        if account_number:
            args["account_number"] = account_number
        return await self._call_tool(workspace_id, "get_equity_positions", args)

    async def get_equity_orders(self, workspace_id: str, status: str = "all", limit: int = 50, account_number: Optional[str] = None) -> Dict[str, Any]:
        """Get equity order status history."""
        args = {"status": status, "limit": limit}
        if account_number:
            args["account_number"] = account_number
        return await self._call_tool(workspace_id, "get_equity_orders", args)

    async def get_realized_pnl(self, workspace_id: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, Any]:
        """View realized profit and loss for account over time window."""
        args = {}
        if start_date: args["start_date"] = start_date
        if end_date: args["end_date"] = end_date
        return await self._call_tool(workspace_id, "get_realized_pnl", args)

    async def get_pnl_trade_history(self, workspace_id: str, limit: int = 50) -> Dict[str, Any]:
        """View trade-by-trade realized P&L history."""
        return await self._call_tool(workspace_id, "get_pnl_trade_history", {"limit": limit})

    # ── Market Data Tools ─────────────────────────────────────────────

    async def get_equity_historicals(self, workspace_id: str, symbol: str, interval: str = "day", span: str = "month") -> Dict[str, Any]:
        """Get OHLCV price bars across a time range."""
        return await self._call_tool(workspace_id, "get_equity_historicals", {
            "symbol": symbol, "interval": interval, "span": span
        })

    async def get_equity_fundamentals(self, workspace_id: str, symbol: str) -> Dict[str, Any]:
        """Get valuation ratios, market cap, 52-week range, dividend info."""
        return await self._call_tool(workspace_id, "get_equity_fundamentals", {"symbol": symbol})

    async def get_financials(self, workspace_id: str, symbols: List[str], period: str = "quarter") -> Dict[str, Any]:
        """Get company's reported financials (revenue, profit, net margin)."""
        return await self._call_tool(workspace_id, "get_financials", {
            "symbols": symbols, "period": period
        })

    async def get_equity_quotes(self, workspace_id: str, symbols: List[str]) -> Dict[str, Any]:
        """Get real-time equity quotes for up to 20 symbols."""
        return await self._call_tool(workspace_id, "get_equity_quotes", {"symbols": symbols})

    async def get_equity_price_book(self, workspace_id: str, symbols: List[str]) -> Dict[str, Any]:
        """Get real-time Level 2 order book for up to 4 stocks."""
        return await self._call_tool(workspace_id, "get_equity_price_book", {"symbols": symbols})

    async def get_equity_technical_indicators(self, workspace_id: str, symbol: str, indicator: str = "rsi", interval: str = "day", span: str = "month") -> Dict[str, Any]:
        """Compute technical indicators (RSI, MACD, Bollinger Bands, etc.)."""
        return await self._call_tool(workspace_id, "get_equity_technical_indicators", {
            "symbol": symbol, "indicator": indicator, "interval": interval, "span": span
        })

    async def get_earnings_results(self, workspace_id: str, symbol: str) -> Dict[str, Any]:
        """Look up stock's earnings history and next report."""
        return await self._call_tool(workspace_id, "get_earnings_results", {"symbol": symbol})

    async def get_earnings_calendar(self, workspace_id: str, start_date: Optional[str] = None, end_date: Optional[str] = None, large_cap_only: bool = False) -> Dict[str, Any]:
        """List earnings reports scheduled across the market."""
        args = {}
        if start_date: args["start_date"] = start_date
        if end_date: args["end_date"] = end_date
        if large_cap_only: args["large_cap_only"] = large_cap_only
        return await self._call_tool(workspace_id, "get_earnings_calendar", args)

    async def get_indexes(self, workspace_id: str, symbols: List[str]) -> Dict[str, Any]:
        """Look up market indexes by symbol."""
        return await self._call_tool(workspace_id, "get_indexes", {"symbols": symbols})

    async def get_index_quotes(self, workspace_id: str, symbols: List[str]) -> Dict[str, Any]:
        """Get real-time index values."""
        return await self._call_tool(workspace_id, "get_index_quotes", {"symbols": symbols})

    async def search(self, workspace_id: str, query: str) -> Dict[str, Any]:
        """Find a company name or partial name to a ticker."""
        return await self._call_tool(workspace_id, "search", {"query": query})

    # ── Watchlist Tools ───────────────────────────────────────────────

    async def get_watchlists(self, workspace_id: str) -> Dict[str, Any]:
        """List user's watchlists."""
        return await self._call_tool(workspace_id, "get_watchlists", {})

    async def get_watchlist_items(self, workspace_id: str, watchlist_id: str) -> Dict[str, Any]:
        """List symbols in a specific watchlist."""
        return await self._call_tool(workspace_id, "get_watchlist_items", {"watchlist_id": watchlist_id})

    async def get_popular_watchlists(self, workspace_id: str) -> Dict[str, Any]:
        """Discover Robinhood lists (like "100 most popular")."""
        return await self._call_tool(workspace_id, "get_popular_watchlists", {})

    async def create_watchlist(self, workspace_id: str, name: str, description: str = "") -> Dict[str, Any]:
        """Make a new custom watchlist."""
        return await self._call_tool(workspace_id, "create_watchlist", {
            "name": name, "description": description
        })

    async def update_watchlist(self, workspace_id: str, watchlist_id: str, name: Optional[str] = None, description: Optional[str] = None) -> Dict[str, Any]:
        """Rename or update a watchlist's name description."""
        args = {"watchlist_id": watchlist_id}
        if name: args["name"] = name
        if description: args["description"] = description
        return await self._call_tool(workspace_id, "update_watchlist", args)

    async def follow_watchlist(self, workspace_id: str, watchlist_id: str) -> Dict[str, Any]:
        """Follow a Robinhood list."""
        return await self._call_tool(workspace_id, "follow_watchlist", {"watchlist_id": watchlist_id})

    async def unfollow_watchlist(self, workspace_id: str, watchlist_id: str) -> Dict[str, Any]:
        """Stop following a Robinhood list."""
        return await self._call_tool(workspace_id, "unfollow_watchlist", {"watchlist_id": watchlist_id})

    async def add_to_watchlist(self, workspace_id: str, watchlist_id: str, symbol: str) -> Dict[str, Any]:
        """Add stocks, crypto, or indexes to a watchlist."""
        return await self._call_tool(workspace_id, "add_to_watchlist", {
            "watchlist_id": watchlist_id, "symbol": symbol
        })

    async def remove_from_watchlist(self, workspace_id: str, watchlist_id: str, symbol: str) -> Dict[str, Any]:
        """Remove stocks, crypto, or indexes from a watchlist."""
        return await self._call_tool(workspace_id, "remove_from_watchlist", {
            "watchlist_id": watchlist_id, "symbol": symbol
        })

    # ── Trading Tools (requires opt-in) ───────────────────────────────

    async def get_equity_tradability(self, workspace_id: str, symbol: str) -> Dict[str, Any]:
        """Check if a symbol can be traded and if it can be traded fractionally."""
        return await self._call_tool(workspace_id, "get_equity_tradability", {"symbol": symbol})

    async def review_equity_order(
        self, workspace_id: str, symbol: str, side: str, quantity: float,
        order_type: str = "market", limit_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """Simulate an equity order and get pre-trade warnings."""
        args = {
            "symbol": symbol, "side": side, "quantity": quantity, "order_type": order_type,
        }
        if limit_price:
            args["limit_price"] = limit_price
        return await self._call_tool(workspace_id, "review_equity_order", args)

    async def place_equity_order(
        self, workspace_id: str, symbol: str, side: str, quantity: float,
        order_type: str = "market", limit_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """Place an equity order (requires 'trade' scope opt-in)."""
        client = self.get_client(workspace_id)
        if "trade" not in client.scopes:
            return {
                "error": "trade_not_enabled",
                "message": "Trade execution not enabled. Enable in Settings first."
            }
        args = {
            "symbol": symbol, "side": side, "quantity": quantity, "order_type": order_type,
        }
        if limit_price:
            args["limit_price"] = limit_price
        return await self._call_tool(workspace_id, "place_equity_order", args)

    async def cancel_equity_order(self, workspace_id: str, order_id: str) -> Dict[str, Any]:
        """Cancel an open equity order."""
        client = self.get_client(workspace_id)
        if "trade" not in client.scopes:
            return {
                "error": "trade_not_enabled",
                "message": "Trade execution not enabled. Enable in Settings first."
            }
        return await self._call_tool(workspace_id, "cancel_equity_order", {"order_id": order_id})

    # ── Options Tools ─────────────────────────────────────────────────

    async def get_option_chains(self, workspace_id: str, symbol: str) -> Dict[str, Any]:
        """Load option chains for a symbol."""
        return await self._call_tool(workspace_id, "get_option_chains", {"symbol": symbol})

    async def get_option_instruments(self, workspace_id: str, chain_id: str, expiry: Optional[str] = None, strike: Optional[float] = None, option_type: Optional[str] = None) -> Dict[str, Any]:
        """Load option contracts filtered by expiry, strike, or type."""
        args = {"chain_id": chain_id}
        if expiry: args["expiry"] = expiry
        if strike: args["strike"] = strike
        if option_type: args["option_type"] = option_type
        return await self._call_tool(workspace_id, "get_option_instruments", args)

    async def get_option_quotes(self, workspace_id: str, symbols: List[str]) -> Dict[str, Any]:
        """Get real-time quotes for option contracts."""
        return await self._call_tool(workspace_id, "get_option_quotes", {"symbols": symbols})

    async def get_option_positions(self, workspace_id: str) -> Dict[str, Any]:
        """View open or closed options positions."""
        return await self._call_tool(workspace_id, "get_option_positions", {})

    async def get_option_orders(self, workspace_id: str, status: str = "all", limit: int = 50) -> Dict[str, Any]:
        """Get options order history."""
        return await self._call_tool(workspace_id, "get_option_orders", {
            "status": status, "limit": limit
        })

    async def review_option_order(
        self, workspace_id: str, symbol: str, side: str, quantity: float,
        order_type: str = "market", limit_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """Simulate an options order with pre-trade alerts."""
        args = {
            "symbol": symbol, "side": side, "quantity": quantity, "order_type": order_type,
        }
        if limit_price:
            args["limit_price"] = limit_price
        return await self._call_tool(workspace_id, "review_option_order", args)

    async def place_option_order(
        self, workspace_id: str, symbol: str, side: str, quantity: float,
        order_type: str = "market", limit_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """Place a real options order (requires 'trade' scope opt-in)."""
        client = self.get_client(workspace_id)
        if "trade" not in client.scopes:
            return {
                "error": "trade_not_enabled",
                "message": "Trade execution not enabled. Enable in Settings first."
            }
        args = {
            "symbol": symbol, "side": side, "quantity": quantity, "order_type": order_type,
        }
        if limit_price:
            args["limit_price"] = limit_price
        return await self._call_tool(workspace_id, "place_option_order", args)

    async def cancel_option_order(self, workspace_id: str, order_id: str) -> Dict[str, Any]:
        """Cancel an open options order."""
        client = self.get_client(workspace_id)
        if "trade" not in client.scopes:
            return {
                "error": "trade_not_enabled",
                "message": "Trade execution not enabled. Enable in Settings first."
            }
        return await self._call_tool(workspace_id, "cancel_option_order", {"order_id": order_id})

    # ── Scanner Tools ─────────────────────────────────────────────────

    async def get_scans(self, workspace_id: str) -> Dict[str, Any]:
        """List your saved scans."""
        return await self._call_tool(workspace_id, "get_scans", {})

    async def get_scanner_filter_specs(self, workspace_id: str) -> Dict[str, Any]:
        """List every available scanner filter and how to use it."""
        return await self._call_tool(workspace_id, "get_scanner_filter_specs", {})

    async def create_scan(self, workspace_id: str, preset: Optional[str] = None, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a new scan from a starting preset or with custom filters."""
        args = {}
        if preset: args["preset"] = preset
        if filters: args["filters"] = filters
        return await self._call_tool(workspace_id, "create_scan", args)

    async def run_scan(self, workspace_id: str, scan_id: str) -> Dict[str, Any]:
        """Run a saved scan and get live market results."""
        return await self._call_tool(workspace_id, "run_scan", {"scan_id": scan_id})

    async def update_scan_filters(self, workspace_id: str, scan_id: str, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Change the filters on an existing saved scan."""
        return await self._call_tool(workspace_id, "update_scan_filters", {
            "scan_id": scan_id, "filters": filters
        })

    async def update_scan_config(self, workspace_id: str, scan_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Change the way results are sorted."""
        return await self._call_tool(workspace_id, "update_scan_config", {
            "scan_id": scan_id, "config": config
        })


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
            "equity": 98500.00,
            "cash": 8500.00,
            "buying_power": 42500.00,
            "portfolio_value": 98500.00,
            "agentic_account_id": "demo-agentic-acct",
        },
        "positions": [
            {"symbol": "NVDA", "quantity": 15, "avg_entry_price": 720.00, "current_price": 980.50, "unrealized_pl": 3907.50, "unrealized_plpc": 0.362},
            {"symbol": "AAPL", "quantity": 25, "avg_entry_price": 195.00, "current_price": 224.31, "unrealized_pl": 732.75, "unrealized_plpc": 0.15},
            {"symbol": "INTC", "quantity": 80, "avg_entry_price": 52.00, "current_price": 61.50, "unrealized_pl": 760.00, "unrealized_plpc": 0.182},
            {"symbol": "TGT", "quantity": 40, "avg_entry_price": 148.00, "current_price": 152.30, "unrealized_pl": 172.00, "unrealized_plpc": 0.029},
            {"symbol": "BTQ", "quantity": 120, "avg_entry_price": 28.50, "current_price": 31.80, "unrealized_pl": 396.00, "unrealized_plpc": 0.115},
            {"symbol": "PANW", "quantity": 10, "avg_entry_price": 380.00, "current_price": 420.50, "unrealized_pl": 405.00, "unrealized_plpc": 0.106},
        ],
        "orders": [
            {"id": "demo-1", "symbol": "NVDA", "side": "buy", "quantity": 15, "type": "market", "status": "filled", "filled_price": 720.00, "created_at": "2025-12-10T14:30:00Z"},
            {"id": "demo-2", "symbol": "AAPL", "side": "buy", "quantity": 25, "type": "market", "status": "filled", "filled_price": 195.00, "created_at": "2026-01-15T10:15:00Z"},
            {"id": "demo-3", "symbol": "INTC", "side": "buy", "quantity": 80, "type": "market", "status": "filled", "filled_price": 52.00, "created_at": "2026-02-01T10:00:00Z"},
        ],
        "watchlist": ["NVDA", "AAPL", "INTC", "TGT", "BTQ", "PANW", "MSFT", "GOOGL"],
    }


async def _demo_quote(workspace_id: str, ticker: str) -> Dict[str, Any]:
    """Demo quote for local dev."""
    import random
    base_prices = {"NVDA": 980.50, "AAPL": 224.31, "INTC": 61.50, "TGT": 152.30, "BTQ": 31.80, "PANW": 420.50, "MSFT": 415.80, "GOOGL": 175.00}
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
