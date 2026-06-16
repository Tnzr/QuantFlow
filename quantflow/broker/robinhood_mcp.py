from __future__ import annotations

from typing import List

from .base import Broker
from .types import Account, Position
from .mcp_client import RobinhoodMCPClient, MCPClientError


class RobinhoodMCPBroker(Broker):
    """Broker implementation scaffold for Robinhood MCP.

    Current behavior:
    - login(): marks connection intent
    - account()/positions(): explicit not-implemented to avoid silent unsafe behavior
    """

    def __init__(self, client: RobinhoodMCPClient | None = None):
        self.client = client or RobinhoodMCPClient()

    def login(self, *args, **kwargs) -> None:
        self.client.connect()

    def account(self) -> Account:
        self.login()
        try:
            payload = self.client.account()
            return Account(
                equity=float(payload.get("equity", 0.0) or 0.0),
                cash=float(payload.get("cash", 0.0) or 0.0),
                buying_power=float(payload.get("buying_power", 0.0) or 0.0),
            )
        except MCPClientError as e:
            raise RuntimeError(f"Robinhood MCP account read failed: {e}") from e

    def positions(self) -> List[Position]:
        self.login()
        try:
            payload = self.client.positions()
            rows = payload.get("positions", []) if isinstance(payload, dict) else []
            out: List[Position] = []
            for p in rows:
                qty = float(p.get("qty", 0.0) or 0.0)
                out.append(
                    Position(
                        ticker=str(p.get("ticker", "")).upper(),
                        qty=qty,
                        avg_price=float(p.get("avg_price", 0.0) or 0.0),
                        side="long" if qty >= 0 else "short",
                    )
                )
            return out
        except MCPClientError as e:
            raise RuntimeError(f"Robinhood MCP positions read failed: {e}") from e
