from __future__ import annotations

from .base import Broker
from .robinhood_mcp import RobinhoodMCPBroker


BROKER_MODE_MCP = "robinhood_mcp"


def make_broker(mode: str = BROKER_MODE_MCP) -> Broker:
    mode = (mode or BROKER_MODE_MCP).strip().lower()
    if mode != BROKER_MODE_MCP:
        raise ValueError("Unsupported broker mode. QuantFlow now supports MCP-only broker mode: robinhood_mcp")
    return RobinhoodMCPBroker()
