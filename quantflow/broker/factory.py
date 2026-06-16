from __future__ import annotations

from .base import Broker
from .robinhood import RobinhoodBroker
from .robinhood_mcp import RobinhoodMCPBroker


BROKER_MODE_LEGACY = "legacy_robin_stocks"
BROKER_MODE_MCP = "robinhood_mcp"


def make_broker(mode: str = BROKER_MODE_LEGACY) -> Broker:
    mode = (mode or BROKER_MODE_LEGACY).strip().lower()
    if mode == BROKER_MODE_MCP:
        return RobinhoodMCPBroker()
    return RobinhoodBroker()
