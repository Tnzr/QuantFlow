from .env import load_dev_env

load_dev_env()

from .recommend.engine import RuleEngine, Rec
from .recommend.options_picker import pick_affordable_contracts
from .data.finviz_client import run_screener, PRESETS
from .data.persistence import create_schema, save_finviz_snapshot, save_recommendations, get_engine
from .broker.factory import make_broker, BROKER_MODE_MCP

__version__ = "0.1.0"

__all__ = [
    "RuleEngine",
    "Rec",
    "pick_affordable_contracts",
    "run_screener",
    "PRESETS",
    "create_schema",
    "save_finviz_snapshot",
    "save_recommendations",
    "get_engine",
    "make_broker",
    "BROKER_MODE_MCP",
    "__version__",
]
