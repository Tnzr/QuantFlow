from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import time
import os
import yaml

from finvizfinance.screener import Screener


@dataclass
class FinvizResult:
    ticker: str
    company: Optional[str]
    sector: Optional[str]
    industry: Optional[str]
    price: Optional[float]
    change: Optional[float]
    rel_volume: Optional[float]
    atr: Optional[float]
    rsi: Optional[float]
    country: Optional[str]


DEFAULT_VIEW_FIELDS = [
    "Ticker",
    "Company",
    "Sector",
    "Industry",
    "Price",
    "Change",
    "Rel Volume",
    "ATR",
    "RSI (14)",
    "Country",
]


# Default built-in presets (fallback)
PRESETS: Dict[str, List[str]] = {
    # 1-week momentum with liquidity and optionable
    "weekly_momo": [
        "sh_avgvol_o300",
        "sh_opt_option",
        "ta_perf_1w10o",
        "ta_sma50_pa",
        "ta_sma200_pa",
        "sh_price_o5",
    ],
    # Bearish weekly momentum (for puts)
    "weekly_bear": [
        "sh_avgvol_o300",
        "sh_opt_option",
        "ta_perf_1w-10u",
        "ta_sma50_pb",
        "sh_price_o5",
    ],
    # 1-month swing with consistent trend and reasonable volatility
    "monthly_swing": [
        "sh_avgvol_o500",
        "sh_opt_option",
        "ta_perf_4w20o",
        "ta_sma50_pa",
        "ta_sma200_pa",
        "ta_beta_1to2",
        "sh_price_o5",
    ],
    # Bearish monthly swing
    "monthly_bear": [
        "sh_avgvol_o500",
        "sh_opt_option",
        "ta_perf_4w-10u",
        "ta_sma50_pb",
        "ta_beta_1to2",
        "sh_price_o5",
    ],
    # 3-6m medium term trenders
    "midterm_trenders": [
        "sh_avgvol_o500",
        "sh_opt_option",
        "ta_perf_half_20o",
        "ta_sma200_pa",
        "ta_beta_1to2",
        "sh_price_o5",
    ],
    # Bearish medium term
    "midterm_bear": [
        "sh_avgvol_o500",
        "sh_opt_option",
        "ta_perf_half-10u",
        "ta_sma200_pb",
        "ta_beta_1to2",
        "sh_price_o5",
    ],
    # Bullish reversal
    "reversal_bull": [
        "sh_avgvol_o300",
        "sh_opt_option",
        "ta_rsi_os40",
        "ta_sma50_pb",
        "sh_price_o5",
    ],
    # Bearish reversal
    "reversal_bear": [
        "sh_avgvol_o300",
        "sh_opt_option",
        "ta_rsi_ob60",
        "ta_sma50_pa",
        "sh_price_o5",
    ],
    # LEAPS candidates: quality, larger caps trending up
    "leaps_quality": [
        "sh_avgvol_o1000",
        "sh_opt_option",
        "sh_curvol_o1000",
        "sh_price_o10",
        "fa_epsqoq_pos",
        "fa_salesqoq_pos",
        "ta_sma200_pa",
    ],
}

# Attempt to load overrides from YAML
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "configs", "finviz_presets.yml")
if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, "r") as f:
            data = yaml.safe_load(f) or {}
            if isinstance(data, dict) and data:
                PRESETS.update({k: list(v) for k, v in data.items() if isinstance(v, list)})
    except Exception:
        pass


def run_screener(preset: str, view: Optional[List[str]] = None, retry: int = 3, sleep: float = 1.2):
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset: {preset}")
    filters = PRESETS[preset]
    sc = Screener(filters=filters, order="-change")

    last_err = None
    for _ in range(retry):
        try:
            df = sc.get_screen(view=view or DEFAULT_VIEW_FIELDS)
            return df
        except Exception as e:
            last_err = e
            time.sleep(sleep)
    raise last_err
