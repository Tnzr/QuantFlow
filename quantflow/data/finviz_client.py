from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import time
import os
import yaml
import sys
import asyncio
import concurrent.futures

from finvizfinance.screener.overview import Overview


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
# NOTE: These must use the human-readable filter names as in finvizfinance docs, not Finviz URL codes.
PRESETS: Dict[str, dict] = {
    # 1-week momentum with liquidity and optionable
    "weekly_momo": {
        "Average Volume": "Over 300K",
        "Option/Short": "Optionable",
        "Performance": "Week Up",
        "50-Day Simple Moving Average": "Price above SMA50",
        "200-Day Simple Moving Average": "Price above SMA200",
        "Price": "Over $5",
    },
    # Bearish weekly momentum (for puts)
    "weekly_bear": {
        "Average Volume": "Over 300K",
        "Option/Short": "Optionable",
        "Performance": "Week Down",
        "50-Day Simple Moving Average": "Price below SMA50",
        "Price": "Over $5",
    },
    # 1-month swing with consistent trend and reasonable volatility
    "monthly_swing": {
        "Average Volume": "Over 500K",
        "Option/Short": "Optionable",
        "Performance": "Month Up",
        "50-Day Simple Moving Average": "Price above SMA50",
        "200-Day Simple Moving Average": "Price above SMA200",
        "Beta": "Over 1",
        "Price": "Over $5",
    },
    # Bearish monthly swing
    "monthly_bear": {
        "Average Volume": "Over 500K",
        "Option/Short": "Optionable",
        "Performance": "Month Down",
        "50-Day Simple Moving Average": "Price below SMA50",
        "Beta": "Over 1",
        "Price": "Over $5",
    },
    # 3-6m medium term trenders
    "midterm_trenders": {
        "Average Volume": "Over 500K",
        "Option/Short": "Optionable",
        "Performance": "Half Up",
        "200-Day Simple Moving Average": "Price above SMA200",
        "Beta": "Over 1",
        "Price": "Over $5",
    },
    # Bearish medium term
    "midterm_bear": {
        "Average Volume": "Over 500K",
        "Option/Short": "Optionable",
        "Performance": "Half Down",
        "200-Day Simple Moving Average": "Price below SMA200",
        "Beta": "Over 1",
        "Price": "Over $5",
    },
    # Bullish reversal
    "reversal_bull": {
        "Average Volume": "Over 300K",
        "Option/Short": "Optionable",
        # Finviz valid RSI options: 'Oversold (40)', 'Oversold (30)', etc.
        "RSI (14)": "Oversold (40)",
        "50-Day Simple Moving Average": "Price below SMA50",
        "Price": "Over $5",
    },
    # Bearish reversal
    "reversal_bear": {
        "Average Volume": "Over 300K",
        "Option/Short": "Optionable",
        # Finviz valid RSI options: 'Overbought (60)', 'Overbought (70)', etc.
        "RSI (14)": "Overbought (60)",
        "50-Day Simple Moving Average": "Price above SMA50",
        "Price": "Over $5",
    },
    # LEAPS candidates: quality, larger caps trending up
    "leaps_quality": {
        "Average Volume": "Over 1M",
        "Option/Short": "Optionable",
        "Price": "Over $10",
        "EPS growthqtr over qtr": ">0%",
        "Sales growthqtr over qtr": ">0%",
        "200-Day Simple Moving Average": "Price above SMA200",
    },
}

# Attempt to load overrides from YAML
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "configs", "finviz_presets.yml")
if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, "r") as f:
            data = yaml.safe_load(f) or {}
            if isinstance(data, dict) and data:
                # Validate YAML presets: only allow filters present in CUSTOM_SCREENER_COLUMNS.values()
                from .constants import CUSTOM_SCREENER_COLUMNS
                valid_filters = set(CUSTOM_SCREENER_COLUMNS.values())
                for k, v in data.items():
                    if isinstance(v, dict):
                        filtered = {kk: vv for kk, vv in v.items() if kk in valid_filters}
                        if filtered:
                            PRESETS[k] = filtered
                    elif isinstance(v, list):
                        # If list, try to convert to dict with valid filters
                        filtered = {str(kk): "" for kk in v if str(kk) in valid_filters}
                        if filtered:
                            PRESETS[k] = filtered
    except Exception as e:
        print(f"[FinvizClient] Error loading YAML presets: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)


def run_screener(preset: str, view: Optional[List[str]] = None, retry: int = 3, sleep: float = 1.2):
    try:
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset: {preset}")
        filters = PRESETS[preset]
        # Per finvizfinance 1.2.0, set_filter expects a dict of filter codes
        if isinstance(filters, list):
            filters_dict = {f: "" for f in filters if isinstance(f, str)}
        elif isinstance(filters, dict):
            filters_dict = {str(k): str(v) for k, v in filters.items()}
        else:
            filters_dict = {str(filters): ""}
        try:
            sc = Overview()
        except Exception as e:
            print(f"[FinvizClient] Error initializing Overview: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            raise
        try:
            sc.set_filter(filters_dict=filters_dict)
        except Exception as e:
            print(f"[FinvizClient] Error in set_filter: {e}\nFilters: {filters_dict}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            raise
        last_err = None
        for _ in range(retry):
            try:
                df = sc.screener_view(order="Ticker")
                return df
            except Exception as e:
                last_err = e
                print(f"[FinvizClient] Error in screener_view: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)
                time.sleep(sleep)
        if last_err:
            raise last_err
    except Exception as e:
        print(f"[FinvizClient] run_screener failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise


def run_screener_async(preset: str, retry: int = 3, sleep: float = 1.2, loop=None):
    # Wrapper for running run_screener in a thread for asyncio
    loop = loop or asyncio.get_event_loop()
    return loop.run_in_executor(None, run_screener, preset)

async def run_screeners_async(preset_names, max_concurrent=4):
    # Run multiple screeners concurrently using asyncio and threads
    sem = asyncio.Semaphore(max_concurrent)
    results = {}
    async def run_one(name):
        async with sem:
            try:
                df = await run_screener_async(name)
                results[name] = df
            except Exception:
                results[name] = None
    tasks = [run_one(name) for name in preset_names]
    await asyncio.gather(*tasks)
    return results

def run_screeners_parallel(preset_names, max_workers=4):
    """
    Run multiple screeners in parallel using ThreadPoolExecutor.
    Returns a dict of preset name -> DataFrame (or None on error).
    """
    results = {}
    def run_one(name):
        try:
            return name, run_screener(name)
        except Exception:
            return name, None
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for name, df in executor.map(run_one, preset_names):
            results[name] = df
    return results
