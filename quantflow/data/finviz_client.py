from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import time
import os
import yaml
import sys
import asyncio
import concurrent.futures
from contextlib import contextmanager

import requests
from bs4 import BeautifulSoup

from finvizfinance.screener.overview import Overview
from finvizfinance.screener.base import filter_dict


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


@dataclass
class ScrapingRuntimeConfig:
    provider: str
    mode: str
    proxy_configured: bool
    proxy_env_source: Optional[str]
    api_endpoint: Optional[str]
    scraping_api_key_set: bool
    scrapingbee_api_key_set: bool
    notes: List[str]


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
        "EPS growthqtr over qtr": "Positive (>0%)",
        "Sales growthqtr over qtr": "Positive (>0%)",
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
                for k, v in data.items():
                    if isinstance(v, dict):
                        # Accept dictionaries directly (human-readable names or code->value mappings).
                        PRESETS[k] = {str(kk): str(vv) for kk, vv in v.items()}
                    elif isinstance(v, list):
                        # Accept URL code lists directly (e.g., sh_avgvol_o300, ta_sma50_pa).
                        PRESETS[k] = [str(kk) for kk in v if str(kk).strip()]
    except Exception as e:
        print(f"[FinvizClient] Error loading YAML presets: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)


def _build_legacy_code_lookup() -> Dict[str, tuple[str, str]]:
    lookup: Dict[str, tuple[str, str]] = {}
    for filter_name, meta in filter_dict.items():
        prefix = str(meta.get("prefix", "")).strip()
        options = meta.get("option", {}) or {}
        for option_name, url_code in options.items():
            code = str(url_code).strip()
            if prefix and code:
                lookup[f"{prefix}_{code}"] = (filter_name, option_name)
    return lookup


LEGACY_CODE_LOOKUP = _build_legacy_code_lookup()


def _resolve_provider() -> str:
    provider = os.getenv("QF_SCRAPING_PROVIDER", "").strip().lower()
    if provider:
        return provider
    if os.getenv("QF_SCRAPING_PROXY_URL", "").strip() or os.getenv("SMARTPROXY_PROXY_URL", "").strip():
        return "smartproxy"
    if os.getenv("SCRAPINGBEE_API_KEY", "").strip() or os.getenv("SCRAPINGBEE_PROXY_URL", "").strip():
        return "scrapingbee"
    if os.getenv("SCRAPING_API_KEY", "").strip() or os.getenv("SCRAPE_DO_PROXY_URL", "").strip():
        return "scrape_do"
    return "direct"


def _resolve_scrape_do_endpoint() -> str:
    return os.getenv("SCRAPE_DO_ENDPOINT", "https://api.scrape.do").strip() or "https://api.scrape.do"


def _resolve_scrapingbee_endpoint() -> str:
    return os.getenv("SCRAPINGBEE_ENDPOINT", "https://app.scrapingbee.com/api/v1/").strip() or "https://app.scrapingbee.com/api/v1/"


def _resolve_proxy_url() -> tuple[str, Optional[str]]:
    # Priority: explicit QuantFlow var, then compatibility aliases used in ops.
    for key in (
        "QF_SCRAPING_PROXY_URL",
        "SENSITIVE_LOOKUP_PROXY_URL",
        "SMARTPROXY_PROXY_URL",
        "SCRAPINGBEE_PROXY_URL",
        "SCRAPE_DO_PROXY_URL",
    ):
        value = os.getenv(key, "").strip()
        if value:
            return value, key
    return "", None


def get_scraping_runtime_config() -> ScrapingRuntimeConfig:
    provider = _resolve_provider()
    proxy_url, source_key = _resolve_proxy_url()
    scrape_api_key = bool(os.getenv("SCRAPING_API_KEY", "").strip())
    scrapingbee_key = bool(os.getenv("SCRAPINGBEE_API_KEY", "").strip())
    notes: List[str] = []
    api_endpoint: Optional[str] = None

    mode = "direct"
    if proxy_url:
        mode = "proxy"
    elif provider in {"scrape_do", "scrapedo"} and scrape_api_key:
        mode = "api"
        api_endpoint = _resolve_scrape_do_endpoint()
    elif provider == "scrapingbee" and scrapingbee_key:
        mode = "api"
        api_endpoint = _resolve_scrapingbee_endpoint()

    if provider in {"smartproxy", "proxy"} and not proxy_url:
        notes.append(
            "No proxy URL configured for Finviz transport. Set QF_SCRAPING_PROXY_URL, "
            "SENSITIVE_LOOKUP_PROXY_URL, SMARTPROXY_PROXY_URL, SCRAPE_DO_PROXY_URL, or SCRAPINGBEE_PROXY_URL."
        )

    if provider in {"scrape_do", "scrapedo"} and not scrape_api_key and not proxy_url:
        notes.append("SCRAPING_API_KEY is not set. Configure key for API mode or set a proxy URL.")

    if provider == "scrapingbee" and not scrapingbee_key and not proxy_url:
        notes.append("SCRAPINGBEE_API_KEY is not set. Configure key for API mode or set a proxy URL.")

    return ScrapingRuntimeConfig(
        provider=provider,
        mode=mode,
        proxy_configured=bool(proxy_url),
        proxy_env_source=source_key,
        api_endpoint=api_endpoint,
        scraping_api_key_set=scrape_api_key,
        scrapingbee_api_key_set=scrapingbee_key,
        notes=notes,
    )


def _append_params_to_url(url: str, params: Optional[dict]) -> str:
    if not params:
        return url
    req = requests.Request("GET", url, params=params)
    prepared = req.prepare()
    return prepared.url


def _provider_web_scrap(cfg: ScrapingRuntimeConfig, url: str, params: Optional[dict] = None):
    target_url = _append_params_to_url(url, params)
    timeout_value = float(os.getenv("QF_SCRAPING_TIMEOUT", "15"))

    if cfg.provider in {"scrape_do", "scrapedo"}:
        api_key = os.getenv("SCRAPING_API_KEY", "").strip()
        if not api_key:
            raise ValueError("SCRAPING_API_KEY is required for scrape_do API mode")
        endpoint = cfg.api_endpoint or _resolve_scrape_do_endpoint()
        api_params = {
            "token": api_key,
            "url": target_url,
        }
        if os.getenv("SCRAPE_DO_SUPER", "").strip():
            api_params["super"] = os.getenv("SCRAPE_DO_SUPER", "").strip()
        response = requests.get(endpoint, params=api_params, timeout=timeout_value)
        response.raise_for_status()
        return BeautifulSoup(response.text, "lxml")

    if cfg.provider == "scrapingbee":
        api_key = os.getenv("SCRAPINGBEE_API_KEY", "").strip()
        if not api_key:
            raise ValueError("SCRAPINGBEE_API_KEY is required for scrapingbee API mode")
        endpoint = cfg.api_endpoint or _resolve_scrapingbee_endpoint()
        api_params = {
            "api_key": api_key,
            "url": target_url,
            "render_js": os.getenv("SCRAPINGBEE_RENDER_JS", "false").strip() or "false",
        }
        country_code = os.getenv("SCRAPINGBEE_COUNTRY_CODE", "").strip()
        if country_code:
            api_params["country_code"] = country_code
        premium_proxy = os.getenv("SCRAPINGBEE_PREMIUM_PROXY", "").strip()
        if premium_proxy:
            api_params["premium_proxy"] = premium_proxy
        response = requests.get(endpoint, params=api_params, timeout=timeout_value)
        response.raise_for_status()
        return BeautifulSoup(response.text, "lxml")

    raise ValueError(f"Unsupported API scraping provider for Finviz transport: {cfg.provider}")


@contextmanager
def _patched_finviz_web_scrap(cfg: ScrapingRuntimeConfig):
    if cfg.mode != "api":
        yield
        return

    import finvizfinance.util as util_mod
    import finvizfinance.screener.base as base_mod

    old_util = util_mod.web_scrap
    old_base = base_mod.web_scrap

    def _patched(url, params=None):
        return _provider_web_scrap(cfg, url, params=params)

    util_mod.web_scrap = _patched
    base_mod.web_scrap = _patched
    try:
        yield
    finally:
        util_mod.web_scrap = old_util
        base_mod.web_scrap = old_base


def _apply_proxy_env_from_qf() -> None:
    """Map QuantFlow proxy env to standard requests proxy env vars.

    finvizfinance uses requests under the hood, which honors HTTP(S)_PROXY.
    """
    proxy_url, _ = _resolve_proxy_url()
    if proxy_url:
        os.environ.setdefault("HTTP_PROXY", proxy_url)
        os.environ.setdefault("HTTPS_PROXY", proxy_url)
        return

    cfg = get_scraping_runtime_config()
    if cfg.provider in {"smartproxy", "proxy"}:
        print(
            "[FinvizClient] Provider configured but no proxy URL found for Finviz transport. "
            "Set one of QF_SCRAPING_PROXY_URL, SENSITIVE_LOOKUP_PROXY_URL, SMARTPROXY_PROXY_URL, "
            "SCRAPE_DO_PROXY_URL, or SCRAPINGBEE_PROXY_URL.",
            file=sys.stderr,
        )


def _convert_legacy_codes(filters) -> Dict[str, str]:
    out: Dict[str, str] = {}

    def _consume_code(raw_code: str) -> None:
        code = str(raw_code).strip()
        if not code:
            return
        if code in LEGACY_CODE_LOOKUP:
            filter_name, option_name = LEGACY_CODE_LOOKUP[code]
            out[filter_name] = option_name
        else:
            # Keep unknown codes so callers can see exact invalid code in downstream errors.
            out[code] = ""

    if isinstance(filters, list):
        for code in filters:
            _consume_code(str(code))
        return out

    if isinstance(filters, dict):
        for k, v in filters.items():
            key = str(k).strip()
            if key in filter_dict:
                out[key] = str(v)
            elif key in LEGACY_CODE_LOOKUP:
                filter_name, option_name = LEGACY_CODE_LOOKUP[key]
                out[filter_name] = option_name
            else:
                out[key] = str(v)
        return out

    _consume_code(str(filters))
    return out


def run_screener(preset: str, view: Optional[List[str]] = None, retry: int = 3, sleep: float = 1.2):
    try:
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset: {preset}")
        filters = PRESETS[preset]
        # Per finvizfinance 1.2.0, set_filter expects a dict of filter codes.
        filters_dict = _convert_legacy_codes(filters)
        cfg = get_scraping_runtime_config()
        _apply_proxy_env_from_qf()
        last_err = None
        with _patched_finviz_web_scrap(cfg):
            for attempt in range(retry):
                try:
                    sc = Overview()
                    sc.set_filter(filters_dict=filters_dict)
                    df = sc.screener_view(order="Ticker")
                    return df
                except Exception as e:
                    last_err = e
                    print(
                        f"[FinvizClient] Screener attempt {attempt + 1}/{retry} failed for {preset}: {e}",
                        file=sys.stderr,
                    )
                    import traceback
                    traceback.print_exc(file=sys.stderr)
                    if attempt + 1 < retry:
                        time.sleep(sleep * (attempt + 1))
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
