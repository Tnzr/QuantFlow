"""Daily OHLCV (Alpaca, yfinance fallback) and FRED macro series with a
Parquet cache, returned as wide date x ticker panels.

Caveats recorded in every experiment:
* Alpaca's free ``iex`` feed volume is a fraction of consolidated volume;
  use volume only in relative (z-score / ratio) form.
* Universes are passed in explicitly. A list of today's tickers is
  survivorship-biased; see ``universe_note`` in the registry.
* FRED values are first-release-agnostic (revised); ``publication_lag_days``
  shifts them so a backtest only sees values after they would be published.
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

FIELDS = ("open", "high", "low", "close", "volume")
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def cache_dir() -> Path:
    p = Path(os.getenv("QUANTFLOW_RESEARCH_CACHE", "data/research_cache"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def _alpaca_daily(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    from quantflow.data.alpaca_client import _ALPACA_DATA_BASE, _get, is_configured

    if not is_configured():
        return pd.DataFrame()
    params = {
        "timeframe": "1Day",
        "start": pd.Timestamp(start).strftime("%Y-%m-%d"),
        "adjustment": "all",
        "feed": os.getenv("ALPACA_DATA_FEED", "iex"),
        "limit": 10000,
        "sort": "asc",
    }
    if end:
        params["end"] = pd.Timestamp(end).strftime("%Y-%m-%d")
    rows, token = [], None
    url = f"{_ALPACA_DATA_BASE}/stocks/{ticker}/bars"
    while True:
        if token:
            params["page_token"] = token
        data = _get(url, params)
        rows.extend(data.get("bars") or [])
        token = data.get("next_page_token")
        if not token:
            break
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).rename(columns={"t": "date", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None).dt.normalize()
    return df.set_index("date")[list(FIELDS)].astype(float)


def _yfinance_daily(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False, threads=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    return df[list(FIELDS)].astype(float).dropna(subset=["close"])


def load_daily(ticker: str, start: str = "2005-01-01", end: str | None = None, source: str = "auto", refresh: bool = False) -> tuple[pd.DataFrame, str]:
    """Return (bars, source_used). ``source`` is "alpaca", "yfinance" or "auto"."""
    ticker = ticker.upper().strip()
    path = cache_dir() / f"{ticker}_{source}_{start}_{end or 'latest'}.parquet"
    if path.exists() and not refresh:
        df = pd.read_parquet(path)
        return df, str(df.attrs.get("source", source))
    df, used = pd.DataFrame(), source
    if source in ("auto", "alpaca"):
        try:
            df, used = _alpaca_daily(ticker, start, end), "alpaca"
        except Exception as exc:  # noqa: BLE001 - any Alpaca failure falls back to yfinance
            log.warning("alpaca bars failed for %s: %s", ticker, exc)
    if df.empty and source in ("auto", "yfinance"):
        df, used = _yfinance_daily(ticker, start, end), "yfinance"
    if not df.empty:
        if end is None:
            # drop today's partial bar so cached history is final
            df = df[df.index < pd.Timestamp.now().normalize()]
        df.attrs["source"] = used
        df.to_parquet(path)
    return df, used


def load_panel(tickers: list[str], start: str = "2005-01-01", end: str | None = None, source: str = "auto", refresh: bool = False) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """Wide panels {field: DataFrame[date x ticker]} plus per-ticker source."""
    frames, sources = {}, {}
    for t in tickers:
        df, used = load_daily(t, start, end, source, refresh)
        if df.empty:
            log.warning("no data for %s", t)
            continue
        frames[t.upper()] = df
        sources[t.upper()] = used
    panel = {f: pd.DataFrame({t: d[f] for t, d in frames.items()}).sort_index() for f in FIELDS}
    if panel["close"].shape[1]:
        panel = {f: v.reindex(panel["close"].index) for f, v in panel.items()}
    panel["volume"] = panel["volume"].where(panel["volume"] > 0)
    return panel, sources


def load_fred(series_id: str, publication_lag_days: int = 1, refresh: bool = False) -> pd.Series:
    """FRED series from the keyless CSV endpoint, shifted by publication lag."""
    path = cache_dir() / f"fred_{series_id}.parquet"
    if path.exists() and not refresh:
        s = pd.read_parquet(path).iloc[:, 0]
    else:
        resp = requests.get(FRED_CSV, params={"id": series_id}, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        date_col = "observation_date" if "observation_date" in df.columns else df.columns[0]
        s = pd.to_numeric(df[series_id], errors="coerce")
        s.index = pd.to_datetime(df[date_col])
        s = s.dropna().rename(series_id)
        s.to_frame().to_parquet(path)
    s.index = pd.DatetimeIndex(s.index) + pd.Timedelta(days=publication_lag_days)
    return s


def asof_align(series: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
    """Last value published on or before each calendar date."""
    s = series.sort_index()
    return s.reindex(s.index.union(calendar)).ffill().reindex(calendar)
