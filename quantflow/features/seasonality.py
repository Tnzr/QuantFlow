from __future__ import annotations

import pandas as pd
import yfinance as yf
from datetime import timedelta
from .indicators import fetch_ohlcv


def seasonality_by_doy(ticker: str, years: int = 20, by_cycle: bool = False) -> pd.DataFrame:
    """Average daily return by day-of-year.

    With ``by_cycle=True`` rows are additionally segmented by ``segment``
    (``election_year`` vs ``non_election_year``) and ``cycle_phase``
    (election / post_election / midterm / pre_election).
    """
    df = fetch_ohlcv(ticker, period=f"{years}y", interval="1d")
    df["ret"] = df["adj close"].pct_change()
    df["doy"] = df.index.dayofyear
    if not by_cycle:
        agg = df.groupby("doy")["ret"].agg(["mean", "std", "count"]).reset_index()
        return agg.rename(columns={"mean": "avg_ret", "std": "std_ret", "count": "n"})
    return seasonality_by_cycle_frame(df)


def seasonality_by_cycle_frame(df: pd.DataFrame) -> pd.DataFrame:
    from ..election.calendar import cycle_phase

    years = df.index.year
    df = df.assign(
        segment=["election_year" if y % 4 == 0 else "non_election_year" for y in years],
        cycle_phase=[cycle_phase(y) for y in years],
    )
    agg = df.groupby(["segment", "cycle_phase", "doy"])["ret"].agg(["mean", "std", "count"]).reset_index()
    return agg.rename(columns={"mean": "avg_ret", "std": "std_ret", "count": "n"})


def cycle_window_stats(df: pd.DataFrame, start_doy: int, end_doy: int) -> pd.DataFrame:
    """Compounded return over a day-of-year window, per year, tagged by cycle phase."""
    from ..election.calendar import cycle_phase

    px = df["adj close"]
    rows = []
    for year, grp in px.groupby(px.index.year):
        w = grp[(grp.index.day_of_year >= start_doy) & (grp.index.day_of_year <= end_doy)]
        if len(w) < 2:
            continue
        rows.append({"year": year, "cycle_phase": cycle_phase(year), "ret": float(w.iloc[-1] / w.iloc[0] - 1)})
    return pd.DataFrame(rows)


def next_earnings_hint(ticker: str):
    t = yf.Ticker(ticker)
    try:
        cal = t.calendar
        if cal is not None and "Earnings Date" in cal.index:
            vals = cal.loc["Earnings Date"].values
            if len(vals) >= 1:
                return pd.to_datetime(vals[0]).date()
    except Exception:
        pass
    return None


def proximity_to_earnings(ticker: str, today=None):
    today = pd.Timestamp.today().date() if today is None else today
    d = next_earnings_hint(ticker)
    if d is None:
        return None
    return (d - today).days
