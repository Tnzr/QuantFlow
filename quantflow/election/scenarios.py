"""Historical election event studies and scenario basket ranking."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import yaml
import yfinance as yf

from .calendar import Election, elections

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "configs", "election_scenarios.yml")
HORIZONS = {"1w": 5, "1m": 21, "3m": 63, "6m": 126}


def load_scenarios(path: str = CONFIG_PATH) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def download_closes(tickers: list[str], start: str = "2000-01-01", end: Optional[str] = None) -> pd.DataFrame:
    data = yf.download(sorted(set(tickers)), start=start, end=end, auto_adjust=True, progress=False, threads=True)
    closes = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]]
    if isinstance(closes, pd.Series):
        closes = closes.to_frame(tickers[0])
    return closes.sort_index()


def _anchor(index: pd.DatetimeIndex, day) -> Optional[int]:
    """Position of the last bar strictly before election day (the pre-result close)."""
    pos = index.searchsorted(pd.Timestamp(day)) - 1
    return int(pos) if pos >= 0 else None


def event_returns(closes: pd.DataFrame, events: list[Election], benchmark: str = "SPY") -> pd.DataFrame:
    """Long-format table of raw and benchmark-excess returns after each election."""
    rows = []
    idx = closes.index
    for ev in events:
        a = _anchor(idx, ev.day)
        if a is None:
            continue
        for ticker in closes.columns:
            p0 = closes[ticker].iloc[a]
            b0 = closes[benchmark].iloc[a] if benchmark in closes else np.nan
            if not np.isfinite(p0) or p0 <= 0:
                continue
            for label, n in HORIZONS.items():
                if a + n >= len(idx):
                    continue
                p1 = closes[ticker].iloc[a + n]
                b1 = closes[benchmark].iloc[a + n] if benchmark in closes else np.nan
                if not np.isfinite(p1):
                    continue
                ret = p1 / p0 - 1
                bench = b1 / b0 - 1 if np.isfinite(b0) and np.isfinite(b1) else np.nan
                rows.append(
                    {
                        "election": ev.label,
                        "year": ev.day.year,
                        "kind": ev.kind,
                        "pres_party_lost_senate": ev.pres_party_lost_senate,
                        "divided_after": ev.divided_after,
                        "ticker": ticker,
                        "horizon": label,
                        "ret": float(ret),
                        "excess": float(ret - bench) if np.isfinite(bench) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


@dataclass
class BasketEvidence:
    basket: str
    side: str
    topic: str
    thesis: str
    tickers: list[str]
    horizon: str
    analog_excess: float  # mean excess in elections where the president's party lost the Senate
    analog_n: int
    divided_excess: float  # mean excess in all midterms producing divided government
    divided_n: int
    all_midterm_excess: float
    hit_rate: float  # share of analog+divided observations agreeing with the side
    consistent: bool
    score: float
    per_election: dict = field(default_factory=dict)


def rank_baskets(
    study: pd.DataFrame,
    scenarios: dict,
    horizon: str = "3m",
) -> list[BasketEvidence]:
    """Score each basket by how well its side agrees with midterm history.

    Score = sign * (0.5 * analog excess + 0.5 * divided-government excess), where
    analog = midterms in which the president's party lost the Senate (2006, 2014).
    """
    out: list[BasketEvidence] = []
    mid = study[(study["kind"] == "midterm") & (study["horizon"] == horizon)]
    for name, b in scenarios["baskets"].items():
        sub = mid[mid["ticker"].isin(b["tickers"])]
        if sub.empty:
            continue
        per_el = sub.groupby("election")["excess"].mean()
        sign = 1.0 if b["side"] == "long" else -1.0
        analog = sub[sub["pres_party_lost_senate"] == True].groupby("election")["excess"].mean()  # noqa: E712
        divided = sub[sub["divided_after"] == True].groupby("election")["excess"].mean()  # noqa: E712
        a_mean = float(analog.mean()) if len(analog) else float("nan")
        d_mean = float(divided.mean()) if len(divided) else 0.0
        votes = pd.concat([analog, divided[~divided.index.isin(analog.index)]])
        hit = float((np.sign(votes) == sign).mean()) if len(votes) else 0.0
        score = sign * (0.5 * (a_mean if len(analog) else 0.0) + 0.5 * d_mean)
        out.append(
            BasketEvidence(
                basket=name,
                side=b["side"],
                topic=b.get("topic", ""),
                thesis=b["thesis"],
                tickers=list(b["tickers"]),
                horizon=horizon,
                analog_excess=a_mean,
                analog_n=int(len(analog)),
                divided_excess=d_mean,
                divided_n=int(len(divided)),
                all_midterm_excess=float(per_el.mean()),
                hit_rate=hit,
                consistent=len(analog) > 0 and score > 0 and hit >= 0.5,
                score=float(score),
                per_election={k: round(float(v), 4) for k, v in per_el.items()},
            )
        )
    return sorted(out, key=lambda e: e.score, reverse=True)


def ticker_table(study: pd.DataFrame, tickers: list[str], horizon: str = "3m") -> pd.DataFrame:
    mid = study[(study["kind"] == "midterm") & (study["horizon"] == horizon) & (study["ticker"].isin(tickers))]
    return mid.pivot_table(index="ticker", columns="year", values="excess")


def run_scenario_study(horizon: str = "3m", end: Optional[str] = None) -> dict:
    cfg = load_scenarios()
    tickers = sorted({t for b in cfg["baskets"].values() for t in b["tickers"]} | {cfg.get("benchmark", "SPY")})
    closes = download_closes(tickers, end=end)
    study = event_returns(closes, elections(resolved_only=True), cfg.get("benchmark", "SPY"))
    ranked = rank_baskets(study, cfg, horizon)
    return {"scenario": cfg["scenario"], "horizon": horizon, "baskets": [asdict(r) for r in ranked], "study": study}
