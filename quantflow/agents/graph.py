"""LangGraph orchestration: technical, seasonal and sentiment agents fan out from
START, a rule-based synthesizer gates them against the baseline, and an optional
DeepSeek reviewer adds a bounded, advisory adjustment (live runs only).

All agents read only bars/records at or before ``as_of`` so the same graph is used
for point-in-time backtests and live recommendations.
"""
from __future__ import annotations

import operator
from functools import lru_cache
from typing import Annotated, Any, Callable, Optional, TypedDict

import numpy as np
import pandas as pd
from langgraph.graph import END, START, StateGraph

from ..election.calendar import cycle_phase
from ..recommend.engine import RuleEngine
from .gating import WeightAdjuster, volatility_regime

SentimentFn = Callable[[str, pd.Timestamp], dict]


class AgentState(TypedDict, total=False):
    ticker: str
    as_of: pd.Timestamp
    df: pd.DataFrame
    use_llm: bool
    technical: dict
    seasonal: dict
    sentiment: dict
    volatility: dict
    gate: dict
    llm_review: Optional[dict]
    final_confidence: float
    bias: str
    trace: Annotated[list, operator.add]


@lru_cache(maxsize=1)
def _ticker_topics() -> dict[str, list[tuple[str, int]]]:
    """ticker -> [(topic, side_sign)] from configs/election_scenarios.yml."""
    from ..election.scenarios import load_scenarios

    out: dict[str, list[tuple[str, int]]] = {}
    try:
        cfg = load_scenarios()
    except Exception:
        return out
    for b in cfg.get("baskets", {}).values():
        sign = 1 if b.get("side") == "long" else -1
        for t in b.get("tickers", []):
            out.setdefault(t.upper(), []).append(("senate_control", sign))
            if b.get("topic"):
                out[t.upper()].append((b["topic"], sign))
    return out


def db_sentiment(db_path: str = "sqlite:///quantflow.db") -> SentimentFn:
    """Sentiment provider reading persisted snapshots point-in-time (created_at <= as_of)."""
    from ..data.queries import sentiment_snapshots_as_of

    def _fn(ticker: str, as_of: pd.Timestamp) -> dict:
        topics = _ticker_topics().get(ticker.upper(), [])
        if not topics:
            return {"polarity": 0.0, "n": 0, "source": "none"}
        try:
            snaps = sentiment_snapshots_as_of(as_of, [t for t, _ in topics], limit=200, db_path=db_path)
        except Exception as exc:
            return {"polarity": 0.0, "n": 0, "source": f"error:{exc}"[:80]}
        if snaps.empty:
            return {"polarity": 0.0, "n": 0, "source": "empty"}
        latest = snaps.sort_values("created_at").groupby(["ticker_or_topic", "source"]).tail(1)
        signed = []
        for topic, sign in topics:
            vals = latest.loc[latest["ticker_or_topic"] == topic, "polarity_score"]
            if len(vals):
                signed.append(sign * float(vals.mean()))
        pol = float(np.clip(np.mean(signed), -1, 1)) if signed else 0.0
        return {"polarity": pol, "n": int(len(latest)), "source": "db"}

    return _fn


def _no_sentiment(ticker: str, as_of: pd.Timestamp) -> dict:
    return {"polarity": 0.0, "n": 0, "source": "disabled"}


def technical_agent(state: AgentState) -> dict:
    df = state["df"]
    eng = RuleEngine()
    base = 0.5 * eng.score_short_term(df) + 0.5 * eng.score_medium_term(df)
    breakdown = eng._score_breakdown(df)
    rich = float(np.mean(list(breakdown.values())))
    return {"technical": {"base_score": float(base), "score": rich, "breakdown": breakdown}, "trace": ["technical"]}


def seasonal_agent(state: AgentState, forward_days: int = 21) -> dict:
    """Average forward return from the same calendar window in prior years with the
    same presidential-cycle phase (strictly before ``as_of``)."""
    df = state["df"]
    as_of = pd.Timestamp(state["as_of"])
    px = df["adj close"]
    phase = cycle_phase(as_of.year)
    doy = as_of.day_of_year
    rets = []
    for year in sorted(set(px.index.year)):
        if year >= as_of.year or cycle_phase(year) != phase:
            continue
        yr = px[px.index.year == year]
        start = yr[yr.index.day_of_year >= doy]
        if len(start) <= forward_days:
            continue
        rets.append(float(start.iloc[forward_days] / start.iloc[0] - 1))
    avg = float(np.mean(rets)) if rets else 0.0
    score = float(np.clip(0.5 + 5.0 * avg, 0.0, 1.0))
    return {"seasonal": {"phase": phase, "avg_fwd_ret": avg, "n_years": len(rets), "score": score}, "trace": ["seasonal"]}


def make_sentiment_agent(sentiment_fn: SentimentFn):
    def sentiment_agent(state: AgentState) -> dict:
        s = sentiment_fn(state["ticker"], pd.Timestamp(state["as_of"]))
        s["score"] = float(np.clip(0.5 + 0.5 * s.get("polarity", 0.0), 0.0, 1.0))
        return {"sentiment": s, "trace": ["sentiment"]}

    return sentiment_agent


def make_synthesis(adjuster: WeightAdjuster, weights: tuple[float, float, float]):
    w_t, w_s, w_n = weights

    def synthesis(state: AgentState) -> dict:
        vol = volatility_regime(state["df"], state["as_of"])
        tech, seas, sent = state["technical"], state["seasonal"], state["sentiment"]
        agent_conf = w_t * tech["score"] + w_s * seas["score"] + w_n * sent["score"]
        gate = adjuster.decide(tech["base_score"], agent_conf, vol["elevated"], sent.get("polarity", 0.0))
        final = gate.final_confidence
        return {
            "volatility": vol,
            "gate": gate.to_dict(),
            "final_confidence": final,
            "bias": _bias(final),
            "trace": ["synthesis"],
        }

    return synthesis


def _bias(conf: float) -> str:
    return "long" if conf >= 0.6 else ("short" if conf <= 0.4 else "neutral")


def llm_review(state: AgentState) -> dict:
    from .deepseek import review_signal

    review = review_signal(dict(state))
    if not review:
        return {"llm_review": None, "trace": ["llm_skipped"]}
    final = float(np.clip(state["final_confidence"] + review.get("adjustment", 0.0), 0.0, 1.0))
    return {"llm_review": review, "final_confidence": final, "bias": _bias(final), "trace": ["llm_review"]}


def _route_llm(state: AgentState) -> str:
    return "llm_review" if state.get("use_llm") else END


class AgentGraph:
    def __init__(
        self,
        sentiment_fn: Optional[SentimentFn] = None,
        adjuster: Optional[WeightAdjuster] = None,
        weights: tuple[float, float, float] = (0.5, 0.25, 0.25),
    ):
        self.adjuster = adjuster or WeightAdjuster()
        g = StateGraph(AgentState)
        g.add_node("technical", technical_agent)
        g.add_node("seasonal", seasonal_agent)
        g.add_node("sentiment", make_sentiment_agent(sentiment_fn or _no_sentiment))
        g.add_node("synthesis", make_synthesis(self.adjuster, weights))
        g.add_node("llm_review", llm_review)
        for n in ("technical", "seasonal", "sentiment"):
            g.add_edge(START, n)
        g.add_edge(["technical", "seasonal", "sentiment"], "synthesis")
        g.add_conditional_edges("synthesis", _route_llm, {"llm_review": "llm_review", END: END})
        g.add_edge("llm_review", END)
        self.app = g.compile()

    def run(self, ticker: str, df: pd.DataFrame, as_of=None, use_llm: bool = False) -> dict[str, Any]:
        as_of = pd.Timestamp(as_of) if as_of is not None else df.index[-1]
        view = df[df.index <= as_of]
        if len(view) < 60:
            raise ValueError(f"not enough history for {ticker} at {as_of.date()}")
        out = self.app.invoke({"ticker": ticker.upper(), "as_of": view.index[-1], "df": view, "use_llm": use_llm, "trace": []})
        out.pop("df", None)
        return out


def run_agents(ticker: str, use_llm: bool = False, db_path: str = "sqlite:///quantflow.db", persist: bool = True) -> dict[str, Any]:
    from ..features.indicators import compute_indicators, fetch_ohlcv

    df = compute_indicators(fetch_ohlcv(ticker, period="20y")).dropna(subset=["adj close"])
    pit = db_sentiment(db_path)
    decided_at = pd.Timestamp.now(tz="UTC").tz_localize(None)

    def live_sentiment(t: str, _bar_as_of: pd.Timestamp) -> dict:
        # Live decisions see news/odds ingested after the latest daily bar's midnight stamp.
        return pit(t, decided_at)

    result = AgentGraph(sentiment_fn=live_sentiment).run(ticker, df, use_llm=use_llm)
    if persist:
        from ..data.persistence import save_agent_states

        gate = result["gate"]
        save_agent_states(
            [
                {
                    "as_of": pd.Timestamp(result["as_of"]).to_pydatetime(),
                    "run_id": "live",
                    "ticker": ticker,
                    "horizon": "1m",
                    "base_confidence": gate["base_confidence"],
                    "agent_confidence": gate["agent_confidence"],
                    "final_confidence": result["final_confidence"],
                    "override_applied": gate["override_applied"],
                    "volatility_triggered": gate["volatility_triggered"],
                    "sentiment_multiplier": gate["sentiment_multiplier"],
                    "state": {k: result.get(k) for k in ("technical", "seasonal", "sentiment", "volatility", "llm_review")},
                }
            ],
            db_path,
        )
    result["as_of"] = str(pd.Timestamp(result["as_of"]).date())
    return result
