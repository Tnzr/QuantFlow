"""Election-aware agents, resilience backtests, verticals and signed proposals."""
from __future__ import annotations

import math
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from .auth import require_write_auth

router = APIRouter(tags=["election"])
DB = "sqlite:///quantflow.db"


def _clean(obj):
    """Recursively replace NaN/inf (not JSON compliant) with None."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        try:
            return _clean(obj.item())
        except (ValueError, TypeError):
            return str(obj)
    return obj


def _records(df):
    return [] if df is None or df.empty else df.astype(object).where(df.notna(), None).to_dict(orient="records")


@router.get("/election/calendar")
def election_calendar():
    from ..election.calendar import ELECTIONS, next_election

    nxt = next_election()
    return {
        "next": None if nxt is None else {"date": str(nxt.day), "kind": nxt.kind, "president_party": nxt.president_party},
        "elections": [
            {"date": str(e.day), "kind": e.kind, "president_party": e.president_party, "senate_after": e.senate_after,
             "house_after": e.house_after, "pres_party_lost_senate": e.pres_party_lost_senate}
            for e in ELECTIONS
        ],
    }


@router.post("/election/ingest")
def election_ingest(_auth: dict = Depends(require_write_auth)):
    from ..election.ingest import ingest_all

    return ingest_all(DB)


@router.get("/election/sentiment")
def election_sentiment(as_of: Optional[str] = None, limit: int = 200):
    from ..data.queries import election_events_as_of, sentiment_snapshots_as_of

    return {
        "snapshots": _records(sentiment_snapshots_as_of(as_of, limit=limit, db_path=DB)),
        "events": _records(election_events_as_of(as_of, limit=limit, db_path=DB)),
    }


@router.get("/election/scenario")
def election_scenario(horizon: str = "3m"):
    from ..election.scenarios import HORIZONS, run_scenario_study

    if horizon not in HORIZONS:
        raise HTTPException(400, f"horizon must be one of {sorted(HORIZONS)}")
    out = run_scenario_study(horizon)
    out.pop("study", None)
    return _clean(out)


@router.get("/agents/analyze")
def agents_analyze(ticker: str, use_llm: bool = False):
    from ..agents.graph import run_agents

    try:
        return _clean(run_agents(ticker.upper(), use_llm=use_llm, db_path=DB))
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@router.get("/agents/states")
def agents_states(ticker: Optional[str] = None, limit: int = 100):
    from ..data.queries import recent_agent_states

    return _clean({"states": _records(recent_agent_states(ticker, limit=limit, db_path=DB))})


@router.get("/backtest/resilience")
def backtest_resilience(ticker: str, start: str = "2018-01-01"):
    from ..agents.graph import db_sentiment
    from ..backtest.agentic_backtest import resilience_report

    try:
        return _clean(resilience_report(ticker.upper(), start=start, sentiment_fn=db_sentiment(DB)))
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@router.get("/options/vertical")
def options_vertical(ticker: str, bias: str = "long", equity: float = 100_000.0):
    from ..options.verticals import recommend_structure

    if bias not in ("long", "short"):
        raise HTTPException(400, "bias must be long or short")
    return _clean(recommend_structure(ticker.upper(), bias, equity))


class ProposalRequest(BaseModel):
    ticker: str
    bias: str = "long"
    equity: float = Field(100_000.0, gt=0)
    publish: bool = False


@router.post("/proposals")
def create_proposal(req: ProposalRequest, _auth: dict = Depends(require_write_auth)):
    """Agents + vertical + risk caps -> Ed25519-signed proposal (optionally mirrored to Supabase)."""
    from ..agents.graph import run_agents
    from ..data.persistence import save_allocation_proposal
    from ..data.supabase_sync import publish_proposal
    from ..execution.proposals import build_proposal, sign_proposal
    from ..options.verticals import recommend_structure

    ticker = req.ticker.upper()
    bias = req.bias.lower()
    if bias not in ("long", "short"):
        raise HTTPException(400, "bias must be 'long' or 'short'")
    agents = run_agents(ticker, db_path=DB)
    if agents["bias"] != bias:
        raise HTTPException(409, f"agents bias is {agents['bias']} ({agents['gate']['reason']}); refusing to sign a {bias} proposal")
    structure = recommend_structure(ticker, bias, req.equity)
    spread = structure.get("spread")
    if not spread or not spread.get("contracts"):
        raise HTTPException(422, structure.get("error") or "spread max loss exceeds 1% risk budget")
    proposal = sign_proposal(
        build_proposal(
            ticker, spread["structure"], spread["legs"],
            confidence=agents["final_confidence"],
            max_loss=spread["contracts"] * spread["max_loss"],
            thesis=f"agents bias={agents['bias']} ({agents['gate']['reason']})",
            scenario="midterm_2026",
            expires_at=spread["expiry"],
            evidence={"gate": agents["gate"], "preference": structure.get("preference"), "iv_crush": structure.get("iv_crush", {}).get("scenario")},
        )
    )
    published, err = (publish_proposal(proposal) if req.publish else (False, None))
    save_allocation_proposal(proposal, DB, published=published, publish_error=None if published else err)
    return _clean({"proposal": proposal, "published": published, "publish_error": err})


@router.get("/proposals")
def list_proposals(limit: int = 50):
    from ..data.queries import recent_allocation_proposals

    return {"proposals": _records(recent_allocation_proposals(limit, DB))}


class VerifyRequest(BaseModel):
    proposal: dict


@router.post("/proposals/verify")
def verify(req: VerifyRequest):
    from ..execution.proposals import verify_proposal

    # Only server-trusted keys count; the proposal's embedded public_key is never trusted.
    return {"valid": verify_proposal(req.proposal)}


class RiskCheckRequest(BaseModel):
    side: str
    qty: float
    entry_price: float
    equity: float
    reference_price: Optional[float] = None
    stop_price: Optional[float] = None
    max_loss_per_unit: Optional[float] = None


@router.post("/risk/check")
def risk_check(req: RiskCheckRequest):
    from ..execution.risk_guards import check_order_risk

    return check_order_risk(
        req.side, req.qty, req.entry_price, req.equity,
        reference_price=req.reference_price, stop_price=req.stop_price, max_loss_per_unit=req.max_loss_per_unit,
    ).to_dict()
