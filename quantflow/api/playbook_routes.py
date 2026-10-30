"""Event playbook: midterm impact top-N, rules-tree signals, alerts, and a cycle trigger."""
from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from .auth import require_write_auth
from .election_routes import DB, _clean

router = APIRouter(tags=["playbook"])
_CACHE: dict = {}
TTL = 900


def _watchlist(n: int, universe: str = "affordable") -> dict:
    from ..playbook.tracker import build_watchlist, resolve_outcome, senate_odds

    hit = _CACHE.get((n, universe))
    if hit and time.time() - hit[0] < TTL:
        return hit[1]
    p = senate_odds()
    rows = build_watchlist(p if p is not None else 0.5, n, universe=universe)
    data = {"p_dem_senate": p, "outcome": resolve_outcome(p), "top": [r.to_dict() for r in rows]}
    _CACHE[(n, universe)] = (time.time(), data)
    return data


@router.get("/playbook/midterm")
def playbook_midterm(n: int = 5, live: bool = True, universe: str = "affordable"):
    from ..playbook.impact import UNIVERSES, live_state

    if universe not in UNIVERSES:
        raise HTTPException(400, f"universe must be one of {sorted(UNIVERSES)}")
    data = _watchlist(n, universe)
    if live:
        data = {**data, "top": [{**r, "live": live_state(r["symbol"], UNIVERSES[universe][r["symbol"]][0])} for r in data["top"]]}
    return _clean(data)


@router.get("/playbook/signals")
def playbook_signals():
    from ..data.persistence import create_schema, load_playbook_states

    create_schema(DB)
    return _clean(list(load_playbook_states(DB).values()))


@router.get("/playbook/alerts")
def playbook_alerts(limit: int = 100):
    from ..data.persistence import create_schema, list_playbook_alerts

    create_schema(DB)
    return _clean(list_playbook_alerts(limit, DB))


class CycleRequest(BaseModel):
    n: int = 5
    universe: str = "affordable"
    outcome: str | None = None  # "D" | "R" override; default resolves from Polymarket
    execute: bool = False  # place Alpaca *paper* option orders on confirmed entries
    equity: float = 10_000.0


@router.post("/playbook/cycle")
def playbook_cycle(req: CycleRequest, _auth: dict = Depends(require_write_auth)):  # noqa: B008
    from ..playbook.impact import UNIVERSES
    from ..playbook.tracker import resolve_outcome, run_cycle

    if req.universe not in UNIVERSES:
        raise HTTPException(400, f"universe must be one of {sorted(UNIVERSES)}")
    data = _watchlist(req.n, req.universe)
    outcome = resolve_outcome(data["p_dem_senate"], req.outcome)
    broker = None
    if req.execute:
        from ..broker.alpaca_paper import AlpacaPaperTrading

        broker = AlpacaPaperTrading()
    return _clean(run_cycle(data["top"], outcome, db_path=DB, equity=req.equity, execute=req.execute, broker=broker))
