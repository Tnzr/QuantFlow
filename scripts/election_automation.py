"""Scheduled election pipeline: ingest -> agents -> resilience gate -> signed proposals.

Proposals are only produced for tickers whose walk-forward out-of-sample Sharpe is
positive and whose agents agree with the scenario basket side. Nothing is sent to a
broker; proposals are Alpaca-paper-only and still pass the 2%/1% risk guards at
execution time.

    python scripts/election_automation.py --tickers XLV,HCA,SPY --publish
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from quantflow.agents.graph import _ticker_topics, db_sentiment, run_agents
from quantflow.backtest.agentic_backtest import resilience_report
from quantflow.data.persistence import create_schema, save_allocation_proposal
from quantflow.data.supabase_sync import publish_proposal
from quantflow.election.ingest import ingest_all
from quantflow.execution.proposals import build_proposal, sign_proposal
from quantflow.options.verticals import recommend_structure


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default="SPY,XLV,HCA,XOP,KRE")
    ap.add_argument("--db", default="sqlite:///quantflow.db")
    ap.add_argument("--equity", type=float, default=100_000.0)
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--out", default="reports/election_automation.json")
    args = ap.parse_args()

    create_schema(args.db)
    summary = {"run_at": datetime.utcnow().isoformat(), "ingest": ingest_all(args.db), "tickers": {}}
    sides = {t: s for t, pairs in _ticker_topics().items() for topic, s in pairs if topic == "senate_control"}
    for t in [x.strip().upper() for x in args.tickers.split(",") if x.strip()]:
        entry: dict = {}
        try:
            agents = run_agents(t, db_path=args.db)
            res = resilience_report(t, start=args.start, sentiment_fn=db_sentiment(args.db))
            oos = res["walk_forward"]["oos_mean_sharpe"]
            entry.update(bias=agents["bias"], confidence=agents["final_confidence"], oos_sharpe=oos, gate=agents["gate"]["reason"])
            side = sides.get(t)
            bias = agents["bias"]
            if bias == "neutral" or oos <= 0 or (side is not None and (side > 0) != (bias == "long")):
                entry["decision"] = "skip: no robust, scenario-consistent edge"
            else:
                st = recommend_structure(t, bias, args.equity)
                spread = st.get("spread")
                if not spread or not spread.get("contracts"):
                    entry["decision"] = f"skip: {st.get('error', 'risk budget too small for spread')}"
                else:
                    prop = sign_proposal(build_proposal(
                        t, spread["structure"], spread["legs"], confidence=agents["final_confidence"],
                        max_loss=spread["contracts"] * spread["max_loss"], scenario="midterm_2026",
                        thesis=f"oos_sharpe={oos:.2f}; {agents['gate']['reason']}", expires_at=spread["expiry"],
                    ))
                    ok, err = publish_proposal(prop) if args.publish else (False, "publish disabled")
                    save_allocation_proposal(prop, args.db, published=ok, publish_error=None if ok else err)
                    entry.update(decision="proposed", proposal_id=prop["proposal_id"], published=ok, publish_error=err)
        except Exception as exc:
            entry["decision"] = f"error: {exc}"
        summary["tickers"][t] = entry
        print(t, entry.get("decision"))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
