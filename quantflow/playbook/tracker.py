"""Run the playbook: resolve the election branch, step every symbol, alert, optionally paper-trade."""
from __future__ import annotations

import logging
import os
import threading

import pandas as pd
import requests

from ..data.persistence import (
    create_schema,
    load_playbook_states,
    save_playbook_alerts,
    save_playbook_state,
)
from ..execution.risk_guards import MAX_RISK_PCT
from .impact import UNIVERSES, ImpactRow, load_closes, screen, top_n, top_n_affordable
from .options import choose_contract, load_swing_chain
from .rules import RuleConfig, SignalState, State, indicators, step

log = logging.getLogger(__name__)
DEFAULT_EQUITY = 10_000.0
RESOLVED_P = 0.97
_CYCLE_LOCK = threading.Lock()  # one cycle at a time per process; broker dedupes via client_order_id


def resolve_outcome(p_dem: float | None, manual: str | None = None) -> str | None:
    """'D'/'R' once the Senate result is known; prediction markets ≥97% count as resolved."""
    manual = (manual or os.getenv("QUANTFLOW_SENATE_OUTCOME") or "").strip().upper()
    if manual in ("D", "R"):
        return manual
    if p_dem is None:
        return None
    if p_dem >= RESOLVED_P:
        return "D"
    if p_dem <= 1 - RESOLVED_P:
        return "R"
    return None


def senate_odds() -> float | None:
    from ..election.ingest import fetch_polymarket_odds

    return next((o["probability"] for o in fetch_polymarket_odds() if o["ticker_or_topic"] == "senate_control"), None)


def build_watchlist(p_dem: float, n: int = 5, closes: pd.DataFrame | None = None, universe: str = "affordable",
                    budget: float | None = DEFAULT_EQUITY * MAX_RISK_PCT, fit_fn=None,
                    executable_only: bool = False, outcome: str | None = None,
                    direction_mode: str = "event") -> list[ImpactRow]:
    """Top-``n`` sector-diverse symbols. With ``budget`` set, only symbols where a long call/put
    (or debit vertical, unless ``executable_only``) in the projected direction costs <= budget are kept.
    For execute mode pass the same broker-derived budget that ``run_cycle`` will size with."""
    cands = UNIVERSES[universe]
    closes = closes if closes is not None else load_closes(candidates=cands)
    rows = screen(closes, p_dem, cands)
    if budget:
        return [r for r, _ in top_n_affordable(rows, n, budget, fit_fn, executable_only=executable_only, outcome=outcome,
                                                     direction_mode=direction_mode)]
    return top_n(rows, n)


def _bars(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame | None:
    from ..features.indicators import fetch_ohlcv

    try:
        df = fetch_ohlcv(symbol, period=period, interval=interval)
    except Exception as exc:  # noqa: BLE001 - one bad feed must not stop the cycle
        log.warning("bars %s %s: %s", symbol, interval, exc)
        return None
    return df.dropna(subset=["close"]) if df is not None and not df.empty else None


def _fallback_impact(symbol: str, state: dict) -> dict:
    """Neutral prior for open positions saved before impact rows were persisted: exits (stop,
    target, time, opposite confidence) still evaluate; the event prior contributes 0.5 both ways."""
    return {"symbol": symbol, "benchmark": "", "z_dem": 0.0, "z_rep": 0.0, "projected_z": 0.0}


def execution_budget(broker, risk_pct: float = MAX_RISK_PCT) -> float:
    """Per-trade option budget from the broker account, matching ``run_cycle``'s execute-mode sizing."""
    return float(broker.get_account().equity) * risk_pct


def notify(alert: dict) -> None:
    """Log and, if ``QUANTFLOW_ALERT_WEBHOOK`` is set, POST ``{"text": ...}`` (Slack/Discord compatible)."""
    log.warning("[playbook] %s", alert["message"])
    url = os.getenv("QUANTFLOW_ALERT_WEBHOOK")
    if url:
        try:
            requests.post(url, json={"text": alert["message"], "content": alert["message"]}, timeout=5)
        except requests.RequestException as exc:
            log.warning("alert webhook failed: %s", exc)


def run_cycle(
    watch: list[dict],
    outcome: str | None,
    *,
    db_path: str = "sqlite:///quantflow.db",
    cfg: RuleConfig | None = None,
    equity: float = DEFAULT_EQUITY,
    risk_pct: float = MAX_RISK_PCT,
    execute: bool = False,
    broker=None,
    send: bool = True,
    bars_fn=_bars,
    chain_fn=load_swing_chain,
) -> dict:
    """One pass over the watchlist. ``watch`` items are ImpactRow dicts.

    In execute mode, sizing uses the broker's account equity (not the caller's ``equity``)
    and a confirmed entry only becomes ``IN_TRADE`` once the broker accepts the order;
    otherwise the symbol drops back to ``ARMED`` so it can re-trigger.
    """
    with _CYCLE_LOCK:
        return _run_cycle(watch, outcome, db_path, cfg or RuleConfig(), equity, risk_pct, execute, broker, send,
                          bars_fn, chain_fn)


def _run_cycle(watch, outcome, db_path, cfg, equity, risk_pct, execute, broker, send, bars_fn, chain_fn) -> dict:
    execute = bool(execute and broker is not None)
    if execute:
        equity = float(broker.get_account().equity)
    create_schema(db_path)
    prev_all = load_playbook_states(db_path)
    out = {"outcome": outcome, "signals": {}, "alerts": []}
    # Open positions are always managed, even if the symbol fell off today's entry watchlist.
    listed = {w["symbol"] for w in watch}
    held = [p.get("impact") or _fallback_impact(s, p) for s, p in prev_all.items()
            if p.get("state") == "in_trade" and s not in listed]
    for imp in list(watch) + held:
        sym = imp["symbol"]
        bench = imp.get("benchmark", "")
        bench = "" if bench in ("", "none") else bench
        bars = bars_fn(sym)
        if bars is None or len(bars) < 60:
            out["signals"][sym] = {"symbol": sym, "error": "insufficient bars"}
            continue
        bench_bars = bars_fn(bench) if bench else None
        intraday = bars_fn(sym, "1mo", "1h")
        ind = indicators(bars, bench_bars["close"] if bench_bars is not None else None, intraday, cfg)
        prev = SignalState.from_dict(prev_all[sym]) if sym in prev_all else SignalState(sym)
        st, alerts = step(prev, ind, imp, outcome, cfg)
        for a in alerts:
            ad = a.to_dict()
            if a.kind == "entry_confirmed":
                try:
                    ad["structure"] = choose_contract(chain_fn(sym), ind["close"], st.side, equity * risk_pct)
                except Exception as exc:  # noqa: BLE001
                    ad["structure"] = {"structure": None, "reason": f"chain error: {exc}"}
                s = ad["structure"]
                if s.get("structure"):
                    ad["message"] += f" | {s['structure']} {s.get('contract') or s.get('legs_occ')} x{s['qty']}"
                if execute:
                    if s.get("structure") in ("long_call", "long_put") and s["qty"] > 0:
                        coid = f"qf-pb-{sym}-{ind['as_of']}-{st.side}"
                        try:
                            order = broker.place_option_order(s["contract"], s["qty"], s["limit_price"], client_order_id=coid)
                            ad["order"] = getattr(order, "__dict__", order)
                        except Exception as exc:  # noqa: BLE001 - risk-guard rejections are reported, not raised
                            ad["order"] = {"error": str(exc)}
                    else:
                        ad["order"] = {"error": "no single-leg structure to execute"}
                    if "error" in ad["order"]:
                        st.state = State.ARMED
                        st.entry = st.stop = st.target = st.entered_on = None
                        ad["kind"] = "entry_not_executed"
                        ad["message"] += f" | NOT EXECUTED: {ad['order']['error']}"
            out["alerts"].append(ad)
            if send:
                notify(ad)
        save_playbook_state({**st.to_dict(), "impact": imp}, db_path)
        out["signals"][sym] = st.to_dict()
    if out["alerts"]:
        save_playbook_alerts(out["alerts"], db_path)
    return out
