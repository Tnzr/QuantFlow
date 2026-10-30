import numpy as np
import pandas as pd
import pytest

from quantflow.election.calendar import Election
from quantflow.playbook import impact
from quantflow.playbook.options import choose_contract, occ_symbol
from quantflow.playbook.rules import (
    RuleConfig,
    SignalState,
    State,
    confidence,
    indicators,
    step,
)
from quantflow.playbook.tracker import resolve_outcome, run_cycle

IMP_UP = {"symbol": "AAA", "z_dem": 1.5, "z_rep": -1.0, "projected_z": 0.8}


def bars(n=120, drift=0.004, last_jump=0.0, last_vol=1.0, seed=0, start="2025-01-02"):
    rng = np.random.default_rng(seed)
    r = drift + 0.004 * rng.standard_normal(n)
    r[-1] += last_jump
    c = 100 * np.exp(np.cumsum(r))
    idx = pd.bdate_range(start, periods=n)
    v = np.full(n, 1e6)
    v[-1] *= last_vol
    return pd.DataFrame({"open": c, "high": c * 1.003, "low": c * 0.997, "close": c, "volume": v}, index=idx)


def test_tree_waits_for_event_and_tracks_both_sides():
    ind = indicators(bars())
    st, alerts = step(SignalState("AAA"), ind, IMP_UP, None)
    assert st.state == State.WATCH and st.side == "long" and not alerts
    assert st.bull > st.bear  # uptrend + positive prior


def test_confidence_mirrors_for_downtrend():
    up, dn = indicators(bars(drift=0.004)), indicators(bars(drift=-0.004))
    neutral = {"z_dem": 0.0, "z_rep": 0.0, "projected_z": 0.0}
    bu, be, _ = confidence(up, neutral, "D", RuleConfig())
    du, de, _ = confidence(dn, neutral, "D", RuleConfig())
    assert bu > be and de > du


def test_confirm_entry_then_trailing_stop_exit():
    b = bars(last_jump=0.03, last_vol=3.0)
    st, alerts = step(SignalState("AAA"), indicators(b), IMP_UP, "D")
    assert st.state == State.IN_TRADE and alerts[0].kind == "entry_confirmed"
    assert alerts[0].instrument == "long_call" and st.stop < st.entry < st.target
    crash = b.copy()
    nxt = crash.iloc[[-1]].copy()
    nxt.index = [crash.index[-1] + pd.offsets.BDay()]
    nxt[["open", "high", "low", "close"]] = st.stop * 0.98
    st2, alerts2 = step(st, indicators(pd.concat([crash, nxt])), IMP_UP, "D")
    assert st2.state == State.EXITED and st2.exit_reason == "stop" and alerts2[0].kind == "exit"
    st3, _ = step(st2, indicators(pd.concat([crash, nxt.set_axis([nxt.index[0] + pd.offsets.BDay()])])), IMP_UP, "D")
    assert st3.state != State.EXITED and st3.entry is None or st3.state == State.IN_TRADE


def test_rep_branch_flips_side_to_long_put():
    b = bars(drift=-0.004, last_jump=-0.03, last_vol=3.0)
    st, alerts = step(SignalState("AAA"), indicators(b), IMP_UP, "R")
    assert st.side == "short"
    assert st.state == State.IN_TRADE and alerts[0].instrument == "long_put" and st.stop > st.entry


def test_no_entry_without_volume_confirmation():
    st, alerts = step(SignalState("AAA"), indicators(bars(last_jump=0.03, last_vol=0.5)), IMP_UP, "D")
    assert st.state in (State.ARMED, State.WATCH) and all(a.kind != "entry_confirmed" for a in alerts)


def test_impact_uses_own_benchmark_and_branch():
    idx = pd.bdate_range("2004-01-01", "2012-12-31")
    rng = np.random.default_rng(1)
    bench = pd.Series(100 * np.exp(np.cumsum(0.0003 + 0.01 * rng.standard_normal(len(idx)))), idx)
    ex = 0.005 * rng.standard_normal(len(idx))
    evs = [Election(pd.Timestamp(f"{y}-11-0{d}").date(), "midterm", "R", "R", s, "R", s)
           for y, d, s in [(2005, 8, "D"), (2006, 7, "D"), (2008, 4, "R"), (2010, 2, "D"), (2011, 8, "R"), (2012, 6, "R")]]
    for e in evs:
        a = idx.searchsorted(pd.Timestamp(e.day))
        ex[a: a + 21] += 0.004 if e.senate_after == "D" else -0.004
    sym = bench * np.exp(np.cumsum(ex))
    closes = pd.DataFrame({"BBB": bench, "AAA": sym})
    rows = impact.screen(closes, 0.7, {"AAA": ("BBB", "x")}, events=evs, with_placebo=False)
    r = rows[0]
    assert r.z_dem > 1 and r.z_rep < -1 and r.projected_direction == "up" and r.agree_dem == 1.0
    assert r.benchmark == "BBB"


def _chain(spot=100.0, ask_scale=1.0):
    rows = []
    for exp in ("2026-12-18", "2027-01-15"):
        for k in np.round(np.arange(spot * 0.8, spot * 1.21, spot * 0.025), 1):
            for right in ("C", "P"):
                intrinsic = max(0, spot - k) if right == "C" else max(0, k - spot)
                mid = (intrinsic + 0.03 * spot * np.exp(-abs(k - spot) / (0.1 * spot))) * ask_scale
                rows.append({"ticker": "AAA", "expiry": exp, "right": right, "strike": float(k), "bid": mid * 0.95,
                             "ask": mid * 1.05, "last": mid, "iv": 0.35, "volume": 100, "open_interest": 500})
    return pd.DataFrame(rows)


def test_choose_contract_respects_budget_and_falls_back():
    today = pd.Timestamp("2026-11-04")
    s = choose_contract(_chain(), 100.0, "long", 1_000.0, today=today)
    assert s["structure"] == "long_call" and s["max_loss"] <= 1_000 and 0.3 <= s["delta"] <= 0.6
    assert s["contract"].startswith("AAA26") and "C" in s["contract"]
    small = choose_contract(_chain(), 100.0, "short", 100.0, today=today)
    assert small["structure"] in ("vertical_debit", None)
    if small["structure"]:
        assert small["max_loss_total"] <= 100.0


def test_occ_symbol_format():
    assert occ_symbol("spy", "2026-12-18", "P", 650) == "SPY261218P00650000"


def test_resolve_outcome(monkeypatch):
    monkeypatch.delenv("QUANTFLOW_SENATE_OUTCOME", raising=False)
    assert resolve_outcome(0.7) is None and resolve_outcome(0.99) == "D" and resolve_outcome(0.01) == "R"
    assert resolve_outcome(0.7, "r") == "R"


def test_run_cycle_persists_alerts_and_paper_orders(tmp_path):
    db = f"sqlite:///{tmp_path}/pb.db"
    b = bars(last_jump=0.03, last_vol=3.0)

    class FakeBroker:
        def __init__(self):
            self.orders = []

        def get_account(self):
            return type("A", (), {"equity": 100_000.0})()

        def place_option_order(self, contract, qty, limit, client_order_id=None):
            self.orders.append((contract, qty, limit, client_order_id))
            return {"id": "paper-1", "status": "accepted"}

    br = FakeBroker()
    out = run_cycle([dict(IMP_UP)], "D", db_path=db, equity=1.0, execute=True, broker=br, send=False,
                    bars_fn=lambda s, *a: None if s != "AAA" else b, chain_fn=lambda s: _chain(float(b["close"].iloc[-1])))
    alert = out["alerts"][0]
    assert alert["kind"] == "entry_confirmed" and alert["structure"]["structure"] == "long_call"
    assert br.orders and alert["order"]["id"] == "paper-1"
    assert br.orders[0][3].startswith("qf-pb-AAA-")  # idempotent client order id; equity came from broker
    from quantflow.data.persistence import list_playbook_alerts, load_playbook_states
    assert load_playbook_states(db)["AAA"]["state"] == "in_trade"
    assert list_playbook_alerts(10, db)[0]["kind"] == "entry_confirmed"


def test_option_order_blocked_above_one_percent(monkeypatch):
    from quantflow.broker.alpaca_paper import AlpacaPaperTrading
    from quantflow.execution.risk_guards import RiskGuardError

    br = AlpacaPaperTrading.__new__(AlpacaPaperTrading)
    monkeypatch.delenv("QUANTFLOW_DEMO_MODE", raising=False)
    monkeypatch.setattr(br, "_require_keys", lambda: None, raising=False)
    monkeypatch.setattr(br, "get_account", lambda: type("A", (), {"equity": 10_000.0})(), raising=False)
    sent = []
    monkeypatch.setattr(br, "_post", lambda path, data: sent.append(data) or {"id": "1", "symbol": data["symbol"],
                        "side": "buy", "qty": data["qty"], "type": "limit", "status": "accepted"}, raising=False)
    with pytest.raises(RiskGuardError):
        br.place_option_order("AAA261218C00100000", 1, 2.00)
    br.place_option_order("AAA261218C00100000", 1, 0.95)
    assert sent[0]["time_in_force"] == "day" and sent[0]["limit_price"] == "0.95"


def test_placebo_events_same_count_as_midterms():
    ev = impact.placebo_events()
    assert len(ev) == 6 and all(e.day.weekday() == 1 and e.day.month == 11 for e in ev)


def test_playbook_api_signals_alerts_and_auth(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from quantflow.api import playbook_routes
    from quantflow.api.server import app
    from quantflow.data.persistence import (
        create_schema,
        save_playbook_alerts,
        save_playbook_state,
    )

    db = f"sqlite:///{tmp_path}/api.db"
    monkeypatch.setattr(playbook_routes, "DB", db)
    create_schema(db)
    save_playbook_state(SignalState("AAA", side="long", bull=0.8, bear=0.2).to_dict(), db)
    save_playbook_alerts([{"symbol": "AAA", "kind": "armed", "message": "AAA armed long"}], db)
    c = TestClient(app)
    sig = c.get("/playbook/signals").json()
    assert sig[0]["symbol"] == "AAA" and sig[0]["state"] == "watch"
    assert c.get("/playbook/alerts").json()[0]["kind"] == "armed"
    monkeypatch.setenv("QF_API_AUTH_ENABLED", "true")
    monkeypatch.setenv("QF_API_TOKEN", "secret-token")
    assert c.post("/playbook/cycle", json={}).status_code in (401, 403)


def test_rejected_order_does_not_open_position(tmp_path):
    db = f"sqlite:///{tmp_path}/pb2.db"
    b = bars(last_jump=0.03, last_vol=3.0)

    class RejectingBroker:
        def get_account(self):
            return type("A", (), {"equity": 100_000.0})()

        def place_option_order(self, *a, **k):
            raise RuntimeError("risk guard: blocked")

    out = run_cycle([dict(IMP_UP)], "D", db_path=db, execute=True, broker=RejectingBroker(), send=False,
                    bars_fn=lambda s, *a: None if s != "AAA" else b, chain_fn=lambda s: _chain(float(b["close"].iloc[-1])))
    assert out["alerts"][0]["kind"] == "entry_not_executed"
    assert out["signals"]["AAA"]["state"] == "armed" and out["signals"]["AAA"]["entry"] is None


def test_option_sell_to_open_is_blocked(monkeypatch):
    from quantflow.broker.alpaca_paper import AlpacaPaperTrading
    from quantflow.execution.risk_guards import RiskGuardError

    br = AlpacaPaperTrading.__new__(AlpacaPaperTrading)
    monkeypatch.delenv("QUANTFLOW_DEMO_MODE", raising=False)
    monkeypatch.setattr(br, "_require_keys", lambda: None, raising=False)
    monkeypatch.setattr(br, "get_positions", lambda: [type("P", (), {"ticker": "AAA261218C00100000", "shares": 2})()],
                        raising=False)
    monkeypatch.setattr(br, "get_orders", lambda status="all", limit=50: [
        {"symbol": "AAA261218C00100000", "side": "sell", "qty": "1", "filled_qty": "0"}], raising=False)
    sent = []
    monkeypatch.setattr(br, "_post", lambda path, data: sent.append(data) or {"id": "1", "symbol": data["symbol"],
                        "side": "sell", "qty": data["qty"], "type": "limit", "status": "accepted"}, raising=False)
    with pytest.raises(RiskGuardError):
        br.place_option_order("AAA261218C00100000", 2, 1.0, side="sell")
    br.place_option_order("AAA261218C00100000", 1, 1.0, side="sell")
    assert len(sent) == 1 and sent[0]["side"] == "sell"


def test_backtest_priors_use_only_earlier_midterms(monkeypatch):
    from quantflow.playbook import backtest

    seen = []
    monkeypatch.setattr(backtest, "event_moves", lambda c, s, b, evs: seen.append([e.day.year for e in evs]) or pd.DataFrame())
    backtest.replay(bars(n=400, start="2015-01-02"), pd.DataFrame({"AAA": [1.0]}), "AAA", "", "x")
    years = [e.day.year for e in backtest.elections(kind="midterm", resolved_only=True)]
    assert seen and all(all(y < years[i] for y in prior) for i, prior in enumerate(seen))


def test_top_n_affordable_skips_unaffordable_and_keeps_sectors_diverse():
    mk = lambda sym, grp, score: impact.ImpactRow(sym, "SPY", grp, 6, 1.0, 1.0, 0.0, 3, 3, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                                                  score, "up", 1.0, score, swing_ratio=1.2)
    rows = [mk("A", "fin", 3.0), mk("B", "fin", 2.9), mk("C", "energy", 2.0), mk("D", "tech", 1.5), mk("E", "util", 1.0)]
    fit = lambda sym, side, budget: {"structure": None} if sym == "C" else {"structure": "long_call", "qty": 1}
    picked = impact.top_n_affordable(rows, n=3, budget=100, fit_fn=fit)
    assert [r.symbol for r, _ in picked] == ["A", "D", "E"]
    fit2 = lambda sym, side, budget: {"structure": "vertical_debit" if sym == "A" else "long_call", "qty": 1}
    assert impact.top_n_affordable(rows, n=2, budget=100, fit_fn=fit2)[0][0].symbol == "A"
    assert [r.symbol for r, _ in impact.top_n_affordable(rows, n=2, budget=100, fit_fn=fit2, executable_only=True)] == ["B", "C"]


def test_affordable_universe_has_benchmarks_and_many_sectors():
    groups = {g for _, g in impact.AFFORDABLE.values()}
    assert len(groups) >= 12 and all(b for b, _ in impact.AFFORDABLE.values())


def test_affordability_checked_on_side_the_tree_will_trade():
    r = impact.ImpactRow("A", "SPY", "fin", 6, 1.0, -1.0, 1.0, 3, 3, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.4, "up", -2.0, 0.4,
                         swing_ratio=1.2)
    r.z_dem, r.z_rep, r.projected_z = 1.0, -1.0, 0.4
    fit = lambda sym, side, budget: {"structure": "long_put", "qty": 1} if side == "short" else {"structure": "vertical_debit"}
    assert impact.top_n_affordable([r], 1, 100, fit, executable_only=True, outcome="R")[0][1]["side"] == "short"
    assert impact.top_n_affordable([r], 1, 100, fit, executable_only=True, outcome="D") == []
    pre = impact.top_n_affordable([r], 1, 100, fit, executable_only=True)
    assert pre and pre[0][1]["side"] == "short"  # opposite side kept before the result


def test_open_positions_are_managed_after_leaving_watchlist(tmp_path):
    db = f"sqlite:///{tmp_path}/pb3.db"
    b = bars(last_jump=0.03, last_vol=3.0)
    kw = {"db_path": db, "send": False, "bars_fn": lambda s, *a: None if s != "AAA" else b,
              "chain_fn": lambda s: _chain(float(b["close"].iloc[-1]))}
    first = run_cycle([dict(IMP_UP)], "D", **kw)
    assert first["signals"]["AAA"]["state"] == "in_trade"
    later = run_cycle([], "D", **kw)
    assert "AAA" in later["signals"]


def test_either_mode_keeps_opposite_side_after_result_and_tree_trades_it():
    r = impact.ImpactRow("AAA", "SPY", "fin", 6, 1.0, 1.0, -1.0, 3, 3, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.4, "up", 2.0, 0.4,
                         swing_ratio=1.2)
    fit = lambda sym, side, budget: {"structure": "long_call", "qty": 1} if side == "long" else {"structure": None}
    assert impact.top_n_affordable([r], 1, 100, fit, executable_only=True, outcome="R") == []
    kept = impact.top_n_affordable([r], 1, 100, fit, executable_only=True, outcome="R", direction_mode="either")
    assert kept[0][1]["side"] == "long"
    st, _ = step(SignalState("AAA"), indicators(bars(last_jump=0.03, last_vol=3.0)), r.to_dict(), "R",
                 RuleConfig(direction_mode="either"))
    assert st.side == "long" and st.state in (State.ARMED, State.IN_TRADE)  # opposite of the GOP-branch prior


def test_legacy_open_state_without_impact_still_gets_exits(tmp_path):
    from quantflow.data.persistence import create_schema, save_playbook_state

    db = f"sqlite:///{tmp_path}/legacy.db"
    create_schema(db)
    b = bars()
    px = float(b["close"].iloc[-1])
    legacy = SignalState("AAA", state=State.IN_TRADE, side="long", entry=px * 1.2, stop=px * 1.1, target=px * 1.5,
                         last_bar="2000-01-01").to_dict()
    save_playbook_state(legacy, db)  # no "impact" key, as written by older versions
    out = run_cycle([], "D", db_path=db, send=False, bars_fn=lambda s, *a: None if s != "AAA" else b, chain_fn=lambda s: None)
    assert out["signals"]["AAA"]["state"] == "exited" and out["alerts"][0]["kind"] == "exit"


def test_either_mode_only_confirms_affordable_side():
    r = impact.ImpactRow("AAA", "SPY", "fin", 6, 1.0, 1.0, -1.0, 3, 3, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.4, "up", 2.0, 0.4,
                         swing_ratio=1.2)
    fit = lambda sym, side, budget: {"structure": "long_put", "qty": 1} if side == "short" else {"structure": None}
    kept = impact.top_n_affordable([r], 1, 100, fit, executable_only=True, outcome="R", direction_mode="either")
    assert kept and kept[0][0].tradeable_sides == ["short"]
    st, alerts = step(SignalState("AAA"), indicators(bars(last_jump=0.03, last_vol=3.0)), kept[0][0].to_dict(), "R",
                      RuleConfig(direction_mode="either"))
    assert st.side == "short" and all(a.instrument != "long_call" for a in alerts)
