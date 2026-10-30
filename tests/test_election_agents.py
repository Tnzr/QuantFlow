"""Offline tests for election persistence, gating, signing, risk guards, verticals
and the point-in-time agentic backtest (synthetic data, no network)."""
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from quantflow.agents.gating import WeightAdjuster, volatility_regime
from quantflow.agents.graph import AgentGraph
from quantflow.backtest.agentic_backtest import SimParams, compute_signal_frame, resilience_report, simulate
from quantflow.data.persistence import create_schema, save_election_events, save_sentiment_snapshots
from quantflow.data.queries import election_events_as_of, sentiment_snapshots_as_of
from quantflow.election.calendar import ELECTIONS, cycle_phase, election_in_window, known_outcomes_as_of
from quantflow.election.ingest import parse_rss, party_lean, to_election_events
from quantflow.election.scenarios import event_returns, rank_baskets
from quantflow.execution.proposals import build_proposal, sign_proposal, verify_proposal
from quantflow.execution.risk_guards import check_order_risk
from quantflow.features.indicators import compute_indicators
from quantflow.features.seasonality import seasonality_by_cycle_frame
from quantflow.options.verticals import build_vertical, iv_crush_scenario


def synthetic_ohlcv(n=1500, seed=1, start="2016-01-04"):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, n)))
    df = pd.DataFrame({"open": close * (1 + rng.normal(0, 0.002, n)), "high": close * 1.01, "low": close * 0.99,
                       "close": close, "adj close": close, "volume": 1e6}, index=idx)
    return compute_indicators(df)


def test_point_in_time_queries(tmp_path):
    db = f"sqlite:///{tmp_path/'t.db'}"
    create_schema(db)
    save_election_events([{"source": "x", "ticker_or_topic": "senate_control", "polarity_score": 0.5,
                           "created_at": datetime(2026, 10, 1), "dedupe_key": "k1"}], db)
    assert save_election_events([{"source": "x", "ticker_or_topic": "senate_control", "polarity_score": 0.5, "dedupe_key": "k1"}], db) == 0
    save_sentiment_snapshots([{"source": "polymarket", "ticker_or_topic": "senate_control", "polarity_score": 0.37,
                               "probability": 0.685, "created_at": datetime(2026, 10, 20)}], db)
    assert election_events_as_of("2026-09-30", db_path=db).empty
    assert len(election_events_as_of("2026-10-02", db_path=db)) == 1
    assert sentiment_snapshots_as_of("2026-10-19", db_path=db).empty
    assert len(sentiment_snapshots_as_of("2026-10-21", db_path=db)) == 1


def test_calendar_and_cycle():
    assert cycle_phase(2026) == "midterm" and cycle_phase(2027) == "pre_election"
    e2006 = next(e for e in ELECTIONS if e.day.year == 2006)
    e2018 = next(e for e in ELECTIONS if e.day.year == 2018)
    assert e2006.pres_party_lost_senate and not e2018.pres_party_lost_senate
    assert election_in_window(date(2026, 10, 20), 30)
    assert all(e.day < date(2014, 11, 5) for e in known_outcomes_as_of("2014-11-05"))


def test_override_gate_requires_vol_and_gap():
    adj = WeightAdjuster()
    assert not adj.decide(0.4, 0.9, volatility_triggered=False).override_applied
    assert not adj.decide(0.4, 0.6, volatility_triggered=True).override_applied
    d = adj.decide(0.4, 0.66, volatility_triggered=True)
    assert d.override_applied and d.final_confidence == pytest.approx(0.66)
    assert adj.decide(0.5, 0.5, False, sentiment_polarity=5).sentiment_multiplier == pytest.approx(1.1)


def test_volatility_regime_ignores_future_bars():
    df = synthetic_ohlcv()
    cut = df.index[800]
    shocked = df.copy()
    shocked.loc[shocked.index > cut, "adj close"] *= np.linspace(1, 3, (shocked.index > cut).sum())
    assert volatility_regime(df, cut) == volatility_regime(shocked, cut)


def test_agent_graph_point_in_time():
    df = synthetic_ohlcv()
    g = AgentGraph()
    as_of = df.index[600]
    a = g.run("TEST", df, as_of=as_of)
    b = g.run("TEST", df.iloc[:601], as_of=as_of)
    assert a["final_confidence"] == b["final_confidence"]
    assert set(a["trace"]) >= {"technical", "seasonal", "sentiment", "synthesis"}


def test_signal_backtest_and_resilience():
    df = synthetic_ohlcv(n=1400)
    sig = compute_signal_frame("TEST", df, start=str(df.index[300].date()))
    res = simulate(df, sig, SimParams())
    assert -1 < res.max_drawdown <= 0 and res.trades >= 0
    rpt = resilience_report("TEST", df=df, signals=sig)
    assert rpt["point_in_time"] and "walk_forward" in rpt and "cost_sensitivity" in rpt
    assert rpt["cost_sensitivity"]["0bps"]["total_return"] >= rpt["cost_sensitivity"]["30bps"]["total_return"]


def test_risk_guards():
    ok = check_order_risk("buy", 10, 100.0, 100_000, reference_price=99.9, stop_price=95)
    assert ok.allow
    assert not check_order_risk("buy", 10, 103.0, 100_000, reference_price=100.0, stop_price=95).allow  # 3% slippage
    big = check_order_risk("buy", 300, 100.0, 100_000, reference_price=100.0, stop_price=95)  # $1500 = 1.5%
    assert not big.allow and big.max_qty_allowed == pytest.approx(200)
    assert not check_order_risk("buy", 20, 100.0, 100_000).allow  # no stop -> full notional at risk
    assert check_order_risk("buy", 2, 4.6, 100_000, max_loss_per_unit=460).allow


def test_ed25519_sign_verify(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTFLOW_ED25519_KEY_FILE", str(tmp_path / "k"))
    from quantflow.execution import proposals

    key = proposals.load_signing_key(tmp_path / "k")
    p = sign_proposal(build_proposal("XLV", "bull_call_vertical", [{"action": "buy"}], max_loss=920), key)
    pub = proposals.public_key_b64(key)
    assert verify_proposal(p, pub)
    assert not verify_proposal(dict(p, max_risk_pct=0.05), pub)
    monkeypatch.setenv("QUANTFLOW_TRUSTED_ED25519_PUBKEYS", pub)
    assert verify_proposal(p)
    forged = sign_proposal(build_proposal("XLV", "bull_call_vertical", []), proposals.Ed25519PrivateKey.generate())
    assert not verify_proposal(forged)  # embedded self-signed key is not trusted
    with pytest.raises(ValueError):
        build_proposal("XLV", "x", [], max_risk_pct=0.02)


def test_vertical_and_iv_crush():
    chain = pd.DataFrame([
        {"ticker": "T", "expiry": "2026-12-18", "strike": k, "right": "C", "last": m, "bid": m - 0.05, "ask": m + 0.05,
         "volume": 10, "open_interest": 100, "iv": 0.3}
        for k, m in [(95, 7.2), (100, 4.1), (105, 2.0), (110, 0.9)]
    ])
    v = build_vertical(chain, 100.0, "long", today=pd.Timestamp("2026-10-29"))
    assert v and v.long_strike == 100 and v.short_strike == 105 and v.max_loss == pytest.approx(210)
    s = iv_crush_scenario(100, 100, 105, 0.35, 50, "C", crush=-0.25)
    assert s["single_pnl_pct"] < s["vertical_pnl_pct"]


def test_rss_and_sentiment_tagging():
    xml = """<rss><channel><item><title>Democrats gain in Senate race polls</title>
    <description>Markets weigh tariffs</description><link>http://x</link>
    <pubDate>Wed, 28 Oct 2026 12:00:00 GMT</pubDate></item></channel></rss>"""
    items = parse_rss(xml, "test")
    assert items and items[0]["published_at"] == datetime(2026, 10, 28, 12)
    ev = to_election_events(items)
    topics = {e["ticker_or_topic"] for e in ev}
    assert {"senate_control", "tariffs"} <= topics
    assert party_lean("Republicans hold the Senate") == -1.0
    assert parse_rss("not xml", "bad") == []


def test_seasonality_cycle_and_event_study():
    df = synthetic_ohlcv(n=2600, start="2005-01-03")
    df["ret"] = df["adj close"].pct_change()
    df["doy"] = df.index.day_of_year
    seg = seasonality_by_cycle_frame(df)
    assert set(seg["segment"]) == {"election_year", "non_election_year"}
    closes = pd.DataFrame({"SPY": df["adj close"], "AAA": df["adj close"] * 1.0})
    study = event_returns(closes, [e for e in ELECTIONS if e.senate_after and 2006 <= e.day.year <= 2012], "SPY")
    assert not study.empty and study["excess"].abs().max() < 1e-9
    ranked = rank_baskets(study, {"baskets": {"b": {"side": "long", "tickers": ["AAA"], "thesis": "t"}}}, "1m")
    assert ranked[0].analog_n == 1


def test_simulate_flat_and_open_at_end():
    df = synthetic_ohlcv(n=300)
    flat = pd.DataFrame({"final": 0.5, "base": 0.5, "override": False}, index=df.index)
    r = simulate(df, flat, SimParams())
    assert r.trades == 0 and r.total_return == 0
    always = pd.DataFrame({"final": 0.9, "base": 0.9, "override": False}, index=df.index)
    r2 = simulate(df, always, SimParams(max_hold_bars=10_000, stop_atr=1e6))
    assert r2.trades == 1 and r2.trade_log[-1]["reason"] == "end_of_data"
    assert r2.total_return == pytest.approx(sum(t["pnl"] for t in r2.trade_log) / 100_000, rel=1e-6)


class _FakeAlpaca:
    from quantflow.broker.alpaca_paper import AlpacaPaperTrading as _A
    check_risk = _A.check_risk

    def __init__(self, quote, shares=0):
        from types import SimpleNamespace as NS
        self.quote, self.pos, self.acct = quote, [NS(ticker="ABC", shares=shares)], NS(equity=100_000.0)

    def get_last_quote(self, t): return self.quote
    def get_last_trade(self, t): return {"price": 100.0}
    def get_positions(self): return self.pos
    def get_account(self): return self.acct


def test_alpaca_check_risk_quote_and_closing():
    from quantflow.execution.risk_guards import RiskGuardError

    with pytest.raises(RiskGuardError):
        _FakeAlpaca({"bid": 0, "ask": 0}).check_risk("ABC", "buy", 1)
    with pytest.raises(RiskGuardError):  # 20 x $100 unstopped = 2% risk
        _FakeAlpaca({"bid": 99.99, "ask": 100.01}).check_risk("ABC", "sell", 20)
    assert _FakeAlpaca({"bid": 99.99, "ask": 100.01}, shares=20).check_risk("ABC", "sell", 20).allow
    with pytest.raises(RiskGuardError):
        _FakeAlpaca({"bid": 99.99, "ask": 100.01}, shares=20).check_risk("ABC", "sell", 0)


def test_signing_key_race_reuses_existing(tmp_path):
    from quantflow.execution import proposals

    a = proposals.load_signing_key(tmp_path / "k")
    b = proposals.load_signing_key(tmp_path / "k")
    assert proposals.public_key_b64(a) == proposals.public_key_b64(b)
    assert oct((tmp_path / "k").stat().st_mode & 0o777) == "0o600"
    assert [p.name for p in tmp_path.iterdir()] == ["k"]
