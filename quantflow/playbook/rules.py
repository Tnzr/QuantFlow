"""Conditional entry-confirmation rules tree for event-driven long calls / long puts.

Tree, evaluated once per bar for every symbol:

    event resolved?  -- no -->  WATCH  (track bull/bear confidence, no entries)
        | yes: branch = Senate winner -> expected side from the impact screen
    side confidence >= arm      -- no -->  WATCH
        | yes                           ARMED
    trend + trigger pass and confidence >= confirm  -->  IN_TRADE (entry alert:
        long call if side is long, long put if short; stop/target from ATR)
    IN_TRADE exits on: stop (trailing ATR) | target | time stop | opposite confidence
        overtakes ours by ``flip_margin`` -> EXITED -> WATCH on the next bar.

Confidence is computed for *both* directions every bar from the same checklist so
the opposite case is always visible. All inputs are causal (bars up to ``as_of``).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum

import numpy as np
import pandas as pd


class State(str, Enum):
    WATCH = "watch"
    ARMED = "armed"
    IN_TRADE = "in_trade"
    EXITED = "exited"


@dataclass
class RuleConfig:
    sma_fast: int = 20
    sma_slow: int = 50
    rsi_long: tuple[float, float] = (50.0, 75.0)
    rsi_short: tuple[float, float] = (25.0, 50.0)
    breakout_lookback: int = 10
    volume_mult: float = 1.3
    rs_lookback: int = 5
    atr_stop: float = 1.5
    target_r: float = 2.0
    time_stop: int = 15
    arm_conf: float = 0.55
    confirm_conf: float = 0.70
    flip_margin: float = 0.20
    pre_event_prior_scale: float = 0.5
    # "event": trade only the side history favours for the resolved branch.
    # "either": the event is a volatility catalyst; trade whichever side confirms (prior still tilts confidence).
    direction_mode: str = "event"
    weights: dict = field(default_factory=lambda: {
        "event": 0.30, "trend": 0.20, "momentum": 0.15, "rel_strength": 0.15, "trigger": 0.15, "intraday": 0.05,
    })


@dataclass
class SignalState:
    symbol: str
    state: State = State.WATCH
    side: str | None = None  # expected "long" | "short"
    bull: float = 0.0
    bear: float = 0.0
    entry: float | None = None
    stop: float | None = None
    target: float | None = None
    entered_on: str | None = None
    bars_held: int = 0
    last_bar: str | None = None
    exit_reason: str | None = None
    checks: dict = field(default_factory=dict)
    indicators: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["state"] = self.state.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> SignalState:
        d = dict(d)
        d["state"] = State(d.get("state", "watch"))
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Alert:
    symbol: str
    kind: str  # armed | disarmed | entry_confirmed | exit | stop_raised
    as_of: str
    message: str
    side: str | None = None
    instrument: str | None = None  # long_call | long_put
    price: float | None = None
    stop: float | None = None
    target: float | None = None
    bull: float = 0.0
    bear: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).where(dn > 0, 100.0)


def _atr(bars: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = bars["close"].shift(1)
    tr = pd.concat([bars["high"] - bars["low"], (bars["high"] - pc).abs(), (bars["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def indicators(bars: pd.DataFrame, bench: pd.Series | None = None, intraday: pd.DataFrame | None = None,
               cfg: RuleConfig | None = None) -> dict:
    """Snapshot at the last bar of ``bars`` (columns open/high/low/close/volume)."""
    cfg = cfg or RuleConfig()
    c = bars["close"]
    n = cfg.breakout_lookback
    vol = bars["volume"].astype(float)
    out = {
        "as_of": str(pd.Timestamp(bars.index[-1]).date()),
        "close": float(c.iloc[-1]),
        "sma_fast": float(c.rolling(cfg.sma_fast).mean().iloc[-1]),
        "sma_slow": float(c.rolling(cfg.sma_slow).mean().iloc[-1]),
        "sma200": float(c.rolling(200).mean().iloc[-1]) if len(c) >= 200 else float("nan"),
        "rsi14": float(_rsi(c).iloc[-1]),
        "atr14": float(_atr(bars).iloc[-1]),
        "prior_high": float(bars["high"].iloc[-n - 1:-1].max()),
        "prior_low": float(bars["low"].iloc[-n - 1:-1].min()),
        "volume_ratio": float(vol.iloc[-1] / vol.iloc[-21:-1].mean()) if vol.iloc[-21:-1].mean() > 0 else float("nan"),
        "ret_5d": float(c.iloc[-1] / c.iloc[-cfg.rs_lookback - 1] - 1),
        "trend_agree": float(np.mean([np.sign(c.iloc[-1] - c.rolling(w).mean().iloc[-1]) for w in (10, 21, 42, 63)])),
    }
    if bench is not None and len(bench.dropna()) > cfg.rs_lookback:
        b = bench.reindex(c.index).ffill()
        out["rel_strength_5d"] = float(out["ret_5d"] - (b.iloc[-1] / b.iloc[-cfg.rs_lookback - 1] - 1))
    else:
        out["rel_strength_5d"] = out["ret_5d"]
    if intraday is not None and len(intraday) >= 20:
        ic = intraday["close"]
        out["intraday_trend"] = float(np.sign(ic.iloc[-1] - ic.rolling(20).mean().iloc[-1]))
    return out


def checks(ind: dict, side: str, cfg: RuleConfig) -> dict[str, bool]:
    s = 1 if side == "long" else -1
    lo, hi = cfg.rsi_long if s > 0 else cfg.rsi_short
    breakout = ind["close"] > ind["prior_high"] if s > 0 else ind["close"] < ind["prior_low"]
    out = {
        "trend": bool(s * (ind["close"] - ind["sma_fast"]) > 0 and s * (ind["sma_fast"] - ind["sma_slow"]) > 0),
        "momentum": bool(lo <= ind["rsi14"] <= hi),
        "rel_strength": bool(s * ind["rel_strength_5d"] > 0),
        "trigger": bool(breakout and ind["volume_ratio"] >= cfg.volume_mult),
    }
    if "intraday_trend" in ind:
        out["intraday"] = bool(s * ind["intraday_trend"] > 0)
    return out


def event_prior(z: float) -> float:
    """Map a vol-scaled expected excess move to P(up) in [0, 1]."""
    return 0.5 * (1 + math.tanh(z))


def branch_z(impact: dict, outcome: str | None, cfg: RuleConfig) -> float:
    if outcome == "D":
        return float(impact["z_dem"])
    if outcome == "R":
        return float(impact["z_rep"])
    return float(impact["projected_z"]) * cfg.pre_event_prior_scale


def confidence(ind: dict, impact: dict, outcome: str | None, cfg: RuleConfig) -> tuple[float, float, dict]:
    pu = event_prior(branch_z(impact, outcome, cfg))
    detail = {}
    scores = {}
    for side, prior in (("long", pu), ("short", 1 - pu)):
        ch = checks(ind, side, cfg)
        w = {k: v for k, v in cfg.weights.items() if k == "event" or k in ch}
        tot = sum(w.values())
        scores[side] = (w["event"] * prior + sum(w[k] * float(ch[k]) for k in ch)) / tot
        detail[side] = ch
    detail["event_prior_up"] = pu
    return scores["long"], scores["short"], detail


def _alert(st: SignalState, kind: str, msg: str, as_of: str, **kw) -> Alert:
    return Alert(st.symbol, kind, as_of, msg, side=st.side, bull=round(st.bull, 3), bear=round(st.bear, 3), **kw)


def step(prev: SignalState, ind: dict, impact: dict, outcome: str | None,
         cfg: RuleConfig | None = None) -> tuple[SignalState, list[Alert]]:
    """Advance one symbol by one bar. Pure: returns the new state and any alerts."""
    cfg = cfg or RuleConfig()
    st = SignalState.from_dict(prev.to_dict())
    as_of = ind["as_of"]
    new_bar = st.last_bar != as_of
    st.last_bar = as_of
    st.bull, st.bear, detail = confidence(ind, impact, outcome, cfg)
    st.checks, st.indicators = detail, ind
    alerts: list[Alert] = []
    px, atr = ind["close"], ind["atr14"]

    if st.state == State.EXITED and new_bar:
        st.state, st.exit_reason = State.WATCH, None
        st.entry = st.stop = st.target = st.entered_on = None
        st.bars_held = 0

    if st.state == State.IN_TRADE:
        sgn = 1 if st.side == "long" else -1
        if new_bar:
            st.bars_held += 1
        own, opp = (st.bull, st.bear) if sgn > 0 else (st.bear, st.bull)
        trail = px - sgn * cfg.atr_stop * atr
        if sgn * (trail - st.stop) > 0:
            st.stop = trail
        reason = None
        if sgn * (px - st.stop) <= 0:
            reason = "stop"
        elif sgn * (px - st.target) >= 0:
            reason = "target"
        elif st.bars_held >= cfg.time_stop:
            reason = "time"
        elif opp - own >= cfg.flip_margin:
            reason = "opposite_confidence"
        if reason:
            st.state, st.exit_reason = State.EXITED, reason
            alerts.append(_alert(st, "exit", f"EXIT {st.symbol} ({reason}) at {px:.2f}; entry {st.entry:.2f}", as_of,
                                 price=px, stop=st.stop, target=st.target))
        return st, alerts

    if outcome is None:
        st.side = "long" if impact["projected_z"] > 0 else "short"
        if st.state == State.ARMED:
            alerts.append(_alert(st, "disarmed", f"{st.symbol}: waiting for the election result", as_of))
        st.state = State.WATCH
        return st, alerts

    if cfg.direction_mode == "either":
        allowed = impact.get("tradeable_sides") or ["long", "short"]
        st.side = max(allowed, key=lambda s: st.bull if s == "long" else st.bear)
    else:
        st.side = "long" if branch_z(impact, outcome, cfg) > 0 else "short"
    own = st.bull if st.side == "long" else st.bear
    ch = detail[st.side]
    if own >= cfg.confirm_conf and ch["trend"] and ch["trigger"]:
        sgn = 1 if st.side == "long" else -1
        st.state, st.entry, st.entered_on, st.bars_held = State.IN_TRADE, px, as_of, 0
        st.stop = px - sgn * cfg.atr_stop * atr
        st.target = px + sgn * cfg.target_r * cfg.atr_stop * atr
        inst = "long_call" if sgn > 0 else "long_put"
        alerts.append(_alert(st, "entry_confirmed",
                             f"ENTRY {st.symbol}: {inst.replace('_', ' ')} — close {px:.2f}, stop {st.stop:.2f}, target {st.target:.2f}",
                             as_of, instrument=inst, price=px, stop=st.stop, target=st.target))
    elif own >= cfg.arm_conf:
        if st.state != State.ARMED:
            alerts.append(_alert(st, "armed", f"{st.symbol} armed {st.side}: confidence {own:.2f}, waiting for trend + breakout", as_of))
        st.state = State.ARMED
    else:
        if st.state == State.ARMED:
            alerts.append(_alert(st, "disarmed", f"{st.symbol} disarmed: confidence {own:.2f}", as_of))
        st.state = State.WATCH
    return st, alerts
