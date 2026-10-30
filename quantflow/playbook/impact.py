"""Midterm impact screen with per-symbol benchmarks.

For every candidate we measure, around each resolved midterm since 2002, the return in
excess of the symbol's *own* benchmark (its sector ETF; SPY for sector ETFs) over 5 and
21 trading days after the pre-election close, scaled by the symbol's normal pre-event
volatility of that excess return (a z-score). Elections are split by which party held
the Senate afterwards, so each symbol gets a conditional projection for each branch and
a probability-weighted projection using live prediction-market odds.

Only data dated on or before each election's pre-result close is used to scale moves.
With six midterms (three per branch) every conditional estimate is small-sample; the
``n`` and ``agree`` fields are reported so callers do not over-read them.
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from ..election.calendar import Election, elections

# symbol -> (benchmark, group). Benchmarks are the closest liquid sector ETF; "" = raw return.
CANDIDATES: dict[str, tuple[str, str]] = {
    "SPY": ("", "market"), "QQQ": ("", "market"), "IWM": ("", "market"),
    "XLV": ("SPY", "healthcare"), "XBI": ("SPY", "healthcare"),
    "UNH": ("XLV", "healthcare"), "HCA": ("XLV", "healthcare"), "UHS": ("XLV", "healthcare"),
    "CNC": ("XLV", "healthcare"), "MOH": ("XLV", "healthcare"), "ELV": ("XLV", "healthcare"),
    "PFE": ("XLV", "healthcare"), "LLY": ("XLV", "healthcare"),
    "XLE": ("SPY", "energy"), "XOP": ("SPY", "energy"),
    "OXY": ("XLE", "energy"), "COP": ("XLE", "energy"), "DVN": ("XLE", "energy"),
    "TAN": ("SPY", "clean_energy"), "ICLN": ("SPY", "clean_energy"),
    "FSLR": ("XLE", "clean_energy"), "ENPH": ("XLE", "clean_energy"),
    "XLF": ("SPY", "financials"), "KRE": ("SPY", "financials"),
    "ZION": ("XLF", "financials"), "KEY": ("XLF", "financials"), "JPM": ("XLF", "financials"),
    "ITA": ("SPY", "defense"), "LMT": ("ITA", "defense"), "NOC": ("ITA", "defense"),
    "GD": ("ITA", "defense"), "RTX": ("ITA", "defense"),
    "GEO": ("SPY", "immigration"), "CXW": ("SPY", "immigration"),
    "XLI": ("SPY", "industrials"), "CAT": ("XLI", "industrials"), "DE": ("XLI", "industrials"),
    "XLU": ("SPY", "utilities"), "XLK": ("SPY", "tech"),
}
# Affordable universe for small accounts: liquid optionable names around $5-$60 (Oct 2026), across 15
# groups, each against its sector ETF (VNQ/VOX used where XLRE/XLC are too young for 2006+ history).
AFFORDABLE: dict[str, tuple[str, str]] = {
    **{s: ("XLF", "financials") for s in ("KEY", "RF", "HBAN", "FNB", "VLY", "ALLY", "BAC")},
    **{s: ("XLV", "healthcare") for s in ("PFE", "VTRS", "TEVA", "ELAN", "HIMS")},
    **{s: ("XLE", "energy") for s in ("DVN", "APA", "HAL", "SLB", "KMI", "AR", "RIG", "EQT")},
    **{s: ("ICLN", "clean_energy") for s in ("RUN", "SEDG", "ENPH", "FCEL", "CSIQ")},
    **{s: ("ITA", "defense_space") for s in ("KTOS", "RKLB", "ACHR", "JOBY")},
    **{s: ("XLK", "tech_hardware") for s in ("HPQ", "SMCI", "NOK", "ERIC", "SNAP")},
    **{s: ("XLY", "consumer") for s in ("F", "RIVN", "CCL", "NCLH", "AAL", "M", "KSS", "NKE")},
    **{s: ("XLU", "utilities") for s in ("AES", "PCG", "EXC", "NI", "PPL", "FE", "CNP")},
    **{s: ("VNQ", "real_estate") for s in ("VICI", "HST", "KIM", "AGNC", "O")},
    **{s: ("VOX", "communications") for s in ("T", "VZ", "CMCSA", "SIRI", "LUMN")},
    **{s: ("XLB", "materials") for s in ("CLF", "AA", "MP", "DOW", "MOS")},
    **{s: ("XLP", "staples") for s in ("KHC", "KDP", "CAG", "KVUE")},
    **{s: ("SPY", "prisons") for s in ("GEO", "CXW")},
    **{s: ("XLI", "industrials") for s in ("CSX", "WU", "JBLU")},
    "XLF": ("SPY", "financials"), "KRE": ("SPY", "financials"), "XLE": ("SPY", "energy"),
    "XLU": ("SPY", "utilities"), "XLP": ("SPY", "staples"),
}
UNIVERSES = {"core": CANDIDATES, "affordable": AFFORDABLE}
EXECUTABLE = ("long_call", "long_put")
WINDOWS = {"5d": 5, "21d": 21}
VOL_LOOKBACK = 126
MIN_ELECTIONS = 4


@dataclass
class ImpactRow:
    symbol: str
    benchmark: str
    group: str
    n: int
    sensitivity: float  # mean |z| of the 21d excess move after midterms (≈0.8 for pure noise)
    z_dem: float  # mean 21d excess z when Democrats held the Senate afterwards
    z_rep: float
    n_dem: int
    n_rep: int
    agree_dem: float  # share of Dem-branch midterms with the same sign as z_dem
    agree_rep: float
    ex21_dem: float  # mean raw 21d excess return, Dem branch
    ex21_rep: float
    ret21_dem: float  # mean raw 21d return of the symbol itself, Dem branch
    ret21_rep: float
    projected_z: float  # p_dem * z_dem + (1 - p_dem) * z_rep
    projected_direction: str  # "up" | "down" vs benchmark
    branch_gap: float  # z_dem - z_rep: how much the outcome itself matters
    score: float
    per_election: dict = field(default_factory=dict)
    placebo_sensitivity: float = float("nan")  # same statistic on post-midterm-year placebo dates
    swing_ratio: float = float("nan")  # sensitivity / placebo_sensitivity; >1 = midterms move it more

    def to_dict(self) -> dict:
        return asdict(self)


def _anchor(index: pd.DatetimeIndex, day) -> int | None:
    pos = int(index.searchsorted(pd.Timestamp(day))) - 1
    return pos if pos >= VOL_LOOKBACK else None


def event_moves(closes: pd.DataFrame, symbol: str, benchmark: str, events: Iterable[Election]) -> pd.DataFrame:
    """Per-election excess move of ``symbol`` vs ``benchmark`` and its vol-scaled z."""
    px = closes[symbol]
    bx = closes[benchmark] if benchmark else None
    rows = []
    for ev in events:
        a = _anchor(px.index, ev.day)
        if a is None or not np.isfinite(px.iloc[a - VOL_LOOKBACK]) or not np.isfinite(px.iloc[a]):
            continue
        r = px.pct_change(fill_method=None)
        er = r - bx.pct_change(fill_method=None) if bx is not None else r
        sd = float(er.iloc[a - VOL_LOOKBACK + 1: a + 1].std())
        if not np.isfinite(sd) or sd <= 0:
            continue
        row = {"election": ev.day.year, "senate_after": ev.senate_after, "pres_party": ev.president_party}
        for label, h in WINDOWS.items():
            if a + h >= len(px) or not np.isfinite(px.iloc[a + h]):
                break
            ret = px.iloc[a + h] / px.iloc[a] - 1
            bret = (bx.iloc[a + h] / bx.iloc[a] - 1) if bx is not None else 0.0
            row[f"ret_{label}"] = float(ret)
            row[f"ex_{label}"] = float(ret - bret)
            row[f"z_{label}"] = float((ret - bret) / (sd * math.sqrt(h)))
        else:
            rows.append(row)
    return pd.DataFrame(rows)


def _agree(z: pd.Series) -> float:
    if z.empty or z.mean() == 0:
        return 0.0
    return float((np.sign(z) == np.sign(z.mean())).mean())


def score_symbol(moves: pd.DataFrame, symbol: str, benchmark: str, group: str, p_dem: float) -> ImpactRow | None:
    if len(moves) < MIN_ELECTIONS:
        return None
    d = moves[moves["senate_after"] == "D"]
    r = moves[moves["senate_after"] == "R"]
    if d.empty or r.empty:
        return None
    z_d, z_r = float(d["z_21d"].mean()), float(r["z_21d"].mean())
    proj = p_dem * z_d + (1 - p_dem) * z_r
    agree_d, agree_r = _agree(d["z_21d"]), _agree(r["z_21d"])
    # Rank by the expected vol-scaled move, discounted when the likely branch disagrees with itself.
    likely_agree = agree_d if p_dem >= 0.5 else agree_r
    score = abs(proj) * likely_agree
    return ImpactRow(
        symbol=symbol, benchmark=benchmark or "none", group=group, n=len(moves),
        sensitivity=float(moves["z_21d"].abs().mean()),
        z_dem=z_d, z_rep=z_r, n_dem=len(d), n_rep=len(r), agree_dem=agree_d, agree_rep=agree_r,
        ex21_dem=float(d["ex_21d"].mean()), ex21_rep=float(r["ex_21d"].mean()),
        ret21_dem=float(d["ret_21d"].mean()), ret21_rep=float(r["ret_21d"].mean()),
        projected_z=float(proj), projected_direction="up" if proj > 0 else "down",
        branch_gap=z_d - z_r, score=float(score),
        per_election={int(k): round(float(v), 3) for k, v in moves.set_index("election")["z_21d"].items()},
    )


def screen(
    closes: pd.DataFrame,
    p_dem_senate: float,
    candidates: dict[str, tuple[str, str]] = CANDIDATES,
    events: list[Election] | None = None,
    with_placebo: bool = True,
) -> list[ImpactRow]:
    events = events if events is not None else elections(kind="midterm", resolved_only=True)
    placebo = placebo_events() if with_placebo else []
    out = []
    for sym, (bench, group) in candidates.items():
        if sym not in closes or (bench and bench not in closes):
            continue
        row = score_symbol(event_moves(closes, sym, bench, events), sym, bench, group, p_dem_senate)
        if row is None:
            continue
        if placebo:
            pm = event_moves(closes, sym, bench, placebo)
            if len(pm):
                row.placebo_sensitivity = float(pm["z_21d"].abs().mean())
                row.swing_ratio = row.sensitivity / row.placebo_sensitivity if row.placebo_sensitivity > 0 else float("nan")
        out.append(row)
    return sorted(out, key=lambda x: x.score, reverse=True)


def top_n(rows: list[ImpactRow], n: int = 5, max_per_group: int = 1, min_swing_ratio: float = 1.0) -> list[ImpactRow]:
    """Highest scores, at most ``max_per_group`` per sector, keeping only symbols that swing
    more after midterms than on placebo dates (``swing_ratio >= min_swing_ratio``; NaN passes)."""
    picked, seen = [], {}
    for r in rows:
        if seen.get(r.group, 0) >= max_per_group or (np.isfinite(r.swing_ratio) and r.swing_ratio < min_swing_ratio):
            continue
        picked.append(r)
        seen[r.group] = seen.get(r.group, 0) + 1
        if len(picked) == n:
            break
    return picked


def top_n_affordable(
    rows: list[ImpactRow],
    n: int = 5,
    budget: float = 100.0,
    fit_fn=None,
    max_per_group: int = 1,
    min_swing_ratio: float = 1.0,
    executable_only: bool = False,
    outcome: str | None = None,
    direction_mode: str = "event",
) -> list[tuple[ImpactRow, dict]]:
    """Like ``top_n`` but only keeps symbols where a long call/put (or debit vertical) in the
    projected direction fits ``budget``. ``fit_fn(symbol, side, budget) -> structure dict``.
    ``executable_only`` drops vertical-only symbols (paper execution is single-leg only).

    Direction checked: once ``outcome`` ("D"/"R") is known, the resolved-branch side that the rules
    tree will trade; before that, the projected side first and, if it does not fit, the opposite side
    (the tree can still flip to it after the result), so no tradeable symbol is dropped early.
    With ``direction_mode="either"`` both sides are always checked, matching ``RuleConfig.direction_mode``."""
    if fit_fn is None:
        fit_fn = _live_fit
    picked, seen = [], {}
    for r in rows:
        if seen.get(r.group, 0) >= max_per_group or (np.isfinite(r.swing_ratio) and r.swing_ratio < min_swing_ratio):
            continue
        def ok(st: dict) -> bool:
            return bool(st.get("structure")) and (not executable_only or st["structure"] in EXECUTABLE)

        if outcome in ("D", "R") and direction_mode != "either":
            sides = ["long" if (r.z_dem if outcome == "D" else r.z_rep) > 0 else "short"]
        else:
            branch_z = r.projected_z if outcome not in ("D", "R") else (r.z_dem if outcome == "D" else r.z_rep)
            first = "long" if branch_z > 0 else "short"
            sides = [first, "short" if first == "long" else "long"]
        st = {}
        for side in sides:
            st = fit_fn(r.symbol, side, budget)
            if ok(st):
                st = {**st, "side": side}
                break
        if not ok(st):
            continue
        picked.append((r, st))
        seen[r.group] = seen.get(r.group, 0) + 1
        if len(picked) == n:
            break
    return picked


def _live_fit(symbol: str, side: str, budget: float) -> dict:
    from ..features.indicators import fetch_ohlcv
    from .options import choose_contract, load_swing_chain

    try:
        spot = float(fetch_ohlcv(symbol, period="1mo")["close"].dropna().iloc[-1])
        return choose_contract(load_swing_chain(symbol), spot, side, budget)
    except Exception as exc:  # noqa: BLE001 - missing chains just make a symbol ineligible
        return {"structure": None, "reason": str(exc)[:120]}


def required_tickers(candidates: dict[str, tuple[str, str]] = CANDIDATES) -> list[str]:
    return sorted({s for s in candidates} | {b for b, _ in candidates.values() if b})


def load_closes(start: str = "2001-06-01", end: str | None = None, candidates=CANDIDATES) -> pd.DataFrame:
    from ..election.scenarios import download_closes

    return download_closes(required_tickers(candidates), start=start, end=end)


def placebo_events(years: Iterable[int] = (2003, 2007, 2011, 2015, 2019, 2023)) -> list[Election]:
    """Same calendar slot (first Tuesday after Nov 1) one year after each midterm.

    Outcome labels are copied from the preceding midterm so the D/R split is preserved;
    a midterm effect is only credible if real-event scores beat these placebo scores.
    """
    real = {e.day.year: e for e in elections(kind="midterm", resolved_only=True)}
    out = []
    for y in years:
        d = pd.Timestamp(f"{y}-11-01")
        d = d + pd.Timedelta(days=(1 - d.weekday()) % 7 or 7) if d.weekday() != 1 else d + pd.Timedelta(days=7)
        src = real.get(y - 1)
        if src is None:
            continue
        out.append(Election(d.date(), "placebo", src.president_party, src.senate_before, src.senate_after,
                            src.house_before, src.house_after))
    return out


def placebo_null(closes: pd.DataFrame, p_dem_senate: float, n_top: int = 5,
                 candidates: dict[str, tuple[str, str]] = CANDIDATES) -> dict:
    """Compare real-midterm top-N scores with the placebo-date top-N scores."""
    real = screen(closes, p_dem_senate, candidates, with_placebo=False)
    fake = screen(closes, p_dem_senate, candidates, events=placebo_events(), with_placebo=False)
    def summ(rows):
        top = top_n(rows, n_top, min_swing_ratio=0.0)
        return {"top_mean_score": float(np.mean([r.score for r in top])) if top else float("nan"),
                "mean_sensitivity": float(np.mean([r.sensitivity for r in rows])) if rows else float("nan"),
                "share_agree_1": float(np.mean([max(r.agree_dem, r.agree_rep) == 1.0 for r in rows])) if rows else float("nan")}
    return {"real": summ(real), "placebo": summ(fake), "n_real": len(real), "n_placebo": len(fake)}


def live_state(symbol: str, benchmark: str = "", period: str = "1y") -> dict:
    """Current technical snapshot plus option-implied move into the first post-election expiry."""
    from ..features.indicators import fetch_ohlcv
    from .rules import indicators

    bars = fetch_ohlcv(symbol, period=period).dropna(subset=["close"])
    bench = fetch_ohlcv(benchmark, period=period).dropna(subset=["close"])["close"] if benchmark else None
    ind = indicators(bars, bench)
    ind.update(implied_move(symbol, ind["close"]))
    return ind


def implied_move(symbol: str, spot: float, after: str = "2026-11-04") -> dict:
    """ATM straddle mid / spot for the first listed expiry on or after ``after``."""
    try:
        import yfinance as yf

        t = yf.Ticker(symbol)
        exp = next((e for e in t.options if e >= after), None)
        if exp is None:
            return {"implied_move": None, "implied_expiry": None, "atm_iv": None}
        ch = t.option_chain(exp)
        k = float(ch.calls.iloc[(ch.calls["strike"] - spot).abs().argsort()[:1]]["strike"].iloc[0])
        c = ch.calls[ch.calls["strike"] == k].iloc[0]
        p = ch.puts[ch.puts["strike"] == k]
        def mid(r):
            b, a = float(r.get("bid") or 0), float(r.get("ask") or 0)
            return (a + b) / 2 if a > 0 and b > 0 else float(r.get("lastPrice") or 0)
        straddle = mid(c) + (mid(p.iloc[0]) if len(p) else mid(c))
        return {"implied_move": straddle / spot if spot else None, "implied_expiry": exp,
                "atm_iv": float(c.get("impliedVolatility") or 0) or None}
    except Exception as exc:  # noqa: BLE001 - network/chain gaps must not break the screen
        return {"implied_move": None, "implied_expiry": None, "atm_iv": None, "implied_error": str(exc)[:120]}
