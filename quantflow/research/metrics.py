"""Performance and significance metrics. Net Sharpe, IC/ICIR, drawdown and
trial-adjusted (deflated) Sharpe are primary; forecast error is diagnostic."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def sharpe(returns: pd.Series, periods_per_year: int = 52) -> float:
    r = pd.Series(returns).dropna()
    sd = r.std()
    return float(r.mean() / sd * np.sqrt(periods_per_year)) if len(r) > 1 and sd > 0 else float("nan")


def max_drawdown(returns: pd.Series) -> float:
    eq = (1 + pd.Series(returns).fillna(0)).cumprod()
    return float((eq / eq.cummax() - 1).min()) if len(eq) else 0.0


def rank_ic(scores: pd.Series, outcomes: pd.Series, min_names: int = 10) -> pd.Series:
    """Per-date Spearman correlation for (date, ticker) indexed series."""
    df = pd.concat({"s": scores, "y": outcomes}, axis=1).dropna()
    out = {}
    for d, g in df.groupby(level=0):
        if len(g) >= min_names and g["s"].nunique() > 1 and g["y"].nunique() > 1:
            out[d] = stats.spearmanr(g["s"], g["y"])[0]
    return pd.Series(out, dtype=float)


def ic_summary(ic: pd.Series, horizon_steps: int = 1, periods_per_year: int = 52) -> dict:
    """Mean IC, ICIR and a t-stat on non-overlapping samples."""
    ic = ic.dropna()
    nono = ic.iloc[:: max(1, horizon_steps)]
    t = float(nono.mean() / nono.std() * np.sqrt(len(nono))) if len(nono) > 2 and nono.std() > 0 else float("nan")
    return {
        "ic_mean": float(ic.mean()) if len(ic) else float("nan"),
        "icir": float(ic.mean() / ic.std() * np.sqrt(periods_per_year)) if len(ic) > 1 and ic.std() > 0 else float("nan"),
        "ic_t": t,
        "ic_hit": float((ic > 0).mean()) if len(ic) else float("nan"),
        "n": len(ic),
    }


def probabilistic_sharpe(returns: pd.Series, sr_benchmark: float = 0.0) -> float:
    """P(true per-period Sharpe > sr_benchmark), Bailey & Lopez de Prado (2012)."""
    r = pd.Series(returns).dropna()
    n = len(r)
    if n < 3 or r.std() == 0:
        return float("nan")
    sr = r.mean() / r.std()
    skew = stats.skew(r)
    kurt = stats.kurtosis(r, fisher=False)
    denom = np.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr**2, 1e-12))
    return float(stats.norm.cdf((sr - sr_benchmark) * np.sqrt(n - 1) / denom))


def expected_max_sharpe(trial_sharpes: list[float], n_trials: int | None = None) -> float:
    """Expected max per-period Sharpe under the null across N trials."""
    n = max(int(n_trials or len(trial_sharpes)), 2)
    sd = float(np.nanstd(trial_sharpes)) if len(trial_sharpes) > 1 else 0.0
    g = np.euler_gamma
    return sd * ((1 - g) * stats.norm.ppf(1 - 1 / n) + g * stats.norm.ppf(1 - 1 / (n * np.e)))


def deflated_sharpe(returns: pd.Series, trial_sharpes: list[float], n_trials: int | None = None) -> float:
    """PSR against the max Sharpe expected by luck after ``n_trials`` attempts.
    ``trial_sharpes`` are per-period (not annualised) Sharpe ratios."""
    return probabilistic_sharpe(returns, expected_max_sharpe(trial_sharpes, n_trials))


def diebold_mariano(loss_a: pd.Series, loss_b: pd.Series, horizon: int = 1) -> dict:
    """DM test on loss differential a - b with Newey-West variance.
    Negative statistic means model A has lower loss."""
    d = (pd.Series(loss_a) - pd.Series(loss_b)).dropna().to_numpy()
    n = len(d)
    if n < 5:
        return {"dm": float("nan"), "p": float("nan")}
    dc = d - d.mean()
    var = dc @ dc / n
    for k in range(1, max(1, horizon)):
        var += 2 * (1 - k / horizon) * (dc[k:] @ dc[:-k]) / n
    dm = d.mean() / np.sqrt(var / n) if var > 0 else float("nan")
    return {"dm": float(dm), "p": float(2 * stats.norm.sf(abs(dm)))}


def summarize(returns: pd.Series, turnover: pd.Series | None = None, periods_per_year: int = 52) -> dict:
    r = pd.Series(returns).dropna()
    out = {
        "sharpe": sharpe(r, periods_per_year),
        "ann_return": float(r.mean() * periods_per_year) if len(r) else float("nan"),
        "ann_vol": float(r.std() * np.sqrt(periods_per_year)) if len(r) > 1 else float("nan"),
        "max_drawdown": max_drawdown(r),
        "psr": probabilistic_sharpe(r),
        "n_periods": len(r),
    }
    if turnover is not None:
        out["turnover"] = float(pd.Series(turnover).mean())
    return out


def by_era(returns: pd.Series, eras: dict[str, tuple[int, int]], periods_per_year: int = 52) -> dict:
    r = pd.Series(returns).dropna()
    yrs = pd.DatetimeIndex(r.index).year
    return {k: sharpe(r[(yrs >= a) & (yrs <= b)], periods_per_year) for k, (a, b) in eras.items()}
