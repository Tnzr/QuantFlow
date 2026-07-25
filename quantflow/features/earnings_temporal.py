from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from ..data.queries import recent_news_articles


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        out = float(value)
        if not np.isfinite(out):
            return None
        return out
    except Exception:
        return None


def _parse_surprise_pct(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().replace("%", "")
        return _safe_float(text)
    return _safe_float(value)


def _normalize_earnings_dates(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()

    data = frame.copy()
    data = data.reset_index()

    date_col = None
    for candidate in ["Earnings Date", "Date", "index"]:
        if candidate in data.columns:
            date_col = candidate
            break
    if date_col is None:
        return pd.DataFrame()

    data["report_dt"] = pd.to_datetime(data[date_col], errors="coerce", utc=True)
    data = data.dropna(subset=["report_dt"]).copy()
    if data.empty:
        return pd.DataFrame()

    data["report_dt"] = data["report_dt"].dt.tz_convert(None)
    data["report_date"] = data["report_dt"].dt.date
    data["year"] = data["report_dt"].dt.year
    data["month"] = data["report_dt"].dt.month
    data["calendar_quarter"] = ((data["month"] - 1) // 3 + 1).astype(int)
    data["day_of_year"] = data["report_dt"].dt.dayofyear
    data["weekday"] = data["report_dt"].dt.day_name()

    eps_est_col = "EPS Estimate" if "EPS Estimate" in data.columns else None
    rep_eps_col = "Reported EPS" if "Reported EPS" in data.columns else None
    surprise_col = "Surprise(%)" if "Surprise(%)" in data.columns else None

    data["eps_estimate"] = data[eps_est_col].map(_safe_float) if eps_est_col else None
    data["reported_eps"] = data[rep_eps_col].map(_safe_float) if rep_eps_col else None
    data["surprise_pct"] = data[surprise_col].map(_parse_surprise_pct) if surprise_col else None

    keep_cols = [
        "report_dt",
        "report_date",
        "year",
        "month",
        "calendar_quarter",
        "day_of_year",
        "weekday",
        "eps_estimate",
        "reported_eps",
        "surprise_pct",
    ]
    out = data[keep_cols].drop_duplicates(subset=["report_date"]).sort_values("report_dt").reset_index(drop=True)
    return out


def _normalize_news_dates(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    out["article_dt"] = pd.to_datetime(out["published_at"].fillna(out["created_at"]), errors="coerce", utc=True)
    out = out.dropna(subset=["article_dt"]).copy()
    if out.empty:
        return pd.DataFrame()
    out["article_dt"] = out["article_dt"].dt.tz_convert(None)
    out["article_date"] = out["article_dt"].dt.date
    out["sentiment_score"] = pd.to_numeric(out["sentiment_score"], errors="coerce")
    return out


def fetch_quarterly_earnings_history(ticker: str, years: int = 8, max_events: int = 60) -> pd.DataFrame:
    symbol = str(ticker or "").strip().upper()
    if not symbol:
        return pd.DataFrame()

    years = max(1, min(int(years), 20))
    max_events = max(8, min(int(max_events), 120))
    limit = max(8, min(120, years * 4 + 12))

    tk = yf.Ticker(symbol)
    raw = pd.DataFrame()
    try:
        raw = tk.get_earnings_dates(limit=limit)
    except Exception:
        raw = pd.DataFrame()

    out = _normalize_earnings_dates(raw)
    if out.empty:
        return out

    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=365 * years)
    out = out[out["report_dt"] >= cutoff].copy()
    out = out.tail(max_events).reset_index(drop=True)
    return out


def _window_sentiment(news: pd.DataFrame, event_date: date, window_days: int) -> dict[str, Any]:
    if news.empty:
        return {
            "news_count_around": 0,
            "news_count_pre": 0,
            "news_count_post": 0,
            "sentiment_avg_around": None,
            "sentiment_avg_pre": None,
            "sentiment_avg_post": None,
            "sentiment_delta_post_minus_pre": None,
        }

    start = event_date - timedelta(days=window_days)
    end = event_date + timedelta(days=window_days)
    around = news[(news["article_date"] >= start) & (news["article_date"] <= end)].copy()
    pre = around[around["article_date"] < event_date]
    post = around[around["article_date"] > event_date]

    pre_mean = _safe_float(pre["sentiment_score"].mean())
    post_mean = _safe_float(post["sentiment_score"].mean())

    delta = None
    if pre_mean is not None and post_mean is not None:
        delta = float(post_mean - pre_mean)

    return {
        "news_count_around": int(len(around)),
        "news_count_pre": int(len(pre)),
        "news_count_post": int(len(post)),
        "sentiment_avg_around": _safe_float(around["sentiment_score"].mean()),
        "sentiment_avg_pre": pre_mean,
        "sentiment_avg_post": post_mean,
        "sentiment_delta_post_minus_pre": delta,
    }


def _aggregate_timing(items: pd.DataFrame) -> dict[str, Any]:
    if items.empty:
        return {
            "event_count": 0,
            "month_histogram": [],
            "quarter_histogram": [],
            "day_of_year_curve": [],
        }

    month_hist = (
        items.groupby("month").size().rename("count").reset_index().sort_values("month")
    )
    month_hist["label"] = month_hist["month"].map(lambda m: pd.Timestamp(year=2024, month=int(m), day=1).strftime("%b"))

    quarter_hist = (
        items.groupby("calendar_quarter").size().rename("count").reset_index().sort_values("calendar_quarter")
    )
    quarter_hist["label"] = quarter_hist["calendar_quarter"].map(lambda q: f"Q{int(q)}")

    doy_hist = items.groupby("day_of_year").size().rename("count").reset_index().sort_values("day_of_year")
    doy_hist["cum_count"] = doy_hist["count"].cumsum()
    total = max(1, int(doy_hist["count"].sum()))
    doy_hist["cum_share"] = doy_hist["cum_count"] / float(total)

    month_hist = month_hist.replace([np.inf, -np.inf], np.nan)
    month_hist = month_hist.astype(object).where(pd.notnull(month_hist), None)
    quarter_hist = quarter_hist.replace([np.inf, -np.inf], np.nan)
    quarter_hist = quarter_hist.astype(object).where(pd.notnull(quarter_hist), None)
    doy_hist = doy_hist.replace([np.inf, -np.inf], np.nan)
    doy_hist = doy_hist.astype(object).where(pd.notnull(doy_hist), None)

    return {
        "event_count": int(len(items)),
        "month_histogram": month_hist.to_dict(orient="records"),
        "quarter_histogram": quarter_hist.to_dict(orient="records"),
        "day_of_year_curve": doy_hist.to_dict(orient="records"),
    }


def build_earnings_temporal_profile(
    ticker: str,
    years: int = 8,
    sentiment_window_days: int = 3,
    db_path: str = "sqlite:///quantflow.db",
) -> dict[str, Any]:
    symbol = str(ticker or "").strip().upper()
    if not symbol:
        raise ValueError("ticker is required")

    events = fetch_quarterly_earnings_history(symbol, years=years)
    if events.empty:
        return {
            "ticker": symbol,
            "years": years,
            "event_count": 0,
            "events": [],
            "timing": _aggregate_timing(events),
            "surprise": {"avg_surprise_pct": None, "median_surprise_pct": None, "beat_rate": None},
            "cadence": {"avg_days_between_reports": None, "median_days_between_reports": None},
            "sentiment_alignment": {"avg_pre": None, "avg_post": None, "avg_delta": None},
        }

    oldest = events["report_date"].min()
    today = pd.Timestamp.utcnow().tz_localize(None).date()
    days = max(90, (today - oldest).days + 30)
    news = recent_news_articles(days=days, db_path=db_path, ticker=symbol, limit=6000)
    news = _normalize_news_dates(news)

    sentiment_window_days = max(1, min(int(sentiment_window_days), 14))
    sentiment_rows = []
    for event_date in events["report_date"].tolist():
        sentiment_rows.append(_window_sentiment(news, event_date, sentiment_window_days))
    sentiment_frame = pd.DataFrame(sentiment_rows)
    merged = pd.concat([events.reset_index(drop=True), sentiment_frame], axis=1)

    intervals = merged["report_dt"].diff().dt.days.dropna()
    cadence = {
        "avg_days_between_reports": _safe_float(intervals.mean()),
        "median_days_between_reports": _safe_float(intervals.median()),
    }

    surprise_col = pd.to_numeric(merged["surprise_pct"], errors="coerce")
    surprise = {
        "avg_surprise_pct": _safe_float(surprise_col.mean()),
        "median_surprise_pct": _safe_float(surprise_col.median()),
        "beat_rate": _safe_float((surprise_col > 0).mean()) if surprise_col.notna().any() else None,
    }

    sentiment_alignment = {
        "avg_pre": _safe_float(pd.to_numeric(merged["sentiment_avg_pre"], errors="coerce").mean()),
        "avg_post": _safe_float(pd.to_numeric(merged["sentiment_avg_post"], errors="coerce").mean()),
        "avg_delta": _safe_float(pd.to_numeric(merged["sentiment_delta_post_minus_pre"], errors="coerce").mean()),
    }

    out = merged.copy()
    out["report_date"] = out["report_date"].astype(str)
    out["report_ts"] = out["report_dt"].astype(str)
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.astype(object).where(pd.notnull(out), None)

    return {
        "ticker": symbol,
        "years": years,
        "event_count": int(len(out)),
        "events": out.drop(columns=["report_dt"]).to_dict(orient="records"),
        "timing": _aggregate_timing(merged),
        "surprise": surprise,
        "cadence": cadence,
        "sentiment_alignment": sentiment_alignment,
    }


def build_earnings_temporal_macro(
    tickers: list[str],
    years: int = 8,
    sentiment_window_days: int = 3,
    db_path: str = "sqlite:///quantflow.db",
) -> dict[str, Any]:
    clean = [str(t).strip().upper() for t in (tickers or []) if str(t).strip()]
    unique = []
    seen = set()
    for t in clean:
        if t in seen:
            continue
        seen.add(t)
        unique.append(t)

    frames = []
    per_ticker = []
    for ticker in unique[:40]:
        profile = build_earnings_temporal_profile(
            ticker=ticker,
            years=years,
            sentiment_window_days=sentiment_window_days,
            db_path=db_path,
        )
        per_ticker.append(
            {
                "ticker": ticker,
                "event_count": int(profile.get("event_count", 0)),
                "avg_surprise_pct": profile.get("surprise", {}).get("avg_surprise_pct"),
                "beat_rate": profile.get("surprise", {}).get("beat_rate"),
                "avg_sentiment_delta": profile.get("sentiment_alignment", {}).get("avg_delta"),
            }
        )
        events = pd.DataFrame(profile.get("events") or [])
        if events.empty:
            continue
        events["ticker"] = ticker
        frames.append(events)

    if not frames:
        return {
            "tickers": unique[:40],
            "ticker_count": len(unique[:40]),
            "event_count": 0,
            "timing": {"event_count": 0, "month_histogram": [], "quarter_histogram": [], "day_of_year_curve": []},
            "sentiment_vs_surprise": {"correlation": None},
            "per_ticker": per_ticker,
            "events": [],
        }

    all_events = pd.concat(frames, ignore_index=True)
    all_events["surprise_pct"] = pd.to_numeric(all_events["surprise_pct"], errors="coerce")
    all_events["sentiment_avg_around"] = pd.to_numeric(all_events["sentiment_avg_around"], errors="coerce")

    corr = None
    pair = all_events[["surprise_pct", "sentiment_avg_around"]].dropna()
    if len(pair) >= 4:
        corr = _safe_float(pair["surprise_pct"].corr(pair["sentiment_avg_around"]))

    out = all_events.copy()
    out = out.sort_values(["report_date", "ticker"]) 
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.astype(object).where(pd.notnull(out), None)

    return {
        "tickers": unique[:40],
        "ticker_count": len(unique[:40]),
        "event_count": int(len(out)),
        "timing": _aggregate_timing(all_events),
        "sentiment_vs_surprise": {"correlation": corr},
        "per_ticker": sorted(per_ticker, key=lambda row: row.get("event_count", 0), reverse=True),
        "events": out.to_dict(orient="records"),
    }
