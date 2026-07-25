from __future__ import annotations

from typing import Any

import pandas as pd

from .indicators import fetch_ohlcv
from ..data.queries import recent_news_articles

POSITIVE_WORDS = {
    "beat", "beats", "bull", "bullish", "buy", "growth", "gain", "gains", "green", "improve",
    "improves", "improving", "upside", "outperform", "positive", "record", "strength", "surge",
    "surges", "rally", "rallies", "upgrade", "upgraded", "strong", "support", "breakout",
}

NEGATIVE_WORDS = {
    "bear", "bearish", "sell", "downgrade", "downgraded", "loss", "losses", "drop", "drops",
    "downside", "weak", "weakness", "miss", "misses", "missed", "warn", "warning", "risk",
    "volatile", "lawsuit", "probe", "slump", "slumps", "decline", "declines",
}


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return None


def simple_sentiment_score(text: str | None) -> float:
    if not text:
        return 0.0
    tokens = [token.strip(".,:;!?()[]{}\"'`").lower() for token in str(text).split()]
    pos = sum(1 for token in tokens if token in POSITIVE_WORDS)
    neg = sum(1 for token in tokens if token in NEGATIVE_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0
    return max(-1.0, min(1.0, (pos - neg) / total))


def sentiment_label(score: float) -> str:
    if score >= 0.2:
        return "bullish"
    if score <= -0.2:
        return "bearish"
    return "neutral"


def build_news_tokens(article: dict[str, Any]) -> dict[str, Any]:
    text = " ".join(str(article.get(field, "") or "") for field in ("title", "summary", "content")).strip()
    score = article.get("sentiment_score")
    score = float(score) if score is not None else simple_sentiment_score(text)
    label = article.get("sentiment_label") or sentiment_label(score)
    tickers = article.get("tickers") or []
    if isinstance(tickers, str):
        tickers = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    tickers = [str(t).upper() for t in tickers if str(t).strip()]

    tone_token = f"news_{label}"
    attention_token = "news_attention_high" if len(text.split()) > 40 else "news_attention_low"
    ticker_tokens = [f"news_ticker_{ticker.lower()}" for ticker in tickers[:6]]
    return {
        "sentiment_score": score,
        "sentiment_label": label,
        "tokens": [tone_token, attention_token, *ticker_tokens],
    }


def summarize_articles(articles: list[dict[str, Any]], max_articles: int = 5, max_chars: int = 220) -> str:
    if not articles:
        return ""
    parts = []
    for article in articles[:max_articles]:
        headline = str(article.get("title", "")).strip()
        summary = str(article.get("summary", "")).strip()
        text = headline or summary
        if headline and summary:
            text = f"{headline}: {summary}"
        if len(text) > max_chars:
            text = f"{text[: max_chars - 1].rstrip()}…"
        parts.append(text)
    return " | ".join(parts)


def _normalize_article_dates(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    out["article_dt"] = pd.to_datetime(out["published_at"].fillna(out["created_at"]), errors="coerce", utc=True)
    out["article_date"] = out["article_dt"].dt.tz_convert(None).dt.normalize()
    out["article_text"] = out.apply(
        lambda row: " ".join(str(row.get(field, "") or "") for field in ("title", "summary", "content")).strip(),
        axis=1,
    )
    out["article_weight"] = out["article_text"].str.split().map(lambda words: max(1.0, min(5.0, len(words) / 20.0 if words else 1.0)))
    return out


def build_news_summary(
    ticker: str | None = None,
    days: int = 30,
    db_path: str = "sqlite:///quantflow.db",
    sentiment: str = "all",
    max_groups: int = 10,
) -> dict[str, Any]:
    articles = recent_news_articles(days=days, db_path=db_path, ticker=ticker.upper() if ticker else None, limit=500)
    if articles.empty:
        return {
            "count": 0,
            "ticker": ticker.upper() if ticker else None,
            "days": days,
            "sentiment": sentiment,
            "summary": "No archived articles matched the current filters.",
            "items": [],
        }

    articles = _normalize_article_dates(articles)
    if sentiment in {"bullish", "bearish", "neutral"}:
        articles = articles[articles["sentiment_label"].fillna("neutral") == sentiment]
    elif sentiment in {"positive", "negative"}:
        target = "bullish" if sentiment == "positive" else "bearish"
        articles = articles[articles["sentiment_label"].fillna("neutral") == target]

    if articles.empty:
        return {
            "count": 0,
            "ticker": ticker.upper() if ticker else None,
            "days": days,
            "sentiment": sentiment,
            "summary": "No archived articles matched the current sentiment filter.",
            "items": [],
        }

    articles = articles.sort_values(["article_date", "created_at"], ascending=[False, False])
    grouped = []
    for article_date, group in articles.groupby("article_date"):
        records = group.to_dict(orient="records")
        avg_sentiment = float(pd.to_numeric(group["sentiment_score"], errors="coerce").fillna(0.0).mean())
        summary = summarize_articles(records, max_articles=4)
        grouped.append(
            {
                "date": str(pd.Timestamp(article_date).date()),
                "count": int(len(group)),
                "avg_sentiment": avg_sentiment,
                "sentiment_label": sentiment_label(avg_sentiment),
                "summary": summary,
                "items": records[:max_groups],
            }
        )

    grouped = grouped[:max_groups]
    narrative = summarize_articles(articles.head(max_groups).to_dict(orient="records"), max_articles=max_groups)
    return {
        "count": int(len(articles)),
        "ticker": ticker.upper() if ticker else None,
        "days": days,
        "sentiment": sentiment,
        "summary": narrative,
        "items": grouped,
    }


def build_news_timeline(
    ticker: str,
    db_path: str = "sqlite:///quantflow.db",
    days: int = 30,
    period: str = "2y",
    interval: str = "1d",
    sentiment: str = "all",
    max_articles: int = 40,
) -> dict[str, Any]:
    price = fetch_ohlcv(ticker.upper(), period=period, interval=interval)
    if price.empty:
        raise ValueError(f"No price history available for {ticker}")

    price = price.tail(max(20, min(len(price), 600))).copy()
    price = price.reset_index()
    price["date"] = pd.to_datetime(price["date"], errors="coerce")
    if getattr(price["date"].dt, "tz", None) is not None:
        price["date"] = price["date"].dt.tz_convert(None)
    price["date"] = price["date"].dt.normalize()

    articles = recent_news_articles(days=days, db_path=db_path, ticker=ticker.upper(), limit=500)
    if articles.empty:
        return {
            "ticker": ticker.upper(),
            "days": days,
            "period": period,
            "interval": interval,
            "price": price.assign(date=price["date"].astype(str)).to_dict(orient="records"),
            "markers": [],
            "summary": "No archived articles matched the current filters.",
            "article_count": 0,
        }

    articles = _normalize_article_dates(articles)
    if sentiment in {"bullish", "bearish", "neutral"}:
        articles = articles[articles["sentiment_label"].fillna("neutral") == sentiment]
    elif sentiment in {"positive", "negative"}:
        target = "bullish" if sentiment == "positive" else "bearish"
        articles = articles[articles["sentiment_label"].fillna("neutral") == target]

    price_dates = pd.DatetimeIndex(pd.to_datetime(price["date"], errors="coerce")).tz_localize(None)
    markers = []
    for row in articles.head(max_articles).sort_values(["article_date", "created_at"], ascending=[True, True]).to_dict(orient="records"):
        article_dt = pd.to_datetime(row.get("article_dt"), errors="coerce")
        if pd.isna(article_dt):
            continue
        if getattr(article_dt, "tzinfo", None) is not None:
            article_dt = article_dt.tz_convert(None)
        target_date = pd.Timestamp(article_dt).normalize()
        if len(price_dates):
            nearest_idx = price_dates.get_indexer([target_date], method="nearest")[0]
            if nearest_idx >= 0:
                target_date = price_dates[nearest_idx]
        marker = {
            "date": str(pd.Timestamp(target_date).date()),
            "title": str(row.get("title", "")).strip(),
            "summary": str(row.get("summary", "")).strip(),
            "source": row.get("source"),
            "url": row.get("url"),
            "sentiment_score": _safe_float(row.get("sentiment_score")),
            "sentiment_label": row.get("sentiment_label") or sentiment_label(_safe_float(row.get("sentiment_score")) or 0.0),
            "weight": float(row.get("article_weight") or 1.0),
        }
        marker["short_summary"] = summarize_articles([row], max_articles=1, max_chars=180)
        markers.append(marker)

    daily = []
    for date_value, group in articles.groupby("article_date"):
        avg_sentiment = float(pd.to_numeric(group["sentiment_score"], errors="coerce").fillna(0.0).mean())
        daily.append(
            {
                "date": str(pd.Timestamp(date_value).date()),
                "count": int(len(group)),
                "avg_sentiment": avg_sentiment,
                "sentiment_label": sentiment_label(avg_sentiment),
                "summary": summarize_articles(group.to_dict(orient="records"), max_articles=4),
            }
        )

    return {
        "ticker": ticker.upper(),
        "days": days,
        "period": period,
        "interval": interval,
        "article_count": int(len(articles)),
        "price": price.assign(date=price["date"].astype(str)).to_dict(orient="records"),
        "markers": markers,
        "daily": daily,
        "summary": summarize_articles(articles.to_dict(orient="records"), max_articles=min(6, max_articles)),
    }