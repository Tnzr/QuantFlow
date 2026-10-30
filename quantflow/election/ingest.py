"""Free election/macro news (RSS) and Polymarket odds ingestion.

Normalises items into ``ElectionEvent`` / ``SentimentSnapshot`` dicts. Every
network call is best-effort: failures log to stderr and return empty lists so
callers fall back to the deterministic engine (sentiment multiplier 1.0).
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Iterable, Optional
from urllib.parse import quote_plus
from xml.etree import ElementTree as ET

import requests

from ..features.news import simple_sentiment_score

USER_AGENT = "QuantFlow/0.1 (+research; election-sentiment)"
TIMEOUT = 12

GOOGLE_NEWS_QUERIES = [
    "2026 midterm elections Senate control",
    "midterm elections stock market",
    "Senate Democrats Republicans polls 2026",
    "tariffs Congress midterms",
    "drug pricing Congress 2026",
    "defense budget Congress 2026",
    "clean energy tax credits Congress",
]

RSS_FEEDS = {
    "cnbc_politics": "https://www.cnbc.com/id/10000113/device/rss/rss.html",
    "cnbc_economy": "https://www.cnbc.com/id/20910258/device/rss/rss.html",
    "marketwatch_top": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "politico_congress": "https://rss.politico.com/congress.xml",
    "npr_politics": "https://feeds.npr.org/1014/rss.xml",
}

POLYMARKET_GAMMA = "https://gamma-api.polymarket.com"
POLYMARKET_EVENTS = {
    "senate_control": "which-party-will-win-the-senate-in-2026",
    "house_control": "which-party-will-win-the-house-in-2026",
}

# Policy topics -> keyword patterns. Topics map to sector baskets in configs/election_scenarios.yml.
TOPIC_PATTERNS = {
    "senate_control": r"\bsenate\b",
    "house_control": r"\bhouse (?:majority|control|republicans|democrats)\b|\bspeaker\b",
    "tariffs": r"tariff|trade war|import tax",
    "healthcare": r"health ?care|medicare|medicaid|obamacare|\baca\b|drug pric|pharma",
    "defense": r"defen[cs]e (?:budget|spending|bill)|pentagon|\bndaa\b|military spending",
    "energy": r"oil|drilling|clean energy|solar|\bira\b|tax credit|fossil|lng|pipeline",
    "financials": r"\bbanks?\b|crypto|stablecoin|deregulat|\bsec\b|capital rules",
    "fiscal": r"government shutdown|debt ceiling|appropriations|deficit|tax cut|spending bill",
    "immigration": r"immigration|deport|border|\bice\b",
}

DEM_WIN = r"democrats? (?:win|gain|flip|take|lead|surge|favored)|blue wave|gop (?:loses|trails)|republicans? (?:lose|trail)"
GOP_WIN = r"republicans? (?:win|gain|hold|keep|lead|favored)|red wave|gop (?:holds|keeps|leads|gains)|democrats? (?:lose|trail)"


def _warn(msg: str) -> None:
    print(f"[election-ingest] WARNING: {msg}", file=sys.stderr)


def _get(url: str) -> Optional[requests.Response]:
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
        r.raise_for_status()
        return r
    except Exception as exc:  # network/rate-limit/timeouts are non-fatal
        _warn(f"{url} failed: {exc}")
        return None


def _parse_date(raw: Optional[str]) -> datetime:
    if raw:
        try:
            return parsedate_to_datetime(raw).astimezone(timezone.utc).replace(tzinfo=None)
        except Exception:
            try:
                return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc).replace(tzinfo=None)
            except Exception:
                pass
    return datetime.utcnow()


def parse_rss(xml_text: str, source: str) -> list[dict]:
    items: list[dict] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        _warn(f"{source}: bad XML ({exc})")
        return items
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        if not title:
            continue
        items.append(
            {
                "source": source,
                "title": title,
                "summary": re.sub(r"<[^>]+>", " ", it.findtext("description") or "").strip()[:600],
                "url": (it.findtext("link") or "").strip(),
                "published_at": _parse_date(it.findtext("pubDate")),
            }
        )
    return items


def tag_topics(text: str) -> list[str]:
    low = text.lower()
    return [topic for topic, pat in TOPIC_PATTERNS.items() if re.search(pat, low)]


def party_lean(text: str) -> float:
    """+1 when the text says Democrats are gaining, -1 for Republicans, 0 otherwise."""
    low = text.lower()
    dem, gop = bool(re.search(DEM_WIN, low)), bool(re.search(GOP_WIN, low))
    return float(dem) - float(gop)


def to_election_events(items: Iterable[dict]) -> list[dict]:
    events: list[dict] = []
    for it in items:
        text = f"{it['title']}. {it.get('summary', '')}"
        topics = tag_topics(text)
        if not topics:
            continue
        tone = simple_sentiment_score(text)
        lean = party_lean(text)
        key_base = hashlib.sha1(f"{it['source']}|{it['title']}".encode()).hexdigest()[:24]
        for topic in topics:
            polarity = lean if topic in ("senate_control", "house_control") and lean else tone
            events.append(
                {
                    "created_at": it["published_at"],
                    "source": it["source"],
                    "ticker_or_topic": topic,
                    "polarity_score": round(float(polarity), 4),
                    "rationale": json.dumps({"tone": tone, "dem_lean": lean, "topics": topics}),
                    "title": it["title"][:500],
                    "url": it.get("url"),
                    "dedupe_key": f"{key_base}:{topic}"[:64],
                }
            )
    return events


def fetch_news_items(queries: Iterable[str] = GOOGLE_NEWS_QUERIES, feeds: dict[str, str] = RSS_FEEDS) -> list[dict]:
    items: list[dict] = []
    for q in queries:
        url = f"https://news.google.com/rss/search?q={quote_plus(q)}+when:7d&hl=en-US&gl=US&ceid=US:en"
        r = _get(url)
        if r is not None:
            items.extend(parse_rss(r.text, "google_news"))
    for name, url in feeds.items():
        r = _get(url)
        if r is not None:
            items.extend(parse_rss(r.text, name))
    return items


def fetch_polymarket_odds(events: dict[str, str] = POLYMARKET_EVENTS) -> list[dict]:
    """Democratic-control probabilities per chamber as SentimentSnapshot dicts.

    ``polarity_score`` = 2 * P(Dem control) - 1 so +1 means Democrats certain.
    """
    snaps: list[dict] = []
    now = datetime.utcnow()
    for topic, slug in events.items():
        r = _get(f"{POLYMARKET_GAMMA}/events?slug={slug}")
        if r is None:
            continue
        try:
            event = r.json()[0]
            prob = None
            for m in event.get("markets", []):
                if "Democratic Party control" in (m.get("question") or "") and m.get("outcomePrices"):
                    prob = float(json.loads(m["outcomePrices"])[0])
                    break
            if prob is None:
                continue
            snaps.append(
                {
                    "created_at": now,
                    "source": "polymarket",
                    "ticker_or_topic": topic,
                    "probability": prob,
                    "polarity_score": round(2 * prob - 1, 4),
                    "n_items": 1,
                    "rationale": json.dumps({"slug": slug, "volume": event.get("volume"), "p_dem": prob}),
                }
            )
        except Exception as exc:
            _warn(f"polymarket {slug}: {exc}")
    return snaps


def aggregate_sentiment(events: list[dict], source: str = "news_aggregate") -> list[dict]:
    by_topic: dict[str, list[float]] = {}
    for e in events:
        by_topic.setdefault(e["ticker_or_topic"], []).append(float(e["polarity_score"]))
    now = datetime.utcnow()
    return [
        {
            "created_at": now,
            "source": source,
            "ticker_or_topic": topic,
            "polarity_score": round(sum(v) / len(v), 4),
            "n_items": len(v),
            "rationale": json.dumps({"n": len(v)}),
        }
        for topic, v in by_topic.items()
    ]


def ingest_all(db_path: str = "sqlite:///quantflow.db", *, fetch_news: bool = True, fetch_odds: bool = True) -> dict:
    from ..data.persistence import save_election_events, save_sentiment_snapshots

    events = to_election_events(fetch_news_items()) if fetch_news else []
    snaps = aggregate_sentiment(events) + (fetch_polymarket_odds() if fetch_odds else [])
    saved_events = save_election_events(events, db_path)
    saved_snaps = save_sentiment_snapshots(snaps, db_path)
    return {
        "events_parsed": len(events),
        "events_saved": saved_events,
        "snapshots_saved": saved_snaps,
        "odds": {s["ticker_or_topic"]: s.get("probability") for s in snaps if s["source"] == "polymarket"},
        "topics": sorted({e["ticker_or_topic"] for e in events}),
    }
