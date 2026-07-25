from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Optional

import requests
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..data.persistence import (
    ExecutionIntent,
    Recommendation,
    TickerSnapshot,
    get_engine,
)


@dataclass
class TableStats:
    count: int
    latest_at: Optional[datetime]


@dataclass
class DbFreshnessReport:
    generated_at: datetime
    db_path: str
    snapshots: TableStats
    recommendations: TableStats
    execution_intents: TableStats

    def age_days(self, latest_at: Optional[datetime]) -> Optional[float]:
        if latest_at is None:
            return None
        delta = self.generated_at - _as_utc(latest_at)
        return max(delta.total_seconds() / 86400.0, 0.0)


@dataclass
class RefreshValidationSummary:
    snapshot_rows_saved: int
    recommendations_saved: int
    successful_presets: int
    failed_presets: list[str]
    successful_recommendation_tickers: int
    failed_recommendation_tickers: list[str]


def default_json_report_path(markdown_output_path: str) -> str:
    target = Path(markdown_output_path)
    return str(target.with_suffix(".json"))


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _table_stats(session: Session, model: Any) -> TableStats:
    count = int(session.execute(select(func.count()).select_from(model)).scalar_one())
    latest = session.execute(select(func.max(model.created_at))).scalar_one_or_none()
    return TableStats(count=count, latest_at=latest)


def build_db_freshness_report(
    db_path: str = "sqlite:///quantflow.db",
    now: Optional[datetime] = None,
) -> DbFreshnessReport:
    generated_at = _as_utc(now or datetime.now(timezone.utc))
    engine = get_engine(db_path)
    with Session(engine) as session:
        snapshots = _table_stats(session, TickerSnapshot)
        recommendations = _table_stats(session, Recommendation)
        intents = _table_stats(session, ExecutionIntent)

    return DbFreshnessReport(
        generated_at=generated_at,
        db_path=db_path,
        snapshots=snapshots,
        recommendations=recommendations,
        execution_intents=intents,
    )


def fetch_api_runtime_status(api_base: str) -> dict[str, Any]:
    base = api_base.rstrip("/")

    def _get(path: str) -> dict[str, Any]:
        resp = requests.get(f"{base}{path}", timeout=8)
        resp.raise_for_status()
        return resp.json()

    runtime: dict[str, Any] = {"api_base": base}
    try:
        runtime["health"] = _get("/health")
    except Exception as exc:
        runtime["health_error"] = str(exc)

    try:
        runtime["mcp_status"] = _get("/broker/mcp/status")
    except Exception as exc:
        runtime["mcp_status_error"] = str(exc)

    try:
        runtime["mcp_readiness"] = _get("/broker/mcp/readiness")
    except Exception as exc:
        runtime["mcp_readiness_error"] = str(exc)

    try:
        runtime["scraping_status"] = _get("/scanner/scraping/status")
    except Exception as exc:
        runtime["scraping_status_error"] = str(exc)

    return runtime


def _fmt_dt(value: Optional[datetime]) -> str:
    if value is None:
        return "none"
    return _as_utc(value).isoformat()


def _fmt_age(days: Optional[float]) -> str:
    if days is None:
        return "none"
    return f"{days:.1f} days"


def _table_line(name: str, stats: TableStats, age: Optional[float]) -> str:
    return f"- {name}: count={stats.count}, latest={_fmt_dt(stats.latest_at)}, age={_fmt_age(age)}"


def _mcp_guidance_lines() -> list[str]:
    return [
        "1. Robinhood Agentic auth is performed in an MCP-capable client session, not in QuantFlow itself.",
        "2. Authenticate your Robinhood Agentic account in that client; this usually opens Robinhood's auth flow.",
        "3. Return to QuantFlow and re-check /broker/mcp/status until authenticated=true.",
        "4. If auth is blocked, set QF_MCP_ACCOUNT_JSON and QF_MCP_POSITIONS_JSON for local fixture tests.",
    ]


def render_textbook_markdown_report(
    db_report: DbFreshnessReport,
    runtime: Optional[dict[str, Any]] = None,
    stale_days: int = 7,
    validation: Optional[RefreshValidationSummary] = None,
) -> str:
    snapshots_age = db_report.age_days(db_report.snapshots.latest_at)
    recs_age = db_report.age_days(db_report.recommendations.latest_at)
    intents_age = db_report.age_days(db_report.execution_intents.latest_at)

    snapshots_stale = snapshots_age is None or snapshots_age > float(stale_days)
    recs_stale = recs_age is None or recs_age > float(stale_days)

    status_line = "REFRESH REQUIRED" if (snapshots_stale or recs_stale) else "FRESH"

    lines = [
        "# QuantFlow Operations Report",
        "",
        "## Executive Summary",
        f"- Generated at: {_fmt_dt(db_report.generated_at)}",
        f"- DB target: {db_report.db_path}",
        f"- Freshness threshold: {stale_days} days",
        f"- Overall DB status: {status_line}",
        "",
        "## Local Database Freshness",
        _table_line("ticker_snapshots", db_report.snapshots, snapshots_age),
        _table_line("recommendations", db_report.recommendations, recs_age),
        _table_line("execution_intents", db_report.execution_intents, intents_age),
        "",
        "## Risk and Action Plan",
        f"- Snapshot refresh needed: {'yes' if snapshots_stale else 'no'}",
        f"- Recommendation refresh needed: {'yes' if recs_stale else 'no'}",
        "- Suggested action: run `python -m quantflow.cli ops-refresh --stale-days 7 --with-report`",
        "",
        "## Robinhood Agentic Authentication Notes",
    ]
    lines.extend([f"- {row}" for row in _mcp_guidance_lines()])

    if runtime is not None:
        health = runtime.get("health")
        mcp = runtime.get("mcp_status")
        readiness = runtime.get("mcp_readiness")
        scraping = runtime.get("scraping_status")
        lines.extend(
            [
                "",
                "## Runtime API Snapshot",
                f"- API base: {runtime.get('api_base', 'unknown')}",
                f"- Health: {health if health is not None else runtime.get('health_error', 'unavailable')}",
                f"- MCP status: {mcp if mcp is not None else runtime.get('mcp_status_error', 'unavailable')}",
                f"- MCP readiness: {readiness if readiness is not None else runtime.get('mcp_readiness_error', 'unavailable')}",
                f"- Scraping status: {scraping if scraping is not None else runtime.get('scraping_status_error', 'unavailable')}",
            ]
        )

    if validation is not None:
        lines.extend(
            [
                "",
                "## Validation Results",
                f"- Snapshot rows saved: {validation.snapshot_rows_saved}",
                f"- Recommendations saved: {validation.recommendations_saved}",
                f"- Successful presets: {validation.successful_presets}",
                f"- Failed presets: {validation.failed_presets if validation.failed_presets else 'none'}",
                f"- Successful recommendation tickers: {validation.successful_recommendation_tickers}",
                f"- Failed recommendation tickers: {validation.failed_recommendation_tickers if validation.failed_recommendation_tickers else 'none'}",
            ]
        )

    lines.extend(
        [
            "",
            "## Appendix: Interpretation",
            "- If MCP is connected=true but authenticated=false, transport works but account auth is incomplete.",
            "- Scraping mode should be proxy or api for resilient Finviz pulls in constrained networks.",
            "- For safe local testing without Robinhood auth, use fixture env vars and keep execution pathways blocked.",
        ]
    )

    return "\n".join(lines) + "\n"


def build_report_payload(
    db_report: DbFreshnessReport,
    runtime: Optional[dict[str, Any]] = None,
    stale_days: int = 7,
    validation: Optional[RefreshValidationSummary] = None,
) -> dict[str, Any]:
    snapshots_age = db_report.age_days(db_report.snapshots.latest_at)
    recs_age = db_report.age_days(db_report.recommendations.latest_at)
    intents_age = db_report.age_days(db_report.execution_intents.latest_at)

    snapshots_stale = snapshots_age is None or snapshots_age > float(stale_days)
    recs_stale = recs_age is None or recs_age > float(stale_days)

    payload: dict[str, Any] = {
        "generated_at": _fmt_dt(db_report.generated_at),
        "db_path": db_report.db_path,
        "stale_days": stale_days,
        "overall_db_status": "REFRESH REQUIRED" if (snapshots_stale or recs_stale) else "FRESH",
        "tables": {
            "ticker_snapshots": {
                "count": db_report.snapshots.count,
                "latest": _fmt_dt(db_report.snapshots.latest_at),
                "age_days": snapshots_age,
                "refresh_needed": snapshots_stale,
            },
            "recommendations": {
                "count": db_report.recommendations.count,
                "latest": _fmt_dt(db_report.recommendations.latest_at),
                "age_days": recs_age,
                "refresh_needed": recs_stale,
            },
            "execution_intents": {
                "count": db_report.execution_intents.count,
                "latest": _fmt_dt(db_report.execution_intents.latest_at),
                "age_days": intents_age,
            },
        },
        "runtime": runtime,
    }

    if validation is not None:
        payload["validation"] = {
            "snapshot_rows_saved": validation.snapshot_rows_saved,
            "recommendations_saved": validation.recommendations_saved,
            "successful_presets": validation.successful_presets,
            "failed_presets": validation.failed_presets,
            "successful_recommendation_tickers": validation.successful_recommendation_tickers,
            "failed_recommendation_tickers": validation.failed_recommendation_tickers,
        }

    return payload


def write_markdown_report(markdown: str, output_path: str) -> str:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(markdown, encoding="utf-8")
    return str(target)


def write_json_report(payload: dict[str, Any], output_path: str) -> str:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return str(target)


def read_json_report(output_path: str) -> dict[str, Any]:
    target = Path(output_path)
    return json.loads(target.read_text(encoding="utf-8"))
