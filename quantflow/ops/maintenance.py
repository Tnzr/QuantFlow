from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .reporting import DbFreshnessReport


@dataclass
class RefreshPlan:
    refresh_snapshots: bool
    refresh_recommendations: bool


@dataclass
class RefreshResult:
    refresh_snapshots: bool
    refresh_recommendations: bool
    snapshots_rows_saved: int
    recommendations_saved: int


def build_refresh_plan(
    report: DbFreshnessReport,
    stale_days: int,
    force: bool,
) -> RefreshPlan:
    snapshots_age = report.age_days(report.snapshots.latest_at)
    recs_age = report.age_days(report.recommendations.latest_at)

    snapshots_stale = snapshots_age is None or snapshots_age > float(stale_days)
    recs_stale = recs_age is None or recs_age > float(stale_days)

    return RefreshPlan(
        refresh_snapshots=force or snapshots_stale,
        refresh_recommendations=force or recs_stale,
    )


def execute_refresh_plan(
    plan: RefreshPlan,
    run_scan: Callable[[], int],
    run_recommend: Callable[[], int],
) -> RefreshResult:
    snapshots_rows_saved = 0
    recommendations_saved = 0

    if plan.refresh_snapshots:
        snapshots_rows_saved = int(run_scan())

    if plan.refresh_recommendations:
        recommendations_saved = int(run_recommend())

    return RefreshResult(
        refresh_snapshots=plan.refresh_snapshots,
        refresh_recommendations=plan.refresh_recommendations,
        snapshots_rows_saved=snapshots_rows_saved,
        recommendations_saved=recommendations_saved,
    )
