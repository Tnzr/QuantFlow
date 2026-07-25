from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

from quantflow.ops.maintenance import build_refresh_plan, execute_refresh_plan
from quantflow.ops.reporting import DbFreshnessReport, TableStats


class MaintenanceTests(unittest.TestCase):
    def test_build_refresh_plan_marks_stale_tables(self):
        now = datetime(2026, 1, 10, tzinfo=timezone.utc)
        report = DbFreshnessReport(
            generated_at=now,
            db_path="sqlite:///tmp.db",
            snapshots=TableStats(count=10, latest_at=now - timedelta(days=40)),
            recommendations=TableStats(count=4, latest_at=now - timedelta(days=1)),
            execution_intents=TableStats(count=0, latest_at=None),
        )

        plan = build_refresh_plan(report, stale_days=7, force=False)
        self.assertTrue(plan.refresh_snapshots)
        self.assertFalse(plan.refresh_recommendations)

    def test_build_refresh_plan_force_refreshes_everything(self):
        now = datetime(2026, 1, 10, tzinfo=timezone.utc)
        report = DbFreshnessReport(
            generated_at=now,
            db_path="sqlite:///tmp.db",
            snapshots=TableStats(count=10, latest_at=now),
            recommendations=TableStats(count=4, latest_at=now),
            execution_intents=TableStats(count=0, latest_at=None),
        )

        plan = build_refresh_plan(report, stale_days=7, force=True)
        self.assertTrue(plan.refresh_snapshots)
        self.assertTrue(plan.refresh_recommendations)

    def test_execute_refresh_plan_runs_expected_callbacks(self):
        calls = {"scan": 0, "rec": 0}

        def run_scan() -> int:
            calls["scan"] += 1
            return 123

        def run_recommend() -> int:
            calls["rec"] += 1
            return 17

        plan = type("Plan", (), {"refresh_snapshots": True, "refresh_recommendations": False})()
        result = execute_refresh_plan(plan, run_scan=run_scan, run_recommend=run_recommend)

        self.assertEqual(calls["scan"], 1)
        self.assertEqual(calls["rec"], 0)
        self.assertEqual(result.snapshots_rows_saved, 123)
        self.assertEqual(result.recommendations_saved, 0)


if __name__ == "__main__":
    unittest.main()
