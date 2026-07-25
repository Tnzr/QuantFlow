from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

from sqlalchemy.orm import Session

from quantflow.data.persistence import Recommendation, TickerSnapshot, create_schema, get_engine
from quantflow.ops.reporting import (
    build_db_freshness_report,
    build_report_payload,
    default_json_report_path,
    RefreshValidationSummary,
    read_json_report,
    render_textbook_markdown_report,
    write_json_report,
    write_markdown_report,
)


class ReportingTests(unittest.TestCase):
    def test_build_db_freshness_report_reads_latest_timestamps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_file = Path(tmpdir) / "quantflow-test.db"
            db_url = f"sqlite:///{db_file}"
            create_schema(db_url)
            engine = get_engine(db_url)

            now = datetime(2026, 1, 10, tzinfo=timezone.utc)
            with Session(engine) as session:
                session.add(
                    TickerSnapshot(
                        source="finviz",
                        preset="weekly_momo",
                        ticker="AAPL",
                        created_at=now - timedelta(days=9),
                    )
                )
                session.add(
                    Recommendation(
                        ticker="AAPL",
                        horizon="1m",
                        bias="long",
                        confidence=0.8,
                        created_at=now - timedelta(days=2),
                    )
                )
                session.commit()

            report = build_db_freshness_report(db_url, now=now)
            self.assertEqual(report.snapshots.count, 1)
            self.assertEqual(report.recommendations.count, 1)
            self.assertAlmostEqual(report.age_days(report.snapshots.latest_at), 9.0, places=1)
            self.assertAlmostEqual(report.age_days(report.recommendations.latest_at), 2.0, places=1)

    def test_render_markdown_contains_auth_guidance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_file = Path(tmpdir) / "empty.db"
            db_url = f"sqlite:///{db_file}"
            create_schema(db_url)

            now = datetime(2026, 1, 10, tzinfo=timezone.utc)
            db_report = build_db_freshness_report(db_url, now=now)
            markdown = render_textbook_markdown_report(db_report, runtime=None, stale_days=7)

            self.assertIn("QuantFlow Operations Report", markdown)
            self.assertIn("Robinhood Agentic auth is performed in an MCP-capable client session", markdown)
            self.assertIn("REFRESH REQUIRED", markdown)

    def test_render_markdown_contains_scraping_status_when_runtime_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_file = Path(tmpdir) / "empty.db"
            db_url = f"sqlite:///{db_file}"
            create_schema(db_url)

            now = datetime(2026, 1, 10, tzinfo=timezone.utc)
            db_report = build_db_freshness_report(db_url, now=now)
            runtime = {
                "api_base": "http://127.0.0.1:8100",
                "health": {"status": "ok"},
                "scraping_status": {"provider": "smartproxy", "mode": "proxy", "proxy_configured": True},
            }
            markdown = render_textbook_markdown_report(db_report, runtime=runtime, stale_days=7)
            self.assertIn("Runtime API Snapshot", markdown)
            self.assertIn("Scraping status:", markdown)
            self.assertIn("smartproxy", markdown)

    def test_render_markdown_contains_validation_results_when_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_file = Path(tmpdir) / "empty.db"
            db_url = f"sqlite:///{db_file}"
            create_schema(db_url)

            now = datetime(2026, 1, 10, tzinfo=timezone.utc)
            db_report = build_db_freshness_report(db_url, now=now)
            validation = RefreshValidationSummary(
                snapshot_rows_saved=321,
                recommendations_saved=17,
                successful_presets=8,
                failed_presets=["leaps_quality"],
                successful_recommendation_tickers=10,
                failed_recommendation_tickers=["SPY"],
            )
            markdown = render_textbook_markdown_report(db_report, runtime=None, stale_days=7, validation=validation)
            self.assertIn("Validation Results", markdown)
            self.assertIn("Snapshot rows saved: 321", markdown)
            self.assertIn("Failed presets: ['leaps_quality']", markdown)

    def test_write_markdown_report_persists_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "reports" / "ops.md"
            path = write_markdown_report("# title\n", str(target))
            self.assertEqual(Path(path), target)
            self.assertTrue(target.exists())
            self.assertEqual(target.read_text(encoding="utf-8"), "# title\n")

    def test_json_report_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_file = Path(tmpdir) / "empty.db"
            db_url = f"sqlite:///{db_file}"
            create_schema(db_url)
            report = build_db_freshness_report(db_url, now=datetime(2026, 1, 10, tzinfo=timezone.utc))
            payload = build_report_payload(report, runtime={"health": {"status": "ok"}}, stale_days=7)
            markdown_path = str(Path(tmpdir) / "reports" / "ops.md")
            json_path = default_json_report_path(markdown_path)
            written = write_json_report(payload, json_path)
            self.assertEqual(written, json_path)
            loaded = read_json_report(json_path)
            self.assertEqual(loaded["overall_db_status"], payload["overall_db_status"])
            self.assertEqual(loaded["runtime"]["health"]["status"], "ok")


if __name__ == "__main__":
    unittest.main()
