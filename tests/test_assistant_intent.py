"""Unit tests for assistant_query intent routing.

These tests call assistant_query directly against a minimal SQLite DB so they
are fast and self-contained — no network required.
"""
from __future__ import annotations

import os
import sys
import unittest

os.environ.setdefault("QF_DB_PATH", "sqlite:///quantflow.db")

from quantflow.api.server import assistant_query, AssistantQueryRequest  # noqa: E402

DB = "sqlite:///quantflow.db"


def _ask(message: str) -> dict:
    return assistant_query(AssistantQueryRequest(message=message, db=DB))


class SmallTalkTests(unittest.TestCase):
    """Greetings and punctuation-only inputs should get a neutral reply with no tool calls."""

    def _assert_neutral(self, msg: str):
        out = _ask(msg)
        self.assertEqual(out.get("suggested_tool_calls", []), [],
                         f"Expected no tool calls for {msg!r}, got {out.get('suggested_tool_calls')}")
        self.assertNotIn("From offline QuantFlow data, top current ideas are",
                         out.get("answer", ""),
                         f"Unexpected top-ideas blurb for {msg!r}")

    def test_hello(self):
        self._assert_neutral("hello")

    def test_hello_caps(self):
        self._assert_neutral("Hello")

    def test_hey(self):
        self._assert_neutral("hey")

    def test_how_are_you_no_punctuation(self):
        self._assert_neutral("how are you")

    def test_how_are_you_with_question_mark(self):
        self._assert_neutral("how are you?")

    def test_period_only(self):
        self._assert_neutral(".")

    def test_thanks(self):
        self._assert_neutral("thanks")

    def test_ok(self):
        self._assert_neutral("ok")


class NonFintechFallbackTests(unittest.TestCase):
    """Non-financial, unrecognised text should not return top-ideas or tool calls."""

    def _assert_fallback(self, msg: str):
        out = _ask(msg)
        self.assertEqual(out.get("suggested_tool_calls", []), [],
                         f"Expected no tool calls for {msg!r}")
        self.assertNotIn("From offline QuantFlow data, top current ideas are",
                         out.get("answer", ""),
                         f"Unexpected top-ideas blurb for {msg!r}")
        self.assertIn("didn't catch", out.get("answer", ""),
                      f"Expected guidance reply for {msg!r}")

    def test_random_string(self):
        self._assert_fallback("sfgsfdsdf")

    def test_conversational_why(self):
        self._assert_fallback("why did you write that?")

    def test_what_question_mark(self):
        self._assert_fallback("what?")

    def test_unrelated_sentence(self):
        self._assert_fallback("what is the weather like today?")


class CapabilityTests(unittest.TestCase):
    """Help/capability queries should return a description with no tool calls."""

    def _assert_capabilities(self, msg: str):
        out = _ask(msg)
        self.assertEqual(out.get("suggested_tool_calls", []), [],
                         f"Expected no tool calls for {msg!r}")
        answer = out.get("answer", "")
        self.assertTrue(
            "can" in answer.lower() or "ask" in answer.lower() or "request" in answer.lower(),
            f"Capability reply looks wrong for {msg!r}: {answer!r}"
        )

    def test_what_can_you_do(self):
        self._assert_capabilities("what can you do?")

    def test_what_else_can_you_do(self):
        self._assert_capabilities("what else can you do?")

    def test_help(self):
        self._assert_capabilities("help")


class IntentRoutingTests(unittest.TestCase):
    """Specific financial keywords should route to the correct intent branch."""

    def test_mcp_keyword_routes_to_mcp(self):
        out = _ask("what is the mcp status?")
        answer = out.get("answer", "").lower()
        self.assertTrue(
            "robinhood" in answer or "mcp" in answer or "authenticated" in answer,
            f"MCP intent not matched: {answer!r}"
        )

    def test_weekly_scan_routes_to_scan(self):
        out = _ask("run a weekly universe scan")
        answer = out.get("answer", "").lower()
        self.assertIn("weekly", answer)

    def test_top_opportunities_routes_to_rankings(self):
        out = _ask("show me the top opportunities")
        answer = out.get("answer", "")
        calls = out.get("suggested_tool_calls", [])
        # Either shows ranked opps or says none are available yet
        self.assertTrue(
            "conviction" in answer.lower()
            or "rank" in answer.lower()
            or "no ranked" in answer.lower(),
            f"Ranking intent not matched: {answer!r}"
        )

    def test_technical_analysis_routes_correctly(self):
        out = _ask("run technical analysis for AAPL")
        calls = [c["name"] for c in (out.get("suggested_tool_calls") or [])]
        self.assertTrue(
            any("indicator" in c or "seasonal" in c or "sentiment" in c for c in calls),
            f"Technical intent should return analysis calls, got: {calls}"
        )

    def test_earnings_keyword_routes_to_earnings(self):
        out = _ask("show me quarterly earnings timing")
        calls = [c["name"] for c in (out.get("suggested_tool_calls") or [])]
        self.assertTrue(
            any("earnings" in c for c in calls),
            f"Earnings intent not matched: {calls}"
        )

    def test_summary_field_always_present(self):
        for msg in ["hello", "top ideas", "sfgsdgsdf"]:
            out = _ask(msg)
            self.assertIn("summary", out, f"summary missing for {msg!r}")
            self.assertIn("recommendations", out["summary"])


if __name__ == "__main__":
    unittest.main()
