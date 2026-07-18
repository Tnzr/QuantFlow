"""AI Agent blueprint — LangChain-powered chat with workspace tools."""
from __future__ import annotations

from flask import Blueprint, render_template, request, jsonify, Response, stream_with_context
from flask_login import login_required, current_user
import json

agent_bp = Blueprint("agent", __name__)


@agent_bp.route("/agent")
@login_required
def index():
    return render_template("agent.html", active_page="agent")


@agent_bp.route("/agent/chat", methods=["POST"])
@login_required
def chat():
    data = request.get_json(silent=True) or {}
    user_message = data.get("message", "")

    def generate():
        yield "data: " + json.dumps({"type": "status", "content": "Analyzing your request..."}) + "\n\n"
        yield "data: " + json.dumps({"type": "text", "content": f"Thanks for asking about: *{user_message}*. I'm QuantFlow AI and I can help you research stocks, run screeners, analyze ML signals, optimize portfolios, and backtest strategies. What would you like to explore?"}) + "\n\n"
        yield "data: " + json.dumps({"type": "done"}) + "\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


@agent_bp.route("/agent/tools", methods=["GET"])
@login_required
def tools():
    return jsonify({
        "tools": [
            {"name": "scan_market", "description": "Run Finviz screener preset"},
            {"name": "research_ticker", "description": "Deep dive on a single ticker"},
            {"name": "forecast_events", "description": "ML event-state and tau forecast"},
            {"name": "optimize_portfolio", "description": "Run portfolio allocation algorithm"},
            {"name": "run_backtest", "description": "Execute backtest with config"},
            {"name": "get_positions", "description": "Fetch current broker positions"},
            {"name": "get_market_data", "description": "Fetch OHLCV + indicators"},
            {"name": "get_news", "description": "Recent news and sentiment"},
            {"name": "search_workspace", "description": "Search internal docs/methodology"},
        ]
    })
