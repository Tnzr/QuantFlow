# QuantFlow TODOs

## Quick Wins (Low-Risk, High-Impact)
- **Streamlit Portfolio UX**
  - Add MCP readiness widget with direct guidance to /broker/mcp/login instructions.
  - Show last MCP account/positions error with retry hint and fixture fallback note.
- **Preset Management UI**
  - Streamlit page to load/edit finviz_presets.yml with YAML validation and safe write; apply without app restart.
- **Robust Retries and Dedupe**
  - Wrap Finviz/yfinance calls with retry/backoff (tenacity), rate-limit sleep, and per-day/preset dedupe on save.

## Options Analytics
- **IV Rank/Percentile**
  - Compute per-ticker IVP from historical IV proxy (e.g., 30d ATM IV derived from option chain) with rolling window; persist to DB; expose in ideas table and RuleEngine score.
- **Skew/Smile Checks**
  - Compare IV across deltas (e.g., 0.25, 0.5, 0.75) and penalize poor skew; add to ranking.
- **Low-Budget Spreads**
  - Add debit verticals generator for target delta with budget cap; rank by probITM and max loss.

## Backtesting
- **Options PnL Approximation**
  - Delta–theta–vega approximation with spot/IV paths; DTE-aware exit rules; simple slippage model.
  - Record trades and equity; persist run records; visualize in Streamlit.

## Scheduling and Reliability
- **Make + Cron/GitHub Actions**
  - Add GH Actions workflow to run “make daily” on market days; artifacts as CSV; optional DB upload.
- **Rate-Limit + Resilience**
  - Centralized request layer with retries, jitter, and backoff; graceful partial failures.

## Portfolio Integration
- **Robinhood MFA Flows**
  - Keep disabled for app runtime; authentication is handled by MCP-capable clients.
- **Broker Abstraction**
  - Extend Broker base with logout/session methods; prep for additional brokers.
- **Robinhood MCP Migration (Priority)**
  - Enforce robinhood_mcp as the only supported broker mode.
  - Implement MCP adapter scaffold and read-only account/position retrieval first.
  - Add execution intent log + policy gate before any live order placement.
  - Build reconciliation job between local intents and broker order history.

## LLM/Sentiment (Scaffold)
- **News Ingestion + Sentiment**
  - Ingest headlines (e.g., yfinance news API), store to DB, basic VADER/FinBERT sentiment, expose in UI; later plug LLM RAG.

## Testing/QA/CI
- **Pytest Suite**
  - Unit tests for indicators, RuleEngine, persistence; notebook smoke tests with papermill; pre-commit (ruff/black).
- **CI**
  - Run tests and lint on PR; cache deps; optional Docker build.

## Frontend-Backend Integration (Priority)
- **Assistant Side Panel Workflow**
  - Keep assistant visible while navigating all workspace tabs.
  - Support multi-step plans and chained API actions from one conversation.
  - Add action execution timeline and rollback guidance for failed steps.
- **Backtest UX Completion**
  - Expand controls: start/end date, entry RSI threshold, max hold days, and scenario presets.
  - Plot equity curve and trade-level diagnostics (entry/exit markers, drawdown panel).
  - Export backtest run metrics and equity series as CSV.
- **Market Data Cache + Plot Baseline**
  - Persist yfinance OHLCV+indicator series in local cache for re-use across tabs.
  - Add dataset builder actions to emit model-ready files from cached series.
  - Show cache metadata (symbol, rows, freshness timestamp, source period/interval).
- **Charts Reliability**
  - Add explicit fetch diagnostics for RSI/indicator charts (API base, endpoint, last error).
  - Add fallback chart placeholders when endpoints return empty series.
  - Add quick smoke check button to validate all chart endpoints from UI.

## Data Model
- **New Tables**
  - iv_metrics, option_ideas (cached), news_items, signals, backtest_runs; simple migrations helper.
  - Add execution_intents, execution_events, broker_reconciliations, policy_decisions for MCP workflow.
