# QuantFlow TODOs

## Quick Wins (Low-Risk, High-Impact)
- **Streamlit Portfolio UX**
  - Add “Save to keyring” checkbox in login form, plus Logout button.
  - Optional session token path preference in sidebar; persist across session.
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
  - Better SMS/app MFA prompts and error feedback; session token save/restore; logout in UI and CLI.
- **Broker Abstraction**
  - Extend Broker base with logout/session methods; prep for additional brokers.
- **Robinhood MCP Migration (Priority)**
  - Add broker mode switch: legacy_robin_stocks vs robinhood_mcp.
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

## Data Model
- **New Tables**
  - iv_metrics, option_ideas (cached), news_items, signals, backtest_runs; simple migrations helper.
  - Add execution_intents, execution_events, broker_reconciliations, policy_decisions for MCP workflow.
