# QuantFlow Reconstructed Master Plan

## Purpose
This document reconstructs and consolidates the long Qwen conversation into a practical, implementation-oriented program for QuantFlow.

It is organized around the same major sections you repeatedly refined:
- Platform vision and operating principles
- Phased technical development
- Data and model methodology
- Backtesting and deployment
- Business, legal, and go-to-market trajectory
- Solidarity fund direction

## 1) Platform Vision and Scope
QuantFlow is a Wealth Management AI platform that combines:
- Technical structure detection (trend, resistance/support, volatility)
- Seasonality (day-of-week and day-of-year effects)
- Sentiment and news impact analysis
- Automated execution support for stocks and options

Near-term objective:
- Build a reliable decision assistant that reduces manual chart/news workload
- Improve entry and exit discipline, especially for options where stop management is operationally harder

Long-term objective:
- Evolve toward a distributed investment platform with optional solidarity allocation from profits

## 2) Risk and Operating Principles
Core operating rules from your prior planning should remain non-negotiable:
- Position risk caps and defined-loss structures first
- Volatility-aware dynamic stops
- Event-aware controls (earnings/macro windows)
- Liquidity-first options selection
- No model signal should bypass risk constraints

Execution hierarchy:
1. Risk constraints
2. Portfolio constraints
3. Signal quality and confidence
4. Execution feasibility (slippage/liquidity)

## 3) Phase Framework (Reconstructed)

### Phase 1: Fast PoC MVP (Technical Signals + Execution Discipline)
Goal:
- Deliver immediate practical value using existing QuantFlow assets
- Improve trend/breakout anticipation and stop management quality

Build scope:
- Strengthen current rule engine with better exit logic and confidence scoring
- Add robust data reliability layer (retry, dedupe, caching, failure fallback)
- Add practical stop management workflows for options
- Add baseline LLM control plane for settings and policy decisions (not autonomous trading)

Outputs:
- Stable daily pipeline from screening to recommendations
- Better signal quality metrics and explainable recommendation notes
- Dashboard support for signal review and stop-adjustment workflow

Success metrics:
- Fewer false-positive entries
- Better realized risk-adjusted return than baseline rule version
- Reduced manual time spent on routine scan/review

### Phase 2: Seasonality Integration
Goal:
- Add systematic calendar effects into forecasting and signal filtering

Build scope:
- Expand seasonality beyond current DOY aggregate to DOY + DOW + regime conditioning
- Store seasonality factors by ticker and sector
- Integrate seasonality factors into confidence and position sizing logic

Outputs:
- Seasonality feature store and diagnostics
- Seasonality-aware signal gating and scoring
- Attribution in dashboard (technical vs seasonality contribution)

Success metrics:
- Lower drawdown during weak seasonal windows
- Improved timing of entries and exits versus non-seasonal baseline

### Phase 3: Sentiment Foundations (Low-Cost First)
Goal:
- Add practical sentiment signal without heavy initial infrastructure

Build scope:
- Ingest structured headlines/news metadata
- Start with light sentiment stack (headline scoring + entity extraction)
- Aggregate sentiment by ticker/sector/theme and time window
- Fuse with technical + seasonality signals using confidence gating

Outputs:
- News ingestion pipeline with persistent storage
- Sentiment features and daily aggregates
- Explainable sentiment contribution in recommendation notes

Success metrics:
- Better handling of event-driven moves
- Improved signal robustness during high-news periods

### Phase 4: Multi-Asset + LLM/MCP Reasoning Layer
Goal:
- Scale from single-asset reliability toward portfolio-level orchestration

Build scope:
- Multi-asset allocation engine with correlation-aware constraints
- RAG/MCP reasoning for cross-asset news impact interpretation
- LLM as supervisor/controller for policy, not direct unrestricted executor

Outputs:
- Portfolio-level recommendation engine
- Cross-asset impact graph and sector propagation logic
- Operator-facing reasoning summaries with uncertainty labels

Success metrics:
- Better portfolio-level risk/return than isolated single-asset signals
- Lower concentration risk and better regime adaptation

## 4) Architecture Methodology (Aligned to Temporal AI Report)
Your Temporal AI architecture principle can be summarized as:
- Multi-scale token generation compresses long horizons while preserving local detail
- Longitudinal coherence smooths noisy local predictions
- Multi-task heads enforce consistency between classification and regression outputs
- Continual adaptation should be constrained to avoid catastrophic forgetting

Practical interpretation for QuantFlow:
- Use parallel multi-scale encoders for micro-to-macro context
- Use coherent heads for regime, support/resistance proximity, and target horizon
- Use confidence smoothing over prediction history to reduce whipsaws
- Keep decision explainability through explicit feature groups and gated fusion

## 5) Data Strategy

### Current strengths
- Finviz screening
- yfinance OHLCV and options baseline
- SQLite persistence and Streamlit inspection

### Required upgrades
- Data quality controls (retry, dedupe, schema expansion, quality flags)
- Incremental enrichment tables (signals, idea cache, sentiment, backtest runs)
- Optional proxy-backed ingestion for web-based sources where allowed

### Finviz optimization direction
- Parallelized preset execution with bounded concurrency
- Request-level resilience and jitter
- Proxy-provider abstraction (Scrape.do / SmartProxy / ScrapingBee) for compliant access paths
- Candidate universe ranking to reduce LLM search space and avoid obvious-leader bias

## 6) ML Training and Loss Strategy (Reconstructed)
Training objective is not pure price prediction. It is decision quality under risk constraints.

Recommended setup:
- Dual objective: directional/forecast output + action quality output
- Loss blend: forecast loss, classification loss, confidence calibration, regime consistency, risk penalty
- Include transaction costs and slippage assumptions early in evaluation

Training workflow:
1. Offline feature and label generation
2. Supervised pretraining on historical windows
3. Walk-forward validation
4. Backtest checkpointing at schedule milestones (not every epoch)
5. Candidate promotion to paper or shadow deployment

## 7) Backtesting and Evaluation Framework
Backtesting must move from simple signal checks toward decision realism.

Required dimensions:
- Costs: slippage, fees, spread penalties
- Realistic exits: stop, trailing logic, time stop
- Regime breakdown: bull/bear/high-vol/low-vol
- Attribution: technical vs seasonality vs sentiment contribution

Primary metrics:
- Risk-adjusted return, drawdown, win/loss asymmetry, profit factor
- Calibration quality and confidence reliability
- Stability across rolling windows and out-of-sample periods

## 8) Deployment and Operations

### Near-term deployment pattern
- Keep CLI and Streamlit as operator control surface
- Add service layer for ingestion, scoring, and signal persistence
- Add scheduled jobs for market-day workflows

### Execution model
- Human-in-the-loop first
- Policy-based automation second
- Fully autonomous actions only after robust safeguards and auditability

### Observability
- Track data freshness, failure rates, and signal drift
- Track model confidence drift and regime misclassification rate
- Keep immutable logs for action traceability

## 9) Business and Legal Direction (Consolidated)
This is not legal advice. Treat this as planning direction.

For a solidarity-oriented investment platform, separate concerns early:
- Investment/advisory entity
- Technology entity
- Impact-distribution entity

Immediate legal posture for PoC and early MVP:
- Avoid public promises of guaranteed returns
- Keep clear educational/research framing while validating system
- Define governance and accounting boundaries for any optional solidarity allocation

## 10) Solidarity Fund Design Direction
Design principles:
- Optional allocation from profits, not principal by default
- Full transparency in allocation rules and impact tracking
- Evidence-based deployment to programs with measurable outcomes

Implementation prerequisites:
- Separate accounting and governance controls
- Public reporting format for contributions and outcomes
- Clear opt-in workflow and user disclosure language

## 11) 90-Day Action Stack

### 0-30 days
- Stabilize data ingestion and Finviz optimization with resilient request handling
- Improve rule-engine exits and confidence notes
- Expand persistence schema for signal/backtest/sentiment-ready tables

### 31-60 days
- Add seasonality feature store and integrate into scoring
- Add baseline news ingestion and lightweight sentiment aggregation
- Add dashboard attribution panels

### 61-90 days
- Introduce low-cost LLM control layer for settings/policy recommendations
- Add shadow-mode inference comparisons and promotion criteria
- Prepare broker integration migration path for MCP-based flow

## 12) Decision Rules for Scope Control
To avoid overbuilding too early:
- If it does not improve risk-adjusted decision quality, delay it
- If it cannot be measured, do not automate it yet
- If it cannot be explained to the operator, gate it behind review mode

## 13) Final Reconstructed Program Statement
QuantFlow should progress as a disciplined, explainable, risk-first investment intelligence platform:
- Start with robust technical decision support
- Layer seasonality for timing edge
- Add sentiment for event awareness
- Scale to multi-asset orchestration with LLM/MCP reasoning only after reliability gates are met

This sequence preserves speed-to-value while building toward your broader enterprise and solidarity vision.