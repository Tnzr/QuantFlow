# QuantFlow Repository Progress Direction Report

## Executive Assessment
QuantFlow has a working PoC foundation for screening, recommendation, options heuristics, portfolio readout, and visualization. The repository is currently in a strong pre-ML, rule-based stage with clear upgrade paths.

Current maturity level:
- Stage: PoC / operator-assisted decision system
- Strength: practical end-to-end flow exists
- Gap: reliability hardening, richer data model, advanced model training, and execution safety infrastructure

## What Is Implemented Today

### Data and Screening
- Finviz preset-based screening with YAML overrides
- Snapshot persistence into SQLite via SQLAlchemy
- Basic parallel screener execution available

### Feature and Signal Layer
- OHLCV retrieval via yfinance
- Indicator computation (RSI, SMA20/50/200, ATR, volatility)
- RuleEngine recommendations for 1w, 1m, 3m, 6m, 1y horizons
- Earnings proximity penalties for short/medium horizons

### Options Candidate Layer
- Contract filtering by budget, OI, volume, spread quality, DTE windows
- Basic Black-Scholes delta-target scoring for candidate ranking

### Backtesting Layer
- Short-term equity proxy backtest with core metrics
- Equity curve and metrics surfaced in dashboard

### UX and Ops
- Streamlit dashboard with pages for scanner/recs/options/portfolio/charts/backtest
- CLI and Make targets for reproducible workflows

## Architectural Strengths
- Clear module boundaries across data, features, recommend, backtest, broker, app
- Persistence layer already present and extensible
- Rule-first approach gives baseline for ML comparison and ablation
- Existing docs and TODO list capture realistic next-step scope

## Primary Gaps and Risks

### 1) Broker Integration Risk
- Legacy robin-stocks credential/MFA handling is deprecated and removed from active broker runtime.
- Remaining risk is MCP onboarding consistency across clients and policy-safe execution controls.

### 2) Data Reliability and Scale
- Need centralized retry/backoff and consistent request policy
- Need dedupe and quality flags to prevent polluted training/inference datasets
- Finviz acquisition pipeline still vulnerable to upstream limits without robust proxy abstraction and caching

### 3) Modeling Gap
- Current system is rule-based; no longitudinal multi-scale model training pipeline yet
- No structured experiment tracking and promotion gating for models

### 4) Backtest Fidelity
- Backtest is currently simplified and equity-proxy oriented
- Missing realistic options PnL approximation, slippage model, and order simulation depth

### 5) Explainability and Governance
- Explanations exist in notes but no formal attribution layer
- No policy engine and execution audit ledger for production-grade automation

## Progress Direction (Recommended)

### Direction A: Reliability Before Complexity
Immediate priority should be data/execution reliability improvements before introducing heavier ML complexity.

### Direction B: MCP-First Broker Evolution
Shift Robinhood integration to MCP-oriented architecture while preserving legacy fallback mode during transition.

### Direction C: Structured Feature Expansion
Prioritize seasonality and sentiment as modular feature streams that can be ablated independently.

### Direction D: Measurable Model Progression
Move from rule engine to model-assisted decisions with explicit baseline comparisons and promotion criteria.

## 12-Week Practical Roadmap

### Weeks 1-4
- Introduce request reliability layer and dataset quality checks
- Add schema extensions for signals, backtest runs, option ideas, and sentiment-ready tables
- Improve Finviz candidate ranking to reduce noisy universe expansion

### Weeks 5-8
- Integrate enhanced seasonality factors (DOY + DOW + sector conditioning)
- Add attribution in recommendation outputs (technical vs seasonality)
- Add improved stop-management logic for options playbooks

### Weeks 9-12
- Build MCP migration scaffolding and approval queue architecture
- Add lightweight sentiment ingestion and scoring pipeline
- Start shadow evaluation framework comparing rule decisions vs model-assisted proposals

## Alignment with Longitudinal Multi-Scale AI Principle
The existing codebase is a suitable substrate for the Temporal AI direction:
- Data ingestion and persistence are already modular
- Signal logic is explicit and can become supervisory labels/baselines
- UI and CLI paths already support iterative operator workflow

Needed to fully align:
- Multi-scale dataset builder
- Model training orchestration and experiment tracking
- Confidence smoothing and consistency constraints
- Policy-safe execution loop with audit and reconciliation

## Conclusion
QuantFlow is on the correct trajectory.

Immediate success depends less on adding many new model ideas and more on:
- hardening ingestion and execution,
- migrating broker integration to MCP-first connectivity,
- and adding measurable, incremental intelligence layers (seasonality then sentiment) before full multi-asset autonomous orchestration.