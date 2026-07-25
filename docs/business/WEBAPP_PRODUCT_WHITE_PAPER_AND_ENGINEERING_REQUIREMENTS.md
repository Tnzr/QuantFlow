# QuantFlow WebApp Product White Paper and Engineering Requirements

## Document Control
- Document ID: QF-WEBAPP-WP-ER-001
- Version: 1.0.0
- Date: 2026-06-16
- Owner: QuantFlow Product and Engineering
- Status: Working Baseline Specification

## Purpose
This document defines the intended QuantFlow WebApp product behavior, engineering requirements, and release acceptance criteria.

It is the single source of truth for:
- What the product should do
- How it should be built and operated
- How progress is measured and validated
- What is complete versus still pending

## Product Vision
QuantFlow is a risk-first AI-assisted trading platform that provides:
- Screened market opportunities
- Rule-assisted recommendations
- Options idea generation
- Backtest diagnostics
- Execution intent governance with audit trails

Primary objective:
Ship a reliable web-first MVP that works end-to-end with deterministic API contracts and visible product telemetry before expanding into advanced ML autonomy.

## Product Scope

### In Scope (MVP)
- FastAPI backend for data, recommendations, options, backtesting, and execution intents
- Expo web frontend with six screens:
  - Overview
  - Scanner
  - Recommend
  - Options
  - Backtest
  - Execution
- SQLite persistence for snapshots, recommendations, and execution logs
- Execution policy gatekeeping and audit logging
- Optional write-auth modes (token and Firebase scaffold)

### Out of Scope (MVP)
- Full autonomous order execution in production markets
- Portfolio-level optimization across many brokers
- Institutional multi-tenant controls and billing
- Guaranteed performance claims or advisory promises

## Personas and Jobs-to-be-Done
- Solo Operator: Wants one place to run scans, inspect ideas, and create governed intents quickly.
- Developer/Researcher: Wants reproducible API behavior and clear quality gates for iterative improvements.
- Compliance-Conscious Reviewer: Wants all write actions policy-checked and auditable.

## Product Principles
- Reliability over novelty
- Explainability over black-box behavior
- Guardrails before automation
- Observable system state at all times

## Functional Requirements

### FR-001 API Root and Health
- The API base URL must return a service payload.
- Health endpoint must indicate status and auth mode.
- Acceptance:
  - GET / returns 200 with name, status, health, docs fields.
  - GET /health returns 200 with status and auth flags.

### FR-002 Scanner Presets and Universe
- Users can view available scanner presets.
- Users can run scanning and later query latest universe.
- Acceptance:
  - GET /scanner/presets returns non-empty list.
  - POST /pipeline/scan returns successful summary.
  - GET /scanner/latest-universe returns rows after scan.

### FR-003 Recommendations Pipeline
- Users can trigger recommendation generation and retrieve latest rows.
- Acceptance:
  - POST /pipeline/recommend returns recommendation_count greater than 0 for at least one supported ticker under normal market data availability.
  - GET /recommend/latest returns persisted rows.

### FR-004 Options Ideas
- Users can request options ideas by ticker, horizon, and budget.
- Acceptance:
  - GET /options/ideas returns structured response with count and items.
  - Empty result is valid only if explicitly justified by liquidity filters and should not be caused by runtime errors.

### FR-005 Backtest Diagnostics
- Users can run short-term backtest summary and equity response.
- Acceptance:
  - GET /backtest/short-term returns summary object and equity array.

### FR-006 Execution Intent Governance
- Users can propose intents; policy engine decides allow/block with reason.
- All proposals are audited.
- Acceptance:
  - POST /execution/propose returns intent_id, allow, reason.
  - GET /execution/intents returns latest intents including the new record.

### FR-007 Frontend MVP Screens
- Web app must load and auto-fetch overview data.
- Each screen must call its corresponding backend route without crashing.
- Acceptance:
  - Overview shows health, presets, recent intents.
  - Scanner shows latest universe and supports run scan action.
  - Recommend supports run recommend and displays rows.
  - Options supports load ideas and displays rows when available.
  - Backtest shows summary when requested.
  - Execution creates and refreshes intents.

### FR-008 Robinhood MCP Status Visibility
- The product must show clear MCP readiness state for connection/auth/account availability.
- Acceptance:
  - GET /broker/mcp/status returns structured status including connected and authenticated flags.
  - Overview screen displays MCP connected/authenticated/account state and last MCP error when present.

### FR-009 Offline Data Assistant and Tool-Call Suggestions
- The product must provide an assistant endpoint that answers from local persisted data and suggests callable API actions.
- Acceptance:
  - POST /assistant/query returns answer, summary, and suggested_tool_calls.
  - Assistant tab allows prompt input and displays returned answer and suggested tool calls.

## Non-Functional Requirements

### NFR-001 Availability (Development Baseline)
- Local API should start consistently on configured port.
- Endpoint error rate for read routes should remain below 1% in local smoke tests.

### NFR-002 Performance (Development Baseline)
- Median API response under 1.5 seconds for lightweight routes (/health, /scanner/presets).
- Median API response under 8 seconds for heavy routes (/pipeline/recommend, /backtest/short-term).

### NFR-003 Observability
- All write actions must be logged in persistence tables.
- API should emit actionable errors for data-source failures.

### NFR-004 Security and Auth
- Write endpoints must support token auth mode.
- Firebase auth integration remains scaffolded for future hardening.

### NFR-005 Data Integrity
- Recommendation and backtest logic requiring long indicators must fetch sufficient lookback history.
- Data-source alias and schema normalization is mandatory before indicators computation.

## Engineering Architecture Requirements

### Backend Requirements
- Python FastAPI service with modular packages under quantflow.
- Broker abstraction with selectable modes:
  - robinhood_mcp
- Execution policy and audit modules must be decoupled from strategy logic.

### Frontend Requirements
- Expo React Native codebase targeting web-first MVP.
- Environment-configured API base URL.
- Auto-load overview on app mount.

### Persistence Requirements
- SQLite schema with tables for:
  - finviz snapshots
  - recommendations
  - execution intents/events/policy decisions
- Schema creation must be idempotent.

## Data and Integration Requirements
- yfinance ingestion must use valid period strings and sufficient history for SMA200-class features.
- Finviz preset loader must tolerate practical YAML structures used in this repo.
- Options chain filtering must return deterministic empty results when market liquidity constraints fail.

## Test and Validation Strategy

### Level 1: API Contract Smoke Tests
- Validate 200 responses and JSON shape for core routes.
- Validate write-route behavior with and without auth (when enabled).

### Level 2: Pipeline Behavior Tests
- Run scan then verify latest universe rows.
- Run recommend then verify latest recommendations rows.
- Propose order then verify intents growth.

### Level 3: UI Behavior Tests
- Browser test for each screen action and expected state transitions.
- Confirm no silent failures on auto-load.

### Level 4: Regression Checks
- Ensure changes do not break:
  - /health
  - /scanner/presets
  - /execution/intents
  - web app Overview rendering

## Requirements Traceability Matrix

| Requirement | API/UI Surface | Validation Method | Owner | Status |
|---|---|---|---|---|
| FR-001 | /, /health | HTTP smoke test | Backend | Complete |
| FR-002 | /scanner/presets, /pipeline/scan, /scanner/latest-universe | API + UI smoke | Data/Backend | Complete (MVP) |
| FR-003 | /pipeline/recommend, /recommend/latest | API + UI smoke | Backend | Complete (MVP) |
| FR-004 | /options/ideas | API + UI smoke | Options | In Progress |
| FR-005 | /backtest/short-term | API + UI smoke | Backtest | Complete (MVP) |
| FR-006 | /execution/propose, /execution/intents | API + UI smoke | Execution | Complete (MVP) |
| FR-007 | Expo six-screen workflow | Browser flow test | Frontend | In Progress |
| FR-008 | /broker/mcp/status + Overview MCP card | API + UI smoke | Broker/Frontend | In Progress |
| FR-009 | /assistant/query + Assistant tab | API + UI smoke | API/Frontend | In Progress |

## Milestones and Gates

### Milestone A: Stable MVP Baseline
Gate criteria:
- FR-001 through FR-003 pass
- FR-005 and FR-006 pass
- Overview and Recommend screens visibly functional

### Milestone B: Full Screen Reliability
Gate criteria:
- FR-007 all screens pass with no uncaught errors
- FR-008 and FR-009 pass in browser and API smoke
- Options screen either returns ideas or explains empty outcomes from filters

### Milestone C: Production Hardening Prep
Gate criteria:
- Auth mode enabled for write routes in staging
- Basic metrics, alerting, and structured logs in place
- Deployable environment matrix (dev, stage, prod)

## Progress Checklist

### Backend
- [x] Root API route implemented
- [x] Health route implemented
- [x] Scanner routes implemented
- [x] Recommendation routes implemented
- [x] Execution policy and audit routes implemented
- [x] Backtest route implemented
- [x] MCP status route implemented
- [x] Offline assistant route implemented
- [ ] Options route validated with consistent non-empty reference ticker set

### Frontend
- [x] API base uses 127.0.0.1:8100 default
- [x] Overview auto-load on mount
- [x] Overview data visible in browser
- [x] Overview MCP status card added
- [x] Recommend action visible in browser
- [x] Assistant tab added
- [ ] Options screen behavior clarified for empty result cases
- [ ] Charts screen added (future extension)

### Reliability and Ops
- [x] API available on port 8100 with direct uvicorn launch path
- [ ] Makefile API target stability on this machine
- [ ] Minimal automated smoke test script committed

## Known Risks and Mitigations
- Data source variability (yfinance and options chain availability)
  - Mitigation: validation, retries, deterministic empty-state messaging.
- Environment startup inconsistencies in shell wrappers
  - Mitigation: one canonical startup command and health probe.
- Hidden runtime failures in pipelines
  - Mitigation: fail-fast logging and smoke tests after each patch.

## Definition of Done (MVP)
MVP is done when:
- Core API and six-screen web flow are functional on a fresh local start.
- Recommendation and execution intent flows are demonstrably working.
- Options screen has deterministic behavior and user-facing explanation for empty outputs.
- This document can be checked line-by-line and each requirement has a verifiable test result.

## Progress Log Template
Use this section to record implementation progress against this document.

- Date:
- Change:
- Requirements impacted:
- Evidence (endpoint response, browser check, test output):
- Status delta:

## Immediate Next Engineering Tasks
1. Add automated smoke test script that checks FR-001 through FR-006.
2. Add deterministic empty-state explanation for Options UI when count is 0.
3. Stabilize Makefile API target for this Linux/WSL setup.
4. Add charts screen requirements and implementation story as Phase 1.1 extension.
