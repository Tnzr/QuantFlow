# Investor Service Implementation Blueprint

Date: 2026-06-18
Project: QuantFlow
Scope: Blueprint for turning the current MVP into a paid, reliable investor subscription service

## 1. Blueprint Objectives

Build a production service that supports:
- Subscription billing and entitlement control
- Reliable signal delivery with observable quality
- Compliance and audit readiness
- Scalable operations and incident response

Primary business objective:
- Launch paid subscriptions for swing and breakout expectation signals with controlled risk and transparent reporting.

## 2. Target Architecture

## 2.1 Logical Components
- Frontend (Expo web/app): account onboarding, plans, subscription status, signal dashboard
- API service (FastAPI): auth, billing webhooks, entitlements, signal APIs, audit endpoints
- Strategy services: signal generation, scoring, and lifecycle state updates
- Data layer: market data, news/sentiment, backtest outputs, signal history
- Billing provider: Stripe Billing (+ Google Pay through Stripe), optional PayPal adapter
- Operations stack: metrics, logs, alerts, incident runbooks

## 2.2 Service Boundaries
- Auth and identity boundary: user identity and roles
- Commercial boundary: plans, subscriptions, invoices, entitlements
- Quant boundary: model and strategy execution
- Compliance boundary: audit logs, disclosures, report snapshots

## 3. Data Model (Minimum Required)

## 3.1 Commerce Entities
- plans
  - id, code, name, price_cents, currency, interval, active
- subscriptions
  - id, user_id, provider, provider_customer_id, provider_subscription_id
  - plan_id, status, trial_end_at, current_period_end, canceled_at
- invoices
  - id, subscription_id, provider_invoice_id, amount_due_cents, amount_paid_cents
  - status, issued_at, paid_at
- entitlement_events
  - id, user_id, source_event, old_state, new_state, reason, created_at

## 3.2 Product Access Entities
- feature_flags
  - id, code, description
- plan_entitlements
  - id, plan_id, feature_code, limit_value
- user_entitlements_cache
  - user_id, feature_code, effective_value, refreshed_at

## 3.3 Quant and Audit Entities
- strategies
  - id, code, version, status, config_json, created_at
- signals
  - id, symbol, strategy_id, strategy_version, timestamp
  - direction, confidence, rationale_json, invalidation_rule_json
  - state (new, triggered, invalidated, closed)
- signal_events
  - id, signal_id, event_type, payload_json, created_at
- model_artifacts
  - id, strategy_id, data_snapshot_ref, metric_bundle_json, created_at
- audit_log
  - id, actor_type, actor_id, action, resource, metadata_json, created_at

## 4. Entitlement State Machine

States:
- none
- trial
- active
- grace
- suspended
- canceled

Transitions:
- none -> trial on successful trial signup
- trial -> active on first successful charge
- active -> grace on payment failure
- grace -> active on payment recovery
- grace -> suspended after grace timeout
- active/suspended -> canceled on user cancellation or permanent provider cancellation

Rules:
- Entitlements are derived from state + plan mapping.
- API enforcement is backend-first; frontend gating is only UX.
- All transitions are event-driven from billing webhooks and stored as immutable events.

## 5. Payments Integration Blueprint

## 5.1 Stripe First Implementation
1. Create products and prices in Stripe.
2. Add checkout session endpoint.
3. Add customer portal endpoint for self-service upgrades/cancel.
4. Implement webhook endpoint:
   - checkout.session.completed
   - invoice.paid
   - invoice.payment_failed
   - customer.subscription.updated
   - customer.subscription.deleted
5. Process webhooks with idempotency table keyed by event_id.
6. Update subscription table, emit entitlement event, refresh cache.

## 5.2 Google Pay Path
- Enable Google Pay in Stripe dashboard.
- Frontend uses Stripe Checkout or Payment Element where Google Pay appears automatically for eligible users.

## 5.3 PayPal Adapter (Phase 2)
- Add provider abstraction layer:
  - BillingProvider interface (create_checkout, parse_webhook, normalize_status)
- Keep one internal subscription status model regardless of processor.

## 6. API Blueprint (Proposed Endpoints)

Auth and profile:
- POST /auth/register
- POST /auth/login
- GET /me

Billing:
- GET /billing/plans
- POST /billing/checkout-session
- POST /billing/portal-session
- POST /billing/webhooks/stripe
- GET /billing/subscription

Entitlements:
- GET /entitlements
- GET /entitlements/features

Signals and backtests:
- GET /signals/live
- GET /signals/history
- GET /signals/{signal_id}
- GET /backtest/summary
- GET /backtest/context

Audit and reporting:
- GET /reports/performance
- GET /reports/methodology
- GET /reports/incidents

## 7. Strategy Quality and Risk Controls

## 7.1 Deployment Guardrails
- Minimum out-of-sample metric thresholds
- Maximum allowable drawdown threshold
- Regime coverage requirement
- Data quality checks before publish

## 7.2 Runtime Guardrails
- Real-time drift score checks
- Abnormal signal volume detector
- Circuit breaker to disable strategy automatically
- Human approval workflow for re-enable

## 7.3 Reporting Standards
- Report gross and net assumptions separately
- Declare slippage, fees, liquidity assumptions
- Publish strategy version and effective period

## 8. Security and Compliance Workstream

## 8.1 Security Baseline
- JWT/session hardening and refresh-token strategy
- Secret storage policy and key rotation cadence
- Rate limiting and abuse controls
- Least-privilege DB and service accounts

## 8.2 Compliance Baseline
- Terms of service and risk disclosure publication
- Marketing copy review checklist for performance claims
- Retention policy for alerts and advisory-related records
- Jurisdiction-specific legal review for advisory positioning

## 9. Delivery Plan (Engineering)

## Sprint 1 (2 weeks): Commerce Foundations
- Data schema for plans/subscriptions/invoices/entitlements
- Stripe checkout + webhook ingestion
- Subscription and entitlements APIs
- Unit tests for webhook and state transitions

Deliverables:
- End-to-end paid signup in test mode
- Deterministic entitlement update on payment events

## Sprint 2 (2 weeks): Product Gating and UX
- Frontend paywall and feature gating
- Account billing page and plan management
- Entitlement-aware signal/backtest access
- Error handling for grace/suspended states

Deliverables:
- Tiered feature access working in web app
- Billing status visible and accurate to user

## Sprint 3 (2 weeks): Quant Governance
- Strategy versioning and artifact tracking
- Signal event ledger and audit endpoints
- Runtime guardrails and kill-switch controls
- Walk-forward report generation automation

Deliverables:
- Reproducible signal lineage
- Auto-disable path for failing strategies

## Sprint 4 (2 weeks): Operations and Beta Readiness
- Metrics/logs/traces dashboards
- Incident and rollback runbooks
- Compliance page publication and methodology docs
- Invite-only beta launch tooling

Deliverables:
- Operational dashboard and alerting in place
- Controlled paid beta launch readiness

## 10. Testing Matrix

Required test suites:
- Unit tests:
  - Billing webhook parser
  - Entitlement state transitions
  - Plan entitlement resolution
- Integration tests:
  - Checkout to active subscription flow
  - Payment failure to grace to suspension flow
  - Signal API access by plan tier
- Non-functional tests:
  - Load tests on signal endpoints
  - Chaos tests for webhook delays/retries
  - Recovery tests for stale entitlement cache

## 11. Operational KPIs

Commercial:
- Trial-to-paid conversion
- Monthly recurring revenue growth
- Churn and involuntary churn

Reliability:
- API uptime and p95 latency
- Webhook processing delay
- Entitlement mismatch incidence

Quant quality:
- Signal hit rate by regime
- Profit factor trend
- Max drawdown and recovery duration

Trust/compliance:
- Disclosure acceptance rate
- Incident count and mean time to resolve
- Number of corrected or retracted signal events

## 12. Definition of Done for Investor-Service MVP

The MVP is service-ready when:
- Billing is production-capable with automated lifecycle handling.
- Entitlements are enforced at API level and validated in tests.
- Signal lifecycle is auditable and strategy versions are traceable.
- Compliance documents are published and legally reviewed.
- Operational monitoring and incident response are active.
- Paid beta users can onboard, receive value, and renew with low support friction.

## 13. Immediate Next Actions (This Week)

1. Create schema migrations for billing and entitlement tables.
2. Add Stripe integration skeleton and webhook endpoint.
3. Implement entitlement middleware and wire to protected APIs.
4. Add plan-tier gating in frontend for live signals and advanced backtests.
5. Publish methodology and risk-disclosure draft pages.

This blueprint is designed to be implemented incrementally without re-platforming the current stack.
