# Robinhood MCP Integration Blueprint for QuantFlow

## Status Summary
Robinhood now provides an MCP endpoint for agentic trading:
- https://agent.robinhood.com/mcp/trading

From Robinhood support guidance:
- The AI platform connects to Robinhood Trading MCP and authenticates
- During authentication, users are prompted to open a dedicated Agentic account
- The dedicated Agentic account is where agent-placed trades occur
- Desktop is required for opening/authenticating the Agentic account

## Why This Matters for QuantFlow
QuantFlow standardizes on Robinhood MCP and removes direct username/password/MFA brokerage handling from the app runtime.

MCP-first migration benefits:
- Officially supported connectivity path
- Better alignment with modern agent workflows
- Cleaner separation between user approval policy and automated execution actions

## What Robinhood MCP Enables (Planning Scope)
From the support article, expected high-level capabilities include:
- Read access to account, balances, positions, and order history
- Agent-driven trading actions in the Agentic account
- Review/approval workflows depending on AI platform configuration

Important constraints:
- User remains responsible for executed trades
- Third-party agent behavior risk must be treated as operational risk

## QuantFlow MCP Migration Plan

### Step 1: Introduce Broker Mode Abstraction
Add a broker mode setting:
- robinhood_mcp

Behavior:
- Enforce MCP-only mode for broker reads/writes

### Step 2: Add MCP Execution Adapter Layer
Create a new adapter module that is independent from strategy logic:
- Build request/response contracts for account state, positions, and order intents
- Keep adapter stateless where possible
- Log every outbound action and inbound response for auditability

### Step 3: Human Approval Policy Engine
Before full automation, enforce approval policies in QuantFlow:
- Require explicit user approval for order placement and cancellation initially
- Allow read-only automation (portfolio analysis, signal generation) without order execution
- Add policy switches per horizon and asset type

### Step 4: Agentic Account Safety Controls
Implement hard controls at QuantFlow layer:
- Max daily notional
- Max orders per day
- Max per-position risk
- Halt on connectivity anomalies or stale data

### Step 5: Reconcile State and Audit
Add periodic reconciliation job:
- Compare QuantFlow local intent log vs broker order ledger
- Alert on mismatch and block further automation until resolved

## Proposed Module Additions
Suggested file additions (implementation phase):
- quantflow/broker/mcp_client.py
- quantflow/broker/robinhood_mcp.py
- quantflow/execution/policy.py
- quantflow/execution/audit.py
- quantflow/execution/reconcile.py

Suggested schema additions:
- execution_intents
- execution_events
- broker_reconciliations
- policy_decisions

## MCP-First Workflow in QuantFlow
1. Run scan and signal generation
2. Build action proposals (no execution yet)
3. Evaluate proposals against risk and policy engine
4. Present approval queue in UI
5. Send approved actions through MCP broker adapter
6. Persist responses and reconcile against account state

## Rollout Stages

### Stage A: Read-Only MCP
- Account and position retrieval only
- No order placement
- Validate data consistency with existing portfolio evaluator

### Stage B: Simulated Execution
- Produce order intents and paper execution logs only
- No live orders
- Validate throughput, reliability, and policy coverage

### Stage C: Guarded Live Execution
- Small notional limits
- Explicit approval required
- Strict stop and kill-switch controls

### Stage D: Conditional Automation
- Allow limited autonomous actions for pre-approved scenarios
- Keep hard risk and anomaly cutoffs mandatory

## Integration with LLM Control Layer
Use LLM as controller and explainer, not unrestricted executor:
- LLM proposes policy adjustments and action rationales
- Risk engine decides feasibility
- Broker adapter executes only policy-compliant intents

This keeps explainability and control while leveraging fast agentic UX.

## Operational Notes
- The onboarding/authentication flow is platform-dependent (Codex, Cursor, ChatGPT, Claude, etc.)
- Keep QuantFlow internal logs independent of any single AI platform to avoid lock-in
- Build for recoverability first: reconnection, retries, idempotency, and replay-safe actions

## Immediate Next Actions
1. Add MCP migration section in README and TODO
2. Finalize MCP-only broker mode and remove legacy API hooks
3. Add execution intent logging tables
4. Expose approval queue in Streamlit before enabling live execution