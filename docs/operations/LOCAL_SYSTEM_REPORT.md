# QuantFlow Operations Report

## Executive Summary
- Generated at: 2026-06-17T00:52:08.323956+00:00
- DB target: sqlite:///quantflow.db
- Freshness threshold: 7 days
- Overall DB status: FRESH

## Local Database Freshness
- ticker_snapshots: count=33520, latest=2026-06-17T00:23:07.492327+00:00, age=0.0 days
- recommendations: count=390, latest=2026-06-16T23:42:51.248477+00:00, age=0.0 days
- execution_intents: count=6, latest=2026-06-16T21:01:38.243902+00:00, age=0.2 days

## Risk and Action Plan
- Snapshot refresh needed: no
- Recommendation refresh needed: no
- Suggested action: run `python -m quantflow.cli ops-refresh --stale-days 7 --with-report`

## Robinhood Agentic Authentication Notes
- 1. Robinhood Agentic auth is performed in an MCP-capable client session, not in QuantFlow itself.
- 2. Authenticate your Robinhood Agentic account in that client; this usually opens Robinhood's auth flow.
- 3. Return to QuantFlow and re-check /broker/mcp/status until authenticated=true.
- 4. If auth is blocked, set QF_MCP_ACCOUNT_JSON and QF_MCP_POSITIONS_JSON for local fixture tests.

## Runtime API Snapshot
- API base: http://127.0.0.1:8100
- Health: {'status': 'ok', 'auth_enabled': False, 'firebase_auth_enabled': False}
- MCP status: {'broker_mode': 'robinhood_mcp', 'connected': True, 'authenticated': False, 'account_available': False, 'positions_available': False, 'positions_count': 0, 'fixture_account_loaded': False, 'fixture_positions_loaded': False, 'endpoint': 'https://agent.robinhood.com/mcp/trading', 'account_error': 'Robinhood MCP account read failed: MCP HTTP error 401: Unauthorized', 'positions_error': 'Robinhood MCP positions read failed: MCP HTTP error 401: Unauthorized'}
- MCP readiness: {'overall_ready': False, 'checks': [{'id': 'transport', 'label': 'MCP transport reachable', 'state': 'pass', 'details': 'QuantFlow can reach Robinhood MCP transport endpoint.'}, {'id': 'auth', 'label': 'Agent account authentication', 'state': 'fail', 'details': 'Authentication not complete yet (typically 401 until the agent account auth flow is completed).'}, {'id': 'fixtures', 'label': 'Local fixture fallback', 'state': 'warn', 'details': 'Optional local fixture env vars are not fully set (QF_MCP_ACCOUNT_JSON and QF_MCP_POSITIONS_JSON).'}, {'id': 'positions', 'label': 'Positions visibility', 'state': 'warn', 'details': 'Positions are not readable yet. This is expected when auth is incomplete.'}], 'next_steps': ['Authenticate the Robinhood Agentic account in your MCP-capable client session.', 'Re-check MCP status from Overview to confirm authenticated=yes.', 'Optional: set QF_MCP_ACCOUNT_JSON and QF_MCP_POSITIONS_JSON for local fixture-based testing.', 'Once authenticated, validate account and positions read before enabling execution pathways.'], 'status': {'broker_mode': 'robinhood_mcp', 'connected': True, 'authenticated': False, 'account_available': False, 'positions_available': False, 'positions_count': 0, 'fixture_account_loaded': False, 'fixture_positions_loaded': False, 'endpoint': 'https://agent.robinhood.com/mcp/trading', 'account_error': 'Robinhood MCP account read failed: MCP HTTP error 401: Unauthorized', 'positions_error': 'Robinhood MCP positions read failed: MCP HTTP error 401: Unauthorized'}}
- Scraping status: {'provider': 'direct', 'mode': 'direct', 'proxy_configured': False, 'proxy_env_source': None, 'api_endpoint': None, 'scraping_api_key_set': False, 'scrapingbee_api_key_set': False, 'notes': []}

## Appendix: Interpretation
- If MCP is connected=true but authenticated=false, transport works but account auth is incomplete.
- Scraping mode should be proxy or api for resilient Finviz pulls in constrained networks.
- For safe local testing without Robinhood auth, use fixture env vars and keep execution pathways blocked.
