# QuantFlow Operations Report

## Executive Summary
- Generated at: 2026-07-25T17:48:44.856382+00:00
- DB target: sqlite:///quantflow.db
- Freshness threshold: 7 days
- Overall DB status: REFRESH REQUIRED

## Local Database Freshness
- ticker_snapshots: count=39350, latest=2026-06-17T18:59:50.275785+00:00, age=38.0 days
- recommendations: count=405, latest=2026-06-17T01:52:12.491208+00:00, age=38.7 days
- execution_intents: count=6, latest=2026-06-16T21:01:38.243902+00:00, age=38.9 days

## Risk and Action Plan
- Snapshot refresh needed: yes
- Recommendation refresh needed: yes
- Suggested action: run `python -m quantflow.cli ops-refresh --stale-days 7 --with-report`

## Robinhood Agentic Authentication Notes
- 1. Robinhood Agentic auth is performed in an MCP-capable client session, not in QuantFlow itself.
- 2. Authenticate your Robinhood Agentic account in that client; this usually opens Robinhood's auth flow.
- 3. Return to QuantFlow and re-check /broker/mcp/status until authenticated=true.
- 4. If auth is blocked, set QF_MCP_ACCOUNT_JSON and QF_MCP_POSITIONS_JSON for local fixture tests.

## Runtime API Snapshot
- API base: http://127.0.0.1:3000
- Health: {'status': 'ok', 'auth_enabled': False, 'firebase_auth_enabled': False}
- MCP status: {'broker_mode': 'robinhood_mcp', 'connected': True, 'authenticated': False, 'account_available': False, 'positions_available': False, 'positions_count': 0, 'fixture_account_loaded': False, 'fixture_positions_loaded': False, 'runtime_auth_override': {'configured': False, 'auth_header': 'Authorization', 'has_bearer': False, 'has_api_key': False, 'api_key_header': 'X-API-Key', 'has_cookie': False, 'custom_header_keys': [], 'secret_provider': None, 'secret_ref_keys': []}, 'endpoint': 'https://agent.robinhood.com/mcp/trading', 'auth': {'configured': False, 'auth_header': 'Authorization', 'has_bearer': False, 'has_api_key': False, 'api_key_header': 'X-API-Key', 'has_cookie': False, 'custom_header_keys': [], 'payload_mode': 'jsonrpc', 'secret_provider': None, 'secret_ref_keys': [], 'secret_resolution_error': None, 'runtime_override_active': False}, 'auth_summary': {'configured': False, 'auth_header': 'Authorization', 'has_bearer': False, 'has_api_key': False, 'api_key_header': 'X-API-Key', 'has_cookie': False, 'custom_header_keys': [], 'payload_mode': 'jsonrpc', 'secret_provider': None, 'secret_ref_keys': [], 'secret_resolution_error': None, 'runtime_override_active': False}, 'account_error': 'Robinhood MCP account read failed: MCP HTTP error 401: Unauthorized | authentication required\n', 'positions_error': 'Robinhood MCP positions read failed: MCP HTTP error 401: Unauthorized | authentication required\n'}
- MCP readiness: {'overall_ready': False, 'checks': [{'id': 'auth_config', 'label': 'Backend MCP auth wiring', 'state': 'warn', 'details': 'No MCP auth token/header configured in backend env. Set QF_MCP_BEARER_TOKEN or QF_MCP_HEADERS_JSON.'}, {'id': 'runtime_auth', 'label': 'Runtime secure MCP config', 'state': 'warn', 'details': 'No runtime MCP auth override is configured. Use PUT /broker/mcp/config for non-env secret wiring.'}, {'id': 'secret_provider', 'label': 'Secret manager provider', 'state': 'warn', 'details': 'No secret provider configured (vault/aws/gcp/azure).'}, {'id': 'transport', 'label': 'MCP transport reachable', 'state': 'pass', 'details': 'QuantFlow can reach Robinhood MCP transport endpoint.'}, {'id': 'auth', 'label': 'Agent account authentication', 'state': 'fail', 'details': 'Authentication not complete yet (typically 401 until the agent account auth flow is completed).'}, {'id': 'fixtures', 'label': 'Local fixture fallback', 'state': 'warn', 'details': 'Optional local fixture env vars are not fully set (QF_MCP_ACCOUNT_JSON and QF_MCP_POSITIONS_JSON).'}, {'id': 'positions', 'label': 'Positions visibility', 'state': 'warn', 'details': 'Positions are not readable yet. This is expected when auth is incomplete.'}], 'next_steps': ['Preferred: run backend OAuth via POST /broker/mcp/oauth/start and complete browser auth callback.', 'Configure MCP auth via PUT /broker/mcp/config (preferred for runtime) or env vars as fallback.', 'For production, point secret_refs to Vault/AWS/GCP/Azure and avoid raw token payloads.', 'Authenticate the Robinhood Agentic account in your MCP-capable client session (see /broker/mcp/login auth_instructions).', 'Re-check MCP status from Overview to confirm authenticated=yes.', 'Optional: set QF_MCP_ACCOUNT_JSON and QF_MCP_POSITIONS_JSON for local fixture-based testing.', 'Once authenticated, validate account and positions read before enabling execution pathways.'], 'status': {'broker_mode': 'robinhood_mcp', 'connected': True, 'authenticated': False, 'account_available': False, 'positions_available': False, 'positions_count': 0, 'fixture_account_loaded': False, 'fixture_positions_loaded': False, 'runtime_auth_override': {'configured': False, 'auth_header': 'Authorization', 'has_bearer': False, 'has_api_key': False, 'api_key_header': 'X-API-Key', 'has_cookie': False, 'custom_header_keys': [], 'secret_provider': None, 'secret_ref_keys': []}, 'endpoint': 'https://agent.robinhood.com/mcp/trading', 'auth': {'configured': False, 'auth_header': 'Authorization', 'has_bearer': False, 'has_api_key': False, 'api_key_header': 'X-API-Key', 'has_cookie': False, 'custom_header_keys': [], 'payload_mode': 'jsonrpc', 'secret_provider': None, 'secret_ref_keys': [], 'secret_resolution_error': None, 'runtime_override_active': False}, 'auth_summary': {'configured': False, 'auth_header': 'Authorization', 'has_bearer': False, 'has_api_key': False, 'api_key_header': 'X-API-Key', 'has_cookie': False, 'custom_header_keys': [], 'payload_mode': 'jsonrpc', 'secret_provider': None, 'secret_ref_keys': [], 'secret_resolution_error': None, 'runtime_override_active': False}, 'account_error': 'Robinhood MCP account read failed: MCP HTTP error 401: Unauthorized | authentication required\n', 'positions_error': 'Robinhood MCP positions read failed: MCP HTTP error 401: Unauthorized | authentication required\n'}}
- Scraping status: {'provider': 'smartproxy', 'mode': 'proxy', 'proxy_configured': True, 'proxy_env_source': 'QF_SCRAPING_PROXY_URL', 'api_endpoint': None, 'scraping_api_key_set': True, 'scrapingbee_api_key_set': True, 'notes': []}

## Appendix: Interpretation
- If MCP is connected=true but authenticated=false, transport works but account auth is incomplete.
- Scraping mode should be proxy or api for resilient Finviz pulls in constrained networks.
- For safe local testing without Robinhood auth, use fixture env vars and keep execution pathways blocked.
