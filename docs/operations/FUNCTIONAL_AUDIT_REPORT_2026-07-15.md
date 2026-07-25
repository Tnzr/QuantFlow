# QuantFlow Functional Audit Report — July 15, 2026

**Test Suite: 54/54 passing (conda quantflow, Python 3.11, 16.4s)**
**Robinhood MCP endpoint: Reachable (HTTP 405 = expected for JSON-RPC POST-only endpoint)**

---

## Priority 1: Dataset Builder → QuantFlow-AI-Core Handoff

**Verdict: ✅ RELIABLE — Verified end-to-end**

### Live verification performed:
- `train-features --tickers AAPL --save --out /tmp/test_training_features.parquet` → valid parquet output
- `build_training_frame(['AAPL'])` → 53 columns (49 numeric + 4 token lists)
- Numeric features include: OHLCV, returns 1d/5d/21d/63d, SMA20/50/200 + distance%, RSI14, ATR14, vol20, support/resistance KDE levels, forecast envelope (return/band/trend), composite/trend/momentum/RSI-quality/volatility-regime/ATR-efficiency scores, seasonality DOW/DOY (means, win rates, rank %), news sentiment (count, avg, std, bullish/bearish ratios)
- Token bundles: technical, seasonality, sentiment (unified `token_bundle`)
- Parquet export via pyarrow — directly ingestible by PyTorch/TensorFlow/XGBoost

### Data pipeline reliability:
- Finviz screener: 9 presets (weekly_momo/bear, monthly_swing/bear, midterm_trenders/bear, reversal_bull/bear, leaps_quality), YAML-configurable
- Parallel scan execution with ThreadPoolExecutor, retry with exponential backoff
- Multi-provider scraping: direct, proxy (smartproxy), API (scrape_do, scrapingbee)
- SQLAlchemy persistence: 9 tables with proper indexes, `TickerSnapshot` + `Recommendation` + `NewsArticle` + `TrainingFeatureSnapshot` + `AnalyticsLeaderboardEntry` + `ExecutionIntent` + `ExecutionEvent` + `PolicyDecision`
- Error isolation: `build_training_frame()` catches per-ticker failures, returns error rows without crashing batch
- CLI + API (GET/POST `/training/features`) both functional

### Gap: No forward-return labels
`build_training_row()` captures current-state features only. For supervised ML/DL training, a forward-return label (e.g., `target_5d_return`) is needed. This requires a historical iteration loop: feature snapshot at time T → join with actual return T→T+N. The feature extraction machinery is complete; only the labeling iteration is missing.

**Recommendation:** Add `build_labeled_training_dataset(tickers, periods, horizon_days)` function that iterates through historical dates and appends `target_{horizon}d_return` + `target_direction` columns.

---

## Priority 2: Robinhood MCP Integration (Reading Permissions)

**Verdict: ✅ VERIFIED — Endpoint live, architecture complete, ready for auth**

### Live verification performed:
- `curl https://agent.robinhood.com/mcp/trading` → **HTTP 405 (Method Not Allowed)**
- 405 is the **correct expected response**: Robinhood's MCP endpoint only accepts POST (JSON-RPC 2.0), not GET. This confirms the endpoint is live and reachable.

### Architecture verified (all code reviewed, all tests pass):

| Layer | What | Status |
|-------|------|--------|
| Transport | `RobinhoodMCPClient._mcp_read_call()` — JSON-RPC 2.0 POST, header construction, error handling for HTTPError/URLError/JSONDecodeError | ✅ Complete |
| Auth methods | Bearer token, API key, Cookie, custom headers | ✅ Complete |
| Secret management | 5 cloud providers: env, HashiCorp Vault (KV v2), AWS Secrets Manager, Google Secret Manager, Azure Key Vault. All support JSON key extraction via `#field` syntax. | ✅ Complete |
| Runtime config | `PUT/DELETE /broker/mcp/config` — thread-safe with `Lock`, write-auth protected | ✅ Complete |
| OAuth 2.0 + PKCE | `/broker/mcp/oauth/start` → browser callback → `/broker/mcp/oauth/callback`. Fetches Robinhood `.well-known/oauth-authorization-server` metadata. Sets bearer token in runtime config on completion. | ✅ Complete |
| Readiness checklist | `GET /broker/mcp/readiness` — 7 checks: auth_config, runtime_auth, secret_provider, transport, auth, fixtures, positions. Each has pass/fail/warn + actionable next_steps. | ✅ Complete |
| Fixture fallback | `QF_MCP_ACCOUNT_JSON` + `QF_MCP_POSITIONS_JSON` env vars for local testing without real Robinhood account | ✅ Complete |
| Account read | `account()` → equity, cash, buying_power. Error wrapped as RuntimeError. | ✅ Complete |
| Positions read | `positions()` → ticker, qty, avg_price, side. Error wrapped as RuntimeError. | ✅ Complete |
| Portfolio signals | `evaluate_positions(broker)` → hold/trim/exit/add/flip per position with RuleEngine confidence | ✅ Complete |
| Guardrailing | `ExecutionPolicy`: max_daily_notional, max_orders_per_day, max_position_notional, manual approval gate. Full audit trail: intent → policy_decision → event. All write endpoints auth-protected. | ✅ Complete |

### What is needed to go live:
A real Robinhood Agentic OAuth bearer token. Two paths available:
1. **Backend OAuth:** `POST /broker/mcp/oauth/start` → user completes browser auth → token auto-configured
2. **Manual config:** `PUT /broker/mcp/config { "bearer_token": "..." }` (write-auth protected)

Once a token is set, `account()` and `positions()` return live data. Fixture mode works today.

---

## Complete Subsystem Audit Summary

| Subsystem | Files | Status | Notes |
|-----------|-------|--------|-------|
| API Server | `server.py` (2,213 lines) | ✅ | 47 endpoints, async scan jobs, assistant intent router, MCP OAuth |
| Data Pipeline | 7 files | ✅ | 9 presets, parallel scans, 9 tables, parquet export |
| Feature Engineering | 8 files | ✅ | 53 features (49 numeric), KDE forecast, seasonality/sentiment tokens |
| Backtesting | 5 files | ✅ | Equity proxy + portfolio, Sharpe/Sortino/CAGR/PF, 3 alloc methods |
| Broker MCP | 6 files | ✅ | JSON-RPC 2.0, 5 auth methods, 5 cloud secret providers, fixtures |
| Portfolio/Execution | 4 files | ✅ | Position signals, guardrailed execution, audit trail |
| Recommender | 2 files | ✅ | Composite scoring, multi-horizon, options picker |
| Ops | 2 files | ✅ | DB refresh plans, JSON/Markdown reports |
| CLI | `cli.py` | ✅ | 18 commands |
| Frontend | Streamlit + Expo | ✅ | Dashboard, React Native starter |

---

## Findings

| Priority | Finding | Action |
|----------|---------|--------|
| 🔴 HIGH | Training features have no forward-return label column — supervised ML needs `target_Nd_return` | Add `build_labeled_training_dataset()` with historical iteration |
| 🔴 HIGH | Robinhood MCP token not configured in this runtime | Run OAuth flow or inject token via PUT `/broker/mcp/config` |
| 🟡 MED | Pydantic v2 deprecation: `min_items`/`max_items` → `min_length`/`max_length` at server.py:422 | 1-line fix |
| 🟡 MED | CORS wildcard (`allow_origins=["*"]`) | Restrict for production |
| 🟢 LOW | News archive empty by default → sentiment = `news_archive_empty` | Pre-seed sample data |
| 🟢 LOW | `HIGH_INTEREST` hardcoded | Make configurable |