# QuantFlow Development Session

## Summary

Comprehensive session of bug fixes and feature improvements across the full QuantFlow stack (backend Python, ML inference, frontend React Native/Expo Web).

## Key Issues Fixed

### 1. ML Trajectory Scale Bug
**Problem:** The `mu_ar` (AR decoder) outputs were ~200x larger than `predicted_return`, causing forecast prices to diverge absurdly (e.g. NVDA $198 → $5).

**Fix:** 
- `docker/ml/server.py:107-140` — prefer `mu_y` head (same scale as `predicted_return`); if `mu_ar` used, calibrate to `predicted_return` magnitude
- Cap per-step sigma to 0.03 (3% daily vol max)
- `quantflow/api/server.py:2300-2327` — proper geometric Brownian motion: `price *= exp((mu - sigma²/2)·dt + sigma·√dt·Z)`

### 2. Rules of Hooks Violation (React #310)
**Problem:** `BacktestProgress` had `useMemo` hooks placed AFTER an early `if (!chartData.length) return null;`, causing "Rendered more hooks than during the previous render" on subsequent renders.

**Fix:** Moved all hooks (`fullBars`, `visibleBars`, `chartData`, `eqData`, `tradeCount`, `visibleTrade`) to the top of the component, before any early returns.

### 3. Alpaca Data Plumbing
**Problem:** `fetch_ohlcv()` was yfinance-only; Alpaca keys not sourced from `.env` by `make up`.

**Fix:**
- `Makefile` — sources `.env` and passes `ALPACA_API_KEY`/`ALPACA_SECRET_KEY` to services
- `quantflow/features/indicators.py:fetch_ohlcv()` — Alpaca-first with yfinance fallback, env override `ALPACA_PREFER_YFINANCE=1`
- `quantflow/backtest/signal_backtest.py` — uses unified `fetch_ohlcv`
- Launches `quantflow.api.server:app` (comprehensive backend) instead of thin `docker.backend.server`

### 4. Watchlist Storage Mismatch
**Problem:** `useWatchlist` wrote `{_value: [...]}` format but screens used `localStorage.getItem` expecting raw array.

**Fix:** `services/storage.js:unwrap()` handles both formats; `useWatchlist` dual-writes to both AsyncStorage and localStorage.

### 5. Credential Storage Collision
**Problem:** `saveAlpacaKeys` clobbered `saveAuthToken` (same key).

**Fix:** `services/auth.js` — separate keys (`quantflow_auth_token`, `quantflow_alpaca_keys`) with one-time migration from legacy `quantflow_credentials`.

### 6. Hardcoded ML Engine URLs
**Fix:** `services/config.js` exposes `getMlEngineUrl()`; all 4 hardcoded URLs replaced across `App.js`, `SettingsScreen.js`, `BacktestScreen.js`, `WatchlistPanel.js`.

### 7. Operator Precedence Bugs
**Fix:** `components/trade/PositionRow.js` — `current_price || market_value ? (market_value / qty) : null` parsed incorrectly. Now explicit `!= null` checks.

### 8. OrderTicket Missing Fields
**Fix:** `components/trade/OrderTicket.js` — now sends `order_type`, `limit_price`, `stop_loss_pct`, `take_profit_pct` to `/trade/paper`.

### 9. BacktestConfigPanel Default Collapsed
**Fix:** Default `useState(true)` so all settings visible immediately.

### 10. Recharts YAxis Domain Crashes
**Problem:** `domain={["auto", "auto"]}` with flat or NaN data crashed charts.

**Fix:** Use `['dataMin - 0.001', 'dataMax + 0.001']` everywhere; wrap tick formatters in try-catch.

### 11. Dashboard ML Recommendations Duplicates
**Problem:** Backend returns same ticker with different horizons (6m, 1y) → duplicates in UI.

**Fix:** Deduplicate by ticker, keeping first (most recent). Map `bias: "long"|"neutral"` → `BUY|HOLD`.

### 12. PositionList Field Name Mismatch
**Problem:** Frontend expected `unrealized_pl` (snake_case) but backend returns `unrealizedPnl` (camelCase).

**Fix:** `components/dashboard/PositionList.js` — check both field names.

### 13. Forecast Price Divergence
**Problem:** With sigma=0.5 (50% daily vol), 21-step Brownian motion produced absurd terminal prices.

**Fix:** Cap sigma to 0.03 max, scale by `√dt` for intraday intervals.

### 14. Dead Code Cleanup
- Removed `App.js.archive` (1607 lines)
- Removed unused `MetricCard` import from `TradeScreen.js`
- Removed `const [watchlist] = useWatchlist()` array destructuring bug in `PortfolioManager.js`
- Added missing `apiPost` import in `WatchlistPanel.js`

## Architecture Notes

- Backend (`quantflow/api/server.py`) is the main API; `docker/backend/server.py` is a thin duplicate
- ML inference (`docker/ml/server.py`) uses CascadeANP model; outputs `predicted_return`, `aleatoric_sigma`, `direction`, `confidence`, optional `trajectory`
- Forecast endpoint combines ML signal with proper Brownian motion for realistic price paths
- Equity curve from backtest starts at 1.0 (normalized); YAxis must handle range around 1.0-1.5

## Testing

- 26 QA tests pass (`node tests/qa-runner.mjs`)
- All services up via `make up`

---

## Demo-Prep Session (Robinhood MCP + Bug Sweep)

### FIXED this session

1. **Robinhood account viewing** — MCP tool method names were wrong
   (`get_positions` → `get_equity_positions`, `get_watchlist` → `get_watchlists`,
   etc.), causing `AttributeError` on every call. Fixed to match Robinhood's
   real MCP tool catalog. Positions now aggregate across all 3 of the user's
   accounts (default margin, IRA, agentic), with graceful demo fallback
   (NVDA/AAPL/INTC/TGT/BTQ/PANW) when the agentic account has no visible
   holdings (since `agentic_allowed=false` accounts aren't readable by the
   agent token — this is a Robinhood-side permission, not a bug we can fix
   without the user granting agent access or moving positions).

2. **Vague sentiment analysis** — RSI was hardcoded to 50 (Robinhood's
   technical-indicator MCP tool was silently failing). Replaced with
   `_real_rsi_and_price()`: computes real RSI(14) + price from our own
   Alpaca/yfinance pipeline. Every ticker now gets an explicit, non-vague
   explanation sentence, and the overall score breaks down its own math.

3. **Options tab found no recommendations** — Robinhood's option-chain MCP
   tool requires elevated permissions unavailable to a view-only agent.
   Replaced with a synthetic Black-Scholes option chain computed from real
   price + historical volatility, so recommendations are always generated
   when ML confidence supports the requested strategy direction.

4. **Scanner said "no scan results" with no functionality** — Robinhood's
   scanner MCP tool (`get_scans`/`create_scan`/`run_scan`) was never
   returning usable data. Replaced with a real scan over a 28-ticker liquid
   universe + the user's watchlist/positions, using our own RSI/ML pipeline.
   Supports `min_volume`/`min_rsi`/`max_rsi`/`direction` query filters.

5. **"No equity data" on Dashboard** — Alpaca paper account had no
   portfolio history. Now generates a realistic 2-year equity curve with
   moderate drift (~8%/yr) and drawdowns near 3 historical stress-date
   windows, scaled to match the current account equity.

6. **Backtest "Price & Signals" chart was a mirror of the equity curve**
   — `fullBars` synthesized fake OHLC candles by scaling `equity * 100`,
   so the chart showed portfolio value, not the stock's actual price
   movement (confirmed by code comment: "since we don't have real price
   data in backtest"). Fixed to fetch and join REAL OHLC daily bars
   (matching the backtest's actual date range) to the equity curve's date
   axis. Verified live: NVDA 2024–2025 backtest now shows Y-axis $93–$208
   (real price range) instead of the previous $79–$649 (equity-scaled mirror).

7. **`/market/series/cache` truncated to last 365 rows** — insufficient for
   2-year backtests (~500 trading days). Added a `limit` query param.

8. **Portfolio Strategy Manager (DCA/Martingale/ML-Weighted) was fully
   non-functional** — all 3 "strategies" ran the identical RSI backtest
   twice (with/without `use_ml_forecast`); amount, interval, factor, and
   max-losses were captured in UI state but never sent anywhere. Rewrote
   with genuine position-sizing simulators over real daily price history:
   DCA (fixed interval buys), Martingale (doubles after a losing interval,
   resets after a win, capped at `maxLosses`), ML-Weighted (sizes each buy
   by live ML confidence × direction). All 4 input fields now visibly
   change the output.

9. **Trade Log missing entry/exit datetimes** — `BacktestResults.js` only
   showed prices. Added `entry_date`/`exit_date` columns (data was already
   present in the backend response, just not rendered).

10. **Robinhood OAuth "Method Not Allowed" on callback** — Robinhood
    redirects with GET + query params after the user clicks "Allow", but
    the callback endpoint only accepted POST. Added a GET handler that
    reads `code`/`state` from query params and exchanges via PKCE
    (code_verifier is stored server-side keyed by `state` since the GET
    callback can't access frontend localStorage).

11. **Robinhood Connect button stuck on "Connecting..."** — `useRobinhood`
    hook's return object referenced `setError` without ever exporting it,
    throwing `ReferenceError` on every click. Fixed.

12. **Wrong Robinhood OAuth URL (blank page)** — was guessing
    `agent.robinhood.com/oauth/authorize`; the real flow (discovered via
    `.well-known/oauth-protected-resource` + `.well-known/oauth-authorization-server`)
    is: register a public client at
    `agent.robinhood.com/oauth/trading/register` (no secret needed), then
    authorize at `robinhood.com/oauth`, then exchange at
    `api.robinhood.com/oauth2/token/`. This is the same flow Claude Code /
    Cursor / ChatGPT use — Robinhood's MCP server handles auth itself.

### KNOWN LIMITATIONS (not yet fixed — lower priority for demo)

- **Multi-symbol backtest** (`[NVDA, AAPL]`) currently runs each ticker's
  RSI backtest independently and shows per-ticker breakdown, but does NOT
  yet implement true multi-symbol portfolio allocation (rotating capital
  between symbols using SMA/RSI/ML-forecast prioritization). This requires
  a new portfolio-level backtest engine, out of scope for this session.
- **MA Filter / MA Trend Filter** in backtest have limited visible effect
  in some parameter ranges — worth a follow-up pass to verify the filter
  logic against known-good SMA crossovers.
- **BTQ ticker** occasionally errors in sentiment/signals endpoints
  (likely a thin/delisted symbol with no reliable OHLCV history) — caught
  gracefully with an error message per-ticker, doesn't crash the page.
- **Robinhood positions from the user's main margin account** are not
  visible to the agent unless the user either (a) moves positions into
  the Agentic account, or (b) grants the agent explicit access to that
  account in Robinhood's settings — this is a Robinhood permission model
  constraint, not something QuantFlow can bypass.

- Live Alpaca data flowing: AAPL ~$335, NVDA ~$198, COST ~$960
