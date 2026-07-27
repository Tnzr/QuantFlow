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
- Live Alpaca data flowing: AAPL ~$335, NVDA ~$198, COST ~$960
