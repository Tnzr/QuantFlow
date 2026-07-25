# QuantFlow Frontend — UX Review & Implementation Gap Analysis

Date: 2026-07-24
Reviewer: User
Status: All components functional but incomplete — significant UX gaps across all tabs

---

## Tab-by-Tab Audit

### 1. Portfolio Tab — Status: PARTIAL

| Feature | Current | Required |
|---------|---------|----------|
| Equity/Cash/Signals metrics | ✅ Shows live data | Keep as-is |
| AI Signals table | ✅ 4 signals with direction/return/confidence | Add per-ticker forecast tiles with sparkline price charts |
| Positions table | ❌ Empty (demo) | Show simulated or live positions with P&L color coding |
| Equity curve chart | ❌ Missing (canvas exists but empty) | Animated equity curve with gradient fill, auto-updating |
| Per-ticker forecast visualization | ❌ Missing | Each ticker tile: price sparkline + 21-step forecast overlay + confidence band |
| Sigma/confidence interpretation | ⚠️ Shows 42% for all tickers | Should vary per ticker (model currently outputs similar sigma) |

### 2. Trade Tab — Status: BASIC

| Feature | Current | Required |
|---------|---------|----------|
| Paper trade form | ✅ Working | Add stop-loss / take-profit fields, pre-trade risk check visualization |
| Order history | ✅ Table structure exists | Populate from Alpaca API or local state |
| Position tracking | ❌ Missing | Show active positions with real-time P&L updates |
| Order confirmation animation | ❌ Missing | Slide-to-confirm or animated checkmark pattern |

### 3. Backtest Tab — CRITICAL GAPS

| Feature | Current | Required |
|---------|---------|----------|
| Ticker selection | ✅ Checkboxes | Keep as-is |
| Capital / Stop / Take inputs | ✅ Present | Add max positions, entry threshold, trailing stop |
| Time period selector | ❌ Missing | Calendar picker for start/end dates |
| Animated progress | ✅ Bar-by-bar equity curve | Keep, improve |
| **Inference forecast at each step** | ❌ **MISSING** | Show model's predicted_return + sigma + direction at each bar |
| **X-axis time labels** | ❌ **MISSING** | Show date/time on chart x-axis |
| **EMA + Bollinger Bands** | ❌ **MISSING** | Overlay EMA(20), Bollinger(20,2) on equity curve |
| **Volume bars** | ❌ **MISSING** | Volume histogram below price chart |
| **RSI indicator** | ❌ **MISSING** | RSI(14) sub-panel below chart |
| **Portfolio evaluation metrics** | ❌ **MISSING** | Sharpe, Sortino, CAGR, Profit Factor, Expectancy (not just Return/Win/DD/Trades) |
| **Per-stock trading activity** | ❌ **MISSING** | For each ticker: trade count, win rate, total P&L, avg hold time |
| **Forecast at each trade point** | ❌ **MISSING** | Hover/tap on a trade marker → show what the model predicted at that point |
| **Rewind / step-through** | ❌ **MISSING** | Slider or ← → buttons to step through each bar |
| **Parallel Martingale DCA** | ❌ **MISSING** | Visualize how position sizing scales with consecutive losses across tickers |
| **Portfolio balancing visualization** | ❌ **MISSING** | Show how capital rotates between tickers based on confidence ranking |
| **Backtest results report** | ❌ Minimal (4 metrics) | Full 6-panel report: equity+DD, rolling Sharpe, trade P&L distribution, sigma-bucketed performance, exit reason analysis, per-ticker P&L |

### 4. Market Tab — Status: MINIMAL

| Feature | Current | Required |
|---------|---------|----------|
| Signal cards | ✅ 4 cards with direction/confidence | Expand to all watched tickers |
| Price chart | ❌ Missing | Candlestick chart with timeframe selector (1m/5m/15m/1H/1D) |
| AI forecast overlay | ❌ Missing | CascadeANP 21-step forecast path as dashed line on chart |
| RSI / Volume indicators | ❌ Missing | Sub-panels below price chart |
| Market depth | ❌ Missing | Order book visualization (bids/asks) |
| News / sentiment | ❌ Missing | News timeline aligned to price |

### 5. Settings Tab — Status: NON-FUNCTIONAL

| Feature | Current | Required |
|---------|---------|----------|
| Backend URL | ✅ Configurable | Keep |
| Alpaca API key management | ❌ Missing | Key input + test connection button + status indicator |
| Stock data API | ❌ Missing | yfinance / Alpaca / Polygon toggle with API key fields |
| LLM API | ❌ Missing | OpenAI / Anthropic / local LLM endpoint configuration |
| Visual preferences | ❌ Missing | Theme toggle (dark only for now), chart color scheme |
| Risk parameters | ❌ Missing | Default stop-loss %, max position %, daily loss limit |
| Model checkpoint selector | ❌ Missing | Dropdown from model_registry.json |

### 6. AI Assistant Chatbot — COMPLETELY MISSING

| Feature | Required |
|---------|----------|
| Right panel placement | Collapsible sidebar on the right of the screen (not a tab) |
| State awareness | Bot must know: current portfolio state, active signals, backtest results, market data |
| Context injection | On each message, inject: `{portfolio: {...}, signals: [...], activeTab: "backtest", backtestState: {...}}` |
| Streaming responses | SSE or WebSocket for token-by-token streaming |
| Model backend | Calls OpenAI-compatible API endpoint (configurable in Settings) |
| Function calling | Bot can trigger: `run_backtest()`, `place_order()`, `refresh_signals()`, `get_market_data()` |
| Conversation memory | Last 20 messages retained in localStorage |
| Suggested prompts | "What's my best performer?", "Run a backtest on AAPL/MSFT", "Explain this signal" |

---

## Implementation Priority

**P0 (next session):**
1. Backtest tab — add inference forecasts, time labels, rewind, Martingale DCA visual
2. Portfolio tab — add equity chart, per-ticker forecast tiles
3. AI Chatbot — right panel with state awareness

**P1:**
4. Market tab — price charts with indicators and AI overlay
5. Settings — Alpaca + LLM API configuration

**P2:**
6. Interactive order confirmation
7. News/sentiment integration
8. Full backtest report with 6 panels
