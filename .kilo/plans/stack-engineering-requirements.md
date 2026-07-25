# QuantFlow Stack — Engineering Requirements v1.0

Date: 2026-07-24  
Status: SPECIFICATION — not yet implemented  
Depends on: CascadeANP v1.0 (frozen), Alpaca dataset (built)

---

## 1. Frontend — Expo React Native (iOS / Android / Web)

### 1.1 Navigation & Layout

```
Bottom Tab Bar
├── Dashboard       — portfolio overview, equity chart, AI signals
├── Trade           — order entry, active positions, order history
├── Market          — watchlist, price charts, market depth
├── Backtest        — interactive backtesting with visual progress
└── Settings        — API keys, risk params, notifications
```

**REQ-F1: Portfolio Dashboard**
- Hero card: total equity, daily PnL ($), daily PnL (%).
- Animated equity curve (Reanimated 3): 6-month history with gradient fill, pinch-to-zoom, tap-for-value.
- Active positions list: ticker, shares, avg price, unrealized PnL, unrealized PnL %.
- Color-coded: green for profit, red for loss. **Entry animation: values count up from 0 on mount** (staggered, 50ms per item).
- AI signal feed: scrolling list of `[BUY/SELL/HOLD] ticker @ confidence` cards, auto-refresh every 30s.
- Pull-to-refresh triggers full portfolio reload with haptic feedback.

**REQ-F2: Interactive Backtesting**
- NOT a spinner. Full real-time visual progress.
- **Config panel** (collapsible): model checkpoint selector, ticker multi-select (searchable), date range (calendar picker), capital amount, risk params (stop loss %, take profit %, max positions, entry threshold).
- **Progress visualization**: As the backtest runs, show:
  - Live equity curve updating bar-by-bar (animated line chart).
  - Trade markers appearing on the equity curve as dots (green=win, red=loss).
  - Current position indicator: "AAPL: +$340 (3.4%), holding 12 bars".
  - Bar counter: "Processing bar 1,847 / 4,200".
  - Speed control: 1× / 4× / 16× slider controlling animation speed.
- **Results panel** (appears on completion):
  - 8 KPI cards: Total Return, Sharpe, Max DD, Win Rate, Trades, Profit Factor, Expectancy, CAGR.
  - Sigma-bucket performance table (low-σ vs mid-σ vs high-σ win rate).
  - 6-panel report image (same as offline viz).
  - Trade log table with sortable columns, export to CSV.
  - "Re-run" button with parameter diffs highlighted.
- **Animation**: KPI cards stagger in with spring animation. Equity curve draws with stroke-dasharray animation (1s duration). Trade dots pop in sequentially (50ms each).

**REQ-F3: Trade Entry**
- Order ticket card: ticker selector (searchable dropdown), quantity input (stepper +/-), order type (market/limit), limit price (conditional), stop loss toggle + %, take profit toggle + %.
- Live price feed: current price with sparkline mini-chart (last 20 bars). Price updates every 2 seconds with animated color flash on change.
- Pre-trade check visualization: "Position would be $4,200 (4.2% of portfolio)" with green/yellow/red risk indicator.
- Confirm button: two-stage (tap → slide right to confirm) to prevent fat-finger errors.
- Order confirmation: full-screen success animation (checkmark + haptic). Undo button for 3 seconds.

**REQ-F4: Market View**
- Watchlist: user-configurable list of tickers with last price, % change, mini-sparkline.
- Full price chart: candlestick chart with RSI indicator below, volume bars.
- Timeframe selector: 1m / 5m / 15m / 1H / 1D.
- Tap on candle to show OHLC tooltip.
- Market depth visualization: order book levels as horizontal bars (bids green, asks red).
- AI forecast overlay toggle: show CascadeANP 21-step forecast path as dashed line on the chart.

**REQ-F5: Settings**
- API key management: Alpaca key + secret input (masked), test connection button.
- Risk parameters: max position size %, daily loss limit $, max positions, stop loss defaults.
- Notification preferences: order filled, stop triggered, daily summary.
- Theme: dark mode only (financial terminal aesthetic). Accent color picker (default: quantflow purple #8e44ad).
- Data management: clear cache, rebuild dataset (triggers Alpaca fetch), model download.

### 1.2 UI/UX Standards

- **Color palette**: Background #0a0a0f (near-black), surface #1a1a2e, accent #8e44ad (purple), profit #00ff88 (green), loss #ff4444 (red), neutral #666 (gray).
- **Typography**: JetBrains Mono for numbers/monospace, Inter for UI text.
- **Animations**: All transitions use spring physics (damping=20, stiffness=300). Page transitions slide left. List items stagger in. Numbers use shared element transitions where possible.
- **Responsive**: Tablet (iPad) landscape split-view (chart left, order ticket right). Phone single column with bottom sheet for details.
- **Accessibility**: Minimum touch target 44px. VoiceOver labels on all interactive elements. High contrast mode toggle.
- **Performance**: FlatList with windowing for long scrolls. Memoize chart components. WebSocket connection pooled, not per-screen.

### 1.3 API Contracts (Frontend ↔ Backend)

```typescript
// GET /api/portfolio
type Portfolio = {
  equity: number; cash: number;
  positions: Position[];
  signals: Signal[];
  equityHistory: { date: string; value: number }[];
}

// POST /api/backtest/start
type BacktestStart = {
  jobId: string;
  tickers: string[];
  capital: number;
  stopLoss: number; takeProfit: number;
  maxPositions: number;
}

// WebSocket /ws/backtest/{jobId}
type BacktestProgress = {
  jobId: string; totalBars: number; currentBar: number;
  equity: number[];  // partial equity curve
  tradeCount: number;
  currentPosition?: { ticker: string; pnl: number; pnlPct: number; };
}
// Final message: { jobId, status: "complete", reportUrl: "/api/backtest/{jobId}/report" }

// GET /api/backtest/{jobId}/report
type BacktestReport = { /* full Phase 2 report */ }
```

---

## 2. Backend — Python FastAPI

### 2.1 Endpoints

| Method | Path | Purpose | Auth |
|--------|------|---------|------|
| GET | `/health` | Service status | No |
| GET | `/api/portfolio` | Portfolio state + AI signals | JWT |
| POST | `/api/orders` | Place new order | JWT |
| GET | `/api/orders` | Order history | JWT |
| POST | `/api/backtest/start` | Start async backtest job | JWT |
| GET | `/api/backtest/{jobId}/report` | Backtest results | JWT |
| GET | `/api/market-data/{ticker}` | Latest OHLCV bar | No |
| GET | `/api/market-data/{ticker}/bars?n=80` | Last N bars | No |
| GET | `/api/signals/{ticker}` | Live AI signal | No |
| POST | `/api/auth/login` | Login → JWT token | No |
| POST | `/api/auth/register` | Register new user | No |

### 2.2 WebSocket Endpoints

| Path | Purpose |
|------|---------|
| `/ws/backtest/{jobId}` | Backtest progress stream |
| `/ws/portfolio/{userId}` | Real-time portfolio updates |
| `/ws/market/{ticker}` | Live price feed |

### 2.3 Backend Requirements

**REQ-B1: Async Backtest Jobs**
- POST `/api/backtest/start` returns immediately with `jobId`.
- Backtest runs in background thread/process. Progress pushed to Redis pub/sub.
- WebSocket `/ws/backtest/{jobId}` streams progress to frontend.
- Completed reports saved to postgres `backtest_reports` table.
- Max 3 concurrent backtest jobs per user. Queue overflow returns 429.

**REQ-B2: Portfolio Management**
- Positions tracked in postgres `positions` table.
- On order fill (from Rust engine or mock), update positions atomically.
- AI signals generated by calling ML engine `/predict/batch` every 30s. Cached in Redis (TTL 60s).
- Portfolio equity history: snapshotted every 5 minutes into `equity_snapshots` table.

**REQ-B3: Rate Limiting**
- 100 req/min for authenticated users.
- 10 req/min for unauthenticated (market data only).
- Redis token bucket implementation.

**REQ-B4: Error Handling**
- All errors return JSON: `{ error: string, code: string, detail?: string }`.
- 400: validation errors (pydantic). 401: invalid token. 429: rate limit. 500: internal (logged, sanitized).

---

## 3. ML Inference Engine — PyTorch (CPU)

### 3.1 Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Model status, loaded features, memory |
| POST | `/predict` | Single ticker prediction |
| POST | `/predict/batch` | Batch prediction (max 32 tickers) |
| GET | `/models` | List available checkpoints |

### 3.2 ML Engine Requirements

**REQ-ML1: Model Versioning**
- `checkpoints/model_registry.json` tracks all available models.
- Each model entry: `modelId`, `epoch`, `valLoss`, `trainingDate`, `datasetHash`.
- `/models` endpoint returns registry.
- `predict` endpoint accepts optional `?model=modelId` parameter.

**REQ-ML2: Pre-cached Encodings**
- First inference for a ticker encodes full trajectory through cascade and caches to `data/cache/{TICKER}_fused.npy`.
- Subsequent inferences: load cache, run only direct path head (skip LSTM + cascade).
- Cache invalidation on model change (different modelId → rebuild cache).
- Memory budget: keep most recent 20 ticker caches in RAM. LRU eviction.

**REQ-ML3: Batch Throughput**
- Target: 32 tickers in < 5 seconds (CPU, 4 threads).
- If exceeded: return partial results with `"truncated": true`.
- Warm up: pre-encode top 100 tickers at startup (configurable).

**REQ-ML4: Monitoring**
- `/health` reports: model loaded (bool), features count, cached tickers count, avg inference ms, memory MB.
- Log every prediction: ticker, predicted_return, sigma, inference_ms. Aggregated to wandb every 5 min.

---

## 4. Rust Core Engine — tokio / serde

### 4.1 Modules

| Module | Purpose | Status |
|--------|---------|--------|
| `order_engine/` | Price-time priority order book + matching | Implemented (src) |
| `market_data/` | Bar aggregation cache, Alpaca WebSocket | Implemented (src) |
| `risk/` | Pre-trade checks, daily loss limit | Implemented (src) |
| `ml_gateway/` | HTTP client to Python ML service | Implemented (src) |
| `api/` | gRPC server for FastAPI ↔ Rust | Not implemented |

### 4.2 Rust Requirements

**REQ-R1: Order Matching Latency**
- Target: < 100μs per order match (benchmarked via criterion).
- In-memory order book, no allocations on hot path.
- `BTreeMap` for price levels, `Vec` for order queues per level.
- Lock-free where possible (initially single-threaded; multi-threaded with `tokio::sync::RwLock` later).

**REQ-R2: Market Data Ingestion**
- Connect to Alpaca WebSocket (IEX or SIP feed).
- Reconstruct 5-min bars from raw trades/quote updates.
- Cache last 500 bars per ticker in memory.
- Expose via gRPC: `GetBars(ticker, n) → Bar[]`.

**REQ-R3: Risk Checks**
- Pre-trade: max position 25% of portfolio, max total exposure 80%.
- Daily loss limit: configurable, halt trading when exceeded.
- Circuit breaker: halt on >10% single-bar move (configurable).

**REQ-R4: gRPC API (to be implemented)**
- `SubmitOrder(OrderRequest) → OrderResponse`
- `CancelOrder(OrderId) → CancelResponse`
- `GetPositions(UserId) → Position[]`
- `GetBars(Ticker, Count) → Bar[]`
- `GetPrediction(Ticker) → Prediction` (proxied to ML engine)

---

## 5. Database — PostgreSQL 16

### 5.1 Schema

```sql
users(id, email, created_at)
trades(id, user_id, ticker, side, quantity, price, order_type, status, filled_at)
signals(id, ticker, predicted_return, sigma, direction, confidence, generated_at)
equity_snapshots(id, equity, cash, positions_json, snapshot_at)
backtest_reports(id, user_id, job_id, config_json, report_json, created_at)
watchlists(id, user_id, name, tickers_json, created_at)
```

### 5.2 DB Requirements

**REQ-DB1: Indexing**
- `trades(user_id, filled_at DESC)` — fast order history queries.
- `signals(ticker, generated_at DESC)` — latest signal lookup.
- `equity_snapshots(snapshot_at DESC)` — equity chart queries.

**REQ-DB2: Retention**
- `trades`: 90 days (then archive to cold storage).
- `signals`: 7 days (high volume, auto-purge via cron).
- `equity_snapshots`: 365 days.
- `backtest_reports`: 30 days per user.

---

## 6. Docker Deployment

### 6.1 Services

| Service | Dockerfile | Port | Scaling |
|---------|-----------|------|---------|
| nginx | (official) | 80 | 1 |
| backend | `docker/backend/Dockerfile` | 3000 | 2+ (uvicorn workers) |
| ml-engine | `docker/ml/Dockerfile` | 8000 | 1 (CPU-only, 4 threads) |
| rust-engine | `engine-rust/Dockerfile` | 50051 | 1 |
| postgres | (official) | 5432 | 1 |
| redis | (official) | 6379 | 1 |

### 6.2 Docker Requirements

**REQ-DK1: Production Readiness**
- All services restart: `unless-stopped`.
- Health checks on every service.
- Logs to stdout (docker logs compatible).
- Volumes for persistent data: `pgdata`, `redisdata`, `ml_cache`.

**REQ-DK2: Resource Limits**
- ml-engine: 4GB memory limit, 512MB reservation.
- backend: 1GB memory limit.
- rust-engine: 256MB memory limit.
- postgres: 1GB memory limit, 2GB shared_buffers.

**REQ-DK3: Networking**
- Internal network: `quantflow` (bridge).
- Only nginx exposes port 80 to host.
- All inter-service communication over internal network.

---

## 7. Testing & QA

**REQ-T1: Unit Tests**
- Rust: `cargo test` — order book matching (price-time priority, partial fills, cancel).
- Python: `pytest` — backtest engine logic, signal generator, position sizer math.
- Frontend: Jest + React Native Testing Library — component rendering, API mock responses.

**REQ-T2: Integration Tests**
- Docker compose test stack: verify all services start and health checks pass.
- API smoke test: `curl /health`, `POST /predict`, `GET /api/portfolio`.
- Backtest job lifecycle: start → progress → complete → fetch report.

**REQ-T3: Performance Benchmarks**
- Rust: `criterion` benchmarks for order matching (<100μs target).
- ML Engine: `pytest-benchmark` for inference latency (32 tickers < 5s target CPU).
- Backend: `wrk` or `k6` for 100 concurrent portfolio queries.

---

## 8. Documentation

**REQ-D1:** `README.md` updated with new stack: FastAPI + Rust + Expo.
**REQ-D2:** `docs/architecture/` — component diagram, data flow, deployment guide.
**REQ-D3:** API docs auto-generated from FastAPI (`/docs` endpoint).
**REQ-D4:** `.kilo/plans/` — all plan documents (this file + architecture freeze + migration plan + issue inventory).

---

*Next: Implement REQ-F2 (interactive backtesting) as the first deliverable, since it directly addresses the user's "spinning wheel" complaint. Then REQ-R1-R4 (Rust completion), REQ-F1 (dashboard), REQ-F3-F4 (trade + market views).*
