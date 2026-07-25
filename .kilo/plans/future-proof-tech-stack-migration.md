# QuantFlow Future-Proof Tech Stack Migration Plan

Date: 2026-07-24
Status: PLANNING — not yet executing

---

## Target Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Expo React Native                         │
│                    (iOS · Android · Web · Desktop)                │
│              Trader dashboard, order entry, alerts                │
└──────────────────────────────┬───────────────────────────────────┘
                               │ REST / WebSocket / tRPC
┌──────────────────────────────▼───────────────────────────────────┐
│                      Effect-TS (Bun/Node)                         │
│           Business logic, risk engine, user auth, API gateway     │
│    ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│    │  Auth    │  │ Portfolio│  │  Orders  │  │  ML Gateway  │   │
│    │ (JWT/OAuth)│ │ Manager │  │  Router  │  │ (REST call)  │   │
│    └──────────┘  └──────────┘  └────┬─────┘  └──────┬───────┘   │
│                                     │                │           │
└─────────────────────────────────────┼────────────────┼───────────┘
                                      │                │
               ┌──────────────────────▼────┐   ┌───────▼──────────┐
               │      Rust Core Engine     │   │   Python ML      │
               │  (sub-millisecond latency)│   │   (offline)      │
               │  • Order matching engine  │   │  • CascadeANP    │
               │  • Market data ingestion  │   │  • Training      │
               │  • Risk checks (pre-trade)│   │  • ONNX export   │
               │  • Position tracking      │   │                  │
               └───────────────────────────┘   └──────────────────┘
```

---

## Phase 1: Python ML Productionization (Week 1-2) — NOW

### What exists (keep)
| Component | File | Status |
|-----------|------|--------|
| CascadeANP model | `quantflow/models/cascade_anp.py` | Frozen v1.0 |
| Training pipeline | `scripts/train_cascade_anp_opt.py` | Working (AMP, grad_accum, caching) |
| Alpaca dataset builder | `scripts/build_alpaca_dataset.py` | Working (140 tickers, 5 years) |
| Backtest engine | `quantflow/backtest/engine.py` | Phase 2 (dynamic sizing, sigma stops) |

### What to build
1. **ONNX export** — convert trained PyTorch model to ONNX for portable inference
   - File: `scripts/export_cascade_onnx.py`
   - Output: `checkpoints/cascade_anp.onnx`
   - Enables inference without PyTorch dependency (Rust can call ONNX via tract)

2. **Model registry** — versioned checkpoint storage with metadata
   - `model_id`, `val_loss`, `training_date`, `dataset_hash`, `hyperparameters`
   - Simple JSON file: `checkpoints/model_registry.json`

3. **Inference API server** — lightweight FastAPI endpoint serving model predictions
   - Endpoint: `POST /predict` → `{ticker, predicted_return, sigma, confidence, direction}`
   - Input: 80-bar OHLCV window
   - Output: JSON with forward 21-step forecast

4. **Clean up**: Remove abandoned code paths
   - Old classification head (`past_head`, `past_pool`)
   - Old autoregressive decoder (`forward_ar`, `return_embed`)
   - Old loss configs with classification weights
   - Old `future_head` references

---

## Phase 2: Effect-TS Backend Foundation (Week 3-5)

### New codebase: `backend-effect/`

```
backend-effect/
├── package.json          # Bun runtime, Effect-TS, Drizzle ORM
├── tsconfig.json
├── src/
│   ├── main.ts           # Entry point — Effect runtime
│   ├── config.ts         # Environment config (from .env)
│   ├── layers/
│   │   ├── auth/         # JWT + OAuth (Firebase optional)
│   │   ├── api/          # REST + WebSocket endpoints
│   │   ├── portfolio/    # Position tracking, PnL calculation
│   │   ├── orders/       # Order creation, validation, routing
│   │   ├── risk/         # Pre-trade risk checks (max exposure, daily loss limit)
│   │   ├── ml/           # ML Gateway — calls Python inference server
│   │   └── market-data/  # Data access layer (Rust or Alpaca direct)
│   ├── db/
│   │   ├── schema.ts     # Drizzle ORM schema
│   │   └── migrations/
│   └── __tests__/
└── docker-compose.yml
```

### API Contracts (Effect-TS ↔ Expo)

```typescript
// GET /api/portfolio
type PortfolioResponse = {
  equity: number
  cash: number
  positions: Array<{
    ticker: string
    shares: number
    avgPrice: number
    unrealizedPnl: number
    unrealizedPnlPct: number
  }>
  signals: Array<{
    ticker: string
    direction: "BUY" | "SELL" | "HOLD"
    confidence: number
    predictedReturn: number
    sigma: number
  }>
}

// POST /api/orders
type OrderRequest = {
  ticker: string
  side: "buy" | "sell"
  quantity: number
  type: "market" | "limit"
  limitPrice?: number
  stopLoss?: number
  takeProfit?: number
}
```

### API Contracts (Effect-TS ↔ Python ML)

```
POST /ml/predict
Request:  { ticker: string, bars: number[][] }  // 80×27 feature window
Response: { predicted_return: number, sigma: number, direction: string, confidence: number }
```

### API Contracts (Effect-TS ↔ Rust Core)

```
Rust exposes a gRPC or Redis-based pub/sub interface:
  - Order submission: { ticker, side, qty, price, type, userId }
  - Order fills: { orderId, fillPrice, fillQty, timestamp }
  - Position updates: { ticker, shares, avgPrice }
  - Market data stream: { ticker, bar: { o, h, l, c, v } }
```

---

## Phase 3: Rust Core Engine (Week 6-10)

### New codebase: `engine-rust/`

```
engine-rust/
├── Cargo.toml
├── src/
│   ├── main.rs                # Entry point
│   ├── order_engine/
│   │   ├── mod.rs
│   │   ├── book.rs            # Order book (price-time priority)
│   │   ├── matching.rs        # Matching engine (continuous auction)
│   │   ├── order.rs           # Order types (market, limit, stop, IOC, FOK)
│   │   └── tests/
│   ├── market_data/
│   │   ├── mod.rs
│   │   ├── alpaca_stream.rs   # Alpaca WebSocket ingestion (IEX/SIP)
│   │   ├── bar_aggregator.rs  # 1-min, 5-min, 15-min bar builder
│   │   └── cache.rs           # In-memory ring buffer of recent bars
│   ├── risk/
│   │   ├── mod.rs
│   │   ├── pre_trade.rs      # Max exposure, daily loss, position limits
│   │   └── circuit_breaker.rs # Volatility halts, trading pauses
│   ├── state/
│   │   ├── mod.rs
│   │   ├── positions.rs      # Position tracking per account
│   │   └── account.rs        # Account balances, margin
│   ├── ml_gateway/
│   │   ├── mod.rs
│   │   └── onnx_runner.rs    # Run ONNX model via tract-rs for inference
│   └── api/
│       ├── mod.rs
│       ├── grpc_server.rs    # gRPC for Effect-TS ↔ Rust
│       └── health.rs
├── benches/
│   └── order_matching.rs     # Latency benchmarks (<100μs target)
└── tests/
    └── integration/
```

### Rust Technology Choices
| Purpose | Crate | Rationale |
|---------|-------|-----------|
| Async runtime | `tokio` | Industry standard, multi-threaded |
| gRPC server | `tonic` | Native Rust, fast, type-safe protobuf |
| Serialization | `serde` + `serde_json` | Standard, zero-cost |
| ONNX inference | `tract` | Pure Rust, no Python dependency |
| Order book | Custom (no crate) | Simple price-time priority, ~200 lines |
| WebSocket | `tokio-tungstenite` | Lightweight, async |
| Logging | `tracing` | Structured, spans, subscriber |
| Metrics | `metrics` crate | Prometheus-compatible |
| Testing | `criterion` (benches) + `tokio::test` | |

---

## Phase 4: Expo Frontend Enhancement (Week 11-12)

### Already exists: `frontend-expo/` (React Native 0.79 + Expo SDK 53)

### What to add:
1. **Trade entry screen** — order ticket with live price, stop/take-profit fields
2. **Portfolio dashboard** — equity curve, PnL breakdown, active positions
3. **Signal feed** — real-time AI signals with confidence indicators
4. **WebSocket connection** — live price updates + order status from Effect-TS
5. **Push notifications** — filled orders, stop-loss triggered, margin calls

```
frontend-expo/
├── screens/
│   ├── Dashboard.tsx       # Portfolio overview + equity chart
│   ├── TradeEntry.tsx      # Order ticket
│   ├── SignalFeed.tsx      # AI signals list
│   ├── Portfolio.tsx       # Position details
│   └── Settings.tsx        # API keys, risk params
├── services/
│   ├── api.ts              # Effect-TS API client (fetch wrapper)
│   ├── ws.ts               # WebSocket client
│   └── auth.ts             # Token management
├── components/
│   ├── OrderTicket.tsx     # Reusable order form
│   ├── EquityChart.tsx     # Equity curve (canvas)
│   ├── SignalCard.tsx      # Single signal display
│   └── PositionRow.tsx     # Position list item
└── hooks/
    ├── usePortfolio.ts     # SWR/React Query for portfolio data
    ├── useSignals.ts       # SWR for signals
    └── useWebSocket.ts     # WebSocket connection hook
```

---

## Phase 5: Deprecation Plan

### Remove / sunset:
| Component | When | Replacement |
|-----------|------|-------------|
| `quantflow/api/server.py` (FastAPI) | Phase 2 complete | Effect-TS API gateway |
| `webapp/` (Flask) | Phase 2 complete | Expo frontend |
| `app/streamlit_app.py` | Phase 2 complete | Expo + Backtest tab in Effect-TS |
| `quantflow/broker/robinhood_mcp.py` | Phase 3 complete | Rust core engine |
| `quantflow/execution/` | Phase 3 complete | Rust order engine |
| `quantflow/data/alpaca_client.py` | Phase 3 complete | Rust market data ingestion |

### Keep (forever):
| Component | Why |
|-----------|-----|
| `quantflow/models/cascade_anp.py` | Model architecture — frozen v1.0 |
| `quantflow/data/labeler.py` | Directional labeling logic |
| `scripts/build_alpaca_dataset.py` | Dataset builder |
| `scripts/train_cascade_anp_opt.py` | Training pipeline |
| `scripts/backtest_cascade.py` | Backtest CLI |
| `quantflow/backtest/` | Phase 2 backtest engine |
| `quantflow/models/cascade_anp_viz.py` | Forecast visualization |
| `scripts/viz_forecast_inference.py` | Offline inference viz |
| `scripts/build_intraday_dataset.py` | yfinance fallback builder |

---

## Immediate Next Steps

1. **Phase 1 — ML Productionization (this session)**
   - [ ] ONNX export script → `scripts/export_cascade_onnx.py`
   - [ ] Model registry → `checkpoints/model_registry.json`
   - [ ] Clean up dead code paths from `cascade_anp.py`
   - [ ] Create `setup-effect.sh` — Bun + Effect-TS scaffolding script

2. **Phase 2 — Effect-TS backend (next session)**
   - [ ] Initialize `backend-effect/` with Bun + Effect-TS + Drizzle
   - [ ] Implement Auth layer (JWT)
   - [ ] Implement Portfolio endpoint (reads from Rust state or mock)
   - [ ] Implement ML Gateway endpoint (calls Python inference)

3. **Phase 3 — Rust core (when Alpaca paper trading is live)**
   - [ ] Initialize `engine-rust/` with Cargo + tokio
   - [ ] Implement order book (price-time priority)
   - [ ] Implement Alpaca WebSocket ingestion
   - [ ] Implement gRPC server for Effect-TS communication

---

*Last updated: 2026-07-24*
