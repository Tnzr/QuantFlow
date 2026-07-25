# QuantFlow WebApp Phase 1 — Full Architecture Plan

**Status:** Design Complete  
**Date:** 2026-07-17  
**Target:** Educationally advisory, automated research + forecasting, ML-driven portfolio management, AI agent integration, subscription payments

---

## Architecture Decision: Flask + Jinja2 SSR (Not Streamlit, Not Expo)

### Rationale
- **Streamlit** is great for rapid data science dashboards but is single-user, stateful, non-scalable, and cannot support auth/payments/agent chat UX
- **Expo React Native** is designed for mobile, the current single-file `App.js` is unmaintainable, and cross-platform adds latency to ML features
- **Flask + Jinja2 SSR** provides: server-rendered HTML (fast first paint), session-based auth, URL routing for SEO/deep-linking, Stripe checkout, HTMX for interactive components, full control over asset pipeline, Conda env compatibility, ONNX runtime integration

### Tech Stack
| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Web Framework | Flask 3.x + Jinja2 | SSR, stable, compatible with conda env |
| Real-time Updates | HTMX + SSE (Server-Sent Events) | Dynamic portfolio/price updates without SPA overhead |
| Charts | Plotly.js (CDN) + lightweight JSON API | Same charting power as Streamlit but in browser |
| CSS Framework | Tailwind CSS (CDN/jit) | Utility-first, minimal bundle |
| Payments | Stripe.js + Stripe Python SDK | Subscription management, webhooks |
| Auth | Firebase Auth + Flask-Login | Token verification, session management |
| ML Inference | ONNX Runtime | Production inference, CPU/GPU agnostic, Conda compatible |
| Multi-GPU Training | PyTorch DDP (DistributedDataParallel) | Scientific rigor, multi-GPU scaling |
| AI Agent | LangChain + custom tools | RAG over workspace, database, ML model, portfolio algorithm |
| DB | SQLite (dev) / PostgreSQL (prod) | Existing SQLite schema, upgrade path |
| Cache | Redis | Session store, rate limiting, ML cache |
| Task Queue | Celery + Redis | Async ML inference, data fetching |
| Deployment | Docker Compose | Flask + Redis + Celery + Nginx |

### NOT using (for Phase 1)
- React/Vue SPA — overkill for Phase 1, slower dev velocity
- Streamlit — can't support auth/payments/agent
- Expo — mobile-first, not web-first, monolith file unmaintainable

---

## Route Architecture

### Public Routes
| Route | Method | Content |
|-------|--------|---------|
| `/` | GET | Landing page — hero, features, pricing tiers |
| `/pricing` | GET | Subscription plans (Free, Pro, Enterprise) |
| `/login` | GET | Firebase auth login page |
| `/register` | GET | Registration / plan selection |
| `/docs` | GET | Educational resources, methodology docs |

### Authenticated Routes (Session Required)
| Route | Method | Content |
|-------|--------|---------|
| `/dashboard` | GET | Main dashboard — portfolio summary, market overview, alerts |
| `/research` | GET | Scanner, screener, recommendations, real-time data |
| `/research/ticker/<ticker>` | GET | Single-ticker deep dive: fundamentals, ML signals, seasonality |
| `/research/scan` | GET | Scanner results page |
| `/research/recommendations` | GET | Rule-engine recommendations |
| `/portfolio` | GET | Portfolio management — positions, allocation, P&L |
| `/portfolio/allocate` | GET/POST | ML-driven portfolio allocation optimizer |
| `/portfolio/rebalance` | POST | Rebalance triggered by user |
| `/backtest` | GET | Backtest runner + results visualization |
| `/backtest/run` | POST | Execute backtest with config |
| `/forecast` | GET | ML forecast dashboard — event-state predictions, tau forecasts |
| `/forecast/ticker/<ticker>` | GET | Per-ticker ML forecast with confidence bands |
| `/agent` | GET | AI Assistant chat interface |
| `/agent/chat` | POST | Chat message endpoint (SSE streaming) |
| `/settings` | GET/POST | User settings, broker connections, notifications |
| `/settings/billing` | GET | Subscription management, invoices |
| `/api/stripe/webhook` | POST | Stripe webhook handler (unauthenticated, signature verified) |

### API Endpoints (JSON, Session Auth)
| Route | Method | Purpose |
|-------|--------|---------|
| `/api/v1/health` | GET | Health check |
| `/api/v1/price/<ticker>` | GET | Real-time price data |
| `/api/v1/ml/signals/<ticker>` | GET | ML inference results per ticker |
| `/api/v1/ml/forecast/<ticker>` | GET | Forecast with uncertainty bands |
| `/api/v1/ml/portfolio-risk` | POST | Portfolio risk assessment |
| `/api/v1/data/timeseries` | GET | Historical price + indicators |
| `/api/v1/data/universe` | GET | Available ticker universe |
| `/api/v1/scan/run` | POST | Trigger scanner |
| `/api/v1/scan/results` | GET | Scanner results |
| `/api/v1/backtest/submit` | POST | Submit backtest job |
| `/api/v1/backtest/status/<job_id>` | GET | Backtest job status |

---

## Database Schema (Extended)

Keep existing SQLite tables, add:

```sql
-- Users & Subscriptions
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    firebase_uid TEXT UNIQUE NOT NULL,
    email TEXT,
    display_name TEXT,
    subscription_tier TEXT DEFAULT 'free',  -- free, pro, enterprise
    stripe_customer_id TEXT,
    stripe_subscription_id TEXT,
    credits_remaining INTEGER DEFAULT 100,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP
);

-- ML Model versions
CREATE TABLE model_versions (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    onnx_path TEXT,
    pytorch_path TEXT,
    metrics_json TEXT,  -- JSON blob of accuracy, loss, confusion matrix
    is_active BOOLEAN DEFAULT 0,
    trained_at TIMESTAMP,
    deployed_at TIMESTAMP
);

-- ML Predictions cache
CREATE TABLE ml_predictions (
    id INTEGER PRIMARY KEY,
    ticker TEXT NOT NULL,
    model_version_id INTEGER REFERENCES model_versions(id),
    prediction_date DATE NOT NULL,
    prob_inter_event REAL,
    prob_pre_event REAL,
    prob_onset REAL,
    forecast_tau_days REAL,
    uncertainty_sigma REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, prediction_date, model_version_id)
);

-- Portfolio snapshots
CREATE TABLE portfolio_snapshots (
    id INTEGER PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    snapshot_date DATE NOT NULL,
    ticker TEXT NOT NULL,
    shares REAL,
    cost_basis REAL,
    current_price REAL,
    allocation_pct REAL,
    ml_signal TEXT,  -- inter_event, pre_event, onset
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Agent conversation history
CREATE TABLE agent_conversations (
    id INTEGER PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    role TEXT NOT NULL,  -- user, assistant, system
    content TEXT NOT NULL,
    tool_calls_json TEXT,  -- JSON: any tool invocations
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Backtest jobs
CREATE TABLE backtest_jobs (
    id INTEGER PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    status TEXT DEFAULT 'queued',  -- queued, running, completed, failed
    config_json TEXT NOT NULL,
    result_json TEXT,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);
```

---

## Subscription Tiers

| Feature | Free | Pro ($29/mo) | Enterprise ($199/mo) |
|---------|------|-------------|---------------------|
| Scanner access | 3 presets | All presets | All + custom |
| Recommendations | 5 tickers | Full universe | Full + API |
| ML signals | 3 tickers/day | 50 tickers/day | Unlimited |
| Backtesting | 1 config | 10 configs/day | Unlimited + history |
| Portfolio mgmt | Manual | ML-optimized | ML + custom allocations |
| AI Agent queries | 10/day | 100/day | Unlimited |
| Data refresh | Weekly | Daily | Real-time (1 min) |
| Broker integration | Read-only | Read + execute | Full MCP |
| API access | — | 1000 req/day | Unlimited |
| Export reports | — | PDF | PDF + CSV + API |
| Priority support | — | Email | Dedicated |

---

## WebApp Structure

```
webapp/
├── app.py                    # Flask app factory
├── config.py                 # Configuration (env vars, tiers, etc.)
├── requirements.txt          # Python deps (Flask, stripe, onnxruntime, etc.)
├── templates/
│   ├── base.html             # Base layout with Tailwind CDN, nav, footer
│   ├── landing.html          # Public landing page
│   ├── pricing.html          # Pricing plans
│   ├── auth/
│   │   ├── login.html
│   │   └── register.html
│   ├── dashboard.html        # Main dashboard
│   ├── research/
│   │   ├── index.html        # Research hub
│   │   ├── scanner.html      # Scanner results
│   │   ├── recommendations.html
│   │   └── ticker.html       # Single ticker deep dive
│   ├── portfolio/
│   │   ├── index.html        # Portfolio overview
│   │   └── allocate.html     # Allocation optimizer
│   ├── backtest/
│   │   ├── index.html        # Backtest config
│   │   └── results.html      # Backtest results
│   ├── forecast/
│   │   ├── index.html        # ML forecast dashboard
│   │   └── ticker.html       # Per-ticker forecast
│   ├── agent.html            # AI Assistant chat
│   ├── settings/
│   │   ├── profile.html
│   │   └── billing.html
│   └── components/
│       ├── nav.html           # Navigation bar
│       ├── charts.html        # Chart macros
│       ├── signal_badge.html  # ML signal badge
│       └── spinner.html       # Loading spinner
├── static/
│   ├── css/
│   │   └── app.css           # Custom styles
│   ├── js/
│   │   ├── charts.js         # Plotly.js chart helpers
│   │   ├── agent.js          # Agent chat UI logic
│   │   ├── realtime.js       # SSE for live updates
│   │   └── stripe.js         # Stripe Elements
│   └── img/                  # Static images
├── blueprints/
│   ├── auth_bp.py            # Auth routes
│   ├── dashboard_bp.py       # Dashboard routes
│   ├── research_bp.py        # Research routes
│   ├── portfolio_bp.py       # Portfolio routes
│   ├── backtest_bp.py        # Backtest routes
│   ├── forecast_bp.py        # ML forecast routes
│   ├── agent_bp.py           # AI Agent routes
│   ├── settings_bp.py        # Settings + billing routes
│   └── api_bp.py             # JSON API endpoints
├── services/
│   ├── auth_service.py       # Firebase auth integration
│   ├── stripe_service.py     # Stripe subscription management
│   ├── ml_service.py         # ONNX inference, model management
│   ├── portfolio_service.py  # Portfolio optimization
│   ├── agent_service.py      # LangChain agent + tools
│   ├── data_service.py       # Data fetching, caching
│   ├── backtest_service.py   # Backtest orchestration
│   └── notification_service.py  # Email, in-app alerts
├── tools/                    # AI Agent tools
│   ├── scanner_tool.py       # Run screener via agent
│   ├── ml_tool.py            # Query ML model via agent
│   ├── portfolio_tool.py     # Portfolio query via agent
│   ├── research_tool.py      # Research ticker via agent
│   ├── backtest_tool.py      # Run backtest via agent
│   └── workspace_tool.py     # Search workspace files
├── models/                   # SQLAlchemy models
│   ├── user.py
│   ├── subscription.py
│   ├── portfolio.py
│   └── agent.py
├── tasks/                    # Celery tasks
│   ├── inference.py          # Async ML inference
│   ├── data_refresh.py       # Scheduled data refresh
│   ├── backtest.py           # Async backtest jobs
│   └── notifications.py      # Email/push notifications
├── tests/
│   ├── test_routes.py
│   ├── test_services.py
│   └── test_tools.py
└── migrations/               # Alembic migrations
    └── versions/
```

---

## ONNX Integration Plan

### Export PyTorch → ONNX
The `TemporalStateModel` architecture exports cleanly to ONNX:
- Input: `(batch_size, lookback, feature_dim)` — fixed dynamism, static shapes
- Output: `past_state_probs`, `future_forecast`, `uncertainty_sigma`
- Export script: `quantflow/models/export_onnx.py`

```python
# quantflow/models/export_onnx.py
import torch
from architectures import create_model, MODELS

def export_to_onnx(checkpoint_path, output_path, lookback=60, feature_dim=21):
    model = create_model("bilstm", input_dim=feature_dim, hidden_dim=128)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    
    dummy_input = torch.randn(1, lookback, feature_dim)
    torch.onnx.export(
        model, dummy_input, output_path,
        input_names=["features"],
        output_names=["past_state_probs", "future_forecast", "uncertainty_sigma"],
        dynamic_axes={"features": {0: "batch_size"}},
        opset_version=17,
    )
```

### ONNX Runtime Inference Service
```python
# webapp/services/ml_service.py
import onnxruntime as ort
import numpy as np

class ONNXInferenceService:
    def __init__(self, onnx_model_path):
        self.session = ort.InferenceSession(onnx_model_path)
    
    def predict(self, features_window: np.ndarray):
        # features_window: (batch, lookback, feature_dim)
        outputs = self.session.run(None, {"features": features_window.astype(np.float32)})
        return {
            "past_state_probs": outputs[0],
            "future_forecast": outputs[1],
            "uncertainty_sigma": outputs[2],
        }
```

---

## Multi-GPU Training Plan

### PyTorch DDP Setup
The trainer already uses a single GPU. Multi-GPU requires:
1. `torch.nn.parallel.DistributedDataParallel` wrapper
2. `torch.distributed.init_process_group("nccl")` initialization
3. `DistributedSampler` in DataLoader
4. Gradient synchronization across GPUs

```python
# quantflow/models/trainer_ddp.py
def train_model_ddp(df, config, rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    model = create_model(...)
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=train_sampler)
    
    # Standard training loop (unchanged - DDP handles gradient sync)
```

### Launch
```bash
torchrun --nproc_per_node=4 scripts/train_full_universe.py --arch bilstm --epochs 100
```

---

## AI Agent Architecture

### Tool Definitions (LangChain)
The agent has access to these tools:

| Tool | Description | Backend |
|------|-------------|---------|
| `scan_market` | Run Finviz screener preset | `quantflow.data.finviz_client` |
| `research_ticker` | Deep dive on single ticker (fundamentals, signals, news) | `quantflow.recommend.engine` |
| `forecast_events` | ML event-state and tau forecast | `quantflow.models` via ONNX |
| `optimize_portfolio` | Run portfolio allocation algorithm | `quantflow.portfolio.allocator` |
| `run_backtest` | Execute backtest with config | `quantflow.models.backtest_engine` |
| `get_positions` | Fetch current broker positions | `quantflow.broker.factory` |
| `get_market_data` | Fetch OHLCV + indicators | `quantflow.features.indicators` |
| `get_news` | Recent news and sentiment | DB query |
| `search_workspace` | Search internal docs/methodology | File system grep |
| `explain_decision` | Explain ML model decision | SHAP/LIME on ONNX output |

### Agent Prompt
```
You are QuantFlow AI, an investment research and portfolio management assistant.
You have access to real market data, ML models for event forecasting, and
portfolio optimization algorithms. You can help users:
- Research stocks and sectors
- Generate trading signals from ML event-state models
- Optimize portfolio allocations
- Run backtests on strategies
- Explain market patterns and ML model decisions

Always cite your data sources and model versions. Be clear about confidence
levels and uncertainty. Never give financial advice — frame everything as
educational research output.
```

### Chat Interface
- SSE (Server-Sent Events) for streaming agent responses
- Tool calls displayed inline with status indicators
- Conversation persistence in `agent_conversations` table
- Token usage tracking per user (subscription tier limits)

---

## Real-Time Updates Architecture

### SSE (Server-Sent Events)
```
GET /api/v1/stream/portfolio  →  SSE stream of portfolio updates
GET /api/v1/stream/signals    →  SSE stream of ML signal changes  
GET /api/v1/stream/prices     →  SSE stream of real-time prices
```

### Client-Side (HTMX + JS)
```html
<div hx-ext="sse" sse-connect="/api/v1/stream/prices">
    <div sse-swap="price:AAPL">Loading...</div>
</div>
```

---

## Migration Path from Existing Code

### What to KEEP (imported as-is)
- `quantflow/` — entire quant engine, no changes needed
- `quantflow/models/` — training, backtest, architectures, losses
- `quantflow/data/` — universe, labeler, finviz, persistence
- `quantflow/features/` — indicators, signals, seasonality
- `quantflow/recommend/` — rule engine, options picker
- `quantflow/broker/` — MCP integration
- `quantflow/portfolio/` — allocator, evaluator

### What to ADD (new, in `webapp/`)
- Flask routes (blueprints) wrapping existing quantflow modules
- ONNX export script at `quantflow/models/export_onnx.py`
- Multi-GPU trainer at `quantflow/models/trainer_ddp.py`
- Celery tasks for async operations
- AI Agent tools wrapping existing functionality
- Subscription/payment infrastructure
- Jinja2 templates and static assets

### What to RETIRE (replaced)
- `app/streamlit_app.py` — replaced by Flask SSR
- `frontend-expo/` — replaced by Flask SSR (web-first)
- `quantflow/api/server.py` — endpoints reimplemented as Flask blueprints

### What to KEEP but NEST under new webapp
- Existing API server for backward compatibility (during migration)
- Docker Compose updated to include Flask + Redis + Celery

---

## Implementation Phases

### Phase 1A: Foundation (Days 1-2)
- [ ] Flask app factory + config
- [ ] Tailwind CSS base templates (base.html, components/)
- [ ] Landing page + Pricing page
- [ ] Firebase authentication (login, register, session management)
- [ ] Database schema extension (users, subscriptions)
- [ ] Docker Compose update (Flask + Redis + Celery)

### Phase 1B: Stripe Integration (Day 3)
- [ ] Stripe product/price setup
- [ ] Checkout flow (stripe.js)
- [ ] Webhook handler (subscription lifecycle)
- [ ] Customer portal
- [ ] Usage metering (credits)

### Phase 1C: Core Features (Days 4-5)
- [ ] Dashboard page (market overview, portfolio summary)
- [ ] Research page (scanner, recommendations, ticker deep-dive)
- [ ] Portfolio page (positions, allocation, P&L)
- [ ] Backtest page (config, run, results)
- [ ] Settings page (profile, billing)

### Phase 1D: ML Integration (Days 6-7)
- [ ] ONNX export script
- [ ] ONNX inference service
- [ ] Forecast page (event-state predictions, tau forecasts, confidence bands)
- [ ] ML signal badges across all pages
- [ ] Per-ticker ML deep-dive
- [ ] Multi-GPU training script

### Phase 1E: AI Agent (Days 8-9)
- [ ] LangChain agent setup
- [ ] Tool implementations (scanner, research, ML, portfolio, backtest, workspace)
- [ ] Chat interface with SSE streaming
- [ ] Conversation persistence
- [ ] Tool call visualization

### Phase 1F: Real-Time + Polish (Days 10-11)
- [ ] SSE endpoints for live data
- [ ] HTMX progressive enhancement
- [ ] Plotly.js charts (portfolio, research, backtest)
- [ ] Responsive design audit
- [ ] Error pages (404, 500, maintenance)
- [ ] Rate limiting + credit enforcement

### Phase 1G: Testing + Deploy (Days 12-13)
- [ ] Unit tests (routes, services, tools)
- [ ] Integration tests (auth → research → backtest flow)
- [ ] Stripe test mode validation
- [ ] Docker production build
- [ ] Nginx config (reverse proxy, static assets, SSL)
- [ ] Deployment documentation
