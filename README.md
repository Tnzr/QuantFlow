# QuantFlow PoC

A proof-of-concept pipeline to screen stocks with Finviz, persist market and news data for training, compute seasonality, and produce option swing recommendations across horizons (1w, 1m, 3m, 6m, 1y). The backend now also exports model-ready parquet datasets and maintains an article archive for sentiment and multimodal token research.

## Installation
- Conda (recommended)
  - conda env create -f environment.yml
  - conda activate quantflow
  - python -m ipykernel install --user --name quantflow --display-name "Python (quantflow)"
- Pip (alternative)
  - python -m venv .venv && source .venv/bin/activate
  - pip install -r requirements.txt

Parquet exports use `pyarrow`, which is included in both dependency manifests.

Local secrets setup for scraping/proxy testing:
- Copy `.env.example` to `.env` and fill your local values.
- Export before running CLI/API in shell sessions (example): `set -a; source .env; set +a`.

## Golden rules (risk-first)
- Position sizing: risk ≤ 1% of account per trade (for $100–300, keep risk $1–3). Prefer defined-risk spreads over naked options.
- Time stops: predefine a max holding window for weeklys (e.g., exit before Theta cliff: 2 trading days left) regardless of P/L.
- Price/vol stops: use ATR-based stops; abort on adverse move >1.5x ATR or if IV crush risk rises (post-earnings fade).
- Event filter: avoid holding through earnings for short-dated options unless it’s an explicit IV play with spread hedges.
- Liquidity: only trade option chains with tight spreads (≤ $0.05–0.10) and sufficient OI/volume.
- Play the averages: don’t average down; instead, recycle into next setup.
- Process over outcome: journal entry reason, exit plan, and stop before entry.

## Indicators for entries and exits
- Entry (trend-follow): price above SMA20/50, RSI>50 rising, positive 3–5 day momentum, ATR not spiking unusually.
- Exit/stop: close below SMA20 for short-term longs, RSI roll-over from >60 to <55, or adverse move >1.5x ATR. Take partials at 2–3x risk.
- Weekly timing: prefer entries Mon–Wed; avoid new weekly entries Thu–Fri unless intraday catalyst + tight risk.

## Finviz presets (scouting)
- weekly_momo: Optionable, avg vol >300k, perf 1w>10%, above 50/200sma, price>5
- weekly_bear: Optionable, avg vol >300k, perf 1w<-10%, below 50sma, price>5
- monthly_swing: Optionable, avg vol >500k, perf 4w>20%, above 50/200sma, beta 1–2
- monthly_bear: Optionable, avg vol >500k, perf 4w<-10%, below 50sma, beta 1–2
- midterm_trenders: Optionable, avg vol >500k, perf 6m>20%, above 200sma, beta 1–2
- midterm_bear: Optionable, avg vol >500k, perf 6m<-10%, below 200sma, beta 1–2
- reversal_bull: Optionable, avg vol >300k, RSI<40, below 50sma
- reversal_bear: Optionable, avg vol >300k, RSI>60, above 50sma
- leaps_quality: Optionable, higher liquidity, positive EPS/Sales QoQ, above 200sma

Presets are configurable via `configs/finviz_presets.yml`. See `docs/FINVIZ_PRESETS.md`.

Finviz fallback notes:
- Scan and refresh flows continue when individual presets fail, and report failed preset names.
- Optional proxy integration: set `QF_SCRAPING_PROXY_URL` and QuantFlow will map it to `HTTP_PROXY`/`HTTPS_PROXY` for finvizfinance requests.
- Compatibility aliases accepted for proxy URL: `SENSITIVE_LOOKUP_PROXY_URL`, `SMARTPROXY_PROXY_URL`.
- If you use API-key scraper services, keep keys in local `.env` only; Finviz transport still requires a proxy URL for this code path.
- Provider env support: `QF_SCRAPING_PROVIDER` values like `smartproxy`, `scrape_do`, `scrapingbee`.
- Transport mode behavior:
  - `proxy` mode when a proxy URL env is configured.
  - `api` mode for `scrape_do` when `SCRAPING_API_KEY` is set and no proxy URL is present.
  - `api` mode for `scrapingbee` when `SCRAPINGBEE_API_KEY` is set and no proxy URL is present.
- API visibility: `GET /scanner/scraping/status` returns active provider/mode, proxy source, API endpoint, and key presence.

## Low-premium approach ($100–300)
- Prefer debit spreads (verticals) or calendars over naked weeklies to control theta/IV risk.
- Use further OTM only with catalyst + momentum; otherwise closer to ATM with time to expiry (≥ 30 DTE) for swings.

## Architecture
- data: finviz screeners, news archive, and persistence (SQLite/SQLAlchemy)
- features: indicators, seasonality, sentiment tokens, forecast envelope, exit rules
- training: parquet-ready feature extraction for technical, seasonality, and sentiment multimodal tokens
- recommend: rule-based horizons with ATR stops/targets; options picker with liquidity and greeks heuristics
- backtest: signal tests and metrics (Sharpe, Sortino, CAGR, PF)
- app: Streamlit dashboard and React/Expo workspace

## Quick start
- python -m pip install -r requirements.txt
- python -m quantflow.cli init-db
- python -m quantflow.cli scan
- python -m quantflow.cli rec --tickers AAPL,MSFT,NVDA

## Usage (CLI)
- Initialize DB: python -m quantflow.cli init-db
- Run screeners: python -m quantflow.cli scan
- Recommendations: python -m quantflow.cli rec --tickers AAPL,MSFT,NVDA --save
- Build training features: python -m quantflow.cli train-features --tickers AAPL,MSFT,NVDA --save --out data/training_features.parquet
- Ingest news archive: python -m quantflow.cli news-ingest --input data/news_archive.jsonl --out data/news_archive.parquet
- Export any SQL table to parquet: python -m quantflow.cli export-table --table news_articles --out data/news_articles.parquet
- Options ideas: python -m quantflow.cli options --tickers AAPL --horizon 1m --bias long --budget 300
- Daily pipeline: python -m quantflow.cli daily --budget 300
- Backtest: python -m quantflow.cli backtest --tickers AAPL,MSFT --start 2018-01-01
- Portfolio (Robinhood): python -m quantflow.cli portfolio
- Refresh stale local DB automatically: python -m quantflow.cli ops-refresh --stale-days 7
- Generate textbook-style system report: python -m quantflow.cli ops-report --include-api --api-base http://127.0.0.1:8100
  - Includes runtime scraping transport snapshot from `/scanner/scraping/status` when `--include-api` is enabled.

## Makefile
- make env-create; make kernel
- make db-init
- make scan DB=sqlite:///quantflow.db
- make rec TICKERS="AAPL,MSFT" SAVE=1
- make options TICKERS="AAPL" HORIZON=1m BIAS=long BUDGET=300
- make daily BUDGET=300
- make backtest TICKERS="AAPL,MSFT" START=2018-01-01
- make refresh-db STALE_DAYS=7 REPORT=1 INCLUDE_API=1
- make report INCLUDE_API=1
- make test
- make app (Streamlit)
- make lab (JupyterLab)

## Notebooks Lab
Recommended order:
1) 01_finviz_scan_and_cache.ipynb — run screeners, inspect cached snapshots.
2) 02_seasonality_and_indicators.ipynb — compute seasonality and indicators.
3) 03_rule_based_recommender.ipynb — generate and save recommendations.
4) 04_option_chain_filters.ipynb — explore chains and liquidity filters.
5) 05_affordable_contract_picker.ipynb — candidate selection within budget.

Before running, ensure env and DB:
- make env-create && make kernel
- make db-init && make scan (or make daily)

## Streamlit App
- make app (or: streamlit run app/streamlit_app.py)
- Dashboard pages: Scanner & Recs, Options Ideas (CSV export included)

## FastAPI Backend (Recommended for React/Expo)
- Start API: `make api`
- Stop API port: `make api-stop`
- Change API port: `make api API_PORT=8100`

Core endpoints:
- `GET /health`
- `GET /scanner/presets`
- `GET /scanner/latest-universe`
- `GET /recommend/latest`
- `GET /recommend/analyze`
- `GET /analytics/leaderboard/latest`
- `GET /analytics/training-features`
- `GET /analytics/earnings-temporal`
- `GET /analytics/earnings-temporal/macro`
- `GET /news/articles`
- `GET /news/summary`
- `GET /news/timeline`
- `POST /pipeline/scan`
- `POST /pipeline/recommend`
- `POST /analytics/leaderboard`
- `POST /training/features`
- `POST /news/articles`
- `GET /options/ideas`
- `GET /charts/indicators`
- `GET /charts/seasonality`
- `GET /charts/price-distribution`
- `GET /charts/forecast`
- `GET /backtest/short-term`
- `POST /execution/propose`
- `GET /execution/intents`

Optional write auth:
- Enable: `QF_API_AUTH_ENABLED=true`
- Token mode: set `QF_API_TOKEN` and pass `x-api-key` or `Authorization: Bearer <token>`
- Firebase mode:
  - `QF_FIREBASE_AUTH_ENABLED=true`
  - Provide Firebase credentials via `GOOGLE_APPLICATION_CREDENTIALS` (service account json path)
  - Write access requires custom claim `quantflow_write=true` by default
  - Override claim key with `QF_FIREBASE_WRITE_CLAIM` and allowed roles with `QF_FIREBASE_WRITE_ROLES` (default: `writer,admin`)
- Auth diagnostics endpoint: `GET /auth/me`

MCP secure backend config endpoints (write-auth protected when enabled):
- `GET /broker/mcp/config` (redacted runtime summary only)
- `PUT /broker/mcp/config` (set runtime MCP auth/token/header config)
- `DELETE /broker/mcp/config` (clear runtime override)

Production note:
- Prefer runtime config with `secret_provider` + `secret_refs` over shell env token injection.
- Supported providers: `vault`, `aws`, `gcp`, `azure`.
- See `docs/MCP_SECRET_MANAGER_PATTERN.md` for ref formats and examples.

## Expo Frontend Starter
- Path: `frontend-expo/`
- Start backend first: `make api`
- Then run frontend:
  - `cd frontend-expo && npm install`
  - `EXPO_PUBLIC_API_BASE_URL=http://localhost:8000 npm start`

Use LAN IP (not localhost) for phone testing on the same Wi-Fi.

## Docker
- docker build -t quantflow .
- docker run -p 8888:8888 -v "$PWD":/app quantflow

## Roadmap to MVP -> App
1) Expand the news archive into a durable ingestion layer with source deduping and article metadata normalization.
2) Add richer sentiment scoring and token aggregation for technical, seasonality, and article context.
3) Backtest rule engine on equities (proxy for directional bias) + options PnL with simplified Greeks.
4) Streamlit dashboard: presets monitor, rec lists, alerts. Later React frontend.
5) Use parquet snapshots and the article archive for RAG and multimodal training datasets.

## Docs
- Finviz presets and UI mapping: `docs/FINVIZ_PRESETS.md`
- Data layer: `quantflow/data` (finviz_client.py loads YAML presets; persistence via SQLAlchemy)
- Features: `quantflow/features` (indicators, seasonality, exit rules)
- Recommender: `quantflow/recommend` (rule engine, options picker)
- Backtest: `quantflow/backtest` (signal tests, metrics)
- App: `app/streamlit_app.py` (dashboard)

## Robinhood Integration Status
- QuantFlow now uses MCP-only Robinhood integration through Agentic Trading.
- Run portfolio signals: make portfolio (or: python -m quantflow.cli portfolio --broker-mode robinhood_mcp)
- Output: per-position actions (hold/add/exit) derived from the 1w/1m rules and confidence.

### MCP-first path (recommended)
- Robinhood support now documents Agentic Trading via MCP endpoint:
  - https://agent.robinhood.com/mcp/trading
- Desktop onboarding/auth is required to open/authenticate the Agentic account.
- Important auth note: Robinhood Agentic authentication happens in your MCP-capable client session (where the MCP connector runs), not directly inside QuantFlow UI. QuantFlow reads the authenticated MCP session state after that step.

### Agentic MCP Login Steps (by client)
- Claude Code
  - Run: `claude mcp add robinhood-trading --transport http https://agent.robinhood.com/mcp/trading`
  - Run `/mcp`, select `robinhood-trading`, then authenticate.
- Claude Desktop
  - Settings -> Connectors -> Add custom connector
  - Add MCP URL: `https://agent.robinhood.com/mcp/trading`
- ChatGPT
  - Enable Developer Mode
  - Settings -> Apps -> Create app
  - Add MCP URL: `https://agent.robinhood.com/mcp/trading`
- Codex
  - Settings -> MCP servers -> Streamable HTTP
  - Add MCP URL: `https://agent.robinhood.com/mcp/trading`
- Codex CLI
  - Run: `codex mcp add robinhood-trading --url https://agent.robinhood.com/mcp/trading`
  - Run `/mcp`, select `robinhood-trading`, then authenticate.
- Cursor
  - Provide the MCP URL to your agent.
  - Settings -> Cursor Settings -> Tools & MCPs -> Connect

- QuantFlow API helper: POST `/broker/mcp/login` now returns `auth_instructions` with these steps.
- Local MCP fixture testing supported:
  - `QF_MCP_ACCOUNT_JSON='{"equity":1000,"cash":200,"buying_power":1200}'`
  - `QF_MCP_POSITIONS_JSON='{"positions":[{"ticker":"AAPL","qty":2,"avg_price":150}]}'`
- See docs/Robinhood_MCP_Integration_Blueprint.md for rollout stages and safeguards.

## Theory and Design
See `docs/THEORY.md` for deeper notes on risk-first design, entry/exit logic, weekly timing, and low-premium tactics.

Disclaimer: This is not financial advice. For education and research only.
