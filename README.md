# QuantFlow PoC

A proof-of-concept pipeline to screen stocks with Finviz, persist data for RAG, compute seasonality, and produce option swing recommendations across horizons (1w, 1m, 3m, 6m, 1y). Starts as a Python library + notebooks, later upgrade to Streamlit/React + LLM agents.

## Installation
- Conda (recommended)
  - conda env create -f environment.yml
  - conda activate quantflow
  - python -m ipykernel install --user --name quantflow --display-name "Python (quantflow)"
- Pip (alternative)
  - python -m venv .venv && source .venv/bin/activate
  - pip install -r requirements.txt

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

## Low-premium approach ($100–300)
- Prefer debit spreads (verticals) or calendars over naked weeklies to control theta/IV risk.
- Use further OTM only with catalyst + momentum; otherwise closer to ATM with time to expiry (≥ 30 DTE) for swings.

## Architecture
- data: finviz screeners + persistence (SQLite/SQLAlchemy)
- features: indicators, seasonality, earnings proximity, exit rules
- recommend: rule-based horizons with ATR stops/targets; options picker with liquidity and greeks heuristics
- backtest: signal tests and metrics (Sharpe, Sortino, CAGR, PF)
- app: Streamlit dashboard

## Quick start
- python -m pip install -r requirements.txt
- python -m quantflow.cli init-db
- python -m quantflow.cli scan
- python -m quantflow.cli rec --tickers AAPL,MSFT,NVDA

## Usage (CLI)
- Initialize DB: python -m quantflow.cli init-db
- Run screeners: python -m quantflow.cli scan
- Recommendations: python -m quantflow.cli rec --tickers AAPL,MSFT,NVDA --save
- Options ideas: python -m quantflow.cli options --tickers AAPL --horizon 1m --bias long --budget 300
- Daily pipeline: python -m quantflow.cli daily --budget 300
- Backtest: python -m quantflow.cli backtest --tickers AAPL,MSFT --start 2018-01-01
- Portfolio (Robinhood): python -m quantflow.cli portfolio

## Makefile
- make env-create; make kernel
- make db-init
- make scan DB=sqlite:///quantflow.db
- make rec TICKERS="AAPL,MSFT" SAVE=1
- make options TICKERS="AAPL" HORIZON=1m BIAS=long BUDGET=300
- make daily BUDGET=300
- make backtest TICKERS="AAPL,MSFT" START=2018-01-01
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

## Docker
- docker build -t quantflow .
- docker run -p 8888:8888 -v "$PWD":/app quantflow

## Roadmap to MVP -> App
1) Stabilize screeners, retries, and daily/hourly snapshots.
2) Add option chain metrics (OI/vol, spread, IV rank) and integrate into rules.
3) Backtest rule engine on equities (proxy for directional bias) + options PnL with simplified Greeks.
4) Streamlit dashboard: presets monitor, rec lists, alerts. Later React frontend.
5) Add sentiment/LLM agents once DB populated; use RAG over cached news/metrics.

## Docs
- Finviz presets and UI mapping: `docs/FINVIZ_PRESETS.md`
- Data layer: `quantflow/data` (finviz_client.py loads YAML presets; persistence via SQLAlchemy)
- Features: `quantflow/features` (indicators, seasonality, exit rules)
- Recommender: `quantflow/recommend` (rule engine, options picker)
- Backtest: `quantflow/backtest` (signal tests, metrics)
- App: `app/streamlit_app.py` (dashboard)

## Robinhood (read-only provider)
- Set env vars: ROBINHOOD_USERNAME, ROBINHOOD_PASSWORD, optionally ROBINHOOD_MFA
- Run: make portfolio (or: python -m quantflow.cli portfolio)
- Output: per-position actions (hold/add/exit) derived from the 1w/1m rules and confidence

## Theory and Design
See `docs/THEORY.md` for deeper notes on risk-first design, entry/exit logic, weekly timing, and low-premium tactics.

Disclaimer: This is not financial advice. For education and research only.
