SHELL := /bin/bash
ENV_NAME := quantflow
CONDA ?= $(shell command -v conda 2>/dev/null || echo $(HOME)/miniforge3/bin/conda)
PY := $(CONDA) run -n $(ENV_NAME) --no-capture-output python
PIP := $(CONDA) run -n $(ENV_NAME) --no-capture-output pip

.PHONY: help env-create env-update env-remove kernel db-init scan rec options daily backtest portfolio refresh-db report smoke-test test lab app api frontend-build clean docker-build docker-run docker-cli docker-up docker-down docker-logs setup
API_PORT ?= 8000
APP_PORT ?= 8501

help:
	@echo "Targets:"
	@echo "  env-create   Create conda env from environment.yml"
	@echo "  env-update   Update existing env from environment.yml"
	@echo "  env-remove   Remove env"
	@echo "  kernel       Install Jupyter kernel for env"
	@echo "  db-init      Initialize SQLite schema"
	@echo "  scan         Run Finviz screeners and cache snapshots"
	@echo "  rec          Run rule-based recommendations (set TICKERS=AAPL,MSFT SAVE=1)"
	@echo "  options      Suggest affordable option contracts (HORIZON=1m BIAS=long BUDGET=300)"
	@echo "  daily        Orchestrate scan+recommend+options in one shot"
	@echo "  backtest     Run short-term backtest (set TICKERS=AAPL,MSFT START=2018-01-01)"
	@echo "  portfolio    Evaluate Robinhood MCP positions for signals (auth via MCP client session)"
	@echo "  refresh-db   Refresh local DB when stale (STALE_DAYS=7 FORCE=1 REPORT=1 INCLUDE_API=1)"
	@echo "  report       Generate textbook-style ops report (STALE_DAYS=7 INCLUDE_API=1)"
	@echo "  smoke-test   One-command API smoke check (health + scraping + latest report)"
	@echo "  test         Run unit tests"
	@echo "  lab          Launch JupyterLab"
	@echo "  app          Launch Streamlit app"
	@echo "  api          Launch FastAPI backend (for React/frontend clients)"
	@echo "  frontend-build Export Expo web app into frontend-expo/dist"
	@echo "  app-stop     Stop Streamlit port ($(APP_PORT))"
	@echo "  api-stop     Stop FastAPI port ($(API_PORT))"
	@echo "  docker-build Build Docker image"
	@echo "  docker-run   Run JupyterLab in Docker"
	@echo "  docker-cli   Run a one-off CLI command in Docker (ex: make docker-cli CMD='python -m quantflow.cli rec')"
	@echo "  docker-up    Start API + exported web frontend with docker compose"
	@echo "  docker-down  Stop docker compose services"
	@echo "  docker-logs  Tail docker compose logs"
	@echo "  setup        Run env-create, db-init, and docker-build for one-stop setup"

env-create:
	$(CONDA) env create -f environment.yml || mamba env create -f environment.yml
	$(CONDA) run -n $(ENV_NAME) --no-capture-output pip install -e .

env-update:
	$(CONDA) env update -f environment.yml --prune || mamba env update -f environment.yml --prune

env-remove:
	$(CONDA) env remove -n $(ENV_NAME) || true

env-activate:
	@echo "conda activate $(ENV_NAME)"

kernel:
	$(PY) -m ipykernel install --user --name $(ENV_NAME) --display-name "Python ($(ENV_NAME))"

# Database and pipelines
DB ?= sqlite:///quantflow.db

db-init:
	$(PY) -m quantflow.cli init-db --db $(DB)

scan:
	$(PY) -m quantflow.cli scan --db $(DB)

TICKERS ?=
SAVE ?=
rec:
	$(PY) -m quantflow.cli rec --tickers "$(TICKERS)" $(if $(SAVE),--save,) --db $(DB)

HORIZON ?= 1m
BIAS ?= long
BUDGET ?= 300
options:
	$(PY) -m quantflow.cli options --tickers "$(TICKERS)" --horizon $(HORIZON) --bias $(BIAS) --budget $(BUDGET)

daily:
	$(PY) -m quantflow.cli daily --db $(DB) --budget $(BUDGET)

STALE_DAYS ?= 7
FORCE ?=
REPORT ?=
INCLUDE_API ?=
API_BASE ?= http://127.0.0.1:8100
REPORT_OUT ?= docs/LOCAL_SYSTEM_REPORT.md

refresh-db:
	$(PY) -m quantflow.cli ops-refresh --db $(DB) --stale-days $(STALE_DAYS) $(if $(FORCE),--force,) $(if $(REPORT),--with-report,) $(if $(INCLUDE_API),--include-api,) --api-base $(API_BASE) --report-out $(REPORT_OUT)

report:
	$(PY) -m quantflow.cli ops-report --db $(DB) --stale-days $(STALE_DAYS) $(if $(INCLUDE_API),--include-api,) --api-base $(API_BASE) --report-out $(REPORT_OUT)

smoke-test:
	$(PY) -m quantflow.cli ops-report --db $(DB) --stale-days $(STALE_DAYS) --include-api --api-base $(API_BASE) --report-out $(REPORT_OUT)
	API_BASE="$(API_BASE)" $(PY) scripts/smoke_test.py

test:
	$(PY) -m unittest discover -s tests -p "test_*.py"

START ?= 2018-01-01
TICKERS ?=
backtest:
	$(PY) -m quantflow.cli backtest --tickers "$(TICKERS)" --start $(START)

portfolio:
	$(PY) -m quantflow.cli portfolio

lab:
	conda run -n $(ENV_NAME) --no-capture-output jupyter lab --NotebookApp.token='' --NotebookApp.password='' --ip=0.0.0.0 --port=8888 --no-browser

app:
	$(CONDA) run -n $(ENV_NAME) streamlit run app/streamlit_app.py --server.headless true --browser.gatherUsageStats false --server.address 0.0.0.0 --server.port $(APP_PORT)

api:
	$(CONDA) run -n $(ENV_NAME) uvicorn quantflow.api.server:app --host 0.0.0.0 --port $(API_PORT) --reload

frontend-build:
	cd frontend-expo && CI=1 EXPO_NO_TELEMETRY=1 npx expo export --platform web --output-dir dist

app-stop:
	-@fuser -k $(APP_PORT)/tcp 2>/dev/null || true

api-stop:
	-@fuser -k $(API_PORT)/tcp 2>/dev/null || true

clean:
	rm -f quantflow.db
	find . -type d -name __pycache__ -exec rm -rf {} +

# Docker helpers
IMAGE ?= quantflow

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm -p 8888:8888 -v "$$PWD":/app $(IMAGE)

docker-cli:
	docker run --rm -it -v "$$PWD":/app $(IMAGE) bash -lc "$(CMD)"

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f --tail=200

setup: env-create db-init docker-build
	@echo "Environment, DB schema, and Docker image ready."
