SHELL := /bin/bash
ENV_NAME := quantflow
PY := conda run -n $(ENV_NAME) --no-capture-output python
PIP := conda run -n $(ENV_NAME) --no-capture-output pip

.PHONY: help env-create env-update env-remove kernel db-init scan rec options daily backtest portfolio lab app clean docker-build docker-run docker-cli

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
	@echo "  portfolio    Evaluate Robinhood positions for signals (requires env vars)"
	@echo "  lab          Launch JupyterLab"
	@echo "  app          Launch Streamlit app"
	@echo "  docker-build Build Docker image"
	@echo "  docker-run   Run JupyterLab in Docker"
	@echo "  docker-cli   Run a one-off CLI command in Docker (ex: make docker-cli CMD='python -m quantflow.cli rec')"

env-create:
	conda env create -f environment.yml || mamba env create -f environment.yml

env-update:
	conda env update -f environment.yml --prune || mamba env update -f environment.yml --prune

env-remove:
	conda env remove -n $(ENV_NAME) || true

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

START ?= 2018-01-01
TICKERS ?=
backtest:
	$(PY) -m quantflow.cli backtest --tickers "$(TICKERS)" --start $(START)

portfolio:
	$(PY) -m quantflow.cli portfolio

lab:
	conda run -n $(ENV_NAME) --no-capture-output jupyter lab --NotebookApp.token='' --NotebookApp.password='' --ip=0.0.0.0 --port=8888 --no-browser

app:
	conda run -n $(ENV_NAME) --no-capture-output streamlit run app/streamlit_app.py

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
