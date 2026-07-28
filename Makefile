# QuantFlow Makefile — v2 Clean Stack

PROJECT := /media/tnzr/HDD11/Dev/QuantFlow
PYTHON := /home/tnzr/.local/share/mamba/envs/quantflow/bin/python
PID_DIR := .pids
SHELL := /bin/bash

.PHONY: docker-up docker-down train build-data export-onnx test-rust backtest up down restart status logs

docker-up:
	docker compose up -d --build

docker-up-full:
	docker compose --profile full up -d --build

docker-down:
	docker compose down

up:
	@for port in 8000 3000 8080; do \
		if ss -tln 2>/dev/null | grep -q ":$$port "; then \
			echo "⚠ Port $$port is already in use. Run 'make down' first."; \
			exit 1; \
		fi; \
	done
	@if [ -f .env ]; then \
		echo "✓ Loaded .env"; \
	else \
		echo "⚠ No .env found — Alpaca will run in fallback mode."; \
	fi
	@ALPACA_KEY=""; ALPACA_SECRET=""; \
	if [ -f .env ]; then \
		ALPACA_KEY=$$(grep '^ALPACA_API_KEY=' .env | head -1 | cut -d= -f2-); \
		ALPACA_SECRET=$$(grep '^ALPACA_SECRET_KEY=' .env | head -1 | cut -d= -f2-); \
	fi; \
	echo "  ALPACA_API_KEY=$${ALPACA_KEY:0:6}..."; \
	mkdir -p $(PID_DIR); \
	echo "╔══════════════════════════════════════════╗"; \
	echo "║      QuantFlow Local Platform            ║"; \
	echo "╚══════════════════════════════════════════╝"; \
	echo ""; \
	echo "Starting ML engine on :8000..."; \
	PYTHONPATH=$(PROJECT) MODEL_PATH=$(PROJECT)/checkpoints/cascade_anp.pt \
		DATA_PATH=$(PROJECT)/data/alpaca_5m.parquet CACHE_DIR=$(PROJECT)/data/cache \
		ALPACA_API_KEY="$$ALPACA_KEY" ALPACA_SECRET_KEY="$$ALPACA_SECRET" \
		nohup $(PYTHON) -m uvicorn docker.ml.server:app --host 0.0.0.0 --port 8000 \
		> /tmp/qf_ml.log 2>&1 & echo $$! > $(PID_DIR)/ml.pid; \
	echo "Starting API on :3000..."; \
	PYTHONPATH=$(PROJECT) ML_ENGINE_URL=http://localhost:8000 PAPER_TRADING=true JWT_SECRET=dev-secret \
		ALPACA_API_KEY="$$ALPACA_KEY" ALPACA_SECRET_KEY="$$ALPACA_SECRET" \
		nohup $(PYTHON) -m uvicorn quantflow.api.server:app --host 0.0.0.0 --port 3000 \
		> /tmp/qf_backend.log 2>&1 & echo $$! > $(PID_DIR)/api.pid; \
	echo "Starting frontend on :8080..."; \
	nohup $(PYTHON) -m http.server 8080 --bind 0.0.0.0 --directory $(PROJECT)/frontend-expo/dist \
		> /tmp/qf_frontend.log 2>&1 & echo $$! > $(PID_DIR)/frontend.pid; \
	echo ""; \
	echo -n "Waiting for services"; \
	ok=0; for i in $$(seq 1 30); do \
		ML_OK=$$(curl -s http://localhost:8000/health 2>/dev/null | grep -c "ok" || true); \
		BE_OK=$$(curl -s http://localhost:3000/health 2>/dev/null | grep -c "ok" || true); \
		if [ "$$ML_OK" -gt 0 ] && [ "$$BE_OK" -gt 0 ]; then \
			ok=1; break; \
		fi; \
		echo -n "."; \
		sleep 2; \
	done; \
	if [ "$$ok" -eq 1 ]; then \
		echo ""; \
		echo ""; \
		echo "✓ All services ready!"; \
		echo ""; \
		echo "  Frontend:  http://localhost:8080"; \
		echo "  API:       http://localhost:3000/docs"; \
		echo "  ML Engine: http://localhost:8000/health"; \
		echo ""; \
	else \
		echo ""; \
		echo "⚠ Some services failed to start. Run 'make logs' for details."; \
		exit 1; \
	fi

down:
	@echo "Stopping services..."
	@fuser -k 3000/tcp 8000/tcp 8080/tcp 2>/dev/null || true
	@for pidfile in $(PID_DIR)/*.pid; do \
		if [ -f "$$pidfile" ]; then \
			pid=$$(cat "$$pidfile"); \
			name=$$(basename "$$pidfile" .pid); \
			kill $$pid 2>/dev/null || true; \
			rm -f "$$pidfile"; \
			echo "  stopped $$name (pid $$pid)"; \
		fi; \
	done
	@rmdir $(PID_DIR) 2>/dev/null || true
	@echo "✓ All services stopped."

restart: down up

status:
	@for pidfile in $(PID_DIR)/*.pid; do \
		if [ -f "$$pidfile" ]; then \
			pid=$$(cat "$$pidfile"); \
			name=$$(basename "$$pidfile" .pid); \
			if kill -0 $$pid 2>/dev/null; then \
				echo "$$name: running (pid $$pid)"; \
			else \
				echo "$$name: not running"; \
			fi; \
		fi; \
	done

logs:
	@echo "=== ML Engine (/tmp/qf_ml.log) ==="
	@tail -20 /tmp/qf_ml.log || true
	@echo ""
	@echo "=== API (/tmp/qf_backend.log) ==="
	@tail -20 /tmp/qf_backend.log || true
	@echo ""
	@echo "=== Frontend (/tmp/qf_frontend.log) ==="
	@tail -20 /tmp/qf_frontend.log || true

build-data:
	python scripts/build_alpaca_dataset.py --tickers 140 --years 5 --output data/alpaca_5m.parquet

train:
	python scripts/train_cascade_anp_opt.py --data data/alpaca_5m.parquet --epochs 50 \
		--hidden-dim 128 --batch-size 16 --grad-accum 4 --use-amp

test-rust:
	cd engine-rust && cargo test

backtest:
	python scripts/backtest_cascade.py --tickers "AAPL,MSFT,GOOGL,META,AMZN,NVDA,TSLA,AMD" --capital 100000
