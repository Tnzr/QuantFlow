# QuantFlow Makefile — v2 Clean Stack

.PHONY: docker-up docker-down train build-data export-onnx test-rust

docker-up:
	docker compose up -d --build

docker-up-full:
	docker compose --profile full up -d --build

docker-down:
	docker compose down

build-data:
	python scripts/build_alpaca_dataset.py --tickers 140 --years 5 --output data/alpaca_5m.parquet

train:
	python scripts/train_cascade_anp_opt.py --data data/alpaca_5m.parquet --epochs 50 \
		--hidden-dim 128 --batch-size 16 --grad-accum 4 --use-amp

test-rust:
	cd engine-rust && cargo test

backtest:
	python scripts/backtest_cascade.py --tickers "AAPL,MSFT,GOOGL,META,AMZN,NVDA,TSLA,AMD" --capital 100000
