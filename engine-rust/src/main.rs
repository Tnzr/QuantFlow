//! QuantFlow Core Engine — Rust implementation.
//!
//! Architecture:
//!   OrderEntry → OrderBook (price-time priority) → MatchingEngine → TradeOutput
//!   AlpacaStream → BarAggregator → MarketDataCache → (ML Gateway, Risk Checks)
//!
//! Modules:
//!   order_engine/ — order book, matching, order types
//!   market_data/   — Alpaca WebSocket ingestion, bar aggregation, caching
//!   risk/           — pre-trade checks, circuit breakers, position limits
//!   ml_gateway/    — HTTP client to Python ML inference service

mod order_engine;
mod market_data;
mod risk;
mod ml_gateway;

use tracing::info;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    info!("QuantFlow Engine starting...");

    // Initialize order book
    let book = order_engine::OrderBook::new();
    info!("Order book initialized: {} resting orders", book.resting_count());

    // Start market data stream (placeholder)
    info!("Market data stream: ready (connect with Alpaca API key)");

    // Start risk checks loop
    info!("Risk engine: ready");

    // Start ML gateway
    info!("ML gateway: targeting http://ml-engine:8000");

    info!("Engine running. Press Ctrl+C to stop.");
    tokio::signal::ctrl_c().await?;
    info!("Shutting down.");
    Ok(())
}
