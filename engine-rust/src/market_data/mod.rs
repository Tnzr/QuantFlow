//! Market data ingestion — Alpaca WebSocket + bar aggregation.

use std::collections::HashMap;

/// Aggregated OHLCV bar
#[derive(Debug, Clone)]
pub struct Bar {
    pub ticker: String,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: u64,
    pub timestamp_ms: u64,
}

/// In-memory ring buffer of recent bars per ticker
#[derive(Default)]
pub struct MarketDataCache {
    bars: HashMap<String, Vec<Bar>>,
    max_bars_per_ticker: usize,
}

impl MarketDataCache {
    pub fn new(max_bars: usize) -> Self {
        Self { bars: HashMap::new(), max_bars_per_ticker: max_bars }
    }

    pub fn push_bar(&mut self, bar: Bar) {
        let entry = self.bars.entry(bar.ticker.clone()).or_default();
        entry.push(bar);
        if entry.len() > self.max_bars_per_ticker {
            entry.drain(0..entry.len() - self.max_bars_per_ticker);
        }
    }

    pub fn get_bars(&self, ticker: &str, n: usize) -> Vec<&Bar> {
        self.bars.get(ticker)
            .map(|bars| bars.iter().rev().take(n).collect())
            .unwrap_or_default()
    }

    pub fn latest_price(&self, ticker: &str) -> Option<f64> {
        self.bars.get(ticker)
            .and_then(|bars| bars.last())
            .map(|b| b.close)
    }
}
