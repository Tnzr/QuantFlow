-- QuantFlow Database Initialization
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    ticker TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('buy', 'sell')),
    quantity INTEGER NOT NULL,
    price REAL NOT NULL,
    order_type TEXT NOT NULL CHECK (order_type IN ('market', 'limit')),
    status TEXT NOT NULL DEFAULT 'filled' CHECK (status IN ('filled', 'canceled', 'pending')),
    filled_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS signals (
    id SERIAL PRIMARY KEY,
    ticker TEXT NOT NULL,
    predicted_return REAL NOT NULL,
    sigma REAL NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('BUY', 'SELL', 'HOLD')),
    confidence REAL NOT NULL,
    generated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS equity_snapshots (
    id SERIAL PRIMARY KEY,
    equity REAL NOT NULL,
    cash REAL NOT NULL,
    positions_json JSONB,
    snapshot_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_trades_user ON trades(user_id, filled_at DESC);
CREATE INDEX IF NOT EXISTS idx_signals_ticker ON signals(ticker, generated_at DESC);
CREATE INDEX IF NOT EXISTS idx_equity_time ON equity_snapshots(snapshot_at DESC);
