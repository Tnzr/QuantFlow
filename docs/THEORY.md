# QuantFlow Theory

Risk-first principles
- Position sizing ≤ 1% risk per trade.
- Use ATR/time stops; avoid earnings risk for weeklies.
- Prefer defined-risk structures when account is small.

Signals by horizon
- 1w: RSI>50 + close>SMA20; exit on SMA20 breach or 5 trading days.
- 1m: close>SMA50/SMA200 and rising MAs; earnings proximity penalty.
- 3–12m: close>SMA200 and low realized vol; trend persistence.

Exits
- Trailing ATR stops (2.0–3.5x) + SMA20 hard stop + time stop.

Options picker
- Budget + liquidity filters (OI/volume, spread).
- Delta targeting (~0.35) via Black–Scholes for convexity/time-decay balance.

Seasonality
- Day-of-year returns average, with earnings proximity overlay.

Roadmap
- Sentiment and news aggregation for LLM overlay once RAG dataset grows.
- Portfolio-aware alerts and journaling.
