# Visualization & Backtest Engine Overhaul Plan

**Status:** Implementation-ready  
**Date:** 2026-07-17  

## Problem Summary

The current backtest engine runs tickers sequentially (ticker-by-ticker), producing trades that appear as a single chronological chain. Visualizations inherit this flaw, causing:

1. **Cumulative return inflated to billions** — `cumprod` chains 1,620 trades back-to-back, each compounding on the last. Real portfolio returns are ~2x, not 1e6x.
2. **Portfolio allocation is a static pie** — no time-series allocation data exists because the engine processes tickers one after another, not simultaneously.
3. **Probability waterfall static/flat** — CLI generates signals from SPY only (fallback code), not from the actual backtested tickers.
4. **Trade timeline y-axis broken** — cumprod chain places cumulative return line in millions of percent.
5. **Training monitor mostly correct** — loss/accuracy/confusion work; forecasting viz needs per-state breakdown.

## Architecture Decision: Simultaneous Multi-Ticker Backtest

Replace sequential `_backtest_ticker()` with a unified timeline approach:

1. **Shared date axis** — fetch OHLCV for all tickers, align to a common date range (`start_date` → today)
2. **Daily model inference** — compute features + model predictions for each ticker on each day
3. **Top-N selection** — each day, rank tickers by risk score `(pre_event + onset * 1.5)`; allocate 1/N equally to top N (default N=5)
4. **Portfolio state machine** — track `{ticker: (entry_price, entry_date, highest_price)}` per position; apply stop-loss/trailing/profit exit rules
5. **Daily equity tracking** — compute portfolio value = sum(positions × current_price) + cash; log daily

### Data Structures Added to `BacktestResult`

```python
@dataclass
class BacktestResult:
    # existing fields...
    daily_allocations: pd.DataFrame    # (dates × tickers) allocation matrix in dollars
    daily_cash: pd.Series              # cash balance per day
    daily_equity: pd.Series            # total portfolio value per day (dates index)
    daily_signals: Dict[str, pd.DataFrame]  # ticker -> signal timeseries
    active_positions: List[Dict]       # snapshots of position state over time
```

## Implementation Tasks

### Phase 1: New Backtest Engine (`backtest_engine.py` rewrite)

**Task 1.1:** Replace `_backtest_ticker()` with `_backtest_portfolio()`
- Accept `tickers` list, build shared `date_range`
- Pre-compute features for all tickers over the shared timeline (batch inference)
- Run daily: rank tickers by risk score, enter new positions if cash available + signals strong, track position P&L
- Exit logic: trailing stop (3%), hard stop (5%), take-profit (10%), max hold (21d), model exit signal (inter > threshold)
- Cash earns 4%/year annualized (daily compound)

**Task 1.2:** Return enriched `BacktestResult` with:
- `daily_allocations`: `pd.DataFrame(index=dates, columns=tickers)` — dollar amount in each ticker each day
- `daily_cash`: `pd.Series(index=dates)` — uninvested cash
- `daily_equity`: `pd.Series(index=dates)` — total portfolio value
- `daily_signals`: `Dict[ticker, DataFrame]` with columns `prob_inter/pre/onset`, `forecast_tau` per day
- `trade_log`: `List[Dict]` with entry/exit dates, ticker, return, confidence per trade

**Task 1.3:** Fix `equity_curve` to use actual date-indexed `pd.Series` from `daily_equity` (not cumprod chain)

**Task 1.4:** Update `BacktestResult` dataclass with new fields

### Phase 2: Visualization Rewrite (`visualization.py`)

**2.1 — Dashboard (6-panel, 22×16")**
| Position | Panel | Content |
|----------|-------|---------|
| Row 1 L | Equity Curve | Portfolio value over time on date axis, trade entry/exit markers |
| Row 1 R | Per-Ticker Equity | Overlaid equity curves for top 5 tickers, each starting at $1 |
| Row 2 L | Drawdown | Portfolio drawdown % over time |
| Row 2 R | Head Probabilities | Date-indexed waterfall (inter/pre/onset) — from portfolio-aggregated signals |
| Row 3 L | Allocation Timeline | Stacked area chart: each ticker's $ allocation + cash over time |
| Row 3 R | Metrics Table | Sharpe, Sortino, CAGR, wins, avg return, max DD |

**2.2 — Allocation Monitor (standalone, 18×10")**
- 3-panel: allocation stacked area (top), per-ticker P&L bars (middle), rolling Sharpe (bottom)
- Date x-axis on all panels
- Cash balance overlaid as dashed line

**2.3 — Training Monitor (existing, 8-panel, fix only)**
- Already has loss/accuracy/LR/confusion/state-transitions/forecast-error
- Fix: `_plot_forecast_error` uses `preds.get("true_tau", 10)` which is always 10 — use actual tau label from test set
- Fix: `_plot_state_transitions` show combined true vs pred states with color differentiation

**2.4 — Inference Monitor (standalone, 12×8")**
- Per-ticker signal waterfall over date axis
- Tau forecast with uncertainty band
- State transition ribbon at bottom (colored bar by predicted state)

### Phase 3: CLI Integration

**Task 3.1:** `cmd_ai_backtest` — remove the fallback SPY signal generation code; signals now come directly from `BacktestResult.daily_signals`

**Task 3.2:** Pass `result.daily_signals` directly to `generate_backtest_visualizations()`

**Task 3.3:** `cmd_train` — already calls `generate_training_visualizations()`; ensure predictions include `true_state` column for confusion matrix

### Phase 4: Cleanup

**Task 4.1:** Remove dead code: `_plot_portfolio_alloc` (static pie), `_plot_trade_timeline` (old cumprod bar chart), `_plot_trade_distribution` (replaced by allocation P&L), `plot_portfolio_allocation` (old static), `plot_token_activation_waterfall` (unused)

**Task 4.2:** Remove unused exports from `__init__.py`

**Task 4.3:** Remove `plot_inference_monitoring`, `plot_portfolio_timeline`, `plot_token_activation_waterfall` from exports

## Files Modified
| File | Change |
|------|--------|
| `quantflow/models/backtest_engine.py` | Rewrite `run()` for simultaneous multi-ticker; new `_backtest_portfolio()`; enriched `BacktestResult` |
| `quantflow/models/visualization.py` | Full rewrite — remove 4 dead functions, add 3 new ones, fix training monitor |
| `quantflow/models/__init__.py` | Update exports |
| `quantflow/cli.py` | Simplfy `cmd_ai_backtest` viz block (remove 50 lines of fallback signal gen) |

## Validation
1. Run backtest on 15 tickers — verify ~2x total return (not 1e6x), verify daily equity is smooth
2. Verify allocation timeline shows portfolio rotating between tickers over time
3. Verify training monitor renders after full-universe training
4. Verify all 6 panels of dashboard have date x-axis labels
5. Verify per-ticker equity curves overlay correctly

## Risks
- **Performance:** Daily inference for N tickers × D days = N×D model forward passes. With 15 tickers × 1250 days = 18,750 passes. Batched inference reduces this to ~36 batches of 512. Acceptable.
- **yfinance rate limiting:** Pre-fetching all tickers sequentially may hit rate limits. Add `time.sleep(0.5)` between downloads.
- **Feature computation:** `_build_features_at` is still per-row — deferred to future perf optimization (not in scope).
