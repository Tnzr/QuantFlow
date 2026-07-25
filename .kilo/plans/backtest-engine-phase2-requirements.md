# Backtesting Engine — Phase 2 Engineering Requirements

Date: 2026-07-23
Status: REQUIREMENTS — not yet implemented
Depends on: CascadeANP v1.0 (frozen architecture)

---

## 1. Dynamic Position Sizing (Certainty-Aware Martingale DCA)

### Goal
Replace fixed-percentage martingale doubling with a continuous sizing function
that accounts for model confidence, forecast magnitude, and realized PnL.

### Requirements

**R1.1 — Confidence-Weighted Base Allocation**
```
base_alloc = capital × max_position_pct × confidence_factor(normalized_sigma)
```
- `confidence_factor(x) = 1.0 / (1.0 + α·x)` where `α` is a tunable aggressiveness parameter.
- Sigma is the model's aleatoric uncertainty (output by `direct_sigma`).
- As sigma → 0 (high confidence), allocation → `max_position_pct`.
- As sigma → ∞ (low confidence), allocation → 0.

**R1.2 — Forecast Magnitude Multiplier**
```
magnitude_factor = min(|predicted_return| / entry_threshold, 2.0)
```
- A forecast of +2% should get 2× the allocation of a forecast of +0.5%.
- Capped at 2.0 to limit exposure on extreme (potentially erroneous) forecasts.

**R1.3 — Consecutive Loss Recovery (Smart Martingale)**
```
recovery_multiplier = 1.0 + β × (consecutive_losses)^γ
```
- After 1 loss: 1.0 + β
- After 2 losses: 1.0 + β × 2^γ
- After 3 losses: 1.0 + β × 3^γ
- `β` = base recovery aggressiveness (default 0.25)
- `γ` = compounding curvature (default 1.5 — sub-exponential to prevent ruin)
- RESET to 1.0 after ANY winning trade (do not compound wins — only recover losses)
- Hard cap at 3.0 (never exceed 3× base allocation regardless of losses)

**R1.4 — Final Position Size**
```
shares = floor(base_alloc × magnitude_factor × recovery_multiplier / price)
```
- Must respect `max_trade` and `min_trade` config limits.
- Must respect available cash (cannot over-allocate).

---

## 2. Multi-Stock Portfolio Rotation

### Goal
At each time step, rank all available tickers by a composite signal strength
score and allocate capital to only the top-K.

### Requirements

**R2.1 — Composite Signal Strength**
```
signal_score(ticker) = w₁ × confidence_factor(sigma)
                     + w₂ × |predicted_return| / normalized_volatility(ticker)
                     + w₃ × (1.0 if existing position and direction unchanged else 0.0)
```
- `w₁, w₂, w₃` are tunable weights (default: 0.4, 0.4, 0.2).
- `normalized_volatility` = rolling 20-bar volatility / population median volatility. Less volatile stocks get higher allocation for the same forecast magnitude.
- Bonus for holding an existing position (reduces turnover, transaction costs).

**R2.2 — Top-K Selection with Constraints**
- Rank all tickers by `signal_score` descending.
- Select top `max_positions` (default 5).
- BUT: skip any ticker already at `max_exposure_per_ticker` (default 25% of portfolio).
- BUT: skip any ticker where `predicted_return` direction conflicts with existing position without a clear exit signal.
- Entry only occurs if `cash_available > min_trade × 2` (always keep a buffer).

**R2.3 — Position Replacement (Rotation)**
- When a ticker drops out of top-K while another enters:
  - Close the existing position (normal exit, not emergency).
  - Open the new position with freed capital.
- When a ticker stays in top-K but direction FLIPS:
  - Immediate exit (signal-driven).
  - Do NOT immediately re-enter — wait for next bar (avoid whip).

---

## 3. Dynamic Stop Loss & Take Profit

### Goal
Replace fixed-percentage stops (3% SL, 5% TP) with model-aware dynamic stops
that adapt to both market conditions and model confidence.

### Requirements

**R3.1 — Certainty-Aware Stop Loss**
```
stop_loss_pct = base_stop × (1.0 + δ × sigma / sigma_baseline)
```
- `base_stop` = minimum stop loss (default 2%).
- `sigma` = model's aleatoric uncertainty for this forecast.
- `sigma_baseline` = rolling average sigma across all tickers.
- `δ` = sensitivity parameter (default 2.0).
- When sigma is HIGH (model uncertain): widen the stop to avoid noise exits.
- When sigma is LOW (model confident): tighten the stop to protect capital.
- Hard bounds: [1%, 8%] — never stop out on <1% or let a position lose >8%.

**R3.2 — Volatility-Aware Take Profit**
```
take_profit_pct = max(base_tp, predicted_return × tp_capture_ratio)
```
- `base_tp` = minimum take profit (default 3%).
- `tp_capture_ratio` = what fraction of the predicted return to capture (default 0.5).
- If the model predicts +4%, take profit at +2%.
- If the model predicts +1%, take profit at minimum (3%).
- If the actual return exceeds the predicted return and still has positive sigma, hold longer (dynamic: use trailing stop instead of hard TP).

**R3.3 — Trailing Stop with Sigma-Adaptive Trail**
```
trail_distance = max(1%, 2 × sigma × price)
```
- For a $300 stock with sigma = 0.02 (2%): trail = $300 × 0.04 = $12 (4%).
- For sigma = 0.01 (1%): trail = $300 × 0.02 = $6 (2%).
- The trail distance ADAPTS each bar based on CURRENT sigma.
- Minimum trail: 1% (protect gains). Maximum trail: 8% (don't give back the farm).

**R3.4 — Time-Based Exit (Max Hold)**
```
max_hold_bars = base_max_hold × (1.0 + epsilon × sigma / sigma_baseline)
```
- `base_max_hold` = default max hold (default 40 bars).
- `epsilon` = time flexibility (default 1.0).
- Higher uncertainty → longer allowed hold (give the trade more time).
- Lower uncertainty → shorter allowed hold (the model is confident — if price isn't moving, cut it).
- Hard cap: 80 bars.

---

## 4. Entry Confirmation Logic

### Goal
Separate the model's forecast from the actual trade entry decision. The model
predicts a return; the entry engine decides whether to act on it.

### Requirements

**R4.1 — Entry Gate Conditions**
A trade is entered ONLY when ALL of the following are true:
1. `|predicted_return| ≥ entry_threshold` (default 0.3%).
2. `sigma ≤ max_entry_sigma` (default 0.08) — don't enter on high uncertainty.
3. `signal_direction == predicted_direction` for the last N bars (default N=2) — confirmation: the model must agree with itself for at least 2 consecutive bars.
4. `volume/price action confirmation` — if available, the real-time price must be moving in the forecast direction (e.g., current bar's return has the same sign as predicted return). If not available (historical backtest), skip.

**R4.2 — Entry Timing**
- Do NOT enter immediately at the forecast bar. Wait for `entry_delay_bars` (default 1) to confirm the signal.
- During the delay, re-evaluate: if the signal weakens or flips, abort entry.
- This reduces false entries on transient signals.

**R4.3 — Partial Entry**
- For large allocations (>15% of portfolio): split into 2-3 smaller entries over consecutive bars.
- `entry_tranches = max(1, floor(allocation / max_position_pct))`.
- Reduces market impact and smooths entry price.

---

## 5. Performance Metrics & Reporting

### Requirements

**R5.1 — Required Metrics**
- Total return (%), CAGR, Sharpe ratio, Sortino ratio, max drawdown (%, duration).
- Win rate (%), profit factor (gross gains / gross losses), expectancy (avg win × win_rate - avg loss × loss_rate).
- Per-ticker breakdown: total PnL, win rate, avg hold time, best/worst trade.
- Per-exit-reason breakdown: what % of trades exited due to stop_loss vs trailing vs take_profit vs max_hold vs model_exit.
- Equity curve with drawdown overlay (matplotlib figure).
- Rolling Sharpe (60-bar window) to show performance stability.
- Monthly returns heatmap.

**R5.2 — Confidence-Calibrated Metrics**
- Bucket trades by `sigma` at entry: low-σ (<0.03), mid-σ (0.03-0.06), high-σ (>0.06).
- Report win rate and avg return per bucket. EXPECTATION: low-σ trades have higher win rate and higher avg return. If not, the sigma calibration is broken.
- This is the primary DIAGNOSTIC tool for model quality.

**R5.3 — Benchmark Comparison**
- Compare against buy-and-hold of an equal-weight basket of the traded tickers.
- Compare against a simple momentum strategy (e.g., buy top-3 by recent return).
- Output: relative alpha, information ratio, tracking error.

---

## 6. Implementation Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   BacktestEngine v2                      │
├─────────────────────────────────────────────────────────┤
│  SignalGenerator                                         │
│    ├─ generate_signals(tickers, bar) → List[Signal]     │
│    └─ Signal: {ticker, pred_return, sigma, direction,    │
│                confidence, magnitude_score}              │
├─────────────────────────────────────────────────────────┤
│  PortfolioManager                                        │
│    ├─ rank_and_select(signals) → List[Allocation]       │
│    ├─ compute_allocation(signal, cash, history) → shares│
│    └─ Allocation: {ticker, shares, entry_price, reason}  │
├─────────────────────────────────────────────────────────┤
│  RiskManager                                             │
│    ├─ compute_stop_loss(position, model_state) → price  │
│    ├─ compute_take_profit(position, model_state) → price│
│    ├─ compute_trailing_stop(position, model_state) → $  │
│    └─ check_exit_conditions(position, current_bar) → str│
├─────────────────────────────────────────────────────────┤
│  PerformanceTracker                                      │
│    ├─ record_trade(trade) → update metrics              │
│    ├─ equity_curve() → List[float]                      │
│    └─ report() → BacktestReport                         │
└─────────────────────────────────────────────────────────┘
```

---

## 7. Configuration Defaults

```python
@dataclass
class BacktestConfig:
    # Capital
    capital: float = 100_000
    max_position_pct: float = 0.20       # max 20% of portfolio per ticker
    max_positions: int = 5               # max concurrent positions
    max_exposure_pct: float = 0.80       # max 80% of portfolio deployed

    # Sizing
    confidence_alpha: float = 2.0        # alpha in confidence_factor
    magnitude_max_mult: float = 2.0      # cap on magnitude multiplier
    recovery_beta: float = 0.25          # martingale base aggressiveness
    recovery_gamma: float = 1.5          # martingale compounding curvature
    recovery_cap: float = 3.0            # hard cap on recovery multiplier

    # Entry
    entry_threshold: float = 0.003       # 0.3% predicted return minimum
    max_entry_sigma: float = 0.08        # don't enter if sigma > 8%
    entry_confirmation_bars: int = 2     # consecutive bars agreeing
    entry_delay_bars: int = 1            # wait N bars before entering
    entry_tranches: int = 1              # split large allocations

    # Stops
    base_stop_loss: float = 0.02         # 2% minimum stop
    stop_sensitivity: float = 2.0        # delta in stop_loss formula
    base_take_profit: float = 0.03       # 3% minimum take profit
    tp_capture_ratio: float = 0.5        # capture 50% of predicted return
    trail_min: float = 0.01              # minimum trail distance
    trail_max: float = 0.08              # maximum trail distance
    max_hold_bars: int = 40              # base max hold
    max_hold_max: int = 80               # absolute max hold

    # Signal ranking
    signal_w_conf: float = 0.4           # weight on confidence
    signal_w_mag: float = 0.4            # weight on magnitude
    signal_w_pos: float = 0.2            # weight on existing position

    # Costs
    slippage_pct: float = 0.0005         # 5bps slippage
    transaction_cost_pct: float = 0.001  # 10bps transaction cost
```

---

## 8. Acceptance Criteria

1. **AC1**: On the 140-ticker Alpaca dataset, the backtest completes in < 30 seconds.
2. **AC2**: Confidence-bucketed metrics show monotonically decreasing win rate as sigma increases.
3. **AC3**: Dynamic stop loss produces fewer -5%+ losing trades than fixed 3% stop.
4. **AC4**: Smart martingale achieves higher Sharpe than equal-weight allocation on the same signals.
5. **AC5**: Entry confirmation (N=2 bars) reduces total number of trades by ≥20% while maintaining or improving win rate.
6. **AC6**: The equity curve matches the profit factor (no hidden biases).
7. **AC7**: Per-ticker breakdown reveals no single ticker contributing >50% of total PnL.

---

## 9. Future: RL Agent Integration (Phase 3)

The backtesting engine is designed to eventually serve as the environment for a
Reinforcement Learning agent:

- **State**: portfolio allocation, per-position PnL/age, current model signals, recent price action.
- **Action space**: {BUY, SELL, HOLD, INCREASE, DECREASE} per ticker.
- **Reward**: Sharpe ratio over a rolling window + drawdown penalty.
- **Policy**: PPO or SAC, trained offline on historical data, fine-tuned live.

The signal ranking and risk management modules from this Phase 2 engine serve
as the baseline policy for RL pre-training and as safety constraints during
live deployment.

---

*Last updated: 2026-07-23*
