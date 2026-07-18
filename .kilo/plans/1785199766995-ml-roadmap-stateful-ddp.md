# QuantFlow ML Roadmap — Multi-Modal to Portfolio RL

**Date:** 2026-07-18  
**Status:** Engineering requirements — Phase: Stateful training validation  
**Supersedes:** `.kilo/plans/1785186141599-training-pipeline-methodology-compliance.md` (incorporated)

---

## §0 Methodology Compliance Status

### Fixed (commit `88adb46`)
- [x] Chronological per-ticker ordering → `sorted(tickers)`, `drop_last=True`
- [x] Stateful GRU hidden state propagation → `stateful=True`, ticker-keyed state dict
- [x] Remove WeightedRandomSampler → loss weights compensate (inter=2.0, pre=0.3, onset=1.0)
- [x] `_to_device` returns `(features, labels, tickers)` for ticker-aware state management
- [x] Engineering requirements doc tracks violations and remediation

### Pending verification
- [ ] Full-scope training (100 epochs, full universe, DDP) with wandb
- [ ] Loss curve must decrease monotonically (not plateau epoch 3-4)
- [ ] Per-class accuracy curves must diverge (not collapse to single class)
- [ ] Forecast MAE must improve over epochs
- [ ] Hidden state norms must stay bounded (not explode >100 or collapse <0.01)

---

## §1 Multi-Modal Attention Architecture (Phase: Design)

### Current State
The architecture (§3 of methodology) defines a `ModalityFusionGate` that accepts multiple encoded modality streams and produces a fused token `u_t`. However, the current implementation only feeds a single modality — a flat 21-feature vector through a single BiLSTM encoder.

```
Current:  [21 features] → BiLSTM encoder → MultiScaleCascade → 4 heads
Target:   [Price stream] → Encoder_A ─┐
          [Volume stream] → Encoder_B ─┤→ FusionGate → Cascade → 4 heads
          [Seasonality]   → Encoder_C ─┤
          [Sector index]  → Encoder_D ─┘
```

### Modality Decomposition
| Modality | Source | Dim | Encoder |
|----------|--------|-----|---------|
| Price technicals | FEATURE_COLUMNS (14 indicators) | 14 | BiLSTM(T=60) |
| Volume & liquidity | OBV, VWAP, volume ratio, ATR | 4 | BiLSTM(T=60) |
| Seasonality vectors | day_of_week, month_of_year, days_to_earnings, week_of_quarter | 4 | MLP embedding |
| Correlation context | Sector ETF daily return, sector avg event_state | 2 | 1D CNN(T=20) |
| Sentiment (future) | News volume, social sentiment score | 2 | Transformer |

### Sector Correlation Feedback
Rather than a static correlation matrix, use lightweight online signals:
- **Sector ETF daily return**: `XLE, XLK, XLV, XLF` daily close % change (5 values)
- **Sector average event-state**: Mean `prob_pre_event` across sector peers
- **Cross-ticker tau-coherence**: If AAPL tau=5d and sector peers avg tau=15d → anomaly flag

### Seasonality Features (Pre-computed, no training cost)
| Feature | Encoding | Rationale |
|---------|----------|-----------|
| Day of week | One-hot(5) | Monday/Friday effects |
| Month of year | One-hot(12) or sin/cos | January effect, summer doldrums |
| Days to earnings | Scalar (0–90, clamped) | Earnings anticipation drift |
| Days to FOMC | Scalar | Macro event anticipation |
| Days to options expiry | Scalar (0–30) | Gamma positioning effects |
| Quarter phase | Scalar (0.0–1.0) | Institutional rebalancing cycles |

These are `< 20 additional features`, pre-computed at dataset construction time, with zero runtime cost.

---

## §2 Latent Tokens as Learned Candlestick Patterns (Phase: Research)

### Hypothesis
The MultiScaleTokenCascade processes tokens at 4 temporal resolutions (S0–S3). At each scale, GRU cells learn to detect recurring patterns in the 128-dim latent space. These learned patterns are analogous to candlestick formations — specific combinations of price, volume, and momentum that precede directional moves.

### Scale → Pattern Mapping (Hypothetical)
| Scale | Temporal Context | Equivalent Candlestick Patterns |
|-------|-----------------|--------------------------------|
| S0 | 1–5 days | Doji, Hammer, Engulfing (single-bar patterns) |
| S1 | 1–3 weeks | Three White Soldiers, Morning/Evening Star |
| S2 | 1–3 months | Head & Shoulders, Double Top/Bottom |
| S3 | 3–12 months | Cup & Handle, Broadening Formations, Regime shifts |

### Validation Approach
1. Extract latent token activations for known pattern dates from labeled datasets
2. Compare activation vectors for matching patterns across different tickers
3. If activation similarity > threshold for same pattern type → latent tokens ARE encoding patterns
4. Use Integrated Gradients to identify which input features drive each latent channel

### Downstream Use
- Latent token activation heatmaps (already implemented in epoch viz Panel 3)
- Pattern recognition: "S0 channel 47 spiked → high probability of bullish engulfing"
- Confidence calibration: "Pattern detected with S0=0.92 activation but S2 contradicts → lower confidence"

---

## §3 Signal Generation: Multi-Scale Trend Segmentation

### Problem
The current 3-state classifier (inter/pre/onset) is coarse — it doesn't capture the fine structure of uptrend/downtrend transitions that candlestick patterns detect.

### Proposed: Continuous Trend Segmentation with Padding
```
Compute rolling max/min over N-day windows
Define:
- Uptrend zone:  price > rolling_max * (1 - padding_pct)   [above -2% from peak]
- Downtrend zone: price < rolling_min * (1 + padding_pct)  [below +2% from valley]
- Chop zone:      between uptrend and downtrend              [2% padding band]
```

This creates shaded zones that a) confirm trends aren't broken by noise, and b) signal reversals when price crosses the padding band.

### Multi-Scale Detection
Run the trend segmentation at 3 scales (5d, 21d, 63d). Alignment of trends across scales = strong signal. Divergence = chop/consolidation.

---

## §4 Portfolio Management Algorithm — Current State Assessment

### Current Capabilities (backtest_engine.py)
- Simultaneous multi-ticker backtest with shared date axis
- Top-N daily rebalancing based on `risk = prob_pre + 1.5 * prob_onset`
- Exit logic: trailing stop (3%), stop loss (5%), model exit, max hold (21d), forecast exit
- Daily allocation tracking per ticker

### Current Limitations
- Entry signals are purely threshold-based (risk >= 0.60) — no confirmation logic
- No portfolio-level risk management (VaR, correlation, max sector exposure)
- No drawdown-aware position sizing — same allocation whether market is calm or volatile
- No breakout/reversal confirmation — enters immediately on signal

### Required: Confirmation-Based Entry Policy
```
Buy signal candidate generated when risk >= entry_threshold
→ WAIT for confirmation:
  - Price must be above 5d moving average (breakout direction aligned)
  - Volume must be above 20d average (conviction behind move)
  - Drawdown from local peak < 3% (not buying into a dip)
  - Sector ETF is NOT in drawdown > 2% (sector headwind filter)
If all 4 confirm → enter position
If any fail → queue signal, re-evaluate next day
```

### Phase 2: Reinforcement Learning Portfolio Agent
- **State space**: portfolio weights, ML signals (probabilities + tau), sector context, drawdown metrics
- **Action space**: continuous allocation weights per ticker + cash
- **Reward**: Sharpe ratio × (1 − drawdown penalty) × (1 − turnover penalty)
- **Training**: PPO with curriculum (start with 5 tickers, scale to full universe)
- **Constraint**: Chronological stateful context preserved (GRU hidden states ARE the RL state encoder)

---

## §5 Multi-GPU DDP Requirements for Chronological+Stateful Training

### Constraint
DDP must NOT violate §4.3 of methodology:
1. Each GPU processes a distinct subset of tickers
2. Within each GPU, tickers are chronological and stateful
3. No random shuffling across GPUs or within GPUs
4. Hidden states are per-GPU, per-ticker — no cross-GPU state sync needed
5. Gradient all-reduce happens after each batch

### Implementation Plan
- **DistributedSampler**: `shuffle=False` — presents tickers in sorted order, partitioned across GPUs
- **State management**: Each GPU maintains its own `hidden_states: dict[ticker → list[GRU_states]]`
- **No cross-GPU state sync**: Each ticker is owned by exactly one GPU (deterministic partitioning)
- **Gradient sync**: Standard `DistributedDataParallel` all-reduce after backward pass

```
GPU 0: tickers[0:N/4]   → chronological batches → hidden_states[GPU0]
GPU 1: tickers[N/4:N/2] → chronological batches → hidden_states[GPU1]
GPU 2: tickers[N/2:3N/4] → chronological batches → hidden_states[GPU2]
GPU 3: tickers[3N/4:]    → chronological batches → hidden_states[GPU3]
        ↓ all-reduce gradients after each batch
```

---

## §6 Training Launch Plan

### Configuration (full universe)
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Architecture | BiLSTM | Proven baseline, compatible with stateful |
| Hidden dim | 128 | Balanced capacity vs memory |
| Epochs | 100 | Full cosine schedule |
| Batch size | 64 | Per-GPU, all-reduce across 4 GPUs → effective 256 |
| LR | 1e-4 → 1e-5 | Cosine decay, no warmup |
| Lookback | 60 | Trading days (~3 months) |
| Stateful | True | Required per methodology |
| Keep hidden | True | Carry GRU states across epochs |
| DDP | 4 GPUs | `torchrun --nproc_per_node=4` |
| Wandb | quantflow project | Full metric + viz logging |
| Data | full_universe_5y.parquet | 140 tickers, 31,655 snapshots |

### Success criteria
- [ ] Loss curve: initial drop, then steady decline (not plateau at epoch 3-4)
- [ ] Per-class accuracy: 3 distinct curves with inter > 20% (not collapsed)
- [ ] Forecast MAE: decreasing from initial ~10d toward < 5d
- [ ] Epoch viz: all 4 panels generated at epochs 1, 5, 10, 15, 20
- [ ] No CUDA OOM or hidden state explosion
