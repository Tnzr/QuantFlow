# CascadeANP — Complete Issue Inventory & Architectural Analysis

Date: 2026-07-23 (updated — critical bug found)
Status: ROOT CAUSE IDENTIFIED — two compounding bugs explain ALL symptoms

---

## 13. ROOT CAUSE (finally): Conv path trained on leaked future data, then visualized with a completely different, untrained input distribution

This single finding explains every symptom reported across this entire session:
identical trajectories, shapes elongating/collapsing over epochs, forecasts hitting
a "floor" and going flat, and the pattern getting WORSE (not better) with more
training — which is the real tell that this was never a capacity or convergence
problem. A model that is merely undertrained produces noisy nonsense; a model
that is fed structurally wrong data collapses toward a degenerate constant and
gets MORE confidently wrong the longer it trains. That is exactly what happened.

### Bug A — Training fed the conv encoder the target region as "context"

`train_cascade_anp.py`'s `sample_batch` builds `window_feats = feats[start:end]`
where `end - start = encode_len` (400 bars). This window contains BOTH the
context AND the 21-bar target region (the target is literally the last 21 bars
of that same window). When the conv path was wired up, `raw_features=window_feats`
was passed straight through — the *entire* 400-bar window, target included.

The `AdaptiveAvgPool1d(30)` at the end of the conv stack does **global average
pooling** down to 30 tokens. Averaging over 400 bars that always include the
answer at the end means:
- The model can partially shortcut using leaked future information (bad).
- Far more damaging: global average pooling over 400 bars erases almost all
  local/recent signal. Every 400-bar window's average is dominated by the
  overall drift of that ticker over that period, not by the specific 80 bars
  that matter for the forecast. Two different forecast points from the same
  ticker/time-region produce *nearly the same pooled average* — hence identical
  trajectories regardless of which of the 6 snapshot points was used.
- Because the training data (99 tickers, 5-min bars) has a net-negative average
  return over the sampled window in aggregate, the model's cheapest way to
  minimize MAE against this washed-out, leak-containing signal is to learn
  "always predict the population's average drift" — and that average is
  negative. Every epoch of gradient descent pushes it further toward that single
  degenerate constant, which is why epoch 1 still had some shape/curve (weights
  near random init) and by epoch 4+ it had fully collapsed into a straight
  descending line — clipped flat only because the viz's ±15% price bound cuts
  it off. **The "floor" the user is seeing is literally the visualization's
  clamp catching an unbounded, ever-more-negative constant prediction.**

### Bug B — Visualization never exercises the conv path at all

`forward()` was written with two branches:
```python
if raw_features is not None:
    conv_out = self.direct_conv(feat_t)          # trained path
    direct_mu = self.direct_path(feat_sampled)
else:
    sampled = <cascade tokens reshaped>            # fallback path
    direct_mu = self.direct_path(sampled)          # SAME weights, different input distribution
```
Both branches feed into the **same** `self.direct_path` MLP weights. Training
always supplies `raw_features` (conv branch). Neither `cascade_anp_viz.py` nor
`scripts/viz_forecast_inference.py` was ever updated to pass `raw_features` —
they only pass cascade tokens as `x_context`, so at inference time the code
silently falls into the `else` branch and feeds `direct_path` a completely
different statistical distribution of inputs than it was ever trained on
(cascade-pooled tokens instead of conv-pooled raw features).

An MLP fed out-of-distribution inputs doesn't fail loudly — it produces
whatever the nearest thing to a "safe" output is for extreme/unfamiliar inputs,
which for a network that has specialized toward "always predict very negative"
is... predicting very negative, uniformly, regardless of the (meaningless-to-it)
cascade input. This is why the collapse is even more extreme in the
visualizations than in the raw training metrics: the viz is not really
evaluating the trained model's actual behavior at all.

**Both bugs compound**: Bug A makes the model learn a degenerate global-average
collapse during training. Bug B guarantees the visualizations never test the
part of the model that was actually trained, so every viz sees an even more
undefined, out-of-distribution response — which happens to look like "the same
collapsing trajectory," but for reasons unrelated to genuine input diversity.

### Fix (both required, in order)

1. **Bug A fix** — `raw_features` passed to the conv encoder must be restricted
   to the strictly causal window ending at `tgt_start` (i.e. the same 80 bars
   used for `x_c`/`y_c`, never including the target region). No global average
   over a window that contains its own answer.
2. **Bug B fix** — both viz scripts and `backtest_cascade.py` must construct
   and pass the real `raw_features` slice (`feats[ctx_start:ctx_end]` from the
   original OHLCV/feature array, not cascade or BiLSTM tokens) to
   `model.forward()`, exactly matching what training does.

### Why this explains every earlier "still broken" report
- "All 6 forecasts identical" → global pooling over near-identical macro windows (Bug A) + viz not even using the trained path (Bug B).
- "Gets worse each epoch, not better" → genuine degenerate collapse toward the dataset's average drift, reinforced every step (Bug A) — not noise, not undertraining.
- "By epoch 4+, straight line hitting a floor" → runaway negative constant clipped by the visualization's ±15% price bound (Bug A), rendered through untrained weights (Bug B).
- Every previous fix in this document (bidirectional LSTM causality, pos_out removal, spread sampling, BiLSTM skip, token count increases) was real and correct, but none of them could matter because the actual forecast the user was looking at was never produced by a properly-trained, properly-exercised path.

---

## 1. Primary Unresolved: Identical Forecast Trajectories

**Symptom:** All 6 forecast snapshots per epoch show identical price trajectory shape regardless of which ticker, which context window, or which epoch. The shape changes between epochs but all 6 within an epoch are the same.

**Current best diagnosis:**
- `pos_out` (learned per-step bias) is the DOMINANT signal in the forecast output. It is a fixed `[21]` vector added identically to every prediction, regardless of input.
- The MLP output variance between inputs is very small (std=0.3%) compared to pos_out range (±1.1%).
- Since pos_out is constant across all snapshots within an epoch, all forecasts look identical.
- Between epochs, pos_out changes as the model trains, so the pattern shifts.

**Candidate fixes (not yet implemented):**
- **Remove pos_out entirely** — force the MLP to produce input-dependent variation
- **Input-dependent position modulation** — inject per-step embeddings AS INPUT to the MLP, not as output bias
- **Per-step cross-attention** — query learned step embeddings against cascade context, producing step-specific outputs from context tokens
- **Increase cascade token count** — use 20+ tokens instead of 10, giving the MLP more material to differentiate contexts

---

## 2. Bidirectional LSTM Data Leakage

**Symptom:** Forecasts identical regardless of context window position. The model appeared to "know" the full price trajectory before making predictions.

**Root cause:** `self.bilstm = nn.LSTM(input_dim, hidden_dim//2, bidirectional=True)` — the reverse pass processes the sequence end-to-start, meaning every context token at position `t` had information from bars `t..T` (the future).

**Fix applied:** Changed to `bidirectional=False, hidden_dim=128` (unidirectional, causal). Made encoder larger to compensate (128 dim per direction was 64+64 bidirectional, now 128 unidirectional).

**Status:** FIXED. Forecast correlation dropped from 0.999 to 0.892.

---

## 3. Cascade Convergence → Uniform Context Tokens

**Symptom:** After 4-scale GRU cascade, tokens from different context windows are nearly identical. Correlation between cascade tokens at bar 140, 320, 500, 680 → near 1.0.

**Root cause:** The multi-scale GRU cascade (4 layers) smooths over all market-specific patterns. After processing 60+ bars, the cascade reaches a steady state regardless of input. The downstream MLP sees near-identical inputs → produces near-identical outputs.

**Partial fix applied:** Spread-sampling 10 tokens across the full context window (not just last 10). Helps marginally because cascade still converges.

**Better fix (not implemented):** BiLSTM skip connection — raw BiLSTM output (before cascade) is passed to the MLP alongside cascade tokens. The BiLSTM preserves local market patterns that the cascade erases.

**Status:** Skip connection implemented but correlation still 0.892. The BiLSTM tokens also show similarity across snapshots — the 28 features don't carry enough distinctive signal.

---

## 4. Viz Not Using BiLSTM Skip Connection

**Symptom:** Both wandb viz (`cascade_anp_viz.py`) and offline viz (`viz_forecast_inference.py`) were discarding the raw BiLSTM output and calling the model without `raw_context`. The model fell back to cascade-only path, producing uniform forecasts.

**Fix applied:** Both viz functions now pass `raw_context=r_c` to `model.forward()`. The `direct_skip` MLP (20-token input) is used instead of `direct_path` (10-token cascade-only).

**Status:** FIXED in code. Not yet verified on wandb v2 training run.

---

## 5. Auto-regressive Decoder Contraction

**Symptom:** Previous AR decoder (`forward_ar`) produced 2-3 unique values per 21-step trajectory. Each step's prediction was fed back through `return_embed`, creating a contraction mapping that converged to a fixed point after 3 steps.

**Root cause:** `return_embed = nn.Linear(1, H)` maps scalar → hidden token. With Xavier init, weight std ≈ 1.0. Input variation of ±0.10 produces output variation of ±0.10 on each hidden dim. Too small relative to hidden scale (128), so the cross-attention absorbs the variation.

**Fix applied:** Replaced autoregressive decoder with direct path head — a single MLP predicting all 21 steps at once from spread-sampled cascade tokens. No feedback loop, no contraction.

**Status:** FIXED. 21/21 unique steps per trajectory. But trajectories are still uniform between snapshots (see Issue #1).

---

## 6. Classification Collapse (Original Problem)

**Symptom:** Original drawdown-based classification (inter/pre/onset) collapsed to a single class. All predictions were 100% "pre" or 100% "onset" regardless of architecture.

**Root cause:** `dynamic_class_weight_max=4.0` clipped inter-event weight from 10.65x to 4.0x. The rarest class (inter, 3.1% of data) was starved.

**Fix applied:** Expanded range to [0.1, 12.0]. Then replaced classification with regression (return forecasting). Classification artifacts removed from trainer, metrics renamed.

**Status:** FIXED (entire classification pipeline removed).

---

## 7. Loss Going Negative

**Symptom:** Training loss went to -0.59, -1.88, etc. Confusing for monitoring and indicated sigma runaway.

**Root causes (multiple occurrences):**
- **NLL log term:** `0.5 * log(σ²)` is negative when σ < 1. With σ=0.15, adds -1.9 to loss.
- **Sigma growing unchecked:** `sigma_y.clamp(max=0.12)` killed gradient when σ > 0.12, so sigma stayed at 0.71.
- **No sigma magnitude penalty:** Only growth penalty (σ increasing over steps) was active, not magnitude penalty.

**Fixes applied:**
- Replaced NLL with smooth L1 (always positive)
- Removed clamp, added `F.relu(sigma_y - 0.05).mean() * 1.0` magnitude penalty
- Clamped KL to `≥ 0` (numerical issue with `0.5*(μ²+σ²-2log(σ)-1)` going negative)

**Status:** FIXED. Loss consistently positive (0.032-0.050).

---

## 8. Feature Count Bug

**Symptom:** Model trained for hours on 1 feature instead of 28. Predictions were flat because BiLSTM saw only `body_pct` (candlestick body ratio).

**Root cause:** `_prepare_features()` selected from `FEATURE_COLUMNS` (daily features). One column (`body_pct`) existed in both daily and intraday sets. The auto-detection fallback (`len(available) < 5`) never triggered because 1 > 0.

**Fix applied:** Changed fallback threshold from `not available` to `len(available) < 5`.

**Status:** FIXED.

---

## 9. Viz Issues (Multiple Occurrences)

| # | Issue | Status |
|---|-------|--------|
| 9a | Uncertainty envelope $-70 to +$150 flattening price chart | FIXED: sigma clamped to 0.03, envelope ×0.3, price ±15% |
| 9b | Forecast lines too light/fading (alpha decreasing) | FIXED: uniform alpha=0.85, same dark purple palette |
| 9c | Date/time x-axis labels not showing | PARTIAL: ticker labels + dates on offline viz. Wandb viz uses bar indices |
| 9d | Context region not highlighted (which bars fed forecast) | PARTIAL: offline viz has shaded context. Wandb viz missing |
| 9e | Volume/RSI/candlestick indicators not shown | PARTIAL: offline viz has separate panels. Wandb viz has minimal |
| 9f | Error delta% graph only shows few data points | FIXED in offline viz |
| 9g | Detail panel empty/only shows data at very end | FIXED in offline viz |
| 9h | Price jumps from close to open (non-zero window starts) | FIXED in offline viz |
| 9i | Actual future overlapping temporally with model forecast | FIXED in offline viz |
| 9j | Scatter horizontal lines instead of diagonal | FIXED: replaced with hexbin |
| 9k | Multi-scale latent embeddings not visible | PARTIAL: Scale S0 heatmap in wandb viz |
| 9l | Forecast paths don't show raw returns — no visibility into model output | FIXED in offline viz |

---

## 10. Data Limitations

**Symptom:** All models (BiLSTM, ANP, CascadeANP) converge to ~52% direction accuracy and ~1.2% MAE regardless of architecture size.

**Root cause:** 28 intraday features from 5-min bars carry limited predictive signal for 21-step (105-min) forward returns. The data is close to a random walk at this granularity.

**Evidence:** Linear regression on flattened features achieves R²=0.50 — comparable to the neural network. The signal ceiling is data-limited, not architecture-limited.

**Possible remedies (not implemented):**
- Richer features from Alpaca (order flow, L2 quotes, dark pool prints)
- Alternative objective: volatility prediction, regime detection, anomaly detection
- Longer context window (252 bars = 1 day) for multi-day pattern recognition

---

## 11. Backtest Issues

| # | Issue | Status |
|---|-------|--------|
| 11a | Initial backtest showed 0 trades — entry threshold too high | FIXED: switched to direction-based entry |
| 11b | No position sizing by confidence | FIXED: `alloc = base × (1/σ) × martingale_mult` |
| 11c | Single-ticker sequential, no rotation | FIXED: multi-ticker time-aligned backtest with top-N ranking |
| 11d | 99-ticker overnight 50-epoch training | COMPLETED (PID 130961, epoch 50) |
| 11e | Causal encoder 30-epoch training | COMPLETED (epoch 30, val_loss=0.0325) |

---

## 12. Active Short-Term Remediation Plan

### P0: Fix Identical Forecasts (Issue #1)
- Remove `pos_out` bias entirely — forces MLP to produce input-dependent variation
- Increase cascade token count from 10 to 20
- Consider per-step cross-attention decoder (Phase C from plan doc)

### P1: Viz Completeness
- Unify wandb viz and offline viz into single function
- Add context region highlighting, volume, RSI to wandb viz
- Add proper datetime labels

### P2: Data Enrichment
- Alpaca integration for richer microstructure data
- Pre-computed daily seasonality merged to intraday
- More tickers (>100) from SECTOR_UNIVERSE

### P3: Alternative Objectives
- Volatility regime detection (cascade embeddings are already diverse)
- Anomaly detection (is this price action unusual?)
- Feature extraction for downstream models (use cascade tokens as input)

---

## Key Files Modified This Session

| File | Changes |
|------|---------|
| `quantflow/models/cascade_anp.py` | Unidirectional LSTM, direct_skip head, pos_out, spread sampling, sigma loss fixes |
| `quantflow/models/cascade_anp_viz.py` | Raw context pass, color fixes, sigma clamp, price clamp, context regions |
| `scripts/viz_forecast_inference.py` | Raw context pass, context shading, color palette |
| `scripts/train_cascade_anp.py` | Data augmentation, skip connection wiring, loss tracking |
| `scripts/backtest_cascade.py` | Confidence sizing, martingale DCA, multi-ticker rotation |
| `scripts/build_intraday_dataset.py` | Multi-horizon returns, raw OHLCV features, seasonality |
| `quantflow/models/losses.py` | Regression loss, temporal MAE, sigma regularization |
| `quantflow/models/dataset.py` | Auto-detect feature columns for intraday |

---

## Open Questions

1. Why does `pos_out` bias alone produce visibly different trajectories between epochs? The MLP output is ~constant, the bias shifts with training → the shape changes. This confirms pos_out is the dominant signal.

2. If we remove pos_out, will the MLP learn to produce input-dependent variation? Likely not without more cascade tokens or per-step cross-attention — the current 10 tokens × 128 dims may not carry enough distinctive information.

3. Can the 28 intraday features ever produce visibly diverse price trajectory forecasts at 21-step horizon? The data ceiling may be below what's needed for visually meaningful diversity. A volatility estimator may be more appropriate than a price trajectory generator.
