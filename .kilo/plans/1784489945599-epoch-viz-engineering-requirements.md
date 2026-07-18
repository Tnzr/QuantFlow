# ML Training Engineering Requirements & Implementation Log

**Last Updated:** 2026-07-18 15:39 EDT  
**Status:** Active development — Phase: Training stabilization + Epoch-vis  

---

## §1 Architecture Decisions

### §1.1 LR Schedule (FIXED — commit bea24e9, re-fixed 719b023)
- **Requirement:** LR must monotonically decrease during training.
- **Bug (v1/v2):** `LinearLR(start_factor=1e-3)` caused LR to ramp UP from 1e-7 to 1e-4 over 5 warmup epochs before cosine descent.
- **Incomplete fix (v3):** Changed to `start_factor=0.01` — still an upward ramp but shorter.
- **Proper fix (v4):** Removed warmup entirely (`warmup_epochs=0`). Pure `CosineAnnealingLR(1e-4 → 1e-5)` over 100 epochs.
- **Verification:** Python trace confirms 1e-4 → 1e-5 monotonic.

### §1.2 Classification Collapse (FIXED — commit bea24e9)
- **Requirement:** Model must learn to discriminate all 3 event states, not just majority class.
- **Bug:** `PastStateHead` received only `fused_seq[:, -1, :]` (single last-timestep vector). With class imbalance (80% pre_event), classifier collapsed to always output pre_event.
- **Fix:** Changed to `fused_seq.mean(dim=1)` — mean-pool across all 60 timesteps.
- **Additional:** Added `WeightedRandomSampler` with inverse-frequency weights, per-class accuracy tracking.

### §1.3 Loss Components (STABILIZED — commit 719b023)
- **Requirement:** Composite loss must correctly penalize minority classes without over-suppressing majority.
- **v3 bug:** Class weights `(3.0, 0.5, 1.5)` + balanced sampler double-counted frequency correction → model overfit to inter (lowest freq).
- **Fix:** Reverted class weights to `(1.0, 1.0, 1.0)`. Sampler alone handles frequency. Added label smoothing (0.05) and gradient noise (eta=0.005).
- **Current weights:** cls=0.40, reg=0.25, distance=0.10, coherence=0.05, dynamics=0.02, uncertainty=0.08, direction=0.05, ranking=0.05

### §1.4 Forecast Metrics (ADDED — commit 719b023)
- **forecast_mae:** Absolute tau prediction error (days)
- **forecast_mae_temporal:** Weighted by `exp(-tau / τ0)` where `τ0=5.0` — emphasizes near-term accuracy
- **forecast_dir_acc:** Sign direction match rate

---

## §2 Current State — V4 Model

| Metric | Train | Val | Test |
|--------|-------|-----|------|
| Overall Acc | 55% | 40% | 49% |
| Inter F1 | — | — | 0.22 |
| Pre F1 | — | — | 0.59 |
| Onset F1 | — | — | 0.39 |
| Forecast MAE | — | — | 5.7 days |
| Temporal MAE | — | — | 0.78 days |

---

## §3 Remaining Issues & Next Tasks

### §3.1 Epoch-by-Epoch Visualization (TODO)
- **Requirement:** Generate forecast visualization at each epoch checkpoint showing:
  1. Price prediction vs actual with error bands
  2. Latent token activation heatmap across 4 cascade scales
  3. Head probability ribbon with buy/sell/hold region overlay
  4. Uptrend/downtrend detection with drawdown-based confirmation
- **Implementation:** Add `_generate_epoch_viz()` to Trainer, called at epoch boundaries.
- **Architecture changes needed:**
  - `MultiScaleTokenCascade.forward()` must return `fused_seq` (full sequence, not just fused)
  - `TemporalStateModel.forward()` must expose `fused_seq` in output dict
  - Extract scale activations from GRU outputs for token activation viz

### §3.2 Latent Token Visualization (TODO)
- **Data source:** `skips` list in `MultiScaleTokenCascade` — 4 GRU output tensors of shape `(B, T, hidden_dim)`
- **Visualization:** 4-row heatmap, one per scale S0→S3, x-axis=trading days, y-axis=hidden dimensions downsampled via PCA(8)
- **Require:** Modify cascade to return skips alongside fused tensor

### §3.3 Buy/Sell/Hold Signal Overlay
- Compute from head probabilities: buy when onset+pre > threshold AND tau < 7d, sell when inter > exit_threshold
- Overlay as colored regions behind probability ribbon
- Drawdown-based confirmation: flag buy signals that occur within drawdown > X% from local peak

### §3.4 Data Pipeline for Forecast Validation
- At each epoch, run model.predict() on a held-out validation ticker
- Overlay predicted tau against actual price movement windows
- Compute forward return windows aligned with tau predictions

---

## §4 Known Quirks

- **Forecast output has 3 values (not 1):** The model outputs `(B, 3)` for `future_forecast`. This is from the 3-class regression heads — must mean-pool or use the dominant-class value.
- **Direction accuracy = 100%:** Spurious — all forecasts are positive and all tau targets are positive, so sign match is trivial. Need to validate against actual forward returns instead of tau sign.
- **Onset recall declining (87%→76%):** Trade-off as inter/pre improve. Acceptable given overall F1 improvement.
- **Loss plateaus at epoch 3-4 then flattens:** Gradient noise helps but may need learning rate restart (cosine with warm restarts) for deeper convergence.
