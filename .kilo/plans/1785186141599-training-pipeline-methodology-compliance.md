# Training Pipeline Audit — Methodology Compliance Gap Analysis

**Date:** 2026-07-18  
**Severity:** Critical — invalidates all training results  
**Root Cause:** Data loading violates fundamental methodology requirements documented in §4.3, §8.2 of `ML_DL_Architecture_Methodology.md`

---

## §1 Violations Found

### Violation 1: Random sampling breaks chronological ordering (§8.2, §4.3)

**Methodology requires:**
> Samples must be presented in temporal order. For multi-sequence datasets: group by sequence/session identifier, sort by timestamp, present groups in consistent order. No random shuffling across groups.
> State is carried across data-loader batch boundaries during training. This requires chronological data ordering.

**Current code (`prepare_dataloaders`):**
```python
# dataset.py:189-192
tickers = df["ticker"].unique()
np.random.seed(seed)
np.random.shuffle(tickers)   # ← RANDOM TICKER ORDER
```

```python
# dataset.py:217
train_sampler = train_ds.get_balanced_sampler()   # WeightedRandomSampler
# ← RANDOM SAMPLE ORDER WITHIN TICKERS
```

**Violation:** Tickers are shuffled, and within tickers samples are drawn randomly via WeightedRandomSampler. This means epoch 3 batch 47 might contain a random Microsoft window from 2023, followed by a random JPMorgan window from 2025, followed by a pre-event Apple window from 2021 — the model has NO temporal continuity whatsoever. The BiLSTM encoder is fed scrambled windows and the GRU states are reset every batch with `stateful=False`.

**Impact:** The model cannot learn temporal dynamics. It's effectively a `window → 3-class` classifier with no sequential context. This explains:
- The rapid plateau after ~3 epochs (model memorizes the static window→class mapping)
- The classification collapse (no sequential signal to distinguish pre-event from onset)
- The 49% accuracy ceiling (random-window classifier ceiling on imbalanced classes)
- The flat forecast tau (no trend information carried between samples)

### Violation 2: No hidden state propagation (§4.3, §8.2)

**Methodology requires:**
> State is carried across data-loader batch boundaries during training. Hidden state detach/transfer between batches. Implement hidden state managers that track (h, c) tuples per sequence.

**Current code (`TrainingConfig`):**
```python
stateful: bool = False          # ← NEVER ENABLED
keep_hidden_across_epochs: bool = True   # ← meaningLESS without stateful=True
```

**Violation:** The `stateful` flag is never set to True. Every batch starts with zero GRU state. The 4-scale cascade's hidden states are reset to zero for every forward pass. The `h_prev` parameter never carries actual state.

**Impact:** Even if samples were chronological, the model would still have no memory. The GRU cells in the `MultiScaleTokenCascade` operate as stateless feature extractors — they process 60-timestep windows independently with zero-carried state.

### Violation 3: Balanced sampling contradicts stateful requirement (§8.2)

**Methodology states:**
> Balanced sampling (oversample minority, undersample majority) is a valid class imbalance strategy.

**But §4.3 also requires:**
> Sequence-level (not sample-level) sampling.

**Conflict:** The `WeightedRandomSampler` draws samples independently with replacement, completely destroying any temporal order. The methodology allows class balancing but requires it to happen *at the sequence level*: oversample entire sequences (ticker histories) of minority classes, not individual samples.

---

## §2 Required Data Loading Architecture

Per §8.2 and §4.3, the correct architecture must:

1. **Group by ticker** — samples from one ticker form a contiguous sequence
2. **Sort chronologically** — within each ticker, `as_of_date` ascending
3. **Present tickers sequentially** — when a ticker's samples run out, move to the next ticker (in consistent order, not randomly shuffled)
4. **Carry hidden state** — GRU hidden states from batch N of ticker X are fed as `h_prev` to batch N+1 of ticker X
5. **Detach state across tickers** — reset hidden state when switching from ticker X to ticker Y
6. **Stateful training loop** — Trainer must track per-sequence `(h, c)` tuples and inject them into `model.forward(x, h_prev=h)`

### Correct DataLoader Structure

```
Epoch N:
  ┌─────────────────────────────────────────────────┐
  │ Ticker: AAPL (chronological, 200-400 samples)    │
  │  Batch 0: days 0-63  → GRU state → h_batch0     │
  │  Batch 1: days 64-127 → GRU state → h_batch1    │  h_batch0 passed as h_prev
  │  Batch 2: days 128-191 → GRU state → h_batch2   │  h_batch1 passed as h_prev
  │  ...                                             │
  │  Batch K: days end → GRU state detached, reset   │
  ├─────────────────────────────────────────────────┤
  │ Ticker: JPM (chronological)                      │
  │  Batch 0: days 0-63 → zeros h_prev               │ ← fresh state
  │  Batch 1: days 64-127 → h_prev from batch 0      │
  │  ...                                             │
  ├─────────────────────────────────────────────────┤
  │ Ticker: WMT (chronological)                      │
  │  ...                                             │
  └─────────────────────────────────────────────────┘
```

### Class Imbalance Handling (without random sampler)

Since WeightedRandomSampler is incompatible with chronological ordering, replace with:
1. **Sequence-level oversampling**: duplicate minority-class ticker histories in the dataset
2. **Per-sample loss weighting**: `_classification_loss` already has class weights (1.0, 1.0, 1.0) — increase to (3.0, 0.6, 2.0) for focal loss
3. **Online data augmentation**: time-warp, scale, and shift minority-class samples during loading (§8.2.5)

---

## §3 Implementation Plan

### Step 1: Rewrite `prepare_dataloaders()` → `prepare_sequential_dataloaders()`
- Group df by ticker, sort each group by `as_of_date`
- Build ticker-indexed batch iterators
- Return ticker-sequential batches
- Remove `WeightedRandomSampler`

### Step 2: Enable stateful training in Trainer
- Set `stateful=True`, `keep_hidden_across_epochs=True`
- Track per-scale GRU states in a dict keyed by ticker
- Reset state when switching tickers
- Inject `h_prev` into `model.forward(x, h_prev=h)`

### Step 3: Fix class imbalance via loss + augmentation
- Class weights: inter=3.0, pre=0.5, onset=2.0
- Add online time-warp and amplitude scaling in `__getitem__`
- Oversample minority-class ticker sequences in dataset construction

### Step 4: Remove `scale_activations` from training path
- The `_generate_epoch_inference_viz` already handles this
- The `fused_sequence` dict key can stay (no impact on training)
- But `scale_activations` return skips — these are `(B, T, 128)` tensors per scale. In stateful mode with chronological ordering, `T` = sequence length per batch (typically 64). OK for viz, not needed for training.

### Step 5: Enable h_prev in model.forward()
- Already supported: `forward(x, h_prev=None)` accepts optional hidden state
- Already in output: `"hidden_state": h_new` returns new state
- Trainer just needs to capture `h_new` and feed it back

### Step 6: Verify with training sanity check
- Metric: forecast MAE should improve over epochs (not plateau immediately)
- Metric: Per-class accuracy should show distinct curves, not flat collinearity
- Metric: Loss should decrease steadily, not plateau at epoch 3-4
- Metric: State norm should stay bounded (not explode or collapse)

---

## §4 What NOT To Change

- **Model architecture** — BiLSTM + MultiScaleCascade + 4 heads are correct per §3
- **Loss function** — CompositeLoss with focal + regression + coherence + uncertainty matches §6
- **Training config** — epochs=100, lr=1e-4, batch=64, cosine schedule are correct per Appendix A.2
- **Feature extraction** — FEATURE_COLUMNS (21 features) correctly implements §2.1 input modality abstraction
- **Evaluation metrics** — per-class accuracy, forecast MAE, temporal MAE are §9.1 requirements
- **Epoch visualization** — the 4-panel viz matches §9.6 requirements
