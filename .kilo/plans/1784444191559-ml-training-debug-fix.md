# ML Training Debug & Fix Plan

**Date:** 2026-07-18
**Issues Found:**

## Bug 1: LR Scheduler Reversed (warmup start_factor=1e-3)
The scheduler multiplies LR by 1e-3 initially, so at epoch 0 with `lr=1e-4`,
the actual LR is `1e-4 * 1e-3 = 1e-7`. It ramps up over 5 warmup epochs to 1e-4
by epoch 5. The wandb plot shows 2e-5 → 1e-4 which matches: by ~epoch 3 it's at
mid-warmup (~5e-5). This is wrong — warmup should start at a small LR and ramp
TO the target, but `start_factor` should be a small value like 0.1 (10% of target
= 1e-5), not 0.001 (0.1% = 1e-7). The early epochs are basically frozen.

Fix: Change `start_factor=0.01` (1% of target LR, = 1e-6) so LR goes 1e-6 → 1e-4.
The methodology specifies LinearLR warmup.

## Bug 2: Accuracy Flatline at 81-82% From Epoch 1
Root cause: The dataset has massive class imbalance:
  - inter_event (0): ~5% (1,691 samples)
  - pre_event (1): ~80% (25,177 samples)
  - onset (2): ~15% (4,787 samples)

With class weights (0.8, 0.6, 1.0), the model can achieve ~80% accuracy by
always predicting class 1 (pre_event). The 82.3% val accuracy from epoch 1
simply means the model quickly learns to output pre_event for everything.

Fix: Report per-class accuracy (precision/recall/F1) instead of just overall
accuracy. Add class-balanced sampling. Display confusion matrix at each epoch
in console output.

## Bug 3: Classification Collapse to Pre-Event Only
The PastStateHead takes only `fused[:, -1, :]` (last timestep only).
With 128-dim hidden → 64 → 32 → 3, and the bottleneck being a single
timestep vector for 3 very imbalanced classes, the classifier collapses
to the majority class.

Fixes:
1. Mean-pool across all timesteps before classification, not just last
2. Class-balanced sampling in DataLoader
3. Higher class weight for minority classes (inter: 2.0, pre: 0.5, onset: 1.0)
4. Add gradient clipping on classifier head specifically

## Bug 4: Per-Batch Loss Not Logged
Wandb only logs epoch-level metrics. This means we can't see intra-epoch
loss dynamics (high variance, gradient spikes, etc).

Fix: Log per-batch loss to wandb using commit=False for batches, commit=True
at epoch end. Also log gradient norms per batch.

## Bug 5: Confusion Matrix Shows Only Pre (class 1)
Directly caused by Bug 3.

## Bug 6: forecast_mae AttributeError
In `fit()`, the forecast_mae computation uses `labels.get("tau_forward_synthetic")`
but the dataset provides `"tau_forward"`. This silently returns zeros.

---

## Implementation Plan

### Fix 1: Trainer.fit — Per-batch wandb logging + LR fix + class accuracy
### Fix 2: Dataset — Class-balanced sampling
### Fix 3: Model — Mean-pool classification (all timesteps)
### Fix 4: Loss — Fix forecast_mae label key, fix class weights
### Fix 5: Visualization — Per-class confusion matrix, clearer labels
### Fix 6: Retrain and verify
