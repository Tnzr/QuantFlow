# Classification Collapse + Misleading Monitoring — Root Cause & Remediation

Date: 2026-07-18
Status: IN PROGRESS

## Confirmed root causes (full code/doc audit)

1. **Loss instability (primary collapse driver)** — `losses.py::_classification_loss` stacks THREE
   independent imbalance-correction mechanisms multiplicatively: focal factor (`γ=5.0`, non-standard —
   methodology/literature default is 2.0) × static `class_weights=(5.0, 0.25, 3.0)` × temporal proximity
   weight (up to 4x). This is a bistable system: whichever class's effective
   `class_weight × focal_factor × temporal_weight` product dominates becomes the sole prediction. Editing
   the static tuple only changes *which* class it collapses to (pre→onset in the user's last run), not
   whether it collapses. Methodology §6.2.1 explicitly calls for **dynamic per-batch** class weighting,
   not hand-tuned static constants — this was never implemented, only edited by hand across 4 commits
   (each one "fixing" a different collapse direction).
2. **`train_full_universe.py` has no CLI surface for `class_weights`/`focal_gamma`** — only aggregate
   stage weights (`--cls-weight` etc.) are exposed. Every edit the user believes changes behavior is
   actually a hardcoded dataclass default edit in `losses.py`. This is why "changing weights" only moves
   the collapse around instead of fixing it — the actual lever (focal γ + static tuple interaction) was
   never being tuned intentionally.
3. **Cross-ticker hidden-state corruption** — `DataLoader` flattens all tickers into one sample list and
   slices sequentially with plain `batch_size=64`. `Trainer.fit()` keys stateful hidden state off
   `tickers[0]` for the *entire batch*, but a batch can contain rows from two different tickers when a
   ticker's sample count isn't a multiple of 64. This silently applies the wrong ticker's hidden state to
   part of a batch — a direct violation of methodology §4.3 point 3 ("sequence-level, not sample-level,
   sampling").
4. **`_dynamics_loss` is dead code** — always called with a dummy `torch.zeros(1)` for `h_prev`, so the
   smoothness penalty never fires (shape mismatch guard always fails).
5. **Misleading monitoring identified by user**:
   - `_generate_epoch_inference_viz` always visualizes the exact same fixed first ~300 rows (one
     val ticker's earliest days) every single epoch/run — `viz_indices` is computed but never used, the
     actual data comes from unconditionally iterating the front of `val_loader`. This is why panels
     "never change."
   - X-axis is a bare sample index, not real dates, even though `as_of_date` is computed at dataset build
     time — it's silently dropped in `__getitem__`'s type filter (`isinstance` check excludes `str`).
   - There is no actual "forecast price vs actual price" panel — Panel 2 plots forecasted *tau*
     (days-to-event) vs target tau, not price. Panel 4 plots real price but with no forecast overlay.
   - No confusion matrix panel per epoch (methodology §9.6 requires one) — only computed once at final
     test time.
   - No automated anomaly detection (flat signal / token collapse / low-variance flags) despite being an
     explicit §9.6 requirement — exactly the instrumentation that would have caught this collapse
     immediately instead of requiring multiple manual training runs.
   - `forecast_dir_acc` metric is structurally broken (`tau_clamped` is always positive so the metric is
     ~always true) — another example of a monitoring number that looks fine but is meaningless.
   - wandb `viz/epoch_forecast_overview` image is logged with `commit=False` *after* the epoch's own
     `commit=True` call, so it is actually flushed together with the *next* epoch's log — every epoch's
     image is off-by-one in the wandb UI.

## Remediation (this session)

1. `losses.py`: focal_gamma 5.0→2.0 (standard), replace static `class_weights` tuple with **dynamic
   per-batch inverse-frequency weighting** (computed from the batch's own class counts, clipped to a
   sane [0.3, 4.0] range) per methodology §6.2.1/§8.2. Fix `_dynamics_loss` call to stop passing a dummy
   zero tensor (either wire real h_prev/h_current or zero out the weight honestly).
2. `dataset.py`: add a ticker-boundary-respecting `BatchSampler` so no batch ever spans two tickers
   (restores true sequence-level sampling per §4.3). Preserve `as_of_date` through `__getitem__` as an
   ordinal-day int64 (not silently dropped).
3. `trainer.py`:
   - Fix wandb step alignment (viz image logged in the same commit as the epoch scalar log).
   - Fix `forecast_dir_acc` to compare actual predicted-return direction vs realized direction, not a
     tautological expression.
   - Rewrite `_generate_epoch_inference_viz` to sample 4 windows *spread across the full val timeline*
     (different offsets/tickers each call) instead of a fixed first-300-row slice, use real dates on the
     x-axis, add a 5th confusion-matrix panel, and add automated anomaly-detection text (flat-probability
     / token-collapse / single-class-domination flags) directly on the figure and as wandb scalars
     (`viz/anomaly_flat_probs`, `viz/anomaly_token_collapse`, `viz/anomaly_class_collapse`).
4. `scripts/train_full_universe.py`: expose `--focal-gamma` and a `--class-weight-mode {static,dynamic}`
   flag so future tuning happens through the actual CLI surface instead of hand-editing dataclass
   defaults.

## Not addressed this session (tracked for later)
- True price-forecast head (current architecture forecasts tau/days-to-event, not price level) — adding
  a literal "forecasted price vs actual price" panel requires either a new regression head or a derived
  price-path reconstruction from tau+drawdown; flagged to user as a scope decision.
- Balanced-sampler wiring (exists in `dataset.py`, unused) — intentionally left disabled since it
  conflicts with chronological/stateful ordering; dynamic per-batch class weighting is the compliant
  alternative per methodology.
- Full 6-panel §9.6 spec (only 5 implemented after this fix: state ribbon, tau forecast, latent tokens,
  price+drawdown, confusion matrix) — "raw signal trace with event markers" and "derived
  physiology/feature overlays" are EEG-monitoring-domain language from the template methodology doc and
  don't map 1:1 onto financial features; interpreted analogues are covered by existing panels.

---

## UPDATE 2026-07-18: Loss/monitoring fixes verified working, but collapse persists — new root cause isolated

### Monitoring fixes: CONFIRMED WORKING (10-epoch live test, wandb run 0woxrjrm)
- Zero wandb step-ordering warnings across all runs (previously present every epoch).
- Zero viz-generation errors; spread-window/confusion-matrix/anomaly-flag panel renders every epoch.
- Automated `CLASS COLLAPSE` detector fired correctly at epoch 1 in every test run below — exactly the
  instrumentation gap this session set out to close.

### Collapse persists — 4 independent ablations, ALL produced byte-identical results
Every configuration below converged to `accuracy=0.804287045666356` (the test set's literal
majority-class fraction) with `accuracy_inter=0.0, accuracy_pre=1.0, accuracy_onset=0.0`:

| # | Test | Config | Result |
|---|------|--------|--------|
| 1 | Dynamic per-batch class weighting (this session's loss fix) | `class_weight_mode=dynamic`, `focal_gamma=2.0` | Collapse (verified weighting itself gives real 2.4-2.6x boost) |
| 2 | Attention-pool classification head (this session's architecture fix) | Same loss config + `AttentionPool` replacing mean-pool | Byte-identical collapse |
| 3 | Decoupled forecast/classification heads | `use_coherent_heads=False` | Byte-identical collapse |
| 4 | Classification-only objective | `classification_weight=1.0`, all 7 other loss weights = 0.0 | Byte-identical collapse |

Gradient-flow sanity check confirmed gradients DO reach `past_head`/`past_pool` parameters (non-zero,
no wiring bug).

### Decisive test: features ARE separable — RandomForest (shuffled, sklearn `class_weight='balanced'`)
on the same underlying features (last-day + window mean/std, 63-dim) achieves:
```
inter:  recall 0.93, precision 0.46
pre:    recall 0.77, precision 0.95
onset:  recall 0.81, precision 0.47
```
This rules out data/label/feature separability as the cause.

### New leading hypothesis
The one variable every failing deep-learning config shares — and that the successful RF test does NOT
share — is **strictly chronological, non-shuffled, per-ticker sequential batch order**, mandated by
methodology §4.3/§8.2 for stateful hidden-state carry. Consecutive daily samples from one ticker are
highly autocorrelated; non-shuffled SGD sees long runs of the same regime per batch, which likely causes
gradient drift toward whichever regime is locally dominant, overwhelming any class-weighting scheme
(static or dynamic) because the weighting's effectiveness depends on batch diversity that rarely exists
in a temporally-local window regardless of raw class counts.

### Decision needed (not resolved this session — user chose to stop and review rather than continue testing)
Two-phase curriculum candidate: Phase A pretrains the encoder + classification head on SHUFFLED samples
(breaking statefulness, matching the RF setup) to learn separable features; Phase B fine-tunes statefully/
chronologically per methodology, ideally preserving the learned decision boundary. Not yet implemented —
requires deciding whether to formalize this as a permanent training phase in `TrainingConfig`/`Trainer`
before writing it, since it's a real methodology-level design change, not a config tweak.

---

## NEXT STEPS FOR NEW SESSION (handoff, 2026-07-18)

### Immediate priority: resolve the chronological-vs-shuffled tension
1. **Confirm the hypothesis first** (cheap, ~10 min): run a quick ablation that temporarily allows
   shuffling (`shuffle=True` in `prepare_dataloaders`, bypass `TickerBatchSampler`, `stateful=False`,
   `h_prev=None` always) for 10 epochs on `data/daily_20_tickers.parquet`. If classification recall for
   inter/onset becomes non-zero (matching the RF result: ~93%/81% recall), the hypothesis is confirmed
   and it's safe to invest in a real fix. If it ALSO collapses, the hypothesis is wrong and the
   investigation must go back to model capacity, LR, or optimizer settings instead.
2. **If confirmed**, implement a two-phase curriculum in `Trainer`/`TrainingConfig`:
   - Phase A: shuffled, non-stateful pretraining of `_encode` + `multi_scale` + `past_pool` + `past_head`
     only (freeze/skip future/uncertainty/dynamics heads or just don't backprop through them) until
     classification recall stabilizes across all 3 classes.
   - Phase B: unfreeze everything, switch to `TickerBatchSampler` + stateful chronological order per
     methodology §4.3/§8.2, fine-tune all heads jointly. Watch whether Phase A's learned separability
     survives the switch to correlated chronological batches — it may re-collapse, which would be
     important new information.
   - Methodology doc (`ML_DL_Architecture_Methodology.md`) may need a new curriculum section documenting
     this as "Phase A2: classification pretraining" distinct from the existing Phase A/B split — check
     what's already there under §8 before adding, since Phase A/B terminology is already used for
     something else (pretrained-weights-then-composite-loss).
3. **If the shuffle ablation doesn't confirm the hypothesis**, next things to check in order of
   cheapness: (a) increase epochs to 30-50 on the existing daily-resolution run to rule out simple
   undertraining, (b) inspect `AdamW` learning rate — 1e-4 may be too low or too high for this
   specific head given weight_decay=1e-5, (c) check weight initialization of `past_head`/`past_pool` —
   verify they aren't starting in a degenerate region, (d) try a much smaller/simpler classification-only
   model (e.g. single linear layer on frozen encoder features) to isolate whether the multi_scale GRU
   cascade encoder itself is the bottleneck rather than the head.

### Also still pending from the original monitoring/collapse remediation work (lower priority)
- Alpaca data provider migration (§9 of this doc) — not started, only planned.
- 5-min intraday yfinance test dataset — not built yet.
- True price-forecast head (model currently forecasts tau/days-to-event, not price level) — deferred,
  would need a new regression head or derived price-path reconstruction.
- Full 140-ticker daily dataset rebuild — deferred until collapse is resolved on the 20-ticker subset.
- Balanced-sampler wiring in `dataset.py` (`get_balanced_sampler`) — intentionally left disabled/dead
  since it conflicts with chronological ordering; may become directly relevant if Phase A shuffled
  pretraining is implemented (it could be reused there).

### Key files touched this session (all committed to `chore/repo-direction-4-3-2-1`)
- `quantflow/models/losses.py` — dynamic per-batch class weighting, focal_gamma=2.0, fixed dead
  `_dynamics_loss` code path.
- `quantflow/models/dataset.py` — `TickerBatchSampler` (no batch spans a ticker boundary), `date_ordinal`
  preserved through `__getitem__`.
- `quantflow/models/trainer.py` — fixed `forecast_dir_acc`, fixed wandb step misalignment, added
  `pred_class_dominance`/per-class prediction fractions + per-epoch confusion matrix, rewrote
  `_generate_epoch_inference_viz` (spread windows, real dates, confusion-matrix + anomaly-flag panel).
- `quantflow/models/trainer_ddp.py` — same `TickerBatchSampler` fix applied.
- `quantflow/models/architectures.py` — `AttentionPool` module, replaces mean-pool for classification
  head input (kept even though it didn't fix collapse — it's still a legitimate improvement over static
  mean-pool and doesn't hurt).
- `scripts/train_full_universe.py` — `--focal-gamma`, `--class-weight-mode`, `--static-class-weights`,
  `--no-coherent-heads` CLI flags added (previously these required hand-editing dataclass defaults).
- Environment: `torch==2.6.0+cu124` and `wandb==0.28.1` installed into the `quantflow` conda env
  (`/root/miniforge3/envs/quantflow`) — were missing at session start, not yet added to
  `environment.yml`/`requirements.txt` (should be added if this becomes the standard training env).

### Test artifacts (wandb runs, all under project `quantflow`, entity `tnzr-pioneer-innovations-collective`)
- `0woxrjrm` (`bilstm-daily-v3-dynamic-weights-test`) — dynamic weighting only.
- `bilstm-daily-v4-attnpool-test` — + attention pool.
- `bilstm-daily-v5-no-coherent-test` — + decoupled coherent heads.
- Classification-only probe was run WITHOUT wandb (`/tmp/kilo/probe_cls_only.py`, not saved to repo —
  recreate from this doc's Update section if needed).
