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
