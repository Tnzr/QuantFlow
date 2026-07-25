# CascadeANP Architecture Freeze — v1.0

Date: 2026-07-23
Status: FROZEN — regression-safe, no structural changes without explicit unfreeze

## Architecture Snapshot

```
Input: 5-min intraday OHLCV + derived features (27 columns)
  │
  ├─→ BiLSTM (unidirectional, causal, 128-dim, 2 layers)
  │   └─→ MultiScale GRU Cascade (4 scales, 128-dim)
  │       └─→ LatentEncoder → z (market regime, 64-dim)
  │       └─→ ANP Decoder (cross-attn + teacher-forced AR, training only)
  │
  └─→ 1D Conv Encoder (causal raw features, 27→128→30 tokens)
      └─→ direct_path MLP (30×128 → 21-step return forecast)
      └─→ direct_sigma MLP (uncertainty per step)
```

## Key Invariants (DO NOT REGRESS)

1. **Causal encoder**: `bidirectional=False`. LSTM processes left-to-right only.
2. **Causal conv path**: `raw_features` MUST be strictly the context window ending at `tgt_start`, never including target bars. Both training and viz must use the same feature slice.
3. **No pos_out bias**: Per-step output variation comes from input-dependent conv tokens.
4. **Strictly causal raw_features in viz**: Both `cascade_anp_viz.py` and `viz_forecast_inference.py` pass actual raw OHLCV features to `raw_features=`, never cascade/BiLSTM tokens.
5. **Sigma penalty active**: `F.relu(sigma_y - 0.05).mean() * 1.0` — keeps sigma calibrated.
6. **Direct path head**: One-shot 21-step prediction from conv-pooled raw features. No autoregressive loop.
7. **Conv architecture**: `Conv1d(27→64, k=5) → GELU → Conv1d(64→128, k=3) → GELU → AdaptiveAvgPool1d(30) → MLP(3840→512→256→21)`

## Files Under Freeze

| File | What | Do Not Change |
|------|------|---------------|
| `quantflow/models/cascade_anp.py` | Full architecture | forward() signature, encode() returns, causal LSTM, conv encoder structure, loss composition |
| `scripts/train_cascade_anp.py` | Training pipeline | sample_batch causal window logic, raw_feats construction, batch return tuple order |
| `scripts/backtest_cascade.py` | Portfolio backtest | signal/confidence/martingale logic, entry/exit flow |
| `scripts/build_intraday_dataset.py` | Dataset builder | feature columns list, seasonality features |
| `quantflow/models/cascade_anp_viz.py` | Wandb viz | raw_features=rf_c call, color/alpha, price clamping |
| `scripts/viz_forecast_inference.py` | Offline viz | raw_features=rf_c call, context shading, panel layout |
| `quantflow/data/labeler.py` | Directional labels | use_directional_labels=True path |

## Unfreeze Protocol

To modify frozen architecture:
1. Tag current commit: `git tag cascade-anp-v1.0-frozen`
2. Create branch from frozen tag
3. Document the change in this file with rationale
4. Run sanity check: `python -c "min_corr < 0.7"` before merging

## Next-Phase Work (data enrichment only — no architecture changes)

- Historical data: 5y+ instead of 60d of 5-min bars
- More tickers: full SECTOR_UNIVERSE (140+)
- Alpaca integration for microstructure (order flow, L2)
- Event features: sentiment, earnings dates, weather (pre-computed, merged as columns)
- Pre-computed daily seasonality merged into intraday
