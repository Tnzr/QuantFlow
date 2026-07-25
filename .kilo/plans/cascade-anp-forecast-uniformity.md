# CascadeANP Forecast Path Uniformity — Root Cause & Remediation Plan

Date: 2026-07-23
Status: IN PROGRESS (50-epoch training running)

## Observed Issue

All forecast snapshots produce near-identical trajectories:
- Same direction (all up or all down)
- Same shape (curvy start → straightening out)
- Pattern repeats across all tickers and snapshots
- First snapshot has slightly more curve; later ones are flatter

## Root Cause: Direct Path Head Expressivity Bottleneck

The direct path head (`cascade_anp.py:direct_path`) takes the last 10 cascade-encoded
tokens and projects through a fixed MLP to predict all 21 forecast steps.

1. **Cascade convergence**: The multi-scale GRU cascade converges to a near-steady state
   after processing ~60+ bars of context. All snapshots see similar cascade tokens
   regardless of which bars they encode → similar MLP inputs.

2. **MLP template**: The MLP learns a single mapping (last_10_tokens → 21 returns).
   Since the inputs are nearly identical across snapshots, the outputs are too.
   The MLP is NOT learning to produce different paths for different market conditions —
   it's learning one path that minimizes MSE across all conditions.

3. **No per-step feedback**: Each of the 21 forecast steps is predicted independently
   from the SAME input. There's no autoregressive state, no step-dependent modulation,
   no mechanism for the model to say "step 5 should differ from step 10."

4. **Viz confirms**: The first cascade-encoded snapshot has slightly higher variance
   (the encoder hasn't fully converged). Later snapshots are fully converged → flat
   predictions. This matches the observation "first viz have more curve and then they
   start flattening."

## Remediation Plan

### Phase A: Training augmentation (current — 50 epochs running)
- Add Gaussian noise (std=0.001) to feature inputs during training
- Increases encoder output diversity → more varied MLP inputs
- Expected: partial improvement, unlikely to solve fully

### Phase B: Per-step positional encoding
- Add learned position embeddings to the direct path head
- `pos_embed[i]` injected at each of the 21 forecast steps
- Gives the MLP a way to differentiate "step 1 return" from "step 21 return"
- Implementation: `direct_path(pos_embed[i] + last_ctx) → return[i]`

### Phase C: Cross-attention direct decoder
- Replace MLP with cross-attention: query = learned step embeddings, key/value = cascade tokens
- `return[i] = cross_attn(step_query[i], cascade_tokens)`
- Each step can attend to different parts of the context
- More expressive than MLP, same parameter count if kept small

### Phase D: Autoregressive with stateful carry
- Return to autoregressive but with a running cascade state (not return_embed)
- `state[i] = GRU(state[i-1], cascade_token[last], predicted_return[i-1])`
- The state accumulates trajectory history naturally
- This was the original intent before the return_embed bottleneck was discovered

## Overnight Training (50 epochs)
- 49 tickers, 28 features, 128 hidden, 64 latent
- Data augmentation: noise=0.001
- Wandb: cascade-50ep-overnight
- Checkpoint: checkpoints/cascade_anp.pt

## Next Session Priority
1. Evaluate overnight training results
2. Implement Phase B (per-step positional encoding)
3. If still insufficient, implement Phase C (cross-attention decoder)
4. Test on backtest with confidence sizing + rotation
