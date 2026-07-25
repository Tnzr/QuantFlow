"""CascadeANP multi-panel visualization — single stock, full trajectory,
multiple forecast snapshots overlaid, with volume, RSI, and latent embeddings."""

import io
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple


def compute_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    deltas = np.diff(prices, prepend=prices[0])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = pd.Series(gains).rolling(period, min_periods=1).mean().values
    avg_loss = pd.Series(losses).rolling(period, min_periods=1).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100)
    return (100 - 100 / (1 + rs)).clip(0, 100)


def generate_cascade_anp_viz(
    model,
    df: pd.DataFrame,
    device: str,
    feature_cols: List[str],
    n_context: int = 80,
    n_target: int = 21,
    n_snapshots: int = 5,
    epoch: int = 0,
    ticker: Optional[str] = None,
) -> Optional:
    """Single-column multi-panel visualization for CascadeANP.

    Panels (shared time x-axis):
      1. Price line + multiple forecast snapshots with confidence envelopes
      2. Volume histogram bars
      3. RSI indicator with overbought/oversold bands
      4. Latent embedding heatmap (scale activations from cascade)
      5. Per-step forecast detail + uncertainty (last snapshot)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        # Pick a ticker with sufficient data
        ticker_groups = {t: g.sort_values("as_of_date") for t, g in df.groupby("ticker")}
        candidates = [t for t, g in ticker_groups.items() if len(g) >= 500]
        if not candidates:
            return None

        if ticker is None:
            np.random.seed(epoch * 42)
            ticker = np.random.choice(candidates)

        group = ticker_groups[ticker]
        feats = group[feature_cols].fillna(0.0).values.astype(np.float32)
        prices = group["adj_close"].fillna(0.0).values.astype(np.float32)
        returns = group["target_return"].fillna(0.0).values.astype(np.float32)
        dates = group["as_of_date"].values

        T = len(feats)
        encode_len = min(T, 800)  # max ~800 bars in one plot
        price_arr = prices[:encode_len]

        # Run full encode through cascade
        feats_t = torch.from_numpy(feats[:encode_len]).unsqueeze(0).to(device)
        with torch.no_grad():
            fused, raw_bilstm, scale_acts = model.encode(feats_t)
        fused = fused[0].cpu().detach().numpy()  # [L, H]
        raw = raw_bilstm[0].cpu().detach().numpy()  # [L, H] — BiLSTM skip
        scale_heatmaps = [s[0].cpu().detach().numpy() for s in scale_acts]  # each [L, H]

        L = fused.shape[0]
        n_snapshots_eff = max(3, n_snapshots)
        step = max(1, (L - n_context - n_target) // n_snapshots_eff)

        snapshots = []
        all_step_errors = []
        for snap_pos in range(0, max(1, L - n_context - n_target), step):
            ctx_start = snap_pos
            ctx_end = min(snap_pos + n_context, L - 1)
            tgt_start = ctx_end
            tgt_end = min(tgt_start + n_target, L)

            if tgt_end - tgt_start < 3:
                continue

            x_c = torch.from_numpy(fused[ctx_start:ctx_end]).unsqueeze(0).to(device)
            y_c = torch.from_numpy(returns[ctx_start:ctx_end, None]).unsqueeze(0).to(device)
            # Raw causal feature window for the conv direct-path — must match
            # training exactly: the actual raw OHLCV/feature slice ending at
            # ctx_end, never the target region. This is NOT cascade or BiLSTM
            # tokens — it's the original feature array itself.
            rf_c = torch.from_numpy(feats[ctx_start:ctx_end]).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(x_c, y_c, n_steps=tgt_end - tgt_start,
                               teacher_forcing=False, raw_features=rf_c)

            mu = outputs["mu_y"][0].cpu().numpy()
            sigma = outputs["sigma_y"][0].cpu().numpy()
            z = outputs["z"][0].cpu().numpy()

            horizon = len(mu)
            ref_price = price_arr[ctx_end - 1]
            for h in range(horizon):
                actual_idx = tgt_start + h
                if actual_idx < L and ref_price > 0:
                    actual_ret = (price_arr[actual_idx] / ref_price) - 1.0
                    all_step_errors.append((h, mu[h] - actual_ret))

            snapshots.append({
                "ctx_end": ctx_end, "tgt_start": tgt_start,
                "tgt_end": tgt_end, "mu": mu, "sigma": sigma, "z": z,
            })

        if len(snapshots) < 2:
            return None

        fig = plt.figure(figsize=(24, 18), dpi=100)
        gs = GridSpec(6, 1, figure=fig, height_ratios=[3.0, 0.6, 0.7, 0.9, 1.2, 0.8], hspace=0.06)

        x_all = np.arange(L)
        rsi_arr = compute_rsi(price_arr)
        vol_col = next((i for i, c in enumerate(feature_cols) if "vol" in c.lower() or "volume" in c.lower()), 0)
        vol_arr = feats[:L, vol_col]
        vol_arr = np.maximum(vol_arr, 0)

        dt_labels = []
        for d in dates[:L]:
            s = str(d)
            if " " in s:
                parts = s.split(" ")
                dt_labels.append(f"{parts[0][5:]}\n{parts[1][:5]}")
            else:
                dt_labels.append(s[:10])
        tick_spacing = max(1, L // 12)
        tick_pos = np.arange(0, L, tick_spacing)

        colors = [  # dark purples, distinct from green price line — uniform alpha
            "#6c3483", "#7d3c98", "#8e44ad", "#9b59b6",
            "#a569bd", "#af7ac5", "#bb8fce", "#c39bd3"] * 3

        # Panel 1: Price + dense forecast snapshots with tight uncertainty
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(x_all, price_arr, color="#2ecc71", linewidth=1.0, alpha=0.9, zorder=2)

        for si, snap in enumerate(snapshots):
            ctx_end = snap["ctx_end"]
            mu = snap["mu"]
            sigma = np.clip(snap["sigma"], 0.0, 0.03)  # clamp to 3% per step
            horizon = len(mu)
            fc_x = np.arange(ctx_end, ctx_end + horizon)
            cum_ret = np.cumsum(mu)
            fc_price = snap.get("ref_price", price_arr[ctx_end - 1]) * (1 + cum_ret)
            fc_price = np.clip(fc_price, price_arr[ctx_end - 1] * 0.85, price_arr[ctx_end - 1] * 1.15)

            show_label = si % max(1, len(snapshots) // 6) == 0 or si < 2 or si >= len(snapshots) - 2
            c = colors[si % len(colors)]
            ax1.plot(fc_x, fc_price, color=c, linewidth=1.2, alpha=0.85,
                    linestyle="--", label=f"F{si+1}" if show_label else "")
            ax1.scatter(ctx_end, price_arr[ctx_end - 1], color=c, s=10, zorder=3)

        ax1.set_ylabel("Price ($)")
        ax1.set_title(f"{ticker} — Price + {len(snapshots)} Forecast Snapshots (tight ±σ bands)", fontweight="bold")
        ax1.legend(loc="upper left", ncol=6, fontsize=5)
        ax1.grid(True, alpha=0.10, linestyle="--")
        ax1.tick_params(labelbottom=False)

        # Panel 2: Volume bars
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax2.fill_between(x_all, 0, vol_arr, color="#3498db", alpha=0.4, linewidth=0, step="mid")
        ax2.set_ylabel("Vol")
        ax2.grid(True, alpha=0.10, linestyle="--")
        ax2.tick_params(labelbottom=False)

        # Panel 3: RSI
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        ax3.plot(x_all, rsi_arr, color="#9b59b6", linewidth=0.7, alpha=0.9)
        ax3.axhline(70, color="#e74c3c", ls="--", alpha=0.25, lw=0.6)
        ax3.axhline(30, color="#27ae60", ls="--", alpha=0.25, lw=0.6)
        ax3.fill_between(x_all, 70, 100, alpha=0.06, color="#e74c3c")
        ax3.fill_between(x_all, 0, 30, alpha=0.06, color="#27ae60")
        ax3.set_ylim(0, 100)
        ax3.set_ylabel("RSI")
        ax3.grid(True, alpha=0.10, linestyle="--")
        ax3.tick_params(labelbottom=False)

        # Panel 4: Latent embedding heatmap (S0 from cascade, top-16 dims)
        ax4 = fig.add_subplot(gs[3], sharex=ax1)
        if scale_heatmaps and len(scale_heatmaps) > 0:
            s0 = scale_heatmaps[0]
            n_dims = min(16, s0.shape[1])
            s0_t = s0[:L, :n_dims].T
            im = ax4.imshow(s0_t, aspect="auto", cmap="RdYlBu_r", interpolation="nearest",
                           extent=[0, L - 1, 0, n_dims], origin="lower")
            fig.colorbar(im, ax=ax4, shrink=0.5, label="Activation")
        ax4.set_ylabel("S0 dims")
        ax4.set_title("Latent Embeddings (Scale S0)", fontsize=8)
        ax4.tick_params(labelbottom=False)

        # Panel 5: Error vs t distribution (hexbin from ALL snapshots)
        ax5 = fig.add_subplot(gs[4], sharex=ax1)
        if all_step_errors:
            steps_arr = np.array([e[0] for e in all_step_errors])
            err_pct = np.array([e[1] for e in all_step_errors]) * 100
            hb = ax5.hexbin(steps_arr, err_pct, gridsize=20, cmap="YlOrRd",
                           mincnt=1, bins='log')
            fig.colorbar(hb, ax=ax5, label="Count", shrink=0.6)

            max_s = int(steps_arr.max()) + 1
            rmse_by_step = np.zeros(max_s)
            for s in range(max_s):
                mask = steps_arr == s
                if mask.any():
                    rmse_by_step[s] = np.sqrt(np.mean(err_pct[mask] ** 2))
            valid = np.arange(max_s)[rmse_by_step > 0]
            ax5_twin = ax5.twiny()
            ax5_twin.plot(valid, rmse_by_step[valid], "k-", linewidth=1.5)
            ax5_twin.set_xlim(ax5.get_xlim());
            ax5_twin.set_xticks([])
            ax5_twin.set_ylabel("RMSE%", fontsize=7)

        ax5.set_ylabel("Error Δ%")
        ax5.set_title(f"Error vs t — {len(all_step_errors)} points from {len(snapshots)} snapshots")
        ax5.grid(True, alpha=0.10, linestyle="--")
        ax5.tick_params(labelbottom=False)

        # Panel 6: Mid-snapshot per-step forecast detail
        ax6 = fig.add_subplot(gs[5], sharex=ax1)
        mid_snap = snapshots[len(snapshots) // 3]  # pick snapshot 1/3 through
        mu = mid_snap["mu"]
        sigma = np.clip(mid_snap["sigma"], 0.0, 0.03)
        steps_x = np.arange(mid_snap["ctx_end"], mid_snap["ctx_end"] + len(mu))
        ax6.plot(steps_x, mu, color="#9b59b6", linewidth=1.0, label="μ")
        ax6.fill_between(steps_x, mu - sigma, mu + sigma, alpha=0.12, color="#9b59b6", label="±σ")
        actual_r = returns[mid_snap["tgt_start"]:mid_snap["tgt_end"]]
        ax6.scatter(steps_x[:len(actual_r)], actual_r, s=8, color="#e74c3c", alpha=0.4, label="Actual")
        ax6.axhline(0, color="gray", ls="--", alpha=0.2, lw=0.6)
        ax6.set_xlabel(f"Timestep — {len(dt_labels)} bars shown ({len(snapshots)} forecasts)")
        ax6.set_ylabel("Return")
        ax6.set_title(f"Forecast detail (snap {len(snapshots)//3}/{len(snapshots)}), |z|={np.linalg.norm(mid_snap['z']):.2f}")
        ax6.legend(fontsize=6, loc="upper right")
        ax6.grid(True, alpha=0.10, linestyle="--")

        if len(tick_pos) <= len(dt_labels):
            ax6.set_xticks(tick_pos)
            ax6.set_xticklabels([dt_labels[i] if i < len(dt_labels) else "" for i in tick_pos],
                                rotation=30, ha="right", fontsize=6)
        fig.suptitle(f"CascadeANP Epoch {epoch} — {ticker} ({L} bars, {len(snapshots)} forecasts)",
                      fontsize=11, fontweight="bold", y=0.98)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white", dpi=100)
        plt.close(fig)
        buf.seek(0)
        import wandb
        from PIL import Image
        return wandb.Image(Image.open(buf), caption=f"CascadeANP Epoch {epoch} — {ticker}")
    except Exception as exc:
        import traceback
        print(f"CascadeANP viz failed: {exc}\n{traceback.format_exc()}")
        return None
