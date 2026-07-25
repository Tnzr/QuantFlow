#!/usr/bin/env python3
"""Standalone CascadeANP inference viz — no training, no wandb, just load+forecast+viz."""
import argparse, numpy as np, pandas as pd, torch

def get_feature_cols(df):
    exclude = {"ticker","event_state","event_state_code","is_volatile",
               "tau_forward","target_return","target_5d","target_21d",
               "target_1h","target_4h","drawdown_5d_max","drawdown_21d_max",
               "as_of_date","adj_close","close","target_direction_5d",
               "target_direction_21d","sector_idx","target_return_path"}
    return sorted([c for c in df.columns if c not in exclude and df[c].dtype!='object'])

def load_model(path, n_features, device="cpu"):
    from quantflow.models.cascade_anp import CascadeANP
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    m = CascadeANP(n_features, cfg.get("hidden_dim",96),
                   cfg.get("latent_dim",48), cfg.get("cascade_layers",4))
    m.load_state_dict(ckpt["model_state_dict"]); m.eval(); m.to(device)
    return m, cfg

def visualize_forecast(model, df, ticker, feature_cols, n_ctx=60, n_fc=21,
                       enc_len=800, n_snap=8, outpath=None):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    g = df[df.ticker==ticker].sort_values("as_of_date")
    feats = g[feature_cols].fillna(0).values.astype(np.float32)
    prices = g["adj_close"].fillna(0).values.astype(np.float32)
    returns = g["target_return"].fillna(0).values.astype(np.float32)
    dates = g["as_of_date"].values
    T = len(feats); L = min(T, enc_len)
    if L < n_ctx + n_fc + 50:
        return print(f"  Too few bars: {T}")

    device = next(model.parameters()).device
    with torch.no_grad():
        fused, raw_bilstm, _ = model.encode(torch.from_numpy(feats[:L]).unsqueeze(0).to(device))
    fused = fused[0].cpu().numpy()
    raw = raw_bilstm[0].cpu().numpy()
    prices = prices[:L]; returns = returns[:L]

    step = max(1, (L - n_ctx - n_fc) // n_snap)
    snapshots = []
    for ctx_end in range(n_ctx, L - n_fc, step):
        if ctx_end + n_fc > L: continue
        ctx_start = ctx_end - n_ctx
        x_c = torch.from_numpy(fused[ctx_start:ctx_end]).unsqueeze(0).to(device)
        y_c = torch.from_numpy(returns[ctx_start:ctx_end,None]).unsqueeze(0).to(device)
        # Raw causal feature window for conv direct-path — must match training:
        # the actual raw OHLCV/feature slice ending at ctx_end, never target bars.
        rf_c = torch.from_numpy(feats[ctx_start:ctx_end]).unsqueeze(0).to(device)
        n_steps = min(n_fc, L - ctx_end)
        with torch.no_grad():
            model_out = model(x_c, y_c, n_steps=n_steps, teacher_forcing=False,
                             raw_features=rf_c)
        mu = model_out["mu_y"][0].cpu().numpy()
        sigma = model_out["sigma_y"][0].cpu().numpy()
        ref = prices[ctx_end-1]
        af = prices[ctx_end:ctx_end+len(mu)]
        ar = (af/ref)-1 if ref>0 else np.zeros_like(mu)
        n_uniq = len(set(mu.round(5)))
        snapshots.append({"bar":ctx_end,"mu":mu,"sigma":sigma,
                         "ref_price":ref,"actual_fwd":af,"actual_ret":ar,"uniq":n_uniq})
        if len(snapshots) == 1:
            print(f"  [DEBUG] sigma mean={sigma.mean():.6f}, max={sigma.max():.6f}, "
                  f"sampled_noise ±{sigma.mean()*2.0:.4f}, mu mean={mu.mean():.6f} std={mu.std():.6f}")
        print(f"  Snap bar {ctx_end}: {n_uniq}/{len(mu)} unique, "
              f"mean={mu.mean()*100:.1f}%, std={mu.std()*100:.1f}%")

    if len(snapshots) < 3:
        return print(f"  Too few snapshots: {len(snapshots)}")
    print(f"  {len(snapshots)} snapshots for {ticker}")

    fig = plt.figure(figsize=(20, 17), dpi=100)
    gs = fig.add_gridspec(5, 1, height_ratios=[2.2, 1.0, 1.0, 1.2, 1.0], hspace=0.35)
    axes = [fig.add_subplot(gs[i]) for i in range(5)]
    x_all = np.arange(L)
    fc_colors = [  # dark purples, distinct from green price
        ("#6c3483", 0.9), ("#7d3c98", 0.85), ("#8e44ad", 0.80),
        ("#9b59b6", 0.75), ("#a569bd", 0.70), ("#af7ac5", 0.65),
        ("#bb8fce", 0.60), ("#c39bd3", 0.55)] * 3
    
    def fc(i):
        """Get forecast color tuple for snapshot i."""
        return fc_colors[i % len(fc_colors)]

    # Panel 1: Price + forecasts + actual future
    ax1 = axes[0]
    ax1.plot(x_all, prices, color="#2ecc71", lw=1.0, alpha=0.85, zorder=1, label=ticker)
    for si, s in enumerate(snapshots):
        bar = s["bar"]; mu = s["mu"]; fl = len(mu)
        fx = np.arange(bar, bar+fl)
        c = fc_colors[si % len(fc_colors)]
        # Shade context region: n_ctx bars before forecast start
        ctx_start = max(0, bar - n_ctx)
        ax1.axvspan(ctx_start, bar, alpha=0.06, color=c[0], zorder=0)
        # Forecast path
        cum = np.cumsum(mu); fp = s["ref_price"]*(1+cum)
        fp = np.clip(fp, s["ref_price"]*0.85, s["ref_price"]*1.15)
        ax1.plot(fx, fp, color=c[0], lw=1.0, alpha=c[1], ls="--",
                label=f"F{si+1}" if si<5 else "")
        ax1.scatter(bar, s["ref_price"], color=c[0], s=12, zorder=3)
        al = min(len(s["actual_fwd"]), fl)
        if al>0:
            ax1.plot(fx[:al], s["actual_fwd"][:al], color=fc(si)[0],
                    lw=0.5, alpha=0.2, ls=":")
    tick_p = np.arange(0, L, max(1, L//8))
    dl = [str(d)[-8:] for d in dates[:L]]
    ax1.set_xticks(tick_p)
    ax1.set_xticklabels([dl[i] if i<len(dl) else "" for i in tick_p], fontsize=5)
    ax1.set_ylabel("Price ($)")
    ax1.set_title(f"{ticker} — {L} bars, {len(snapshots)} forecasts (green=price, dash=pred, dot=actual)")
    ax1.legend(loc="upper left", ncol=4, fontsize=6)
    ax1.grid(True, alpha=0.10)

    # Panel 2: Raw per-step returns (proof of non-flatness)
    ax2 = axes[1]
    for si, s in enumerate(snapshots):
        ax2.plot(np.arange(len(s["mu"])), s["mu"]*100, color=fc(si)[0],
                lw=0.7, alpha=0.7, label=f"F{si+1}" if si<6 else "")
    ax2.axhline(0, color="gray", ls="--", alpha=0.3)
    ax2.set_ylabel("Return/step (%)")
    ax2.set_title("Raw per-step predictions — each line = 21 autoregressive steps")
    ax2.legend(loc="upper right", ncol=4, fontsize=5)
    ax2.grid(True, alpha=0.10)

    # Panel 3: Cumulative return paths
    ax3 = axes[2]
    for si, s in enumerate(snapshots):
        cum_p = np.cumsum(s["mu"])*100
        cum_a = np.cumsum(s["actual_ret"])*100
        st = np.arange(len(cum_p))
        ax3.plot(st, cum_p, color=fc(si)[0], lw=0.8, alpha=0.7,
                label=f"F{si+1}" if si<6 else "")
        ax3.plot(st[:len(cum_a)], cum_a, color=fc(si)[0],
                lw=0.4, alpha=0.2, ls=":")
    ax3.axhline(0, color="gray", ls="--", alpha=0.3)
    ax3.set_ylabel("Cum return (%)")
    ax3.set_title("Cumulative return (solid=pred, dotted=actual)")
    ax3.grid(True, alpha=0.10)

    # Panel 4: Error vs forecast distance
    ax4 = axes[3]
    all_err = []
    for s in snapshots:
        for h in range(min(len(s["mu"]), len(s["actual_ret"]))):
            all_err.append((h, (s["mu"][h]-s["actual_ret"][h])*100))
    if all_err:
        se = np.array([e[0] for e in all_err])
        ea = np.array([e[1] for e in all_err])
        hb = ax4.hexbin(se, ea, gridsize=15, cmap="YlOrRd", mincnt=1)
        plt.colorbar(hb, ax=ax4, label="Count", shrink=0.6)
        max_s = int(se.max())+1
        rmse = np.array([np.sqrt(np.mean(ea[se==s]**2)) if (se==s).any() else 0
                         for s in range(max_s)])
        ax4.plot(np.where(rmse>0)[0], rmse[rmse>0], "k-", lw=1.5, label="RMSE%")
        ax4.legend(fontsize=6)
    ax4.set_xlabel("Forecast step"); ax4.set_ylabel("Error (%)")
    ax4.set_title(f"Error vs Distance — {len(all_err)} points")
    ax4.grid(True, alpha=0.10)

    # Panel 5: Mid-snapshot detail with uncertainty
    ax5 = axes[4]
    s = snapshots[len(snapshots)//2]
    mu = s["mu"]; sigma = np.clip(s["sigma"], 0, 0.10)
    st = np.arange(len(mu))
    ax5.plot(st, mu*100, color="#9b59b6", lw=1.2, label="Pred mu")
    ax5.fill_between(st, (mu-sigma)*100, (mu+sigma)*100,
                     alpha=0.15, color="#9b59b6", label="pm sigma")
    al = min(len(s["actual_ret"]), len(mu))
    ax5.scatter(st[:al], s["actual_ret"][:al]*100, s=12,
               color="#e74c3c", alpha=0.6, label="Actual")
    ax5.axhline(0, color="gray", ls="--", alpha=0.3)
    ax5.set_xlabel("Forecast step"); ax5.set_ylabel("Return (%)")
    ax5.set_title(f"Detail bar {s['bar']} (${s['ref_price']:.0f}), "
                  f"{s['uniq']}/{len(mu)} unique steps")
    ax5.legend(fontsize=7); ax5.grid(True, alpha=0.10)

    fig.tight_layout()
    if outpath:
        fig.savefig(outpath, dpi=120, bbox_inches="tight", facecolor="white")
        print(f"  Saved {outpath}")
    plt.close(fig)
    return fig


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/cascade_anp.pt")
    p.add_argument("--data", default="data/intraday_5m.parquet")
    p.add_argument("--ticker", default="AAPL")
    p.add_argument("--n-context", type=int, default=60)
    p.add_argument("--n-forecast", type=int, default=21)
    p.add_argument("--encode-len", type=int, default=800)
    p.add_argument("--n-snapshots", type=int, default=8)
    p.add_argument("--output", default=None)
    p.add_argument("--all-tickers", action="store_true")
    args = p.parse_args()

    df = pd.read_parquet(args.data)
    fc = get_feature_cols(df)
    model, cfg = load_model(args.checkpoint, len(fc))
    print(f"Loaded: {sum(p.numel() for p in model.parameters()):,} params")

    if args.all_tickers:
        for t in [t for t,g in df.groupby("ticker") if len(g)>=500][:5]:
            print(f"\n--- {t} ---")
            visualize_forecast(model, df, t, fc,
                              args.n_context, args.n_forecast,
                              args.encode_len, args.n_snapshots,
                              f"forecast_viz_{t}.png")
    else:
        visualize_forecast(model, df, args.ticker, fc,
                          args.n_context, args.n_forecast,
                          args.encode_len, args.n_snapshots, args.output)
