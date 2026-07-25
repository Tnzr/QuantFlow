#!/usr/bin/env python3
"""Backtest CascadeANP with portfolio management, stop loss, and take profit.

Usage:
    python scripts/backtest_cascade.py --tickers AAPL,MSFT,GOOGL --capital 100000

Signals: predicted_return > 0.2% => BUY, < -0.2% => SELL.
Exits: stop_loss (3%), take_profit (5%), trailing_stop (2%), max_hold (21 bars).
"""
import argparse, time, numpy as np, pandas as pd, torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path

@dataclass
class TradeRecord:
    ticker: str; entry_bar: int; exit_bar: int; entry_price: float; exit_price: float
    return_pct: float; exit_reason: str; highest_price: float

@dataclass
class BacktestResult:
    equity_curve: List[float]; trades: List[TradeRecord]
    total_return: float; sharpe: float; max_drawdown: float
    win_rate: float; num_trades: int

def run_backtest(model, df, tickers, feature_cols, config):
    """Multi-ticker portfolio backtest with confidence-based sizing + martingale DCA."""
    all_trades = []
    equity = [config.capital]
    cash = config.capital
    positions = {}  # ticker -> {entry_price, shares, direction, highest, entry_bar, consecutive_losses}
    # Precompute fused representations per ticker
    ticker_data = {}
    device = next(model.parameters()).device

    for t in tickers:
        g = df[df.ticker == t].sort_values("as_of_date")
        if len(g) < config.context_bars + 60: continue
        feats = g[feature_cols].fillna(0).values.astype(np.float32)
        prices = g["adj_close"].values.astype(np.float32)
        returns = g["target_return"].values.astype(np.float32)
        with torch.no_grad():
            fused, _, _ = model.encode(torch.from_numpy(feats).unsqueeze(0).to(device))
        ticker_data[t] = {"fused": fused[0].cpu().numpy(), "prices": prices,
                          "returns": returns, "feats": feats, "T": len(feats)}

    # Find common time range
    min_len = min(d["T"] for d in ticker_data.values())
    bar_range = range(config.context_bars, min_len - config.forecast_bars, config.step_bars)

    for bar in bar_range:
        # 1. Generate signals for ALL tickers at this bar
        signals = []  # (ticker, pred_return, sigma, confidence, price, direction)
        for t, data in ticker_data.items():
            if bar >= data["T"] - config.forecast_bars: continue
            ctx_s = bar - config.context_bars
            xc = torch.from_numpy(data["fused"][ctx_s:bar]).unsqueeze(0).to(device)
            yc = torch.from_numpy(data["returns"][ctx_s:bar, None]).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(xc, yc, n_steps=config.forecast_bars, teacher_forcing=False)
            pred = out["predicted_return"].item()
            sig = out["aleatoric_sigma"].item()
            price = data["prices"][bar]
            direction = 1 if pred > 0 else -1 if pred < 0 else 0
            if direction == 0: continue
            # Confidence: inverse sigma, capped
            confidence = 1.0 / max(sig, 0.005)
            signals.append((t, pred, sig, confidence, price, direction))

        if not signals: continue
        # Rank by confidence descending
        signals.sort(key=lambda x: -x[3])

        # 2. Manage exits for existing positions
        for t, pos in list(positions.items()):
            price = ticker_data[t]["prices"][bar]
            pos["highest"] = max(pos["highest"], price)
            ret = (price / pos["entry_price"] - 1.0) * pos["direction"]
            exit_reason = None

            if ret <= -config.stop_loss: exit_reason = "stop_loss"
            elif ret >= config.take_profit: exit_reason = "take_profit"
            elif pos["direction"] > 0 and price <= pos["highest"] * (1 - config.trailing_stop):
                exit_reason = "trailing_stop"
            elif holding := bar - pos["entry_bar"] >= config.max_hold_bars:
                exit_reason = "max_hold"
            # Check if this ticker is still in top signals with same direction
            t_signal = next((s for s in signals if s[0] == t), None)
            if t_signal and t_signal[5] != pos["direction"]:
                exit_reason = exit_reason or "model_exit"

            if exit_reason:
                exit_val = pos["shares"] * price
                cash += exit_val
                pnl = ret * 100
                all_trades.append(TradeRecord(t, pos["entry_bar"], bar,
                    pos["entry_price"], price, pnl, exit_reason, pos["highest"]))
                # Track consecutive losses for martingale
                if pnl < 0:
                    pos["consecutive_losses"] = pos.get("consecutive_losses", 0) + 1
                del positions[t]

        # 3. Enter new positions: top-N by confidence, with martingale sizing
        available = [s for s in signals if s[0] not in positions]
        for i, (t, pred, sig, conf, price, direction) in enumerate(available):
            if cash < config.min_trade: break
            if len(positions) >= config.max_positions: break

            # Confidence-based sizing: higher confidence = bigger allocation
            conf_scale = min(conf / 10.0, 1.0)  # normalize confidence to [0,1]
            base_alloc = cash * config.position_size * conf_scale

            # Martingale DCA: scale up after consecutive losses for this ticker
            prev_losses = sum(1 for tr in all_trades if tr.ticker == t and tr.return_pct < 0)
            martingale_mult = min(2.0 ** (prev_losses % 4), 4.0)  # cap at 4x
            alloc = min(base_alloc * martingale_mult, config.max_trade)

            shares = alloc / price
            if shares * price > config.min_trade:
                positions[t] = {"entry_price": price, "shares": shares,
                                "highest": price, "entry_bar": bar,
                                "direction": direction, "consecutive_losses": 0}
                cash -= shares * price

        # Track equity
        pos_value = sum(p["shares"] * ticker_data[p_t]["prices"][bar]
                        for p_t, p in positions.items() if p_t in ticker_data)
        equity.append(cash + pos_value)

    # Close remaining
    for t, pos in list(positions.items()):
        lp = ticker_data[t]["prices"][-1]
        ret = (lp / pos["entry_price"] - 1.0) * pos["direction"]
        cash += pos["shares"] * lp
        all_trades.append(TradeRecord(t, pos["entry_bar"], ticker_data[t]["T"]-1,
            pos["entry_price"], lp, ret*100, "end_of_data", pos["highest"]))
        del positions[t]
    equity.append(cash)
    return compute_metrics(all_trades, equity)


def compute_metrics(trades: List[TradeRecord], equity: List[float]) -> BacktestResult:
    eq = np.array(equity)
    total_ret = eq[-1] / eq[0] - 1.0
    daily_ret = np.diff(eq) / eq[:-1] + 1e-10
    sharpe = float(np.mean(daily_ret) / max(np.std(daily_ret), 1e-10)) * np.sqrt(252)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    max_dd = float(dd.min())
    wins = sum(1 for t in trades if t.return_pct > 0)
    wr = wins / max(len(trades), 1)
    return BacktestResult(
        equity_curve=list(eq), trades=trades,
        total_return=total_ret, sharpe=sharpe, max_drawdown=max_dd,
        win_rate=wr, num_trades=len(trades),
    )


def print_results(r: BacktestResult):
    print(f"\n{'='*60}")
    print(f"BACKTEST RESULTS")
    print(f"{'='*60}")
    print(f"Total return:  {r.total_return*100:.1f}%")
    print(f"Sharpe ratio:  {r.sharpe:.2f}")
    print(f"Max drawdown:  {r.max_drawdown*100:.1f}%")
    print(f"Win rate:      {r.win_rate*100:.1f}% ({r.num_trades} trades)")
    if r.trades:
        rets = [t.return_pct for t in r.trades]
        print(f"Avg trade ret: {np.mean(rets):.1f}%")
        print(f"Best trade:    {np.max(rets):.1f}%")
        print(f"Worst trade:   {np.min(rets):.1f}%")
        reasons = {}
        for t in r.trades:
            reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
        print(f"Exit reasons:  {reasons}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/cascade_anp.pt")
    p.add_argument("--data", default="data/intraday_5m.parquet")
    p.add_argument("--tickers", default="AAPL,MSFT,GOOGL,META,AMZN,NVDA,TSLA,AMD")
    p.add_argument("--capital", type=float, default=100000)
    p.add_argument("--context-bars", type=int, default=60)
    p.add_argument("--forecast-bars", type=int, default=21)
    p.add_argument("--step-bars", type=int, default=5)
    p.add_argument("--entry-threshold", type=float, default=0.002)
    p.add_argument("--stop-loss", type=float, default=0.03)
    p.add_argument("--take-profit", type=float, default=0.05)
    p.add_argument("--trailing-stop", type=float, default=0.02)
    p.add_argument("--max-hold-bars", type=int, default=40)
    p.add_argument("--position-size", type=float, default=0.2)
    p.add_argument("--min-trade", type=float, default=500)
    p.add_argument("--max-trade", type=float, default=50000)
    p.add_argument("--max-positions", type=int, default=5)
    args = p.parse_args()

    df = pd.read_parquet(args.data)
    label_cols = {"ticker","event_state","event_state_code","is_volatile",
        "tau_forward","target_return","target_5d","target_21d",
        "target_1h","target_4h","drawdown_5d_max","drawdown_21d_max",
        "as_of_date","adj_close","close","target_direction_5d",
        "target_direction_21d","sector_idx","target_return_path"}
    fc = sorted([c for c in df.columns if c not in label_cols and df[c].dtype != 'object'])

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from quantflow.models.cascade_anp import CascadeANP

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = CascadeANP(len(fc), cfg.get("hidden_dim",128),
                       cfg.get("latent_dim",64), cfg.get("cascade_layers",4))
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval().to("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loaded: {sum(p.numel() for p in model.parameters()):,} params, "
          f"epoch={ckpt.get('epoch','?')}, val_loss={ckpt.get('val_loss','?'):.4f}")

    tickers = [t.strip() for t in args.tickers.split(",")]
    tickers = [t for t in tickers if t in df.ticker.unique()]
    print(f"Backtesting {len(tickers)} tickers with ${args.capital:,.0f}")
    print(f"Stop: {args.stop_loss*100}%, TP: {args.take_profit*100}%, "
          f"Trail: {args.trailing_stop*100}%, MaxHold: {args.max_hold_bars} bars")

    result = run_backtest(model, df, tickers, fc, args)
    print_results(result)
