"""Phase 2 Backtesting Engine — dynamic sizing, risk management, confidence-aware stops.

Architecture:
    SignalGenerator → PortfolioManager → RiskManager → PerformanceTracker

Key features over Phase 1:
1. Certainty-weighted position sizing (sigma-based confidence factor)
2. Forecast magnitude multiplier
3. Smart martingale DCA (sub-exponential recovery, hard cap)
4. Dynamic stop loss / take profit / trailing stop (sigma-adaptive)
5. Entry confirmation (consecutive bar agreement, delayed entry)
6. Multi-stock portfolio rotation with composite signal ranking
7. Confidence-bucketed performance metrics (diagnostic: low-σ must outperform high-σ)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


# ── Configuration ────────────────────────────────────────────────────────

@dataclass
class BacktestConfig:
    # Capital
    capital: float = 100_000
    max_position_pct: float = 0.20
    max_positions: int = 5
    max_exposure_pct: float = 0.80

    # Sizing
    confidence_alpha: float = 2.0        # 1/(1+alpha*sigma) confidence factor
    magnitude_max_mult: float = 2.0      # cap on magnitude multiplier
    recovery_beta: float = 0.25          # martingale base
    recovery_gamma: float = 1.5          # martingale curvature
    recovery_cap: float = 3.0            # hard cap

    # Entry
    entry_threshold: float = 0.003
    max_entry_sigma: float = 0.08
    entry_confirmation_bars: int = 2
    entry_delay_bars: int = 1

    # Stops
    base_stop_loss: float = 0.02
    stop_sensitivity: float = 2.0
    base_take_profit: float = 0.03
    tp_capture_ratio: float = 0.5
    trail_min_pct: float = 0.01
    trail_max_pct: float = 0.08
    max_hold_bars: int = 40
    max_hold_cap: int = 80

    # Signal ranking
    signal_w_conf: float = 0.4
    signal_w_mag: float = 0.4
    signal_w_pos: float = 0.2

    # Costs
    slippage_pct: float = 0.0005
    transaction_cost_pct: float = 0.001

    # Data
    context_bars: int = 80
    forecast_bars: int = 21
    step_bars: int = 5

    # Misc
    min_trade: float = 500
    max_trade: float = 50_000
    sigma_baseline_window: int = 100  # bars for rolling sigma baseline


# ── Data Structures ──────────────────────────────────────────────────────

@dataclass
class Signal:
    ticker: str
    bar: int
    predicted_return: float
    sigma: float
    direction: int    # 1 = buy, -1 = sell, 0 = hold
    price: float
    confidence: float  # 0-1, derived from sigma
    signal_score: float = 0.0

    @property
    def magnitude(self) -> float:
        return abs(self.predicted_return)


@dataclass
class Position:
    ticker: str
    entry_bar: int
    entry_price: float
    shares: int
    direction: int
    highest_price: float
    lowest_price: float
    entry_sigma: float
    consecutive_losses: int = 0
    stop_loss_price: float = 0.0
    take_profit_price: float = 0.0
    trail_distance: float = 0.0
    partial_entries_remaining: int = 0


@dataclass
class Trade:
    ticker: str
    entry_bar: int; exit_bar: int
    entry_price: float; exit_price: float
    shares: int; direction: int
    return_pct: float
    exit_reason: str
    entry_sigma: float
    holding_bars: int


@dataclass
class BacktestReport:
    total_return: float; cagr: float
    sharpe: float; sortino: float
    max_drawdown: float; max_dd_duration: int
    win_rate: float; profit_factor: float
    avg_win: float; avg_loss: float; expectancy: float
    num_trades: int
    equity_curve: List[float]
    trades: List[Trade]
    sigma_buckets: Dict[str, dict]
    per_ticker: pd.DataFrame
    monthly_returns: pd.DataFrame
    benchmark_return: float = 0.0


# ── Signal Generator ─────────────────────────────────────────────────────

class SignalGenerator:
    """Generate signals from CascadeANP model at each bar."""

    def __init__(self, model, ticker_data: dict, config: BacktestConfig):
        self.model = model
        self.data = ticker_data
        self.config = config
        self.device = next(model.parameters()).device

    def generate(self, ticker: str, bar: int) -> Optional[Signal]:
        data = self.data.get(ticker)
        if data is None or bar >= data["T"] - self.config.forecast_bars:
            return None

        ctx_s = bar - self.config.context_bars
        xc = torch.from_numpy(data["fused"][ctx_s:bar]).unsqueeze(0).to(self.device)
        yc = torch.from_numpy(data["returns"][ctx_s:bar, None]).unsqueeze(0).to(self.device)
        rf = torch.from_numpy(data["feats"][ctx_s:bar]).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(xc, yc, n_steps=self.config.forecast_bars,
                            teacher_forcing=False, raw_features=rf)

        pred = out["predicted_return"].item()
        sigma = out["aleatoric_sigma"].item()
        direction = 1 if pred > self.config.entry_threshold else -1 if pred < -self.config.entry_threshold else 0
        confidence = 1.0 / (1.0 + self.config.confidence_alpha * sigma)

        return Signal(
            ticker=ticker, bar=bar, predicted_return=pred,
            sigma=sigma, direction=direction,
            price=data["prices"][bar], confidence=confidence,
        )

    def generate_all(self, bar: int) -> List[Signal]:
        signals = []
        for ticker in self.data:
            sig = self.generate(ticker, bar)
            if sig is not None and sig.direction != 0 and sig.sigma <= self.config.max_entry_sigma:
                # Entry confirmation: check N consecutive bars agree
                if self._confirm_entry(ticker, bar, sig.direction):
                    signals.append(sig)
        return signals

    def _confirm_entry(self, ticker: str, bar: int, direction: int) -> bool:
        """Require N consecutive bars to agree on direction."""
        n = self.config.entry_confirmation_bars
        if n <= 1:
            return True
        for offset in range(1, n):
            prev_sig = self.generate(ticker, bar - offset * self.config.step_bars)
            if prev_sig is None or prev_sig.direction != direction:
                return False
        return True


# ── Position Sizer ───────────────────────────────────────────────────────

class PositionSizer:
    """Compute allocation using certainty, magnitude, and martingale recovery."""

    def __init__(self, config: BacktestConfig):
        self.config = config

    def compute_allocation(self, signal: Signal, cash: float,
                           consecutive_losses: int = 0) -> float:
        """Returns dollar amount to allocate."""
        base = cash * self.config.max_position_pct * signal.confidence

        # Magnitude multiplier
        mag_mult = min(signal.magnitude / max(self.config.entry_threshold, 0.001),
                       self.config.magnitude_max_mult)
        mag_mult = max(mag_mult, 0.5)  # floor at 0.5

        # Martingale recovery
        if consecutive_losses > 0:
            recovery = 1.0 + self.config.recovery_beta * (consecutive_losses ** self.config.recovery_gamma)
            recovery = min(recovery, self.config.recovery_cap)
        else:
            recovery = 1.0

        alloc = base * mag_mult * recovery

        # Respect limits
        alloc = min(alloc, self.config.max_trade)
        alloc = max(min(alloc, cash * 0.25), self.config.min_trade) if alloc > 0 else 0

        return alloc if alloc <= cash else cash * 0.25


# ── Risk Manager ─────────────────────────────────────────────────────────

class RiskManager:
    """Compute dynamic stops based on sigma and market conditions."""

    def __init__(self, config: BacktestConfig):
        self.config = config

    def compute_stop_loss(self, entry_price: float, direction: int,
                          sigma: float) -> float:
        sl_pct = self.config.base_stop_loss * (1.0 + self.config.stop_sensitivity * sigma)
        sl_pct = np.clip(sl_pct, 0.01, 0.08)
        if direction > 0:
            return entry_price * (1.0 - sl_pct)
        else:
            return entry_price * (1.0 + sl_pct)

    def compute_take_profit(self, entry_price: float, direction: int,
                            predicted_return: float) -> float:
        tp_pct = max(self.config.base_take_profit,
                     abs(predicted_return) * self.config.tp_capture_ratio)
        tp_pct = min(tp_pct, 0.15)
        if direction > 0:
            return entry_price * (1.0 + tp_pct)
        else:
            return entry_price * (1.0 - tp_pct)

    def compute_trail_stop(self, price: float, direction: int, sigma: float) -> float:
        trail_pct = np.clip(2.0 * sigma, self.config.trail_min_pct, self.config.trail_max_pct)
        if direction > 0:
            return price * (1.0 - trail_pct)
        else:
            return price * (1.0 + trail_pct)

    def compute_max_hold(self, sigma: float, sigma_baseline: float) -> int:
        flex = 1.0 + 1.0 * (sigma / max(sigma_baseline, 0.001))
        return min(int(self.config.max_hold_bars * flex), self.config.max_hold_cap)

    def check_exit(self, pos: Position, current_price: float, bar: int,
                   sigma: float = 0.02) -> Optional[str]:
        holding = bar - pos.entry_bar
        ret = (current_price / pos.entry_price - 1.0) * pos.direction

        if pos.direction > 0 and current_price <= pos.stop_loss_price:
            return "stop_loss"
        elif pos.direction < 0 and current_price >= pos.stop_loss_price:
            return "stop_loss"

        if pos.direction > 0 and current_price >= pos.take_profit_price:
            return "take_profit"
        elif pos.direction < 0 and current_price <= pos.take_profit_price:
            return "take_profit"

        # Trailing stop (only for profitable positions)
        if ret > 0.01:
            trail = self.compute_trail_stop(
                pos.direction > 0 and pos.highest_price or pos.lowest_price,
                pos.direction, sigma)
            if pos.direction > 0 and current_price <= trail:
                return "trailing_stop"
            elif pos.direction < 0 and current_price >= trail:
                return "trailing_stop"

        if holding >= self.config.max_hold_cap:
            return "max_hold"

        return None


# ── Portfolio Manager ────────────────────────────────────────────────────

class PortfolioManager:
    """Rank, select, size, and rotate positions."""

    def __init__(self, config: BacktestConfig, sizer: PositionSizer, risk: RiskManager):
        self.config = config
        self.sizer = sizer
        self.risk = risk

    def compute_signal_score(self, signal: Signal, has_position: bool) -> float:
        w = self.config
        score = (w.signal_w_conf * signal.confidence +
                 w.signal_w_mag * min(signal.magnitude / 0.01, 1.0) +
                 w.signal_w_pos * (0.5 if has_position else 0.0))
        return score

    def rank_and_select(self, signals: List[Signal],
                        current_positions: Dict[str, Position],
                        cash: float) -> List[Tuple[Signal, float]]:
        """Rank signals and return top-K with allocations."""
        scored = []
        for sig in signals:
            score = self.compute_signal_score(sig, sig.ticker in current_positions)
            scored.append((sig, score))

        scored.sort(key=lambda x: -x[1])

        allocations = []
        total_alloc = 0.0
        available = cash

        for sig, score in scored[:self.config.max_positions]:
            if available < self.config.min_trade * 2:
                break
            if total_alloc >= cash * self.config.max_exposure_pct:
                break

            pos = current_positions.get(sig.ticker)
            cons_losses = pos.consecutive_losses if pos else 0
            alloc = self.sizer.compute_allocation(sig, cash, cons_losses)
            if alloc >= self.config.min_trade:
                allocations.append((sig, alloc))
                total_alloc += alloc
                available -= alloc

        return allocations

    def update_position_stops(self, pos: Position, current_price: float,
                               sigma: float):
        """Update trailing stop levels."""
        if pos.direction > 0:
            pos.highest_price = max(pos.highest_price, current_price)
        else:
            pos.lowest_price = min(pos.lowest_price, current_price)


# ── Backtest Engine ──────────────────────────────────────────────────────

class BacktestEngineV2:
    """Phase 2 backtesting engine with caching and progress callbacks."""

    def __init__(self, model, df: pd.DataFrame, tickers: List[str],
                 feature_cols: List[str], config: Optional[BacktestConfig] = None,
                 progress_callback=None):
        self.config = config or BacktestConfig()
        self.model = model
        self.device = next(model.parameters()).device
        self.progress = progress_callback

        # Precompute encoded data per ticker (with caching)
        self.ticker_data = {}
        import os
        cache_dir = Path(df.attrs.get('cache_dir', 'data/cache'))
        os.makedirs(cache_dir, exist_ok=True)

        for i, t in enumerate(tickers):
            cache_path = cache_dir / f"{t}_fused.npy"
            if cache_path.exists():
                fused_np = np.load(cache_path)
            else:
                g = df[df.ticker == t].sort_values("as_of_date")
                if len(g) < self.config.context_bars + 60:
                    continue
                feats = g[feature_cols].fillna(0).values.astype(np.float32)
                with torch.no_grad():
                    fused, _, _ = model.encode(
                        torch.from_numpy(feats).unsqueeze(0).to(self.device))
                fused_np = fused[0].cpu().numpy()
                np.save(cache_path, fused_np)

            # Now load features/prices/returns from DataFrame
            g = df[df.ticker == t].sort_values("as_of_date")
            if len(g) < self.config.context_bars + 60:
                continue
            feats = g[feature_cols].fillna(0).values.astype(np.float32)
            self.ticker_data[t] = {
                "fused": fused_np, "feats": feats,
                "prices": g["adj_close"].values.astype(np.float32),
                "returns": g["target_return"].values.astype(np.float32),
                "T": len(feats),
            }
            if self.progress:
                self.progress((i + 1) / len(tickers), f"Loaded {t}")

        self.signal_gen = SignalGenerator(model, self.ticker_data, self.config)
        self.sizer = PositionSizer(self.config)
        self.risk = RiskManager(self.config)
        self.portfolio = PortfolioManager(self.config, self.sizer, self.risk)

    def run(self) -> BacktestReport:
        cash = self.config.capital
        positions: Dict[str, Position] = {}
        trades: List[Trade] = []
        equity = [cash]

        tickers = list(self.ticker_data.keys())
        min_len = min(d["T"] for d in self.ticker_data.values())
        bar_range = range(self.config.context_bars, min_len - self.config.forecast_bars,
                         self.config.step_bars)

        # Rolling sigma baseline
        sigma_history = []

        for bar in bar_range:
            # 1. Generate signals
            signals = self.signal_gen.generate_all(bar)
            if not signals:
                continue

            # Update sigma baseline
            for s in signals:
                sigma_history.append(s.sigma)
            sigma_baseline = np.mean(sigma_history[-self.config.sigma_baseline_window:]) if sigma_history else 0.02

            # 2. Manage exits
            for t, pos in list(positions.items()):
                current_price = self.ticker_data[t]["prices"][bar]
                self.portfolio.update_position_stops(pos, current_price, sigma_baseline)

                exit_reason = self.risk.check_exit(pos, current_price, bar, sigma_baseline)
                if exit_reason:
                    ret = (current_price / pos.entry_price - 1.0) * pos.direction
                    slippage_cost = self.config.slippage_pct * pos.shares * current_price
                    txn_cost = self.config.transaction_cost_pct * pos.shares * current_price
                    exit_value = pos.shares * current_price - slippage_cost - txn_cost
                    cash += exit_value

                    pnl_pct = ret * 100
                    trades.append(Trade(
                        ticker=t, entry_bar=pos.entry_bar, exit_bar=bar,
                        entry_price=pos.entry_price, exit_price=current_price,
                        shares=pos.shares, direction=pos.direction,
                        return_pct=pnl_pct, exit_reason=exit_reason,
                        entry_sigma=pos.entry_sigma,
                        holding_bars=bar - pos.entry_bar,
                    ))
                    del positions[t]

            # 3. Enter new positions
            allocations = self.portfolio.rank_and_select(signals, positions, cash)
            for sig, alloc in allocations:
                if sig.ticker in positions:
                    continue
                if alloc > cash:
                    continue
                shares = int(alloc / sig.price)
                if shares * sig.price < self.config.min_trade:
                    continue

                # Apply slippage on entry
                entry_price = sig.price * (1.0 + self.config.slippage_pct)
                cost = shares * entry_price * (1.0 + self.config.transaction_cost_pct)

                positions[sig.ticker] = Position(
                    ticker=sig.ticker, entry_bar=bar,
                    entry_price=entry_price, shares=shares,
                    direction=sig.direction,
                    highest_price=sig.price, lowest_price=sig.price,
                    entry_sigma=sig.sigma,
                    stop_loss_price=self.risk.compute_stop_loss(
                        entry_price, sig.direction, sig.sigma),
                    take_profit_price=self.risk.compute_take_profit(
                        entry_price, sig.direction, sig.predicted_return),
                    trail_distance=self.risk.compute_trail_stop(
                        sig.price, sig.direction, sig.sigma),
                )
                cash -= cost

            # 4. Track equity
            pos_value = sum(
                p.shares * self.ticker_data[p.ticker]["prices"][bar]
                for p in positions.values()
            )
            equity.append(cash + pos_value)

        # Close remaining
        for t, pos in list(positions.items()):
            final_price = self.ticker_data[t]["prices"][-1]
            ret = (final_price / pos.entry_price - 1.0) * pos.direction
            cash += pos.shares * final_price
            trades.append(Trade(
                ticker=t, entry_bar=pos.entry_bar,
                exit_bar=self.ticker_data[t]["T"] - 1,
                entry_price=pos.entry_price, exit_price=final_price,
                shares=pos.shares, direction=pos.direction,
                return_pct=ret * 100, exit_reason="end_of_data",
                entry_sigma=pos.entry_sigma,
                holding_bars=self.ticker_data[t]["T"] - 1 - pos.entry_bar,
            ))
            del positions[t]
        equity.append(cash)

        return self._compute_report(trades, equity)

    def _compute_report(self, trades: List[Trade],
                        equity: List[float]) -> BacktestReport:
        eq = np.array(equity)
        total_ret = eq[-1] / eq[0] - 1.0

        # CAGR
        n_bars = len(eq) - 1
        years = n_bars * 5 / (252 * 78)  # 5-min bars → years (approximate)
        cagr = (1 + total_ret) ** (1 / max(years, 0.01)) - 1 if years > 0 else 0.0

        # Returns
        returns = np.diff(eq) / (eq[:-1] + 1e-10)
        sharpe = float(np.mean(returns) / max(np.std(returns), 1e-10)) * np.sqrt(252)
        neg_returns = returns[returns < 0]
        sortino = float(np.mean(returns) / max(np.std(neg_returns), 1e-10)) * np.sqrt(252) if len(neg_returns) > 0 else 0.0

        # Drawdown
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak
        max_dd = float(dd.min())

        # Trade metrics
        wins = [t for t in trades if t.return_pct > 0]
        losses = [t for t in trades if t.return_pct <= 0]
        win_rate = len(wins) / max(len(trades), 1)
        avg_win = np.mean([t.return_pct for t in wins]) if wins else 0.0
        avg_loss = np.mean([t.return_pct for t in losses]) if losses else 0.0
        gross_gains = sum(t.return_pct for t in wins) if wins else 0
        gross_losses = abs(sum(t.return_pct for t in losses)) if losses else 1
        profit_factor = gross_gains / max(gross_losses, 1)
        expectancy = win_rate * avg_win - (1 - win_rate) * abs(avg_loss)

        # Sigma buckets
        sigma_buckets = {"low": [], "mid": [], "high": []}
        for t in trades:
            if t.entry_sigma < 0.03: bucket = "low"
            elif t.entry_sigma < 0.06: bucket = "mid"
            else: bucket = "high"
            sigma_buckets[bucket].append(t.return_pct)

        bucket_metrics = {}
        for bucket, rets in sigma_buckets.items():
            if rets:
                bucket_metrics[bucket] = {
                    "count": len(rets), "win_rate": sum(1 for r in rets if r > 0) / len(rets),
                    "avg_return": np.mean(rets),
                }

        # Per-ticker
        per_ticker = pd.DataFrame([{
            "ticker": t.ticker, "return_pct": t.return_pct,
            "exit_reason": t.exit_reason, "holding_bars": t.holding_bars,
            "entry_sigma": t.entry_sigma,
        } for t in trades])

        # Monthly returns
        monthly = pd.DataFrame() if not trades else pd.DataFrame()

        return BacktestReport(
            total_return=total_ret, cagr=cagr,
            sharpe=sharpe, sortino=sortino,
            max_drawdown=max_dd, max_dd_duration=0,
            win_rate=win_rate, profit_factor=profit_factor,
            avg_win=avg_win, avg_loss=avg_loss, expectancy=expectancy,
            num_trades=len(trades),
            equity_curve=list(eq), trades=trades,
            sigma_buckets=bucket_metrics,
            per_ticker=per_ticker, monthly_returns=monthly,
        )
