from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from .trainer import Trainer
from ..features.indicators import fetch_ohlcv, compute_indicators

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    ticker: str
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    direction: str
    ret: float
    signal_confidence: float = 0.0


@dataclass
class BacktestResult:
    trades: List[Trade]
    equity_curve: pd.Series
    n_trades: int
    win_rate: float
    avg_ret: float
    median_ret: float
    avg_hold_days: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    cagr: float
    profit_factor: float
    total_return: float
    by_ticker: pd.DataFrame
    params: Dict[str, any] = field(default_factory=dict)
    daily_allocations: pd.DataFrame = field(default_factory=pd.DataFrame)
    daily_cash: pd.Series = field(default_factory=pd.Series)
    daily_equity: pd.Series = field(default_factory=pd.Series)
    daily_signals: Dict[str, pd.DataFrame] = field(default_factory=dict)
    trade_log: List[Dict] = field(default_factory=list)


class AIBacktestEngine:
    """Backtesting engine for AI-generated trading signals.

    Uses the TemporalStateModel to generate event-state probabilities and
    time-to-event forecasts, then simulates a simultaneous multi-ticker
    portfolio with daily rebalancing and position management.
    """

    def __init__(
        self,
        trainer: Trainer,
        entry_threshold: float = 0.60,
        exit_threshold: float = 0.40,
        max_hold_days: int = 21,
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.0,
        use_forecast_for_exit: bool = True,
        top_n: int = 5,
        cash_return_annual: float = 0.04,
    ):
        self.trainer = trainer
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.max_hold_days = max_hold_days
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.use_forecast_for_exit = use_forecast_for_exit
        self.top_n = top_n
        self.cash_return_annual = cash_return_annual

    def run(
        self,
        tickers: List[str],
        period: str = "5y",
        interval: str = "1d",
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None,
    ) -> BacktestResult:
        return self._backtest_portfolio(
            tickers=tickers,
            period=period,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
        )

    def _backtest_portfolio(
        self,
        tickers: List[str],
        period: str = "5y",
        interval: str = "1d",
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None,
    ) -> BacktestResult:
        ticker_dfs: Dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            df = fetch_ohlcv(ticker, period=period, interval=interval)
            if df.empty or len(df) < 100:
                logger.warning(f"Skipping {ticker}: insufficient data ({len(df)} rows)")
                continue
            df = compute_indicators(df)
            df = df[df.index >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df.index <= pd.to_datetime(end_date)]
            if len(df) < 100:
                continue
            ticker_dfs[ticker] = df
            time.sleep(0.5)

        valid_tickers = sorted(ticker_dfs.keys())
        if not valid_tickers:
            return self._empty_result()

        all_dates = sorted(set(
            d for df in ticker_dfs.values()
            for d in df.index
        ))
        if not all_dates:
            return self._empty_result()

        date_range = all_dates
        min_warmup = self.trainer.config.lookback

        daily_signals: Dict[str, dict] = {}
        for ticker, df in ticker_dfs.items():
            if len(df) <= min_warmup:
                continue
            sig = self._generate_signals_for_ticker(df, date_range, min_warmup)
            daily_signals[ticker] = sig

        initial_cash = 1.0
        cash = initial_cash
        initial_equity = initial_cash

        per_ticker_allocation = pd.DataFrame(
            0.0,
            index=pd.DatetimeIndex(date_range),
            columns=valid_tickers,
        )
        daily_cash_series = pd.Series(0.0, index=pd.DatetimeIndex(date_range))
        daily_equity_series = pd.Series(0.0, index=pd.DatetimeIndex(date_range))

        positions: Dict[str, Optional[dict]] = {t: None for t in valid_tickers}

        trade_list: List[Trade] = []
        trade_log: List[Dict] = []
        active_position_snapshots: List[Dict] = []

        daily_risk_cash_return = (1 + self.cash_return_annual) ** (1 / 252) - 1

        prev_portfolio_value = initial_equity

        for day_idx, current_date in enumerate(date_range):
            if day_idx == 0:
                daily_equity_series.loc[current_date] = initial_equity
                daily_cash_series.loc[current_date] = cash
                continue

            prev_portfolio_value = daily_equity_series.iloc[day_idx - 1]

            for ticker, pos in positions.items():
                if pos is None:
                    continue
                if current_date in ticker_dfs[ticker].index:
                    current_price = float(
                        ticker_dfs[ticker].loc[current_date, "adj close"]
                        if "adj close" in ticker_dfs[ticker].columns
                        else ticker_dfs[ticker].loc[current_date, "Close"]
                    )
                    pos["highest_price"] = max(pos.get("highest_price", current_price), current_price)

            positions_pnl = {}
            for ticker, pos in positions.items():
                if pos is None:
                    continue
                if current_date in ticker_dfs[ticker].index:
                    pnl = pos["shares"] * ticker_dfs[ticker].loc[current_date]["adj close"] if "adj close" in ticker_dfs[ticker].columns else pos["shares"] * ticker_dfs[ticker].loc[current_date]["Close"]
                    positions_pnl[ticker] = float(pnl)

            portfolio_value = cash + sum(positions_pnl.values())
            cash += cash * daily_risk_cash_return

            positions_to_close: List[str] = []
            for ticker, pos in positions.items():
                if pos is None:
                    continue
                if current_date not in ticker_dfs[ticker].index:
                    continue

                current_price = float(
                    ticker_dfs[ticker].loc[current_date, "adj close"]
                    if "adj close" in ticker_dfs[ticker].columns
                    else ticker_dfs[ticker].loc[current_date, "Close"]
                )
                hold_days = (current_date - pos["entry_date"]).days
                exit_signal = False
                exit_reason = ""

                sig = daily_signals.get(ticker, {})
                prob_inter = sig.get("prob_inter_event", np.zeros(1))[day_idx] if day_idx < len(sig.get("prob_inter_event", np.zeros(1))) else 0.0
                forecast_tau = sig.get("forecast_tau", np.full(1, 21.0))[day_idx] if day_idx < len(sig.get("forecast_tau", np.full(1, 21.0))) else 21.0

                if prob_inter >= self.exit_threshold:
                    exit_signal = True
                    exit_reason = "model_exit"

                if self.stop_loss_pct > 0 and current_price <= pos["entry_price"] * (1 - self.stop_loss_pct):
                    exit_signal = True
                    exit_reason = "stop_loss"

                if self.take_profit_pct > 0 and current_price >= pos["entry_price"] * (1 + self.take_profit_pct):
                    exit_signal = True
                    exit_reason = "take_profit"

                highest = pos.get("highest_price", current_price)
                if current_price <= highest * (1 - 0.03):
                    exit_signal = True
                    exit_reason = "trailing_stop"

                if hold_days >= self.max_hold_days:
                    exit_signal = True
                    exit_reason = "max_hold"

                if self.use_forecast_for_exit and forecast_tau > self.max_hold_days * 0.8:
                    exit_signal = True
                    exit_reason = "forecast_exit"

                if exit_signal:
                    positions_to_close.append(ticker)
                    ret = current_price / pos["entry_price"] - 1.0
                    cash += pos["shares"] * current_price
                    trade = Trade(
                        ticker=ticker,
                        entry_date=pos["entry_date"],
                        entry_price=pos["entry_price"],
                        exit_date=current_date,
                        exit_price=current_price,
                        direction="long",
                        ret=ret,
                        signal_confidence=float(pos.get("entry_confidence", 0.0)),
                    )
                    trade_list.append(trade)
                    trade_log.append({
                        "ticker": ticker,
                        "entry_date": pos["entry_date"],
                        "exit_date": current_date,
                        "entry_price": pos["entry_price"],
                        "exit_price": current_price,
                        "return": ret,
                        "hold_days": hold_days,
                        "exit_reason": exit_reason,
                        "confidence": float(pos.get("entry_confidence", 0.0)),
                    })

            for ticker in positions_to_close:
                positions[ticker] = None

            risk_scores = {}
            for ticker in valid_tickers:
                if positions[ticker] is not None:
                    continue
                sig = daily_signals.get(ticker, {})
                if not sig:
                    continue
                prob_pre = sig.get("prob_pre_event", np.zeros(1))[day_idx] if day_idx < len(sig.get("prob_pre_event", np.zeros(1))) else 0.0
                prob_onset = sig.get("prob_onset", np.zeros(1))[day_idx] if day_idx < len(sig.get("prob_onset", np.zeros(1))) else 0.0
                risk = prob_pre + prob_onset * 1.5
                if risk >= self.entry_threshold:
                    risk_scores[ticker] = risk

            if risk_scores and cash > 0:
                ranked = sorted(risk_scores.items(), key=lambda x: x[1], reverse=True)
                top_candidates = ranked[:self.top_n]
                if top_candidates:
                    allocation_per_ticker = cash / len(top_candidates)
                    for ticker, risk in top_candidates:
                        if current_date not in ticker_dfs[ticker].index:
                            continue
                        entry_price = float(
                            ticker_dfs[ticker].loc[current_date, "adj close"]
                            if "adj close" in ticker_dfs[ticker].columns
                            else ticker_dfs[ticker].loc[current_date, "Close"]
                        )
                        shares = allocation_per_ticker / entry_price
                        positions[ticker] = {
                            "entry_date": current_date,
                            "entry_price": entry_price,
                            "shares": shares,
                            "highest_price": entry_price,
                            "entry_confidence": risk,
                        }
                        cash -= allocation_per_ticker
                        per_ticker_allocation.loc[current_date, ticker] = allocation_per_ticker

            new_positions_pnl = {}
            for ticker, pos in positions.items():
                if pos is None:
                    continue
                if current_date in ticker_dfs[ticker].index:
                    price = float(
                        ticker_dfs[ticker].loc[current_date, "adj close"]
                        if "adj close" in ticker_dfs[ticker].columns
                        else ticker_dfs[ticker].loc[current_date, "Close"]
                    )
                    new_positions_pnl[ticker] = pos["shares"] * price

            portfolio_value = cash + sum(new_positions_pnl.values())
            daily_equity_series.loc[current_date] = portfolio_value
            daily_cash_series.loc[current_date] = cash

            for ticker in valid_tickers:
                if positions[ticker] is not None and current_date in ticker_dfs[ticker].index:
                    price = float(
                        ticker_dfs[ticker].loc[current_date, "adj close"]
                        if "adj close" in ticker_dfs[ticker].columns
                        else ticker_dfs[ticker].loc[current_date, "Close"]
                    )
                    per_ticker_allocation.loc[current_date, ticker] = positions[ticker]["shares"] * price

        daily_allocations = per_ticker_allocation

        daily_signals_dfs: Dict[str, pd.DataFrame] = {}
        for ticker, sig_dict in daily_signals.items():
            df_sig = pd.DataFrame({
                "prob_inter_event": sig_dict.get("prob_inter_event", np.zeros(len(date_range)))[:len(date_range)],
                "prob_pre_event": sig_dict.get("prob_pre_event", np.zeros(len(date_range)))[:len(date_range)],
                "prob_onset": sig_dict.get("prob_onset", np.zeros(len(date_range)))[:len(date_range)],
                "forecast_tau": sig_dict.get("forecast_tau", np.full(len(date_range), 21.0))[:len(date_range)],
            }, index=pd.DatetimeIndex(date_range[:len(sig_dict.get("prob_inter_event", np.zeros(len(date_range))))]))
            daily_signals_dfs[ticker] = df_sig

        if trade_list:
            trade_list.sort(key=lambda t: t.exit_date)
            rets = np.array([t.ret for t in trade_list])

            daily_ret = daily_equity_series.pct_change().fillna(0.0)

            max_dd = self._max_drawdown(daily_equity_series.values)
            sharpe = self._sharpe(daily_ret.values)
            sortino = self._sortino(daily_ret.values)
            cagr_val = self._cagr(daily_equity_series.values)
            profit_factor = self._profit_factor(rets)

            win_rate = float((rets > 0).mean())
            avg_ret = float(np.mean(rets))
            median_ret = float(np.median(rets))
            avg_hold = float(np.mean([
                (t.exit_date - t.entry_date).days for t in trade_list
            ]))

            total_return = float(daily_equity_series.iloc[-1] / daily_equity_series.iloc[0] - 1.0)

            equity_curve = daily_equity_series.copy()
        else:
            equity_curve = pd.Series([1.0])
            max_dd = 0.0
            sharpe = 0.0
            sortino = 0.0
            cagr_val = 0.0
            profit_factor = 0.0
            win_rate = 0.0
            avg_ret = 0.0
            median_ret = 0.0
            avg_hold = 0.0
            total_return = 0.0

        ticker_stats: List[Dict] = []
        for ticker in valid_tickers:
            ticker_trades = [t for t in trade_list if t.ticker == ticker]
            if ticker_trades:
                ticker_rets = [t.ret for t in ticker_trades]
                ticker_stats.append({
                    "ticker": ticker,
                    "n_trades": len(ticker_trades),
                    "win_rate": float((np.array(ticker_rets) > 0).mean()),
                    "avg_ret": float(np.mean(ticker_rets)),
                    "total_return": float(np.prod([1 + r for r in ticker_rets]) - 1),
                })

        return BacktestResult(
            trades=trade_list,
            equity_curve=equity_curve,
            n_trades=len(trade_list),
            win_rate=win_rate,
            avg_ret=avg_ret,
            median_ret=median_ret,
            avg_hold_days=avg_hold,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            cagr=cagr_val,
            profit_factor=profit_factor,
            total_return=total_return,
            by_ticker=pd.DataFrame(ticker_stats) if ticker_stats else pd.DataFrame(),
            params={
                "entry_threshold": self.entry_threshold,
                "max_hold_days": self.max_hold_days,
                "stop_loss_pct": self.stop_loss_pct,
                "top_n": self.top_n,
            },
            daily_allocations=daily_allocations,
            daily_cash=daily_cash_series,
            daily_equity=daily_equity_series,
            daily_signals=daily_signals_dfs,
            trade_log=trade_log,
        )

    def _generate_signals_for_ticker(
        self, df: pd.DataFrame, shared_dates: List[pd.Timestamp], min_warmup: int,
    ) -> Dict[str, np.ndarray]:
        from ..data.labeler import _build_features_at
        from .dataset import FEATURE_COLUMNS, _prepare_features

        n = len(df)
        lookback = self.trainer.config.lookback
        feature_dim = self.trainer.model.input_dim

        feature_rows = []
        for i in range(lookback, n):
            feat = _build_features_at(df, i, ticker="ticker", forecast_horizon=21)
            feature_rows.append(feat)

        n_shared = len(shared_dates)
        probs_inter = np.zeros(n_shared)
        probs_pre = np.zeros(n_shared)
        probs_onset = np.zeros(n_shared)
        forecast_tau = np.full(n_shared, 21.0)

        if not feature_rows:
            return {
                "prob_inter_event": probs_inter,
                "prob_pre_event": probs_pre,
                "prob_onset": probs_onset,
                "forecast_tau": forecast_tau,
            }

        feat_df = pd.DataFrame(feature_rows)
        feat_df = _prepare_features(feat_df)
        expected_cols = [c for c in FEATURE_COLUMNS if c in feat_df.columns]

        if len(expected_cols) != feature_dim:
            if len(expected_cols) > feature_dim:
                expected_cols = expected_cols[:feature_dim]
            else:
                expected_cols = expected_cols + [expected_cols[0]] * (feature_dim - len(expected_cols))
                expected_cols = expected_cols[:feature_dim]

        feature_data = feat_df[expected_cols].fillna(0.0).values.astype(np.float32)
        if feature_data.shape[1] != feature_dim:
            if feature_data.shape[1] > feature_dim:
                feature_data = feature_data[:, :feature_dim]
            else:
                pad = np.zeros((feature_data.shape[0], feature_dim - feature_data.shape[1]), dtype=np.float32)
                feature_data = np.concatenate([feature_data, pad], axis=1)

        n_features = len(feature_data)

        self.trainer.model.eval()
        with torch.no_grad():
            for i in range(min(lookback, n_features - 1) + 1, n_features):
                if i < lookback:
                    continue
                window = feature_data[i - lookback:i]
                if window.shape[0] != lookback:
                    continue
                x = torch.from_numpy(window).unsqueeze(0).to(self.trainer.device)
                try:
                    outputs = self.trainer.model(x)
                    probs = outputs["past_state_probs"].cpu().numpy()[0]
                    idx_in_df = min(lookback + i, n - 1)
                    if idx_in_df < n_shared:
                        probs_inter[idx_in_df] = probs[0]
                        probs_pre[idx_in_df] = probs[1]
                        probs_onset[idx_in_df] = probs[2]
                        forecast_tau[idx_in_df] = float(outputs["future_forecast"].cpu().numpy()[0].mean())
                except Exception:
                    continue

        return {
            "prob_inter_event": probs_inter,
            "prob_pre_event": probs_pre,
            "prob_onset": probs_onset,
            "forecast_tau": forecast_tau,
        }

    @staticmethod
    def _max_drawdown(equity: np.ndarray) -> float:
        if len(equity) < 2:
            return 0.0
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        return float(drawdown.min())

    @staticmethod
    def _sharpe(returns: np.ndarray, rf: float = 0.02) -> float:
        if len(returns) < 2:
            return 0.0
        excess = returns - (rf / 252)
        if excess.std() == 0:
            return 0.0
        return float(np.sqrt(252) * excess.mean() / excess.std())

    @staticmethod
    def _sortino(returns: np.ndarray, rf: float = 0.02) -> float:
        if len(returns) < 2:
            return 0.0
        excess = returns - (rf / 252)
        downside = excess[excess < 0]
        if len(downside) == 0 or downside.std() == 0:
            return 0.0
        return float(np.sqrt(252) * excess.mean() / downside.std())

    @staticmethod
    def _cagr(equity: np.ndarray) -> float:
        if len(equity) < 2:
            return 0.0
        total_return = equity[-1] / equity[0]
        years = len(equity) / 252
        if years <= 0 or total_return <= 0:
            return 0.0
        return float(total_return ** (1 / years) - 1)

    @staticmethod
    def _profit_factor(returns: np.ndarray) -> float:
        winners = returns[returns > 0].sum()
        losers = abs(returns[returns < 0].sum())
        if losers == 0:
            return float("inf") if winners > 0 else 0.0
        return float(winners / losers)

    @staticmethod
    def _empty_result() -> BacktestResult:
        return BacktestResult(
            trades=[],
            equity_curve=pd.Series([1.0]),
            n_trades=0,
            win_rate=0.0,
            avg_ret=0.0,
            median_ret=0.0,
            avg_hold_days=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            cagr=0.0,
            profit_factor=0.0,
            total_return=0.0,
            by_ticker=pd.DataFrame(),
        )
