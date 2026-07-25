"""Simple RSI-based short-term backtest engine.
Returns a report object with summary statistics and equity curve.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class ShortTermBacktestReport:
    n_trades: int = 0
    win_rate: float = 0.0
    avg_ret: float = 0.0
    median_ret: float = 0.0
    avg_hold_days: float = 0.0
    max_dd: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    cagr: float = 0.0
    profit_factor: float = 0.0
    equity_curve: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["date", "equity"]))
    trades: list[dict] = field(default_factory=list)


def backtest_short_term(
    ticker: str,
    start: str = "2018-01-01",
    end: str | None = None,
    entry_rsi_threshold: float = 50.0,
    max_hold_days: int = 7,
    stop_loss_pct: float = 0.0,
    take_profit_pct: float = 0.0,
    ma_filter: str = "sma20",
    ma_trend_filter: str = "none",
) -> ShortTermBacktestReport:
    """Run a short-term RSI-based backtest for a single ticker.

    Strategy:
      - Entry when RSI(14) drops below `entry_rsi_threshold` AND price is above MA
      - Exit when max_hold_days reached OR stop_loss_pct hit OR take_profit_pct hit
    """
    # ── Fetch data ──────────────────────────────────────────────────────
    end_str = end if end and str(end).strip() else None
    df = yf.download(ticker, start=start, end=end_str, progress=False, auto_adjust=True)
    if df.empty:
        return ShortTermBacktestReport()

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = df["Close"].squeeze()
    if close.empty or len(close) < 50:
        return ShortTermBacktestReport()

    # ── Compute indicators ───────────────────────────────────────────────
    rsi = _rsi(close, 14)
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()

    # Select MA filter
    ma_map = {"sma20": sma20, "sma50": sma50, "sma200": sma200}
    ma_series = ma_map.get(ma_filter, sma20)

    # MA trend filter
    trend_ok = pd.Series(True, index=close.index)
    if ma_trend_filter == "above_sma50":
        trend_ok = close > sma50
    elif ma_trend_filter == "above_sma200":
        trend_ok = close > sma200

    # ── Simulate trades ──────────────────────────────────────────────────
    equity = [1.0]
    trades_list = []
    in_position = False
    entry_idx = 0
    entry_price = 0.0

    for i in range(50, len(close)):
        if not in_position:
            # Entry: RSI below threshold AND price above MA AND trend OK
            if (
                rsi.iloc[i] < entry_rsi_threshold
                and close.iloc[i] > ma_series.iloc[i]
                and trend_ok.iloc[i]
            ):
                in_position = True
                entry_idx = i
                entry_price = float(close.iloc[i])
        else:
            hold = i - entry_idx
            current_price = float(close.iloc[i])
            ret = (current_price - entry_price) / entry_price
            exit_reason = None

            # Time exit
            if hold >= max_hold_days:
                exit_reason = "time"
            # Stop loss
            elif stop_loss_pct > 0 and ret <= -stop_loss_pct:
                exit_reason = "stop_loss"
            # Take profit
            elif take_profit_pct > 0 and ret >= take_profit_pct:
                exit_reason = "take_profit"

            if exit_reason:
                trades_list.append({
                    "ticker": ticker,
                    "action": "buy",
                    "entry_price": round(entry_price, 2),
                    "exit_price": round(current_price, 2),
                    "return_pct": round(ret, 6),
                    "entry_date": str(close.index[entry_idx].date()),
                    "exit_date": str(close.index[i].date()),
                    "hold_days": hold,
                    "exit_reason": exit_reason,
                    "model_sigma": 0.02 + abs(ret) * 0.5,
                })
                equity.append(equity[-1] * (1.0 + ret))
                in_position = False
                continue

    # Close any open position at the end
    if in_position:
        current_price = float(close.iloc[-1])
        ret = (current_price - entry_price) / entry_price
        trades_list.append({
            "ticker": ticker,
            "action": "buy",
            "entry_price": round(entry_price, 2),
            "exit_price": round(current_price, 2),
            "return_pct": round(ret, 6),
            "entry_date": str(close.index[entry_idx].date()),
            "exit_date": str(close.index[-1].date()),
            "hold_days": len(close) - entry_idx,
            "exit_reason": "end",
            "model_sigma": 0.02 + abs(ret) * 0.5,
        })
        equity.append(equity[-1] * (1.0 + ret))

    # Fill equity to match date range
    eq_series = pd.Series(equity, index=close.index[:len(equity)])
    eq_series = eq_series.reindex(close.index).ffill()
    eq_df = eq_series.reset_index()
    eq_df.columns = ["date", "equity"]

    # ── Compute metrics ──────────────────────────────────────────────────
    n_trades = len(trades_list)
    if n_trades == 0:
        return ShortTermBacktestReport(
            n_trades=0,
            equity_curve=eq_df,
        )

    returns = np.array([t["return_pct"] for t in trades_list])
    win_rate = float(np.mean(returns > 0))
    avg_ret = float(np.mean(returns))
    median_ret = float(np.median(returns))
    avg_hold = float(np.mean([t["hold_days"] for t in trades_list]))

    # Daily returns for Sharpe / Sortino
    daily = eq_series.pct_change().dropna()
    sharpe = 0.0
    sortino = 0.0
    if len(daily) > 1 and daily.std() > 1e-12:
        sharpe = float(daily.mean() / daily.std() * np.sqrt(252))
        downside = daily[daily < 0]
        if len(downside) > 1 and downside.std() > 1e-12:
            sortino = float(daily.mean() / downside.std() * np.sqrt(252))

    # Max drawdown
    roll_max = eq_series.cummax()
    dd = (eq_series - roll_max) / roll_max
    max_dd = float(abs(dd.min())) if len(dd) > 0 else 0.0

    # CAGR
    total_days = (close.index[-1] - close.index[0]).days
    years = total_days / 365.25
    cagr = 0.0
    if years > 0 and eq_series.iloc[0] > 0:
        cagr = float((eq_series.iloc[-1] / eq_series.iloc[0]) ** (1.0 / years) - 1.0)

    # Profit factor
    gains = max(0.0, returns[returns > 0].sum())
    losses = max(1e-12, abs(returns[returns < 0].sum()))
    profit_factor = float(gains / losses)

    return ShortTermBacktestReport(
        n_trades=n_trades,
        win_rate=win_rate,
        avg_ret=avg_ret,
        median_ret=median_ret,
        avg_hold_days=avg_hold,
        max_dd=max_dd,
        sharpe=sharpe,
        sortino=sortino,
        cagr=cagr,
        profit_factor=profit_factor,
        equity_curve=eq_df,
        trades=trades_list,
    )


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI indicator."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))