"""Simple RSI-based short-term backtest engine.
Returns a report object with summary statistics and equity curve.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import requests


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


def _load_backtest_bars(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    """Fetch backtest data — Alpaca preferred, yfinance fallback."""
    from quantflow.features.indicators import fetch_ohlcv

    interval = "1d"
    # If yfinance fallback path is forced, period is derived from start..end
    import os
    from datetime import datetime
    prefer_yf = os.environ.get("ALPACA_PREFER_YFINANCE", "").strip() in ("1", "true", "yes")
    if prefer_yf:
        # Map start..end to yfinance period
        try:
            s = datetime.fromisoformat(str(start)[:10])
            e = datetime.fromisoformat(str(end)[:10]) if end else datetime.now()
            days = max((e - s).days, 30)
            if days <= 7:
                period = "5d"
            elif days <= 30:
                period = "1mo"
            elif days <= 90:
                period = "3mo"
            elif days <= 180:
                period = "6mo"
            elif days <= 365:
                period = "1y"
            elif days <= 365 * 2:
                period = "2y"
            elif days <= 365 * 5:
                period = "5y"
            else:
                period = "10y"
        except Exception:
            period = "5y"
        return fetch_ohlcv(ticker, period=period, interval=interval)
    # Alpaca path — derive a window large enough to cover [start, end]
    try:
        from datetime import datetime, timezone
        end_dt = datetime.fromisoformat(str(end)[:10]).replace(tzinfo=timezone.utc) if end else datetime.now(timezone.utc)
        start_dt = datetime.fromisoformat(str(start)[:10]).replace(tzinfo=timezone.utc)
        days = max((end_dt - start_dt).days, 30)
        # map to yfinance-equivalent period for the unified fetch_ohlcv path
        if days <= 7:
            period = "5d"
        elif days <= 30:
            period = "1mo"
        elif days <= 90:
            period = "3mo"
        elif days <= 180:
            period = "6mo"
        elif days <= 365:
            period = "1y"
        elif days <= 365 * 2:
            period = "2y"
        elif days <= 365 * 5:
            period = "5y"
        else:
            period = "10y"
        df = fetch_ohlcv(ticker, period=period, interval=interval)
        if df is not None and not df.empty:
            # Clip to the requested [start, end] window
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            mask = df.index >= pd.Timestamp(start_dt)
            if end:
                mask &= df.index <= pd.Timestamp(end_dt)
            df = df.loc[mask]
        return df
    except Exception:
        return fetch_ohlcv(ticker, period="5y", interval=interval)


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
    use_ml_forecast: bool = False,
    ml_api_url: str = "http://localhost:8000",
    ml_threshold: float = 0.002,
    ml_min_confidence: float = 0.6,
) -> ShortTermBacktestReport:
    """Run a short-term RSI-based backtest for a single ticker.

    Strategy:
      - Entry when RSI(14) drops below `entry_rsi_threshold` AND price is above MA
      - Exit when max_hold_days reached OR stop_loss_pct hit OR take_profit_pct hit
    """
    # ── Fetch data (Alpaca-first) ────────────────────────────────────────
    df = _load_backtest_bars(ticker, start, end)
    if df.empty:
        return ShortTermBacktestReport()

    # Alpaca path uses lowercase columns; yfinance may be title-cased
    if "close" not in df.columns and "Close" in df.columns:
        df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume", "Adj Close": "adj close"})
    if "close" not in df.columns:
        return ShortTermBacktestReport()

    close = df["close"].squeeze()
    if close.empty or len(close) < 50:
        return ShortTermBacktestReport()

    # ── Compute indicators ───────────────────────────────────────────────
    rsi = _rsi(close, 14)
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()

    # ── ML forecast (one-time call) ──────────────────────────────────────
    ml_pred = None
    if use_ml_forecast:
        try:
            resp = requests.post(f"{ml_api_url}/predict",
                                 json={"ticker": ticker},
                                 timeout=15)
            if resp.status_code == 200:
                ml_pred = resp.json()
        except Exception:
            ml_pred = None

    # Select MA filter — "none" means skip MA check
    use_ma = ma_filter and ma_filter.lower() != "none"
    ma_map = {"sma20": sma20, "sma50": sma50, "sma200": sma200}
    ma_series = ma_map.get(ma_filter, sma20) if use_ma else None

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
            # Entry: RSI condition, optionally filtered by ML direction
            # RSI entry with ML-adjusted threshold or pure ML signal
            if use_ml_forecast and ml_pred:
                ml_dir = ml_pred.get("direction", "HOLD")
                ml_conf = float(ml_pred.get("confidence", 0) or 0)
                # Design decision: When ML direction is SELL, we use a very strict RSI threshold (0.5x)
                # to avoid going long against a bearish signal. This may result in 0 trades for some tickers.
                # Users who want more trades can:
                #   1. Use RSI-only mode (use_ml_forecast=False)
                #   2. Increase entry_rsi_threshold parameter
                #   3. Modify the 0.5 multiplier to 0.7 (allows RSI < 21 instead of < 15)
                if ml_dir == "SELL":
                    # ML bearish: only enter on deep oversold
                    entry_threshold = entry_rsi_threshold * 0.5
                    rsi_ok = rsi.iloc[i] < entry_threshold and ml_conf < 0.55
                elif ml_dir == "BUY":
                    # ML bullish: enter on mild pullback or even pure ML when confidence high
                    if ml_conf >= ml_min_confidence and abs(float(ml_pred.get("predicted_return",0))) > ml_threshold:
                        rsi_ok = True  # pure ML entry
                    else:
                        rsi_ok = rsi.iloc[i] < entry_rsi_threshold * 1.4
                else:
                    rsi_ok = rsi.iloc[i] < entry_rsi_threshold
            else:
                entry_threshold = entry_rsi_threshold
                rsi_ok = rsi.iloc[i] < entry_threshold

            if (
                rsi_ok
                and (ma_series is None or close.iloc[i] > ma_series.iloc[i])
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
                    "ml_direction": ml_pred.get("direction") if ml_pred else None,
                    "ml_confidence": ml_pred.get("confidence") if ml_pred else None,
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
                    "ml_direction": ml_pred.get("direction") if ml_pred else None,
                    "ml_confidence": ml_pred.get("confidence") if ml_pred else None,
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