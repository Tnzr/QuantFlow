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
    rsi_series: list[dict] = field(default_factory=list)  # [{date, rsi, signal}]
    signals: list[dict] = field(default_factory=list)  # [{date, type: "buy"/"sell", price, reason}]


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

    # Select MA filter — "none" means skip MA check
    # For mean-reversion (RSI-based) strategy: we WANT price below MA when entering
    # (buying the dip). So the MA filter checks that price is NOT in a strong uptrend
    # (i.e., price should be near or below MA for a pullback entry).
    use_ma = ma_filter and ma_filter.lower() != "none"
    ma_map = {"sma20": sma20, "sma50": sma50, "sma200": sma200}
    ma_series = ma_map.get(ma_filter, sma20) if use_ma else None

    # MA trend filter
    trend_ok = pd.Series(True, index=close.index)
    if ma_trend_filter == "above_sma50":
        trend_ok = (close > sma50).fillna(False)
    elif ma_trend_filter == "above_sma200":
        trend_ok = (close > sma200).fillna(False)

    # ── Simulate trades ──────────────────────────────────────────────────
    # Track equity at every bar so the curve is continuous
    equity = [1.0]  # equity at bar 0
    trades_list = []
    in_position = False
    entry_idx = 0
    entry_price = 0.0
    current_equity = 1.0  # tracks portfolio value bar-by-bar

    # Pre-compute ML signals for the backtest period
    # For short backtests, call ML periodically; for long ones, use a single call
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

    # Use a local RSI threshold that adapts to ML direction
    # When ML says BUY: be more aggressive (higher RSI threshold = easier entry)
    # When ML says SELL: be very conservative (much lower threshold)
    # When ML says HOLD: use base threshold
    ml_dir = (ml_pred or {}).get("direction", "HOLD")
    ml_conf = float((ml_pred or {}).get("confidence", 0) or 0)

    if ml_dir == "BUY":
        # ML bullish: easier to enter (higher threshold)
        effective_rsi_threshold = entry_rsi_threshold * 1.5
    elif ml_dir == "SELL":
        # ML bearish: much harder to enter (very low threshold)
        effective_rsi_threshold = entry_rsi_threshold * 0.3
    else:
        # HOLD: use base threshold
        effective_rsi_threshold = entry_rsi_threshold

    # Track signals and RSI for visualization
    signals_list = []
    rsi_series_list = []

    for i in range(50, len(close)):
        # Record RSI for this bar
        rsi_val = float(rsi.iloc[i]) if not pd.isna(rsi.iloc[i]) else 50.0
        bar_date = str(close.index[i].date())
        rsi_series_list.append({"date": bar_date, "rsi": round(rsi_val, 2)})

        if not in_position:
            # Entry: RSI condition, optionally filtered by ML direction
            rsi_ok = rsi.iloc[i] < effective_rsi_threshold

            # MA filter: allow entry when price is near or below MA (pullback)
            # This makes sense for mean-reversion RSI strategy
            if ma_series is not None and not pd.isna(ma_series.iloc[i]):
                # Allow entry if price is within 5% above MA or below MA
                ma_val = ma_series.iloc[i]
                price = close.iloc[i]
                ma_ok = price <= ma_val * 1.05  # near or below MA
            else:
                ma_ok = True

            if (
                rsi_ok
                and ma_ok
                and trend_ok.iloc[i]
            ):
                in_position = True
                entry_idx = i
                entry_price = float(close.iloc[i])
                signals_list.append({
                    "date": bar_date,
                    "type": "buy",
                    "price": round(entry_price, 2),
                    "reason": f"RSI {rsi_val:.1f} < {effective_rsi_threshold:.0f}" + (f" + ML {ml_dir}" if use_ml_forecast else ""),
                })
        else:
            hold = i - entry_idx
            current_price = float(close.iloc[i])
            ret = (current_price - entry_price) / entry_price
            exit_reason = None

            if hold >= max_hold_days:
                exit_reason = "time"
            elif stop_loss_pct > 0 and ret <= -stop_loss_pct:
                exit_reason = "stop_loss"
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
                signals_list.append({
                    "date": bar_date,
                    "type": "sell",
                    "price": round(current_price, 2),
                    "reason": f"{exit_reason} ({ret*100:+.1f}%)",
                })
                current_equity = current_equity * (1.0 + ret)
                in_position = False

        # Track equity at every bar
        if in_position:
            unrealized = (float(close.iloc[i]) - entry_price) / entry_price
            bar_equity = current_equity * (1.0 + unrealized)
        else:
            bar_equity = current_equity
        equity.append(bar_equity)

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
        current_equity = current_equity * (1.0 + ret)
        if equity:
            equity[-1] = current_equity
        signals_list.append({
            "date": str(close.index[-1].date()),
            "type": "sell",
            "price": round(current_price, 2),
            "reason": f"end ({ret*100:+.1f}%)",
        })

    # Build equity series aligned to close index
    # equity[0] = 1.0 at bar 0, equity[i] tracks portfolio value at bar i+50 (first tradable bar)
    eq_index = close.index[:50] if len(close) >= 50 else close.index
    eq_values = [1.0] * len(eq_index)
    for i, val in enumerate(equity[1:]):
        bar_idx = 50 + i
        if bar_idx < len(close.index):
            eq_values.append(val)
    eq_series = pd.Series(equity[:len(close.index)], index=close.index[:len(equity)])
    eq_series = eq_series.reindex(close.index).ffill()
    eq_df = eq_series.reset_index()
    eq_df.columns = ["date", "equity"]

    # ── Compute metrics ──────────────────────────────────────────────────
    n_trades = len(trades_list)
    if n_trades == 0:
        return ShortTermBacktestReport(
            n_trades=0,
            equity_curve=eq_df,
            rsi_series=rsi_series_list,
            signals=signals_list,
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
        rsi_series=rsi_series_list,
        signals=signals_list,
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