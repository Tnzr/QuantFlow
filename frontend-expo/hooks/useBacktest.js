import { useState, useCallback } from "react";
import { apiGet } from "../services/api";

export default function useBacktest(token) {
  const [config, setConfig] = useState({});
  const [equityCurve, setEquityCurve] = useState([]);
  const [trades, setTrades] = useState([]);
  const [summary, setSummary] = useState(null);
  const [multiResults, setMultiResults] = useState([]);
  const [portfolioEquity, setPortfolioEquity] = useState([]);
  const [sigmaBuckets, setSigmaBuckets] = useState(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(null);
  const [running, setRunning] = useState(false);
  const [animating, setAnimating] = useState(false);

  const run = useCallback(
    async (cfg) => {
      if (!cfg.tickers || cfg.tickers.length === 0) {
        setError("Select at least one ticker");
        return;
      }
      setRunning(true);
      setLoading(true);
      setError(null);
      setProgress(0);
      setConfig(cfg);
      setMultiResults([]);
      setPortfolioEquity([]);
      setAnimating(false);

      try {
        const params = new URLSearchParams();
        if (cfg.start) params.set("start", cfg.start);
        if (cfg.end) params.set("end", cfg.end);
        if (cfg.rsi_threshold != null) params.set("entry_rsi_threshold", cfg.rsi_threshold);
        if (cfg.max_hold_days) params.set("max_hold_days", cfg.max_hold_days);
        if (cfg.stop_loss) params.set("stop_loss_pct", cfg.stop_loss);
        if (cfg.take_profit) params.set("take_profit_pct", cfg.take_profit);
        if (cfg.ma_filter) params.set("ma_filter", cfg.ma_filter);
        if (cfg.ma_trend) params.set("ma_trend_filter", cfg.ma_trend);
        if (cfg.capital) params.set("capital", cfg.capital);
        if (cfg.use_ml_forecast) {
          params.set("use_ml_forecast", "true");
          if (cfg.ml_threshold != null) params.set("ml_threshold", cfg.ml_threshold);
          if (cfg.ml_min_confidence != null) params.set("ml_min_confidence", cfg.ml_min_confidence);
        }

        const isMulti = cfg.tickers.length > 1;
        const path = isMulti ? "/backtest/multi" : "/backtest/short-term";
        if (isMulti) {
          params.set("tickers", cfg.tickers.join(","));
        } else {
          params.set("ticker", cfg.tickers.join(","));
        }

        const data = await apiGet(`${path}?${params.toString()}`, token);

        if (isMulti) {
          setMultiResults(data.results || []);
          setPortfolioEquity(data.portfolio_equity || []);
          const firstResult = data.results?.find(r => r.n_trades > 0) || data.results?.[0];
          setSummary(firstResult ? {
            n_trades: data.results.reduce((s, r) => s + (r.n_trades || 0), 0),
            win_rate: data.results.length ? data.results.reduce((s, r) => s + (r.win_rate || 0), 0) / data.results.length : 0,
            avg_ret: data.results.length ? data.results.reduce((s, r) => s + (r.avg_ret || 0), 0) / data.results.length : 0,
            sharpe: data.results.length ? data.results.reduce((s, r) => s + (r.sharpe || 0), 0) / data.results.length : 0,
            max_dd: Math.max(...data.results.map(r => r.max_dd || 0), 0),
            cagr: 0,
            profit_factor: 0,
            sortino: 0,
            isMulti: true,
          } : null);
          setEquityCurve(data.portfolio_equity || []);
        } else {
          setEquityCurve(data.equity || []);
          setTrades(data.trades || []);
          setSummary(data.summary || null);
          setSigmaBuckets(data.sigma_buckets || data.sigma || null);
        }
        setProgress(100);
        setAnimating(true);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
        setRunning(false);
      }
    },
    [token]
  );

  return {
    config,
    equityCurve,
    trades,
    summary,
    multiResults,
    portfolioEquity,
    sigmaBuckets,
    loading,
    progress,
    error,
    running,
    animating,
    setAnimating,
    run,
  };
}
