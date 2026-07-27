import { useState, useCallback } from "react";
import { apiGet, cacheMarketSeries } from "../services/api";

function toYfInterval(tf) {
  const map = { "1m": "1m", "5m": "5m", "15m": "15m", "1H": "1h", "1D": "1d", "1W": "1wk", "1M": "1mo" };
  return map[tf] || "1d";
}

export default function useMarketData(token) {
  const [ohlcv, setOhlcv] = useState([]);
  const [forecast, setForecast] = useState(null);
  const [ticker, setTicker] = useState("");
  const [timeframe, setTimeframe] = useState("1D");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const loadData = useCallback(
    async (sym) => {
      const symbol = sym || ticker;
      if (!symbol) return;
      setLoading(true);
      setError(null);
      try {
        const interval = toYfInterval(timeframe);
        const [cacheRes, foreRes] = await Promise.allSettled([
          cacheMarketSeries(symbol, interval, null, token),
          apiGet(`/charts/forecast?ticker=${encodeURIComponent(symbol)}&interval=${encodeURIComponent(timeframe)}`, token),
        ]);

        if (cacheRes.status === "fulfilled") {
          setOhlcv(cacheRes.value?.items || cacheRes.value?.data || cacheRes.value?.series || cacheRes.value || []);
        } else {
          setError(cacheRes.reason?.message || "Failed to load data");
        }
        if (foreRes.status === "fulfilled") {
          setForecast(foreRes.value);
        }
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    },
    [ticker, timeframe, token]
  );

  return {
    ohlcv,
    forecast,
    ticker,
    setTicker,
    timeframe,
    setTimeframe,
    loading,
    error,
    loadData,
  };
}
