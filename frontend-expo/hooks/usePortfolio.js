import { useState, useEffect, useCallback } from "react";
import { apiGet } from "../services/api";

export default function usePortfolio(token) {
  const [account, setAccount] = useState(null);
  const [positions, setPositions] = useState([]);
  const [signals, setSignals] = useState([]);
  const [equityCurve, setEquityCurve] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchPortfolio = useCallback(async () => {
    try {
      setError(null);
      const data = await apiGet("/portfolio/positions", token);
      setAccount(data.account || null);
      setPositions(data.positions || []);
      setEquityCurve(data.equity_curve || []);
    } catch (err) {
      setError(err.message);
    }
  }, [token]);

  const fetchSignals = useCallback(async () => {
    try {
      const data = await apiGet("/portfolio/signals", token);
      setSignals(data.signals || data || []);
    } catch {
      // signals are optional
    }
  }, [token]);

  const refresh = useCallback(async () => {
    setLoading(true);
    await Promise.all([fetchPortfolio(), fetchSignals()]);
    setLoading(false);
  }, [fetchPortfolio, fetchSignals]);

  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, 30000);
    return () => clearInterval(interval);
  }, [refresh]);

  return { account, positions, signals, equityCurve, loading, error, refresh };
}
