import { useState, useCallback, useEffect } from "react";
import { apiGet, apiPost } from "../services/api";

/**
 * Hook for managing Robinhood MCP connection.
 *
 * Each workspace has its own isolated MCP client (token storage
 * is keyed by workspace_id derived from the auth token).
 */
export default function useRobinhood(token) {
  const [status, setStatus] = useState(null);
  const [portfolio, setPortfolio] = useState(null);
  const [orders, setOrders] = useState(null);
  const [watchlist, setWatchlist] = useState(null);
  const [forecast, setForecast] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const refreshStatus = useCallback(async () => {
    try {
      const data = await apiGet("/robinhood/status", token);
      setStatus(data);
    } catch (e) {
      setError(e.message);
    }
  }, [token]);

  const connect = useCallback(async () => {
    setError(null);
    try {
      // Check current demo status first
      const status = await apiGet("/robinhood/status", token);
      if (status.demo_mode) {
        // In demo mode: auto-complete OAuth without opening popup
        // This skips the Robinhood login flow and uses simulated data
        await apiPost("/robinhood/oauth/callback", { code: "demo_code_" + Date.now() }, token);
        await refreshStatus();
        return;
      }
      // Real OAuth: open Robinhood in new window, poll for completion
      const { url } = await apiGet("/robinhood/oauth/url", token);
      if (typeof window !== "undefined") {
        const popup = window.open(url, "_blank", "width=600,height=700");
        if (!popup) {
          setError("Popup blocked. Please allow popups for Robinhood OAuth.");
          return;
        }
        // Poll for popup close
        const pollInterval = setInterval(async () => {
          if (popup && popup.closed) {
            clearInterval(pollInterval);
            await refreshStatus();
          }
        }, 1000);
      } else {
        console.log("Open this URL to connect:", url);
      }
    } catch (e) {
      setError(e.message);
    }
  }, [token, refreshStatus]);

  const disconnect = useCallback(async () => {
    setError(null);
    try {
      await apiPost("/robinhood/disconnect", {}, token);
      setStatus(null);
      setPortfolio(null);
      setForecast(null);
    } catch (e) {
      setError(e.message);
    }
  }, [token]);

  const setTradeOptIn = useCallback(async (enable) => {
    try {
      const data = await apiPost("/robinhood/trading/opt-in", { enable }, token);
      await refreshStatus();
      return data;
    } catch (e) {
      setError(e.message);
    }
  }, [token, refreshStatus]);

  const fetchPortfolio = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await apiGet("/robinhood/portfolio", token);
      setPortfolio(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [token]);

  const fetchOrders = useCallback(async () => {
    try {
      const data = await apiGet("/robinhood/orders?limit=20", token);
      setOrders(data);
    } catch (e) {
      setError(e.message);
    }
  }, [token]);

  const fetchWatchlist = useCallback(async () => {
    try {
      const data = await apiGet("/robinhood/watchlist", token);
      setWatchlist(data);
    } catch (e) {
      setError(e.message);
    }
  }, [token]);

  const importWatchlist = useCallback(async () => {
    try {
      const data = await apiPost("/robinhood/watchlist/import", {}, token);
      return data;
    } catch (e) {
      setError(e.message);
      return null;
    }
  }, [token]);

  const fetchForecast = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await apiGet("/robinhood/personalized/forecast", token);
      setForecast(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [token]);

  const addToWatchlist = useCallback(async (ticker) => {
    try {
      await apiPost("/robinhood/watchlist/add", { ticker }, token);
      await fetchWatchlist();
    } catch (e) {
      setError(e.message);
    }
  }, [token, fetchWatchlist]);

  const removeFromWatchlist = useCallback(async (ticker) => {
    try {
      await apiPost("/robinhood/watchlist/remove", { ticker }, token);
      await fetchWatchlist();
    } catch (e) {
      setError(e.message);
    }
  }, [token, fetchWatchlist]);

  const reviewOrder = useCallback(async (ticker, side, qty, orderType = "market", limitPrice = null) => {
    try {
      return await apiPost("/robinhood/order/review", {
        ticker, side, qty, order_type: orderType, limit_price: limitPrice,
      }, token);
    } catch (e) {
      setError(e.message);
      return null;
    }
  }, [token]);

  const placeOrder = useCallback(async (ticker, side, qty, orderType = "market", limitPrice = null) => {
    try {
      return await apiPost("/robinhood/order/place", {
        ticker, side, qty, order_type: orderType, limit_price: limitPrice,
      }, token);
    } catch (e) {
      setError(e.message);
      return null;
    }
  }, [token]);

  const llmResearch = useCallback(async (query, includePortfolio = true) => {
    try {
      return await apiPost("/robinhood/llm/research", { query, include_portfolio: includePortfolio }, token);
    } catch (e) {
      setError(e.message);
      return null;
    }
  }, [token]);

  useEffect(() => {
    refreshStatus();
  }, [refreshStatus]);

  return {
    status,
    portfolio,
    orders,
    watchlist,
    forecast,
    loading,
    error,
    connect,
    disconnect,
    setTradeOptIn,
    completeWithDemoCode,
    fetchPortfolio,
    fetchOrders,
    fetchWatchlist,
    importWatchlist,
    fetchForecast,
    addToWatchlist,
    removeFromWatchlist,
    reviewOrder,
    placeOrder,
    llmResearch,
    refreshStatus,
  };
}
