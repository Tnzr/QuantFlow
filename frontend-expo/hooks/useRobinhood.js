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
      const { url, workspace_id } = await apiGet("/robinhood/oauth/url", token);
      // In a real React Native app this would open the URL in a browser.
      // For web, we open a new window and listen for the callback.
      if (typeof window !== "undefined") {
        const popup = window.open(url, "_blank", "width=600,height=700");
        // Poll for completion (popup will close after OAuth completes)
        const pollInterval = setInterval(async () => {
          if (popup && popup.closed) {
            clearInterval(pollInterval);
            // For demo mode, auto-complete with a code
            await completeWithDemoCode();
          }
        }, 500);
      } else {
        // Native: would use Linking
        console.log("Open this URL to connect:", url);
      }
    } catch (e) {
      setError(e.message);
    }
  }, [token]);

  const completeWithDemoCode = useCallback(async () => {
    // For demo mode: directly complete OAuth with a simulated code
    try {
      await apiPost("/robinhood/oauth/callback", { code: "demo_code_" + Date.now() }, token);
      await refreshStatus();
      await fetchPortfolio();
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
