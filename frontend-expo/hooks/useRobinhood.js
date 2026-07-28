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
    try {
      setError(null);
      // Check current demo status first
      const status = await apiGet("/robinhood/status", token);
      if (status.demo_mode) {
        // In demo mode: auto-complete OAuth without opening popup
        // This skips the Robinhood login flow and uses simulated data
        await apiPost("/robinhood/oauth/callback", { code: "demo_code_" + Date.now() }, token);
        await refreshStatus();
        return true;
      }
      // Real OAuth: get URL with PKCE, open in popup, poll for callback
      const oauthInfo = await apiGet("/robinhood/oauth/url", token);
      if (typeof window !== "undefined") {
        const popup = window.open(oauthInfo.url, "_blank", "width=600,height=700");
        if (!popup) {
          setError("Popup blocked. Please allow popups for Robinhood OAuth.");
          return false;
        }
        // Store PKCE info for callback
        if (oauthInfo.code_verifier) {
          localStorage.setItem("robinhood_code_verifier", oauthInfo.code_verifier);
          localStorage.setItem("robinhood_state", oauthInfo.state);
        }
        // Poll for popup close
        const pollInterval = setInterval(async () => {
          if (popup && popup.closed) {
            clearInterval(pollInterval);
            await refreshStatus();
          }
        }, 1000);
        // Auto-stop polling after 5 minutes
        setTimeout(() => clearInterval(pollInterval), 300000);
      } else {
        console.log("Open this URL to connect:", oauthInfo.url);
      }
      return true;
    } catch (e) {
      setError(e.message || "Connection failed");
      return false;
    }
  }, [token, refreshStatus]);

  // Handle OAuth callback (called from popup redirect)
  const handleOAuthCallback = useCallback(async (code, codeVerifier) => {
    try {
      setError(null);
      const payload = { code };
      if (codeVerifier) payload.code_verifier = codeVerifier;
      const result = await apiPost("/robinhood/oauth/callback", payload, token);
      if (result.connected) {
        localStorage.removeItem("robinhood_code_verifier");
        localStorage.removeItem("robinhood_state");
        await refreshStatus();
        return true;
      } else {
        setError(result.error || "OAuth callback failed");
        return false;
      }
    } catch (e) {
      setError(e.message || "OAuth callback failed");
      return false;
    }
  }, [token, refreshStatus]);

  // Manual token entry (for OAuth completed externally)
  const setManualToken = useCallback(async (accessToken, refreshToken = null, expiresIn = 3600) => {
    try {
      setError(null);
      await apiPost("/robinhood/oauth/manual-token", {
        access_token: accessToken,
        refresh_token: refreshToken,
        expires_in: expiresIn,
      }, token);
      await refreshStatus();
      return true;
    } catch (e) {
      setError(e.message || "Failed to set token");
      return false;
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
    setError,
    handleOAuthCallback,
    setManualToken,
  };
}
