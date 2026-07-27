import { useState, useEffect, useCallback } from "react";
import { loadAuthToken, saveAuthToken, clearAuthToken } from "../services/auth";
import { apiGet } from "../services/api";

export default function useAuth() {
  const [token, setToken] = useState(null);
  const [unlocked, setUnlocked] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadAuthToken()
      .then((stored) => {
        if (stored) {
          setToken(stored);
          setUnlocked(true);
        }
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  const unlock = useCallback(async (key) => {
    setError(null);
    try {
      await apiGet("/health", key);
      await saveAuthToken(key);
      setToken(key);
      setUnlocked(true);
      return true;
    } catch (err) {
      setError(err.message || "Invalid key");
      return false;
    }
  }, []);

  const lock = useCallback(async () => {
    await clearAuthToken();
    setToken(null);
    setUnlocked(false);
  }, []);

  return { token, unlocked, loading, error, unlock, lock };
}
