import { useState, useEffect, useCallback } from "react";
import { storageGet, storageSet, storageSetRaw, storageGetRaw } from "../services/storage";

const WATCHLIST_KEY = "quantflow_watchlist";

function parseList(raw) {
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed)) return parsed;
    if (parsed && Array.isArray(parsed._value)) return parsed._value;
  } catch {}
  return [];
}

export default function useWatchlist() {
  const [watchlist, setWatchlist] = useState([]);

  useEffect(() => {
    let mounted = true;
    (async () => {
      let list = await storageGet(WATCHLIST_KEY);
      if (!Array.isArray(list)) {
        const raw = await storageGetRaw(WATCHLIST_KEY);
        list = parseList(raw);
      }
      if (mounted && Array.isArray(list)) setWatchlist(list);
    })();
    return () => { mounted = false; };
  }, []);

  const persist = useCallback((next) => {
    storageSet(WATCHLIST_KEY, next).catch(() => {});
    try { storageSetRaw(WATCHLIST_KEY, JSON.stringify(next)); } catch {}
  }, []);

  const addTicker = useCallback((ticker) => {
    const sym = String(ticker).trim().toUpperCase();
    if (!sym) return;
    setWatchlist((prev) => {
      if (prev.includes(sym)) return prev;
      const next = [...prev, sym];
      persist(next);
      return next;
    });
  }, [persist]);

  const removeTicker = useCallback((ticker) => {
    setWatchlist((prev) => {
      const next = prev.filter((t) => t !== ticker.toUpperCase());
      persist(next);
      return next;
    });
  }, [persist]);

  return { watchlist, addTicker, removeTicker };
}
