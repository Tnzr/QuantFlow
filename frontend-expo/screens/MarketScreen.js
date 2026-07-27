import React, { useState, useEffect, useCallback } from "react";
import { View, Text, TextInput, Pressable, ScrollView, StyleSheet } from "react-native";
import { apiGet, cacheMarketSeries } from "../services/api";
import { getMlEngineUrl } from "../services/config";
import useWatchlist from "../hooks/useWatchlist";
import CandlestickChart from "../components/charts/CandlestickChart";
import ErrorBanner from "../components/shared/ErrorBanner";
import LoadingSpinner from "../components/shared/LoadingSpinner";
import ScreenTitle from "../components/shared/ScreenTitle";
import THEME from "../theme/colors";

const TIMEFRAMES = ["1D", "1H", "15m", "5m", "1m"];
const INTERVAL_MAP = { "1m": "1m", "5m": "5m", "15m": "15m", "1H": "1h", "1D": "1d" };

function MarketTile({ ticker, timeframe, token, onRemove }) {
  const [ohlcv, setOhlcv] = useState([]);
  const [forecast, setForecast] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showForecast, setShowForecast] = useState(false);
  const [mlSignal, setMlSignal] = useState(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const interval = INTERVAL_MAP[timeframe] || "1d";
      const [cacheRes, foreRes, mlRes] = await Promise.allSettled([
        cacheMarketSeries(ticker, interval, null, token),
        apiGet(`/charts/forecast?ticker=${encodeURIComponent(ticker)}&interval=${encodeURIComponent(timeframe)}`, token),
        fetch(`${getMlEngineUrl()}/predict`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ticker, n_context: 80 }),
        }).then(r => r.ok ? r.json() : null).catch(() => null),
      ]);
      if (cacheRes.status === "fulfilled") setOhlcv(cacheRes.value?.items || []);
      else setError(cacheRes.reason?.message || "Failed");
      if (foreRes.status === "fulfilled") setForecast(foreRes.value);
      if (mlRes.status === "fulfilled" && mlRes.value) setMlSignal(mlRes.value);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [ticker, timeframe, token]);

  useEffect(() => { load(); }, [load]);

  return (
    <View style={tileStyles.wrap}>
      <View style={tileStyles.header}>
        <Text style={tileStyles.ticker}>{ticker}</Text>
        <View style={tileStyles.controls}>
          {mlSignal && (
            <View style={[tileStyles.sigBadge, { backgroundColor: mlSignal.direction === "BUY" ? THEME.profitDim : mlSignal.direction === "SELL" ? THEME.lossDim : THEME.surfaceLight, borderColor: mlSignal.direction === "BUY" ? THEME.profit : mlSignal.direction === "SELL" ? THEME.loss : THEME.border }]}>
              <Text style={[tileStyles.sigText, { color: mlSignal.direction === "BUY" ? THEME.profit : mlSignal.direction === "SELL" ? THEME.loss : THEME.textMuted }]}>
                {mlSignal.direction} {(mlSignal.confidence * 100).toFixed(0)}%
              </Text>
            </View>
          )}
          <Pressable style={[tileStyles.fcastBtn, showForecast && tileStyles.fcastOn]} onPress={() => setShowForecast(v => !v)}>
            <Text style={[tileStyles.fcastText, showForecast && tileStyles.fcastOnText]}>Forecast</Text>
          </Pressable>
          {onRemove && <Pressable onPress={() => onRemove(ticker)}><Text style={tileStyles.remove}>X</Text></Pressable>}
        </View>
      </View>
      {error && <ErrorBanner message={error} onRetry={load} />}
      {loading && !ohlcv.length && <LoadingSpinner />}
      {(ohlcv.length > 0 && (
        <CandlestickChart data={ohlcv} forecast={forecast?.forecast} showForecast={showForecast} />
      )) || (loading && <LoadingSpinner />)}
    </View>
  );
}

const tileStyles = StyleSheet.create({
  wrap: { marginBottom: 10, backgroundColor: THEME.surface, borderRadius: 8, padding: 10, borderLeftWidth: 3, borderLeftColor: THEME.accent },
  header: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginBottom: 2 },
  ticker: { color: THEME.text, fontWeight: "800", fontSize: 15 },
  controls: { flexDirection: "row", alignItems: "center", gap: 8 },
  fcastBtn: { paddingHorizontal: 8, paddingVertical: 3, borderRadius: 3, backgroundColor: THEME.surfaceLight, borderWidth: 1, borderColor: THEME.border },
  fcastOn: { backgroundColor: THEME.accentGlow, borderColor: THEME.accent },
  fcastText: { color: THEME.textMuted, fontSize: 10 },
  fcastOnText: { color: THEME.accent, fontWeight: "600" },
  remove: { color: THEME.loss, fontWeight: "700", fontSize: 14 },
  sigBadge: { paddingHorizontal: 6, paddingVertical: 2, borderRadius: 3, borderWidth: 1 },
  sigText: { fontWeight: "800", fontSize: 9 },
});

export default function MarketScreen({ token }) {
  const { watchlist, removeTicker } = useWatchlist();
  const [timeframe, setTimeframe] = useState("1D");
  const [customTicker, setCustomTicker] = useState("");
  const [customData, setCustomData] = useState(null);

  const loadCustomTicker = useCallback(() => {
    if (!customTicker.trim()) return;
    setCustomData(customTicker.trim().toUpperCase());
  }, [customTicker]);

  return (
    <ScrollView style={styles.wrap} contentContainerStyle={styles.content}>
      <ScreenTitle title="Market" subtitle="Charts, forecasts, and watchlist" icon="M" />
      <View style={styles.controls}>
        <View style={styles.inputWrap}>
          <Text style={styles.inputLabel}>Ticker</Text>
          <TextInput
            style={styles.input}
            placeholder="e.g. NVDA"
            placeholderTextColor={THEME.neutral}
            value={customTicker}
            onChangeText={(t) => setCustomTicker(t.toUpperCase())}
            autoCapitalize="characters"
            returnKeyType="search"
            onSubmitEditing={loadCustomTicker}
            accessibilityLabel="Ticker symbol"
          />
        </View>
        <Pressable style={styles.loadBtn} onPress={loadCustomTicker}>
          <Text style={styles.loadBtnText}>Load</Text>
        </Pressable>
      </View>

      <Text style={styles.inputLabel}>Timeframe</Text>
      <View style={styles.tfRow}>
        {TIMEFRAMES.map((tf) => (
          <Pressable key={tf} style={[styles.tfBtn, timeframe === tf && styles.tfActive]} onPress={() => setTimeframe(tf)}>
            <Text style={[styles.tfText, timeframe === tf && styles.tfTextActive]}>{tf}</Text>
          </Pressable>
        ))}
      </View>

      <Text style={styles.sectionTitle}>Watchlist ({watchlist.length})</Text>
      {watchlist.length === 0 && (
        <Text style={styles.emptyText}>No tickers. Add to watchlist from the Dashboard tab.</Text>
      )}

      {watchlist.map((t) => (
        <MarketTile key={t} ticker={t} timeframe={timeframe} token={token} onRemove={removeTicker} />
      ))}

      {customData && !watchlist.includes(customData) && (
        <View style={styles.customWrap}>
          <Text style={styles.sectionTitle}>Custom: {customData}</Text>
          <MarketTile ticker={customData} timeframe={timeframe} token={token} />
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  wrap: { flex: 1, backgroundColor: THEME.bg },
  content: { padding: 16, paddingBottom: 32 },
  controls: { flexDirection: "row", gap: 8, marginBottom: 10, alignItems: "flex-end" },
  inputWrap: { flex: 1 },
  inputLabel: { color: THEME.textMuted, fontSize: 11, marginBottom: 4, fontWeight: "600" },
  input: { backgroundColor: THEME.surface, borderWidth: 1, borderColor: THEME.border, borderRadius: 6, paddingHorizontal: 12, paddingVertical: 10, color: THEME.text, fontSize: 14 },
  loadBtn: { backgroundColor: THEME.accent, borderRadius: 6, paddingHorizontal: 18, paddingVertical: 10, justifyContent: "center" },
  loadBtnText: { color: THEME.textBright, fontWeight: "700", fontSize: 13 },
  tfRow: { flexDirection: "row", gap: 4, marginBottom: 14 },
  tfBtn: { flex: 1, paddingVertical: 6, borderRadius: 4, backgroundColor: THEME.surface, alignItems: "center", borderWidth: 1, borderColor: THEME.border },
  tfActive: { backgroundColor: THEME.accentGlow, borderColor: THEME.accent },
  tfText: { color: THEME.textMuted, fontSize: 11, fontWeight: "600" },
  tfTextActive: { color: THEME.accent },
  sectionTitle: { color: THEME.text, fontSize: 13, fontWeight: "700", marginBottom: 8, marginTop: 4 },
  emptyText: { color: THEME.textMuted, fontSize: 12, marginBottom: 12 },
  customWrap: { marginTop: 8 },
});
