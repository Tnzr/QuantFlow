import React, { useState, useEffect, useCallback } from "react";
import { View, Text, Pressable, TextInput, StyleSheet, FlatList } from "react-native";
import { apiGet, apiPost, cacheMarketSeries } from "../../services/api";
import { getMlEngineUrl } from "../../services/config";
import useWatchlist from "../../hooks/useWatchlist";
import { AreaChart, Area, XAxis, YAxis, ResponsiveContainer, Tooltip, Line, ComposedChart } from "recharts";
import THEME from "../../theme/colors";

function MiniChart({ data, forecast }) {
  if (!data || data.length === 0) {
    return (
      <View style={miniStyles.emptyChart}>
        <Text style={miniStyles.emptyText}>No data</Text>
      </View>
    );
  }

  const chartData = data.slice(-60).map((d, i) => ({
    idx: i,
    value: Number(d.close) || Number(d.price) || 0,
    date: d.date || String(i),
    fcast: null,
  }));

  if (forecast && forecast.length > 0) {
    const offset = chartData.length;
    forecast.forEach((f, i) => {
      chartData.push({
        idx: offset + i,
        value: null,
        date: f.date || "",
        fcast: Number(f.price) || 0,
      });
    });
  }

  const allVals = chartData.reduce((acc, d) => {
    if (d.value != null) acc.push(d.value);
    if (d.fcast != null) acc.push(d.fcast);
    return acc;
  }, []);
  const minV = Math.min(...allVals);
  const maxV = Math.max(...allVals);

  return (
    <View style={miniStyles.chartWrap}>
      <ResponsiveContainer width="100%" height={100}>
        <ComposedChart data={chartData} margin={{ top: 2, right: 2, bottom: 2, left: 2 }}>
          <XAxis dataKey="idx" tickFormatter={(i) => (i % 8 === 0 ? (chartData[i]?.date || "").slice(5,10) : "")} tick={{ fill: THEME.textMuted, fontSize: 8 }} axisLine={{ stroke: THEME.border }} tickLine={false} />
          <YAxis domain={[minV * 0.995, maxV * 1.005]} tick={{ fill: THEME.textMuted, fontSize: 8 }} width={35} axisLine={{ stroke: THEME.border }} tickLine={false} tickFormatter={(v) => v.toFixed(0)} />
          <Tooltip
            contentStyle={{ background: THEME.surface, border: `1px solid ${THEME.border}`, borderRadius: 4, fontSize: 10 }}
            formatter={(val) => val != null ? [`$${Number(val).toFixed(2)}`] : null}
          />
          <Area
            type="monotone"
            dataKey="value"
            stroke={THEME.accent}
            strokeWidth={1}
            fill={THEME.accentGlow}
            fillOpacity={0.2}
            dot={false}
            connectNulls={false}
          />
          <Line
            type="monotone"
            dataKey="fcast"
            stroke={THEME.warn}
            strokeWidth={2.5}
            strokeDasharray="5 3"
            dot={{ r: 2, fill: THEME.warn }}
            connectNulls
          />
        </ComposedChart>
      </ResponsiveContainer>
    </View>
  );
}

const miniStyles = StyleSheet.create({
  chartWrap: {},
  emptyChart: { alignItems: "center", justifyContent: "center", height: 80 },
  emptyText: { color: THEME.textMuted, fontSize: 11 },
});

function WatchTile({ ticker, token, onRemove }) {
  const [forecast, setForecast] = useState(null);
  const [ohlcv, setOhlcv] = useState([]);
  const [mlSignal, setMlSignal] = useState(null);
  const [expanded, setExpanded] = useState(false);
  const [timeframe, setTimeframe] = useState("1D");

  const TF_OPTIONS = [
    { key: "1D", label: "1D" },
    { key: "1H", label: "1H" },
    { key: "15m", label: "15m" },
    { key: "5m", label: "5m" },
  ];

  const intervalMap = { "1D": "1d", "1H": "1h", "15m": "15m", "5m": "5m" };

  const loadData = useCallback(async () => {
    try {
      const interval = intervalMap[timeframe] || "1d";
      const [fcRes, ohlcvRes] = await Promise.allSettled([
        apiGet(`/charts/forecast?ticker=${encodeURIComponent(ticker)}&interval=${timeframe}`, token),
        cacheMarketSeries(ticker, interval, null, token),
      ]);
      if (fcRes.status === "fulfilled") setForecast(fcRes.value);
      if (ohlcvRes.status === "fulfilled") setOhlcv(ohlcvRes.value?.items || []);
    } catch (e) {
      console.warn("WatchTile load failed for", ticker, e);
    }
  }, [ticker, token, timeframe]);

  const loadMl = useCallback(async () => {
    try {
      const data = await apiPost(`${getMlEngineUrl()}/predict`, { ticker, n_context: 80 }, token);
      setMlSignal(data);
    } catch {}
  }, [ticker, token]);

  useEffect(() => { loadData(); }, [loadData]);
  useEffect(() => { loadMl(); }, [loadMl]);

  const predictedReturn = forecast?.forecast_return_pct;
  const fcPoints = forecast?.forecast || [];
  const direction = mlSignal?.direction || (predictedReturn != null ? (predictedReturn > 0 ? "BUY" : "SELL") : null);
  const conf = mlSignal?.confidence;

  return (
    <View style={tileStyles.tile}>
      <View style={tileStyles.header}>
        <Text style={tileStyles.ticker}>{ticker}</Text>
        <View style={tileStyles.headerRight}>
          {direction && (
            <View style={[tileStyles.signalBadge, {
              backgroundColor: direction === "BUY" ? THEME.profitDim : direction === "SELL" ? THEME.lossDim : THEME.surfaceLight,
              borderColor: direction === "BUY" ? THEME.profit : direction === "SELL" ? THEME.loss : THEME.border,
            }]}>
              <Text style={[tileStyles.signalText, {
                color: direction === "BUY" ? THEME.profit : direction === "SELL" ? THEME.loss : THEME.textMuted,
              }]}>{direction}</Text>
            </View>
          )}
          {conf != null && (
            <Text style={tileStyles.conf}>{(conf * 100).toFixed(0)}%</Text>
          )}
          <View style={tileStyles.tfRow}>
            {TF_OPTIONS.map((tf) => (
              <Pressable
                key={tf.key}
                style={[tileStyles.tfBtn, timeframe === tf.key && tileStyles.tfBtnActive]}
                onPress={() => setTimeframe(tf.key)}
              >
                <Text style={[tileStyles.tfText, timeframe === tf.key && tileStyles.tfTextActive]}>
                  {tf.label}
                </Text>
              </Pressable>
            ))}
          </View>
          <Pressable onPress={() => setExpanded(v => !v)}>
            <Text style={tileStyles.expandBtn}>{expanded ? "−" : "+"}</Text>
          </Pressable>
          <Pressable onPress={() => onRemove(ticker)}>
            <Text style={tileStyles.removeBtn}>X</Text>
          </Pressable>
        </View>
      </View>

      <MiniChart data={ohlcv} forecast={fcPoints} />

      {expanded && (
        <View style={tileStyles.expanded}>
          <View style={tileStyles.statsRow}>
            {predictedReturn != null && (
              <View style={tileStyles.stat}>
                <Text style={tileStyles.statLabel}>Return</Text>
                <Text style={[tileStyles.statValue, { color: predictedReturn >= 0 ? THEME.profit : THEME.loss }]}>
                  {predictedReturn >= 0 ? "+" : ""}{(predictedReturn * 100).toFixed(2)}%
                </Text>
              </View>
            )}
            {forecast?.daily_trend_pct != null && (
              <View style={tileStyles.stat}>
                <Text style={tileStyles.statLabel}>Trend</Text>
                <Text style={[tileStyles.statValue, { color: forecast.daily_trend_pct >= 0 ? THEME.profit : THEME.loss }]}>
                  {(forecast.daily_trend_pct * 100).toFixed(2)}%
                </Text>
              </View>
            )}
            {mlSignal?.aleatoric_sigma != null && (
              <View style={tileStyles.stat}>
                <Text style={tileStyles.statLabel}>Sigma</Text>
                <Text style={[tileStyles.statValue, { color: THEME.text }]}>
                  {mlSignal.aleatoric_sigma.toFixed(3)}
                </Text>
              </View>
            )}
            {mlSignal?.predicted_return != null && (
              <View style={tileStyles.stat}>
                <Text style={tileStyles.statLabel}>Pred Ret</Text>
                <Text style={[tileStyles.statValue, { color: mlSignal.predicted_return >= 0 ? THEME.profit : THEME.loss }]}>
                  {(mlSignal.predicted_return * 100).toFixed(2)}%
                </Text>
              </View>
            )}
          </View>
        </View>
      )}
    </View>
  );
}

const tileStyles = StyleSheet.create({
  tile: {
    backgroundColor: THEME.surfaceLight,
    borderRadius: 8,
    padding: 10,
    marginBottom: 6,
    borderWidth: 1,
    borderColor: THEME.border,
  },
  header: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 4,
  },
  headerRight: { flexDirection: "row", alignItems: "center", gap: 6 },
  ticker: { color: THEME.text, fontWeight: "800", fontSize: 15 },
  signalBadge: {
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 3,
    borderWidth: 1,
  },
  signalText: { fontWeight: "800", fontSize: 10 },
  conf: { color: THEME.textMuted, fontSize: 11, fontWeight: "600" },
  expandBtn: { color: THEME.textMuted, fontWeight: "700", fontSize: 16, paddingHorizontal: 4 },
  removeBtn: { color: THEME.loss, fontWeight: "700", fontSize: 14 },
  expanded: { marginTop: 8, borderTopWidth: 1, borderTopColor: THEME.border, paddingTop: 8 },
  statsRow: { flexDirection: "row", gap: 12, flexWrap: "wrap" },
  stat: {},
  statLabel: { color: THEME.textMuted, fontSize: 10, textTransform: "uppercase" },
  statValue: { fontSize: 13, fontWeight: "700", fontVariant: ["tabular-nums"] },
  tfRow: { flexDirection: "row", gap: 2, marginLeft: 4 },
  tfBtn: { paddingHorizontal: 4, paddingVertical: 2, borderRadius: 3, borderWidth: 1, borderColor: THEME.border, backgroundColor: THEME.bg },
  tfBtnActive: { backgroundColor: THEME.accentGlow, borderColor: THEME.accent },
  tfText: { color: THEME.textMuted, fontSize: 8, fontWeight: "600" },
  tfTextActive: { color: THEME.accent },
});

export default function WatchlistPanel({ token }) {
  const { watchlist, addTicker, removeTicker } = useWatchlist();
  const [adding, setAdding] = useState(false);
  const [newTicker, setNewTicker] = useState("");

  const handleAdd = useCallback(() => {
    const sym = newTicker.trim().toUpperCase();
    if (!sym) return;
    addTicker(sym);
    setNewTicker("");
    setAdding(false);
  }, [addTicker, newTicker]);

  return (
    <View style={styles.wrap}>
      <View style={styles.panelHeader}>
        <Text style={styles.panelTitle}>Watchlist ({watchlist.length})</Text>
        <Pressable style={styles.addBtn} onPress={() => setAdding((v) => !v)}>
          <Text style={styles.addBtnText}>{adding ? "Cancel" : "+ Add"}</Text>
        </Pressable>
      </View>

      {adding && (
        <View style={styles.addRow}>
          <TextInput
            style={styles.addInput}
            placeholder="Ticker (e.g. AAPL)"
            placeholderTextColor={THEME.neutral}
            value={newTicker}
            onChangeText={(t) => setNewTicker(t.toUpperCase())}
            onSubmitEditing={addTicker}
            autoCapitalize="characters"
            autoFocus
          />
          <Pressable style={styles.confirmAddBtn} onPress={handleAdd}>
            <Text style={styles.confirmAddText}>Add</Text>
          </Pressable>
        </View>
      )}

      {watchlist.length === 0 ? (
        <Text style={styles.emptyText}>No tickers yet. Click + Add to start watching.</Text>
      ) : (
        watchlist.map((t) => (
          <WatchTile key={t} ticker={t} token={token} onRemove={removeTicker} />
        ))
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    marginBottom: 16,
    padding: 14,
    backgroundColor: THEME.surface,
    borderRadius: 8,
    borderLeftWidth: 3,
    borderLeftColor: THEME.profit,
  },
  panelHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 10,
  },
  panelTitle: { color: THEME.text, fontSize: 14, fontWeight: "700" },
  addBtn: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 4,
    backgroundColor: THEME.accent,
  },
  addBtnText: { color: THEME.textBright, fontSize: 11, fontWeight: "600" },
  addRow: { flexDirection: "row", gap: 6, marginBottom: 10, alignItems: "center" },
  addInput: {
    flex: 1,
    backgroundColor: THEME.surfaceLight,
    borderWidth: 1,
    borderColor: THEME.border,
    borderRadius: 6,
    paddingHorizontal: 10,
    paddingVertical: 7,
    color: THEME.text,
    fontSize: 13,
  },
  confirmAddBtn: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 4,
    backgroundColor: THEME.profit,
  },
  confirmAddText: { color: THEME.bg, fontWeight: "700", fontSize: 12 },
  emptyText: { color: THEME.textMuted, fontSize: 12, paddingVertical: 8 },
});
