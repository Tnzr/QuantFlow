import React, { useMemo, useState } from "react";
import { SafeAreaView, ScrollView, Text, View, TextInput, Pressable, StyleSheet } from "react-native";
import { StatusBar } from "expo-status-bar";

const API_BASE = process.env.EXPO_PUBLIC_API_BASE_URL || "http://localhost:8000";
const API_KEY = process.env.EXPO_PUBLIC_API_KEY || "";

const SCREENS = ["Overview", "Scanner", "Recommend", "Options", "Backtest", "Execution"];

function authHeaders() {
  return API_KEY ? { "x-api-key": API_KEY } : {};
}

async function apiGet(path) {
  const res = await fetch(`${API_BASE}${path}`, { headers: { ...authHeaders() } });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

async function apiPost(path, payload) {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export default function App() {
  const [screen, setScreen] = useState("Overview");
  const [status, setStatus] = useState("idle");
  const [health, setHealth] = useState(null);
  const [presets, setPresets] = useState([]);
  const [latestUniverse, setLatestUniverse] = useState([]);
  const [latestRecs, setLatestRecs] = useState([]);
  const [intents, setIntents] = useState([]);
  const [optionIdeas, setOptionIdeas] = useState([]);
  const [backtestSummary, setBacktestSummary] = useState(null);

  const [ticker, setTicker] = useState("AAPL");
  const [notional, setNotional] = useState("250");
  const [horizon, setHorizon] = useState("1m");

  const baseLabel = useMemo(() => `API: ${API_BASE}`, []);

  const loadOverview = async () => {
    setStatus("loading");
    try {
      const [h, p, i, u, r] = await Promise.all([
        apiGet("/health"),
        apiGet("/scanner/presets"),
        apiGet("/execution/intents?limit=10"),
        apiGet("/scanner/latest-universe"),
        apiGet("/recommend/latest"),
      ]);
      setHealth(h);
      setPresets(p.items || []);
      setIntents(i.items || []);
      setLatestUniverse(u.items || []);
      setLatestRecs(r.items || []);
      setStatus("ready");
    } catch (e) {
      setStatus(`error: ${e.message}`);
    }
  };

  const runScanPipeline = async () => {
    setStatus("scan running");
    try {
      await apiPost("/pipeline/scan", { parallel: true, max_workers: 4 });
      await loadOverview();
    } catch (e) {
      setStatus(`error: ${e.message}`);
    }
  };

  const runRecommendPipeline = async () => {
    setStatus("recommend running");
    try {
      await apiPost("/pipeline/recommend", { tickers: [ticker], save: true });
      await loadOverview();
    } catch (e) {
      setStatus(`error: ${e.message}`);
    }
  };

  const loadOptions = async () => {
    setStatus("loading options");
    try {
      const data = await apiGet(`/options/ideas?ticker=${encodeURIComponent(ticker)}&horizon=${encodeURIComponent(horizon)}&bias=long&budget=300`);
      setOptionIdeas(data.items || []);
      setStatus("ready");
    } catch (e) {
      setStatus(`error: ${e.message}`);
    }
  };

  const loadBacktest = async () => {
    setStatus("running backtest");
    try {
      const data = await apiGet(`/backtest/short-term?ticker=${encodeURIComponent(ticker)}&start=2020-01-01`);
      setBacktestSummary(data.summary || null);
      setStatus("ready");
    } catch (e) {
      setStatus(`error: ${e.message}`);
    }
  };

  const proposeOrder = async () => {
    setStatus("proposing");
    try {
      await apiPost("/execution/propose", {
        ticker,
        action: "buy",
        qty: 1,
        notional: Number(notional || 0),
        orders_today: 0,
        position_notional_after: Number(notional || 0),
        broker_mode: "robinhood_mcp",
        auto: false,
      });
      await loadOverview();
    } catch (e) {
      setStatus(`error: ${e.message}`);
    }
  };

  const renderOverview = () => (
    <>
      <View style={styles.row}>
        <Pressable style={styles.button} onPress={loadOverview}>
          <Text style={styles.buttonText}>Refresh</Text>
        </Pressable>
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Health</Text>
        <Text>{health ? JSON.stringify(health) : "not loaded"}</Text>
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Presets ({presets.length})</Text>
        {(presets || []).slice(0, 8).map((p) => (
          <Text key={p} style={styles.item}>{p}</Text>
        ))}
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Recent Intents ({intents.length})</Text>
        {(intents || []).slice(0, 8).map((i) => (
          <Text key={String(i.id)} style={styles.item}>
            #{i.id} {i.ticker} {i.action} ${i.notional} [{i.status}]
          </Text>
        ))}
      </View>
    </>
  );

  const renderScanner = () => (
    <View style={styles.card}>
      <Text style={styles.cardTitle}>Scanner Pipeline</Text>
      <Pressable style={styles.button} onPress={runScanPipeline}>
        <Text style={styles.buttonText}>Run Scan</Text>
      </Pressable>
      <Text style={styles.cardTitle}>Latest Universe ({latestUniverse.length})</Text>
      {(latestUniverse || []).slice(0, 12).map((row, idx) => (
        <Text key={String(idx)} style={styles.item}>{row.ticker || row.symbol || "?"} | {row.strategy || row.preset || "-"}</Text>
      ))}
    </View>
  );

  const renderRecommend = () => (
    <View style={styles.card}>
      <Text style={styles.cardTitle}>Recommendations</Text>
      <TextInput value={ticker} onChangeText={setTicker} style={styles.input} placeholder="Ticker" />
      <Pressable style={styles.button} onPress={runRecommendPipeline}>
        <Text style={styles.buttonText}>Run Recommend</Text>
      </Pressable>
      <Text style={styles.cardTitle}>Latest Recs ({latestRecs.length})</Text>
      {(latestRecs || []).slice(0, 12).map((row, idx) => (
        <Text key={String(idx)} style={styles.item}>{row.ticker} {row.action || row.signal || "-"} ({row.horizon || "?"})</Text>
      ))}
    </View>
  );

  const renderOptions = () => (
    <View style={styles.card}>
      <Text style={styles.cardTitle}>Options Ideas</Text>
      <TextInput value={ticker} onChangeText={setTicker} style={styles.input} placeholder="Ticker" />
      <TextInput value={horizon} onChangeText={setHorizon} style={styles.input} placeholder="Horizon: 1w/1m/3m" />
      <Pressable style={styles.button} onPress={loadOptions}>
        <Text style={styles.buttonText}>Load Ideas</Text>
      </Pressable>
      {(optionIdeas || []).slice(0, 12).map((row, idx) => (
        <Text key={String(idx)} style={styles.item}>
          {row.contract || row.symbol || "contract"} | score {String(row.score ?? "-")}
        </Text>
      ))}
    </View>
  );

  const renderBacktest = () => (
    <View style={styles.card}>
      <Text style={styles.cardTitle}>Backtest</Text>
      <TextInput value={ticker} onChangeText={setTicker} style={styles.input} placeholder="Ticker" />
      <Pressable style={styles.button} onPress={loadBacktest}>
        <Text style={styles.buttonText}>Run Short-Term Backtest</Text>
      </Pressable>
      <Text>{backtestSummary ? JSON.stringify(backtestSummary) : "no backtest loaded"}</Text>
    </View>
  );

  const renderExecution = () => (
    <View style={styles.card}>
      <Text style={styles.cardTitle}>Propose Order</Text>
      <TextInput value={ticker} onChangeText={setTicker} style={styles.input} placeholder="Ticker" />
      <TextInput value={notional} onChangeText={setNotional} style={styles.input} placeholder="Notional" keyboardType="numeric" />
      <Pressable style={styles.button} onPress={proposeOrder}>
        <Text style={styles.buttonText}>Create Intent</Text>
      </Pressable>
      <Text style={styles.cardTitle}>Recent Intents ({intents.length})</Text>
      {(intents || []).slice(0, 12).map((i) => (
        <Text key={String(i.id)} style={styles.item}>
          #{i.id} {i.ticker} {i.action} ${i.notional} [{i.status}]
        </Text>
      ))}
    </View>
  );

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar style="dark" />
      <ScrollView contentContainerStyle={styles.content}>
        <Text style={styles.title}>QuantFlow Mobile MVP</Text>
        <Text style={styles.meta}>{baseLabel}</Text>
        <Text style={styles.meta}>Auth key configured: {API_KEY ? "yes" : "no"}</Text>
        <Text style={styles.meta}>Status: {status}</Text>
        <View style={styles.tabRow}>
          {SCREENS.map((name) => (
            <Pressable
              key={name}
              style={[styles.tab, screen === name ? styles.tabActive : null]}
              onPress={() => setScreen(name)}
            >
              <Text style={[styles.tabText, screen === name ? styles.tabTextActive : null]}>{name}</Text>
            </Pressable>
          ))}
        </View>

        {screen === "Overview" && renderOverview()}
        {screen === "Scanner" && renderScanner()}
        {screen === "Recommend" && renderRecommend()}
        {screen === "Options" && renderOptions()}
        {screen === "Backtest" && renderBacktest()}
        {screen === "Execution" && renderExecution()}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#f6f8fb" },
  content: { padding: 16, gap: 12 },
  title: { fontSize: 24, fontWeight: "700", color: "#0f172a" },
  meta: { color: "#334155" },
  row: { flexDirection: "row", gap: 8 },
  tabRow: { flexDirection: "row", gap: 8, flexWrap: "wrap" },
  tab: { borderWidth: 1, borderColor: "#cbd5e1", borderRadius: 999, paddingVertical: 6, paddingHorizontal: 10, backgroundColor: "#fff" },
  tabActive: { backgroundColor: "#e0f2fe", borderColor: "#0284c7" },
  tabText: { color: "#334155", fontWeight: "600" },
  tabTextActive: { color: "#075985" },
  card: { backgroundColor: "white", borderRadius: 12, padding: 12, gap: 8, borderWidth: 1, borderColor: "#e2e8f0" },
  cardTitle: { fontSize: 16, fontWeight: "700", color: "#0f172a" },
  item: { color: "#1e293b" },
  input: { borderWidth: 1, borderColor: "#cbd5e1", borderRadius: 10, padding: 10, backgroundColor: "#fff" },
  button: { backgroundColor: "#0ea5e9", paddingVertical: 10, paddingHorizontal: 14, borderRadius: 10, alignSelf: "flex-start" },
  buttonText: { color: "white", fontWeight: "700" },
});
