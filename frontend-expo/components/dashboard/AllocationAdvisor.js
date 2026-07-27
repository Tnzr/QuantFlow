import React, { useState, useCallback } from "react";
import { View, Text, Pressable, StyleSheet } from "react-native";
import { apiPost } from "../../services/api";
import useWatchlist from "../../hooks/useWatchlist";
import THEME from "../../theme/colors";

export default function AllocationAdvisor({ token }) {
  const [result, setResult] = useState(null);
  const [method, setMethod] = useState("mean_variance");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const { watchlist } = useWatchlist();

  const methods = [
    { key: "mean_variance", label: "Mean-Variance" },
    { key: "hrp", label: "HRP" },
    { key: "volatility_scaled", label: "Risk Parity" },
    { key: "equal_weight", label: "Equal Weight" },
  ];

  const runAllocation = async () => {
    if (!Array.isArray(watchlist) || watchlist.length < 2) {
      setError("Need at least 2 tickers in watchlist");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const req = {
        tickers: watchlist,
        method: method,
        max_weight: 0.25,
        min_weight: 0.02,
        target_volatility: 0.15,
        lookback_days: 252,
      };
      const data = await apiPost("/portfolio/allocate", req, token);
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <View style={styles.wrap}>
      <View style={styles.headerRow}>
        <Text style={styles.title}>Portfolio Allocation</Text>
        <Pressable style={styles.runBtn} onPress={runAllocation} disabled={loading}>
          <Text style={styles.runText}>{loading ? "Running..." : "Optimize"}</Text>
        </Pressable>
      </View>

      <View style={styles.methodRow}>
        {methods.map((m) => (
          <Pressable
            key={m.key}
            style={[styles.methodChip, method === m.key && styles.methodChipActive]}
            onPress={() => setMethod(m.key)}
          >
            <Text style={[styles.methodText, method === m.key && styles.methodTextActive]}>
              {m.label}
            </Text>
          </Pressable>
        ))}
      </View>

      {error && <Text style={styles.error}>{error}</Text>}

      {result && !error && (
        <View style={styles.resultSection}>
          <View style={styles.statsRow}>
            <View style={styles.statBox}>
              <Text style={styles.statLabel}>Exp Return</Text>
              <Text style={[styles.statValue, { color: (result.expected_return || 0) >= 0 ? THEME.profit : THEME.loss }]}>
                {((result.expected_return || 0) * 100).toFixed(2)}%
              </Text>
            </View>
            <View style={styles.statBox}>
              <Text style={styles.statLabel}>Volatility</Text>
              <Text style={styles.statValue}>
                {((result.expected_volatility || 0) * 100).toFixed(2)}%
              </Text>
            </View>
            <View style={styles.statBox}>
              <Text style={styles.statLabel}>Sharpe</Text>
              <Text style={styles.statValue}>{((result.sharpe_ratio || 0)).toFixed(2)}</Text>
            </View>
          </View>

          <Text style={styles.sectionTitle}>Target Weights</Text>
          {result.weights &&
            Object.entries(result.weights)
              .sort((a, b) => b[1] - a[1])
              .map(([ticker, weight]) => (
                <View key={ticker} style={styles.weightRow}>
                  <Text style={styles.weightTicker}>{ticker}</Text>
                  <View style={styles.weightBarBg}>
                    <View
                      style={[
                        styles.weightBar,
                        { width: `${(weight * 100).toFixed(1)}%` },
                      ]}
                    />
                  </View>
                  <Text style={styles.weightPct}>{(weight * 100).toFixed(1)}%</Text>
                </View>
              ))}
        </View>
      )}

      {!result && !error && (
        <Text style={styles.help}>
          Select optimization method and run to get target weights for your watchlist.
        </Text>
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
    borderLeftColor: THEME.warn,
  },
  headerRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 8,
  },
  title: { color: THEME.text, fontSize: 14, fontWeight: "700" },
  runBtn: {
    paddingHorizontal: 12,
    paddingVertical: 5,
    borderRadius: 4,
    backgroundColor: THEME.accent,
  },
  runText: { color: THEME.textBright, fontWeight: "600", fontSize: 11 },
  methodRow: { flexDirection: "row", gap: 4, marginBottom: 10, flexWrap: "wrap" },
  methodChip: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
    backgroundColor: THEME.surfaceLight,
    borderWidth: 1,
    borderColor: THEME.border,
  },
  methodChipActive: { backgroundColor: THEME.accentGlow, borderColor: THEME.accent },
  methodText: { color: THEME.textMuted, fontSize: 10 },
  methodTextActive: { color: THEME.accent, fontWeight: "600" },
  error: { color: THEME.loss, fontSize: 12, marginBottom: 8 },
  help: { color: THEME.textMuted, fontSize: 12, fontStyle: "italic" },
  resultSection: {},
  statsRow: { flexDirection: "row", gap: 8, marginBottom: 10 },
  statBox: { flex: 1, backgroundColor: THEME.surfaceLight, borderRadius: 6, padding: 8, alignItems: "center" },
  statLabel: { color: THEME.textMuted, fontSize: 10, textTransform: "uppercase", marginBottom: 2 },
  statValue: { fontSize: 16, fontWeight: "700", fontVariant: ["tabular-nums"] },
  sectionTitle: { color: THEME.textMuted, fontSize: 11, fontWeight: "600", textTransform: "uppercase", marginBottom: 6 },
  weightRow: { flexDirection: "row", alignItems: "center", marginBottom: 4, gap: 8 },
  weightTicker: { color: THEME.text, fontWeight: "700", fontSize: 12, minWidth: 50 },
  weightBarBg: { flex: 1, height: 14, backgroundColor: THEME.border, borderRadius: 4, overflow: "hidden" },
  weightBar: { height: "100%", backgroundColor: THEME.accent, borderRadius: 4 },
  weightPct: { color: THEME.textMuted, fontSize: 11, minWidth: 42, textAlign: "right", fontVariant: ["tabular-nums"] },
});
