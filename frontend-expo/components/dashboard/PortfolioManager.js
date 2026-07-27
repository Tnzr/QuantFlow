import React, { useState, useCallback } from "react";
import { View, Text, Pressable, TextInput, StyleSheet } from "react-native";
import { apiGet } from "../../services/api";
import useWatchlist from "../../hooks/useWatchlist";
import THEME from "../../theme/colors";

const STRATEGIES = [
  { key: "dca", label: "DCA", desc: "Invest fixed amount at regular intervals" },
  { key: "martingale", label: "Martingale", desc: "Double position after loss, reset after win" },
  { key: "ml_weighted", label: "ML Weighted", desc: "Allocate based on ML confidence scores" },
];

export default function PortfolioManager({ token }) {
  const [strategy, setStrategy] = useState("dca");
  const [amount, setAmount] = useState("1000");
  const [intervalDays, setIntervalDays] = useState("7");
  const [martingaleFactor, setMartingaleFactor] = useState("2");
  const [maxLosses, setMaxLosses] = useState("3");
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const { watchlist } = useWatchlist();

  const runBacktest = useCallback(async () => {
    if (watchlist.length === 0) { setError("Add tickers to your watchlist first"); return; }
    setRunning(true); setError(null);
    try {
      const ticker = watchlist[0];
      const params = new URLSearchParams();
      params.set("ticker", ticker);
      params.set("start", "2024-01-01");
      params.set("end", "2025-12-31");
      params.set("entry_rsi_threshold", "50");
      params.set("max_hold_days", String(parseInt(intervalDays) || 7));
      params.set("ma_filter", "none");
      params.set("ma_trend_filter", "none");
      const [base, ml] = await Promise.all([
        apiGet("/backtest/short-term?" + params.toString(), token),
        apiGet("/backtest/short-term?" + params.toString() + "&use_ml_forecast=true", token),
      ]);
      const be = base?.equity || [];
      const me = ml?.equity || [];
      setResult({
        ticker, strategy,
        base: { trades: base?.summary?.n_trades || 0, winRate: (base?.summary?.win_rate || 0) * 100, finalEquity: be.length ? be[be.length - 1]?.equity : 1, sharpe: base?.summary?.sharpe || 0 },
        ml: { trades: ml?.summary?.n_trades || 0, winRate: (ml?.summary?.win_rate || 0) * 100, finalEquity: me.length ? me[me.length - 1]?.equity : 1, sharpe: ml?.summary?.sharpe || 0 },
      });
    } catch (err) { setError(err.message); } finally { setRunning(false); }
  }, [watchlist, strategy, intervalDays, martingaleFactor, maxLosses, token]);

  return (
    <View style={styles.wrap}>
      <Text style={styles.title}>Portfolio Strategy Manager</Text>
      <View style={styles.stratRow}>
        {STRATEGIES.map((s) => (
          <Pressable key={s.key} style={[styles.stratChip, strategy === s.key && styles.stratOn]} onPress={() => setStrategy(s.key)}>
            <Text style={[styles.stratText, strategy === s.key && styles.stratOnText]}>{s.label}</Text>
          </Pressable>
        ))}
      </View>
      <Text style={styles.desc}>{STRATEGIES.find((s) => s.key === strategy)?.desc}</Text>
      <View style={styles.inputRow}>
        <View style={styles.inputWrap}>
          <Text style={styles.inputLabel}>Amount $</Text>
          <TextInput style={styles.input} value={amount} onChangeText={setAmount} keyboardType="decimal-pad" placeholderTextColor={THEME.neutral} />
        </View>
        <View style={styles.inputWrap}>
          <Text style={styles.inputLabel}>Interval (d)</Text>
          <TextInput style={styles.input} value={intervalDays} onChangeText={setIntervalDays} keyboardType="number-pad" placeholderTextColor={THEME.neutral} />
        </View>
        {strategy === "martingale" && (
          <>
            <View style={styles.inputWrap}>
              <Text style={styles.inputLabel}>Factor</Text>
              <TextInput style={styles.input} value={martingaleFactor} onChangeText={setMartingaleFactor} keyboardType="decimal-pad" placeholderTextColor={THEME.neutral} />
            </View>
            <View style={styles.inputWrap}>
              <Text style={styles.inputLabel}>Max Losses</Text>
              <TextInput style={styles.input} value={maxLosses} onChangeText={setMaxLosses} keyboardType="number-pad" placeholderTextColor={THEME.neutral} />
            </View>
          </>
        )}
      </View>
      <Pressable style={[styles.runBtn, running && styles.runBtnDisabled]} onPress={runBacktest} disabled={running || watchlist.length === 0}>
        <Text style={styles.runText}>{running ? "Running..." : "Simulate on " + (watchlist[0] || "watchlist")}</Text>
      </Pressable>
      {error && <Text style={styles.error}>{error}</Text>}
      {result && (
        <View style={styles.resultSection}>
          <Text style={styles.resultTitle}>{result.ticker} — Standard vs ML-Weighted</Text>
          <View style={styles.compareRow}>
            <View style={styles.compareCol}>
              <Text style={styles.compareLabel}>Standard</Text>
              <Text style={styles.compareVal}>{result.base.trades} trades</Text>
              <Text style={styles.compareVal}>{result.base.winRate.toFixed(0)}% win</Text>
              <Text style={[styles.compareVal, { color: result.base.finalEquity >= 1 ? THEME.profit : THEME.loss }]}>
                {((result.base.finalEquity - 1) * 100).toFixed(1)}% return
              </Text>
            </View>
            <View style={styles.compareCol}>
              <Text style={styles.compareLabel}>ML-Driven</Text>
              <Text style={styles.compareVal}>{result.ml.trades} trades</Text>
              <Text style={styles.compareVal}>{result.ml.winRate.toFixed(0)}% win</Text>
              <Text style={[styles.compareVal, { color: result.ml.finalEquity >= 1 ? THEME.profit : THEME.loss }]}>
                {((result.ml.finalEquity - 1) * 100).toFixed(1)}% return
              </Text>
            </View>
          </View>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: { marginBottom: 16, padding: 14, backgroundColor: THEME.surface, borderRadius: 8, borderLeftWidth: 3, borderLeftColor: THEME.warn },
  title: { color: THEME.text, fontSize: 14, fontWeight: "700", marginBottom: 8 },
  stratRow: { flexDirection: "row", gap: 4, marginBottom: 6 },
  stratChip: { paddingHorizontal: 10, paddingVertical: 5, borderRadius: 4, backgroundColor: THEME.surfaceLight, borderWidth: 1, borderColor: THEME.border },
  stratOn: { backgroundColor: THEME.accentGlow, borderColor: THEME.accent },
  stratText: { color: THEME.textMuted, fontSize: 11, fontWeight: "600" },
  stratOnText: { color: THEME.accent },
  desc: { color: THEME.textMuted, fontSize: 11, marginBottom: 8 },
  inputRow: { flexDirection: "row", gap: 8, marginBottom: 8, flexWrap: "wrap" },
  inputWrap: { flex: 1, minWidth: 70 },
  inputLabel: { color: THEME.textMuted, fontSize: 9, textTransform: "uppercase", marginBottom: 2 },
  input: { backgroundColor: THEME.surfaceLight, borderWidth: 1, borderColor: THEME.border, borderRadius: 4, paddingHorizontal: 8, paddingVertical: 6, color: THEME.text, fontSize: 12 },
  runBtn: { backgroundColor: THEME.accent, borderRadius: 6, paddingVertical: 10, alignItems: "center", marginBottom: 8 },
  runBtnDisabled: { opacity: 0.6 },
  runText: { color: THEME.textBright, fontWeight: "700", fontSize: 13 },
  error: { color: THEME.loss, fontSize: 12, marginBottom: 8 },
  resultSection: { marginTop: 8, borderTopWidth: 1, borderTopColor: THEME.border, paddingTop: 8 },
  resultTitle: { color: THEME.text, fontSize: 12, fontWeight: "700", marginBottom: 6 },
  compareRow: { flexDirection: "row", gap: 8 },
  compareCol: { flex: 1, backgroundColor: THEME.surfaceLight, borderRadius: 6, padding: 8 },
  compareLabel: { color: THEME.textMuted, fontSize: 9, textTransform: "uppercase", marginBottom: 4 },
  compareVal: { color: THEME.text, fontSize: 12, fontWeight: "600", fontVariant: ["tabular-nums"] },
  compareSub: { color: THEME.textMuted, fontSize: 10, marginTop: 2 },
});
