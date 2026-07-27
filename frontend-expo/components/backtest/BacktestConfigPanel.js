import React, { useState } from "react";
import { View, Text, TextInput, Pressable, StyleSheet } from "react-native";
import THEME from "../../theme/colors";

const MA_OPTIONS = ["none", "sma20", "sma50"];
const TREND_OPTIONS = ["none", "above_sma50", "above_sma200"];

export default function BacktestConfigPanel({ onRun, running }) {
  const [expanded, setExpanded] = useState(true);
  const [tickers, setTickers] = useState("");
  const [startDate, setStartDate] = useState("2024-01-01");
  const [endDate, setEndDate] = useState("2025-12-31");
  const [capital, setCapital] = useState("100000");
  const [rsiThreshold, setRsiThreshold] = useState("30");
  const [maxHoldDays, setMaxHoldDays] = useState("10");
  const [stopLoss, setStopLoss] = useState("5");
  const [takeProfit, setTakeProfit] = useState("15");
  const [maFilter, setMaFilter] = useState("none");
  const [maTrend, setMaTrend] = useState("none");
  const [useMl, setUseMl] = useState(false);
  const [mlThreshold, setMlThreshold] = useState("0.0");
  const [mlMinConf, setMlMinConf] = useState("0.4");

  const handleRun = () => {
    const tickerList = tickers
      .split(",")
      .map((t) => t.trim().toUpperCase())
      .filter(Boolean);

    onRun({
      tickers: tickerList,
      start: startDate || undefined,
      end: endDate || undefined,
      capital: parseFloat(capital) || 100000,
      rsi_threshold: parseFloat(rsiThreshold) || undefined,
      max_hold_days: parseInt(maxHoldDays) || undefined,
      stop_loss: parseFloat(stopLoss) || undefined,
      take_profit: parseFloat(takeProfit) || undefined,
      ma_filter: maFilter !== "none" ? maFilter : undefined,
      ma_trend: maTrend !== "none" ? maTrend : undefined,
      use_ml_forecast: useMl,
      ml_threshold: isNaN(parseFloat(mlThreshold)) ? 0.0 : parseFloat(mlThreshold),
      ml_min_confidence: isNaN(parseFloat(mlMinConf)) ? 0.4 : parseFloat(mlMinConf),
    });
  };

  return (
    <View style={styles.wrap}>
      <Pressable style={styles.header} onPress={() => setExpanded((v) => !v)}>
        <Text style={styles.heading}>Backtest Configuration</Text>
        <Text style={styles.toggle}>{expanded ? "Collapse" : "Expand"}</Text>
      </Pressable>

      <Text style={styles.label}>Tickers (comma-separated)</Text>
      <View style={styles.row}>
        <TextInput
          style={[styles.input, styles.flex1]}
          placeholder="e.g. AAPL,MSFT,NVDA"
          placeholderTextColor={THEME.neutral}
          value={tickers}
          onChangeText={setTickers}
          autoCapitalize="characters"
          accessibilityLabel="Tickers"
        />
        <Pressable style={[styles.runBtn, running && styles.btnDisabled]} onPress={handleRun} disabled={running}>
          <Text style={styles.runText}>{running ? "Running..." : "Run"}</Text>
        </Pressable>
      </View>

      {expanded && (
        <View style={styles.expanded}>
          <View style={styles.row}>
            <View style={styles.half}>
              <Text style={styles.label}>Start Date</Text>
              <TextInput
                style={styles.input}
                placeholder="YYYY-MM-DD"
                placeholderTextColor={THEME.neutral}
                value={startDate}
                onChangeText={setStartDate}
                accessibilityLabel="Start date"
              />
            </View>
            <View style={styles.half}>
              <Text style={styles.label}>End Date</Text>
              <TextInput
                style={styles.input}
                placeholder="YYYY-MM-DD"
                placeholderTextColor={THEME.neutral}
                value={endDate}
                onChangeText={setEndDate}
                accessibilityLabel="End date"
              />
            </View>
          </View>

          <View style={styles.row}>
            <View style={styles.half}>
              <Text style={styles.label}>Initial Capital</Text>
              <TextInput
                style={styles.input}
                placeholder="e.g. 100000"
                placeholderTextColor={THEME.neutral}
                value={capital}
                onChangeText={setCapital}
                keyboardType="decimal-pad"
                accessibilityLabel="Initial capital"
              />
            </View>
            <View style={styles.half}>
              <Text style={styles.label}>RSI Entry Threshold</Text>
              <TextInput
                style={styles.input}
                placeholder="e.g. 30"
                placeholderTextColor={THEME.neutral}
                value={rsiThreshold}
                onChangeText={setRsiThreshold}
                keyboardType="decimal-pad"
                accessibilityLabel="RSI entry threshold"
              />
            </View>
          </View>

          <View style={styles.row}>
            <View style={styles.half}>
              <Text style={styles.label}>Max Hold Days</Text>
              <TextInput
                style={styles.input}
                placeholder="e.g. 10"
                placeholderTextColor={THEME.neutral}
                value={maxHoldDays}
                onChangeText={setMaxHoldDays}
                keyboardType="number-pad"
                accessibilityLabel="Max hold days"
              />
            </View>
            <View style={styles.half}>
              <Text style={styles.label}>Stop Loss %</Text>
              <TextInput
                style={styles.input}
                placeholder="e.g. 5"
                placeholderTextColor={THEME.neutral}
                value={stopLoss}
                onChangeText={setStopLoss}
                keyboardType="decimal-pad"
                accessibilityLabel="Stop loss percent"
              />
            </View>
          </View>

          <View>
            <Text style={styles.label}>Take Profit %</Text>
            <TextInput
              style={styles.input}
              placeholder="e.g. 15"
              placeholderTextColor={THEME.neutral}
              value={takeProfit}
              onChangeText={setTakeProfit}
              keyboardType="decimal-pad"
              accessibilityLabel="Take profit percent"
            />
          </View>

          <Text style={styles.label}>MA Filter</Text>
          <View style={styles.chipRow}>
            {MA_OPTIONS.map((opt) => (
              <Pressable
                key={opt}
                style={[styles.chip, maFilter === opt && styles.chipActive]}
                onPress={() => setMaFilter(opt)}
              >
                <Text style={[styles.chipText, maFilter === opt && styles.chipTextActive]}>{opt}</Text>
              </Pressable>
            ))}
          </View>

          <Text style={styles.label}>MA Trend</Text>
          <View style={styles.chipRow}>
            {TREND_OPTIONS.map((opt) => (
              <Pressable
                key={opt}
                style={[styles.chip, maTrend === opt && styles.chipActive]}
                onPress={() => setMaTrend(opt)}
              >
                <Text style={[styles.chipText, maTrend === opt && styles.chipTextActive]}>{opt}</Text>
              </Pressable>
            ))}
          </View>

          <Text style={styles.label}>Entry Signal</Text>
          <View style={styles.chipRow}>
            <Pressable
              style={[styles.chip, !useMl && styles.chipActive]}
              onPress={() => setUseMl(false)}
            >
              <Text style={[styles.chipText, !useMl && styles.chipTextActive]}>RSI</Text>
            </Pressable>
            <Pressable
              style={[styles.chip, useMl && styles.chipActive]}
              onPress={() => setUseMl(true)}
            >
              <Text style={[styles.chipText, useMl && styles.chipTextActive]}>ML CascadeANP</Text>
            </Pressable>
          </View>
          {useMl && (
            <View style={styles.row}>
              <View style={styles.half}>
                <Text style={styles.label}>ML Threshold</Text>
                <TextInput
                  style={styles.input}
                  placeholder="e.g. 0.002"
                  placeholderTextColor={THEME.neutral}
                  value={mlThreshold}
                  onChangeText={setMlThreshold}
                  keyboardType="decimal-pad"
                  accessibilityLabel="ML threshold"
                />
              </View>
              <View style={styles.half}>
                <Text style={styles.label}>ML Min Confidence</Text>
                <TextInput
                  style={styles.input}
                  placeholder="e.g. 0.4"
                  placeholderTextColor={THEME.neutral}
                  value={mlMinConf}
                  onChangeText={setMlMinConf}
                  keyboardType="decimal-pad"
                  accessibilityLabel="ML minimum confidence"
                />
              </View>
            </View>
          )}
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: { marginBottom: 16 },
  header: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 8,
  },
  heading: { color: THEME.text, fontSize: 14, fontWeight: "700" },
  toggle: { color: THEME.accent, fontSize: 12 },
  row: { flexDirection: "row", gap: 8, marginBottom: 8 },
  input: {
    backgroundColor: THEME.surface,
    borderWidth: 1,
    borderColor: THEME.border,
    borderRadius: 6,
    paddingHorizontal: 12,
    paddingVertical: 10,
    color: THEME.text,
    fontSize: 13,
  },
  flex1: { flex: 1 },
  half: { flex: 1 },
  runBtn: {
    backgroundColor: THEME.accent,
    borderRadius: 6,
    paddingHorizontal: 20,
    justifyContent: "center",
  },
  btnDisabled: { opacity: 0.6 },
  runText: { color: THEME.textBright, fontWeight: "700", fontSize: 13 },
  expanded: { marginTop: 4 },
  label: { color: THEME.textMuted, fontSize: 12, marginBottom: 4, marginTop: 4 },
  chipRow: { flexDirection: "row", gap: 6, marginBottom: 8 },
  chip: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 4,
    backgroundColor: THEME.surface,
    borderWidth: 1,
    borderColor: THEME.border,
  },
  chipActive: { backgroundColor: THEME.accentGlow, borderColor: THEME.accent },
  chipText: { color: THEME.textMuted, fontSize: 12 },
  chipTextActive: { color: THEME.accent, fontWeight: "600" },
});
