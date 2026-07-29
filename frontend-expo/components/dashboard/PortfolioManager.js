import React, { useState, useCallback } from "react";
import { View, Text, Pressable, TextInput, StyleSheet } from "react-native";
import { cacheMarketSeries } from "../../services/api";
import { getMlEngineUrl } from "../../services/config";
import useWatchlist from "../../hooks/useWatchlist";
import THEME from "../../theme/colors";

const STRATEGIES = [
  { key: "dca", label: "DCA", desc: "Invest fixed amount at regular intervals" },
  { key: "martingale", label: "Martingale", desc: "Double position after loss, reset after win" },
  { key: "ml_weighted", label: "ML Weighted", desc: "Size each buy by ML confidence x direction" },
];

/**
 * Simulate a fixed-interval DCA strategy on real daily closes.
 * Buys `amount` worth of shares every `intervalDays` trading days.
 */
function simulateDCA(closes, amount) {
  let shares = 0;
  let invested = 0;
  let trades = 0;
  for (const c of closes) {
    if (c <= 0) continue;
    shares += amount / c;
    invested += amount;
    trades += 1;
  }
  const finalValue = shares * (closes[closes.length - 1] || 0);
  return { trades, invested, finalValue, shares };
}

/**
 * Simulate a Martingale strategy: doubles investment after each losing
 * interval (price dropped vs previous buy), resets to base amount after
 * a winning interval, capped at maxLosses consecutive doublings.
 */
function simulateMartingale(closes, amount, factor, maxLosses) {
  let shares = 0;
  let invested = 0;
  let trades = 0;
  let consecutiveLosses = 0;
  let prevPrice = null;
  for (const c of closes) {
    if (c <= 0) continue;
    let size = amount;
    if (prevPrice != null) {
      if (c < prevPrice) {
        consecutiveLosses = Math.min(consecutiveLosses + 1, maxLosses);
      } else {
        consecutiveLosses = 0;
      }
    }
    size = amount * Math.pow(factor, consecutiveLosses);
    shares += size / c;
    invested += size;
    trades += 1;
    prevPrice = c;
  }
  const finalValue = shares * (closes[closes.length - 1] || 0);
  return { trades, invested, finalValue, shares };
}

/**
 * ML-Weighted: sizes each interval buy by the ML model's directional
 * confidence (fetched once for the ticker). BUY direction scales up the
 * position size; SELL direction scales it down; HOLD keeps base amount.
 */
function simulateMLWeighted(closes, amount, mlDirection, mlConfidence) {
  let multiplier = 1;
  if (mlDirection === "BUY") multiplier = 1 + mlConfidence; // up to 2x
  else if (mlDirection === "SELL") multiplier = Math.max(0.2, 1 - mlConfidence); // down to 0.2x
  const sizedAmount = amount * multiplier;
  const result = simulateDCA(closes, sizedAmount);
  return { ...result, multiplier };
}

function winRateFromCloses(closes) {
  // Win rate proxy: % of intervals where price increased vs prior interval
  if (closes.length < 2) return 0;
  let wins = 0;
  for (let i = 1; i < closes.length; i++) {
    if (closes[i] > closes[i - 1]) wins++;
  }
  return (wins / (closes.length - 1)) * 100;
}

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
      const amt = parseFloat(amount) || 1000;
      const interval = Math.max(1, parseInt(intervalDays) || 7);
      const factor = parseFloat(martingaleFactor) || 2;
      const maxL = Math.max(1, parseInt(maxLosses) || 3);

      // Fetch real daily price history for this ticker
      const seriesData = await cacheMarketSeries(ticker, "1d", "1y", token);
      const items = seriesData?.items || [];
      const allCloses = items
        .map((it) => Number(it.close))
        .filter((c) => !isNaN(c) && c > 0);

      if (allCloses.length < interval * 2) {
        setError(`Not enough price history for ${ticker} to simulate at ${interval}-day intervals`);
        setRunning(false);
        return;
      }

      // Sample closes at the configured interval (every Nth trading day)
      const sampledCloses = [];
      for (let i = 0; i < allCloses.length; i += interval) {
        sampledCloses.push(allCloses[i]);
      }
      if (sampledCloses[sampledCloses.length - 1] !== allCloses[allCloses.length - 1]) {
        sampledCloses.push(allCloses[allCloses.length - 1]);
      }

      // Standard baseline: always plain DCA with the same amount/interval,
      // regardless of selected strategy, so the comparison is meaningful.
      const baseline = simulateDCA(sampledCloses, amt);

      // Selected strategy result
      let selected;
      if (strategy === "martingale") {
        selected = simulateMartingale(sampledCloses, amt, factor, maxL);
      } else if (strategy === "ml_weighted") {
        let mlDirection = "HOLD";
        let mlConfidence = 0;
        try {
          const mlUrl = getMlEngineUrl();
          const resp = await fetch(`${mlUrl}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ ticker, n_context: 80 }),
          });
          if (resp.ok) {
            const pred = await resp.json();
            mlDirection = pred.direction || "HOLD";
            mlConfidence = pred.confidence || 0;
          }
        } catch {}
        selected = simulateMLWeighted(sampledCloses, amt, mlDirection, mlConfidence);
        selected.mlDirection = mlDirection;
        selected.mlConfidence = mlConfidence;
      } else {
        // dca strategy selected = same as baseline, but still show real numbers
        selected = simulateDCA(sampledCloses, amt);
      }

      const winRate = winRateFromCloses(sampledCloses);

      setResult({
        ticker,
        strategy,
        intervalDays: interval,
        base: {
          trades: baseline.trades,
          winRate,
          invested: baseline.invested,
          finalValue: baseline.finalValue,
          returnPct: baseline.invested > 0 ? ((baseline.finalValue - baseline.invested) / baseline.invested) * 100 : 0,
        },
        ml: {
          trades: selected.trades,
          winRate,
          invested: selected.invested,
          finalValue: selected.finalValue,
          returnPct: selected.invested > 0 ? ((selected.finalValue - selected.invested) / selected.invested) * 100 : 0,
          mlDirection: selected.mlDirection,
          mlConfidence: selected.mlConfidence,
          multiplier: selected.multiplier,
        },
      });
    } catch (err) {
      setError(err.message);
    } finally {
      setRunning(false);
    }
  }, [watchlist, strategy, amount, intervalDays, martingaleFactor, maxLosses, token]);

  const selectedLabel = STRATEGIES.find((s) => s.key === strategy)?.label || "Strategy";

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
          <Text style={styles.resultTitle}>{result.ticker} — DCA Baseline vs {selectedLabel}</Text>
          <Text style={styles.resultSub}>
            {result.base.trades} buys every {result.intervalDays}d over the last year (real prices)
          </Text>
          <View style={styles.compareRow}>
            <View style={styles.compareCol}>
              <Text style={styles.compareLabel}>DCA Baseline</Text>
              <Text style={styles.compareVal}>{result.base.trades} buys</Text>
              <Text style={styles.compareVal}>${result.base.invested.toFixed(0)} invested</Text>
              <Text style={[styles.compareVal, { color: result.base.returnPct >= 0 ? THEME.profit : THEME.loss }]}>
                {result.base.returnPct >= 0 ? "+" : ""}{result.base.returnPct.toFixed(1)}% return
              </Text>
            </View>
            <View style={styles.compareCol}>
              <Text style={styles.compareLabel}>{selectedLabel}</Text>
              <Text style={styles.compareVal}>{result.ml.trades} buys</Text>
              <Text style={styles.compareVal}>${result.ml.invested.toFixed(0)} invested</Text>
              <Text style={[styles.compareVal, { color: result.ml.returnPct >= 0 ? THEME.profit : THEME.loss }]}>
                {result.ml.returnPct >= 0 ? "+" : ""}{result.ml.returnPct.toFixed(1)}% return
              </Text>
              {strategy === "ml_weighted" && result.ml.mlDirection && (
                <Text style={styles.compareSub}>
                  ML: {result.ml.mlDirection} ({(result.ml.mlConfidence * 100).toFixed(0)}%) → {result.ml.multiplier?.toFixed(2)}x sizing
                </Text>
              )}
              {strategy === "martingale" && (
                <Text style={styles.compareSub}>
                  Factor {martingaleFactor}x, max {maxLosses} doublings
                </Text>
              )}
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
  resultTitle: { color: THEME.text, fontSize: 12, fontWeight: "700", marginBottom: 2 },
  resultSub: { color: THEME.textMuted, fontSize: 10, marginBottom: 6 },
  compareRow: { flexDirection: "row", gap: 8 },
  compareCol: { flex: 1, backgroundColor: THEME.surfaceLight, borderRadius: 6, padding: 8 },
  compareLabel: { color: THEME.textMuted, fontSize: 9, textTransform: "uppercase", marginBottom: 4 },
  compareVal: { color: THEME.text, fontSize: 12, fontWeight: "600", fontVariant: ["tabular-nums"] },
  compareSub: { color: THEME.textMuted, fontSize: 10, marginTop: 4 },
});
