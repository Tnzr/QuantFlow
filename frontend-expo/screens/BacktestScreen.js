import React, { useState } from "react";
import { ScrollView, StyleSheet, View, Text, Pressable } from "react-native";
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import THEME from "../theme/colors";
import useBacktest from "../hooks/useBacktest";
import { apiPost } from "../services/api";
import { getMlEngineUrl } from "../services/config";
import ErrorBoundary from "../components/shared/ErrorBoundary";
import BacktestConfigPanel from "../components/backtest/BacktestConfigPanel";
import BacktestProgress from "../components/backtest/BacktestProgress";
import BacktestResults from "../components/backtest/BacktestResults";
import LoadingSpinner from "../components/shared/LoadingSpinner";
import ErrorBanner from "../components/shared/ErrorBanner";
import MetricCard from "../components/shared/MetricCard";
import ScreenTitle from "../components/shared/ScreenTitle";

export default function BacktestScreen({ token }) {
  const {
    config,
    equityCurve,
    trades,
    signals,
    rsiSeries,
    summary,
    multiResults,
    portfolioEquity,
    sigmaBuckets,
    loading,
    progress,
    error,
    running,
    animating,
    setAnimating,
    run,
  } = useBacktest(token);

  const [mlSignals, setMlSignals] = useState(null);

  const handleRerun = () => {
    if (config.tickers && config.tickers.length > 0) {
      setMlSignals(null);
      run(config);
    }
  };

  const handleAnimationComplete = () => {
    setAnimating(false);
  };

  const fetchMlSignal = async (ticker) => {
    try {
      const data = await apiPost(`${getMlEngineUrl()}/predict`, { ticker, n_context: 80 }, null);
      setMlSignals(data);
    } catch (err) {
      setMlSignals({ error: err.message });
    }
  };

  return (
    <ScrollView style={styles.wrap} contentContainerStyle={styles.content}>
      <ScreenTitle title="Backtest" subtitle="Test strategies against historical data" icon="B" />
      {error && <ErrorBanner message={error} />}
      <BacktestConfigPanel onRun={run} running={running} />
      {loading && <LoadingSpinner />}

      {animating && equityCurve.length > 0 && (
        <ErrorBoundary>
          <BacktestProgress
            equityCurve={equityCurve}
            trades={trades}
            signals={signals}
            rsiSeries={rsiSeries}
            running={true}
            onComplete={handleAnimationComplete}
            ticker={config.tickers?.[0]}
            token={token}
          />
        </ErrorBoundary>
      )}

      {summary && !animating && (
        <ErrorBoundary>
          <>
            <BacktestResults
              summary={summary}
              sigmaBuckets={sigmaBuckets}
              trades={trades}
              onRerun={handleRerun}
            />

          {equityCurve.length > 0 && (
            <View style={styles.eqSection}>
              <View style={styles.eqHeader}>
                <Text style={styles.mlHeading}>Equity Curve</Text>
                <Pressable style={styles.replayBtn} onPress={() => setAnimating(true)}>
                  <Text style={styles.replayText}>Replay Animation</Text>
                </Pressable>
              </View>
              <View style={styles.chartWrap}>
                <ResponsiveContainer width="100%" height={220}>
                  <AreaChart data={(() => {
                    const mapped = equityCurve.map((d, i) => ({
                      i, label: String(d.date).slice(0, 10),
                      value: Number(d.equity) || 1
                    }));
                    const vals = mapped.map(d => d.value);
                    return { mapped, min: Math.min(...vals), max: Math.max(...vals) };
                  })().mapped}>
                    <XAxis
                      dataKey="label"
                      tick={{ fill: THEME.textMuted, fontSize: 9 }}
                      axisLine={{ stroke: THEME.border }}
                      tickLine={false}
                      interval={Math.floor(equityCurve.length / 6)}
                    />
                    <YAxis hide domain={['dataMin - 0.001', 'dataMax + 0.001']} />
                    <Tooltip contentStyle={{ backgroundColor: THEME.surface, border: `1px solid ${THEME.border}`, borderRadius: 4 }} labelStyle={{ color: THEME.textMuted, fontSize: 11 }} formatter={(val) => [Number(val).toFixed(4)]} />
                    <defs>
                      <linearGradient id="eqGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor={THEME.profit} stopOpacity={0.3} />
                        <stop offset="100%" stopColor={THEME.profit} stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <Area type="monotone" dataKey="value" stroke={THEME.profit} strokeWidth={2} fill="url(#eqGrad)" />
                  </AreaChart>
                </ResponsiveContainer>
              </View>
            </View>
          )}

          {multiResults.length > 1 && (
            <View style={styles.multiSection}>
              <Text style={styles.mlHeading}>Per-Ticker Breakdown</Text>
              {multiResults
                .filter((r) => !r.error)
                .map((r) => (
                  <View key={r.ticker} style={styles.multiRow}>
                    <Text style={styles.multiTicker}>{r.ticker}</Text>
                    <Text style={styles.multiStat}>{r.n_trades || 0} trades</Text>
                    <Text style={[styles.multiStat, { color: (r.avg_ret || 0) >= 0 ? THEME.profit : THEME.loss }]}>
                      {r.avg_ret != null ? `${(r.avg_ret * 100).toFixed(1)}%` : "—"}
                    </Text>
                    <Text style={styles.multiStat}>{(r.win_rate || 0) * 100 > 0 ? `${(r.win_rate * 100).toFixed(0)}% W` : "—"}</Text>
                    <Text style={styles.multiStat}>S {r.sharpe?.toFixed(2) || "—"}</Text>
                  </View>
                ))}
              {portfolioEquity.length > 0 && (
                <View style={{ marginTop: 12 }}>
                  <Text style={styles.mlHeading}>Portfolio Equity</Text>
                  <View style={styles.chartWrap}>
                    <ResponsiveContainer width="100%" height={180}>
                      <AreaChart data={portfolioEquity.map((d, i) => ({ i, label: String(d.date).slice(0, 10), value: Number(d.equity) || 1 }))}>
                        <XAxis dataKey="label" hide />
                        <YAxis hide domain={['dataMin - 0.001', 'dataMax + 0.001']} />
                        <Tooltip
                          contentStyle={{ backgroundColor: THEME.surface, border: `1px solid ${THEME.border}`, borderRadius: 4 }}
                          labelStyle={{ color: THEME.textMuted, fontSize: 11 }}
                          formatter={(val) => [`$${Number(val).toFixed(2)}`]}
                        />
                        <Area type="monotone" dataKey="value" stroke={THEME.profit} strokeWidth={2} fill={THEME.profitDim} fillOpacity={0.3} />
                      </AreaChart>
                    </ResponsiveContainer>
                  </View>
                </View>
              )}
            </View>
          )}

          {config.tickers && config.tickers.length > 0 && (
            <View style={styles.mlSection}>
              <Text style={styles.mlHeading}>ML Model Signal</Text>
              <Pressable
                style={styles.mlBtn}
                onPress={() => fetchMlSignal(config.tickers[0])}
                disabled={mlSignals != null && !mlSignals.error}
              >
                <Text style={styles.mlBtnText}>
                  {mlSignals != null ? "Refresh Prediction" : `Get Prediction for ${config.tickers[0]}`}
                </Text>
              </Pressable>
              {mlSignals && !mlSignals.error && (
                <View style={styles.mlResult}>
                  <View style={styles.mlRow}>
                    <MetricCard
                      label="Direction"
                      value={mlSignals.direction}
                      color={mlSignals.direction === "BUY" ? THEME.profit : THEME.loss}
                    />
                    <MetricCard
                      label="Confidence"
                      value={`${(mlSignals.confidence * 100).toFixed(1)}%`}
                    />
                  </View>
                  <View style={styles.mlRow}>
                    <MetricCard
                      label="Predicted Return"
                      value={`${(mlSignals.predicted_return * 100).toFixed(2)}%`}
                    />
                    <MetricCard
                      label="Sigma"
                      value={mlSignals.aleatoric_sigma?.toFixed(4)}
                    />
                  </View>
                </View>
              )}
              {mlSignals?.error && (
                <Text style={styles.mlError}>Model unavailable: {mlSignals.error}</Text>
              )}
            </View>
          )}
        </>
        </ErrorBoundary>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  wrap: { flex: 1, backgroundColor: THEME.bg },
  content: { padding: 16, paddingBottom: 32 },
  mlSection: {
    marginTop: 16,
    padding: 14,
    backgroundColor: THEME.surface,
    borderRadius: 8,
    borderLeftWidth: 3,
    borderLeftColor: THEME.accent,
  },
  mlHeading: { color: THEME.text, fontSize: 14, fontWeight: "700", marginBottom: 8 },
  mlBtn: {
    backgroundColor: THEME.accent,
    borderRadius: 6,
    paddingVertical: 10,
    alignItems: "center",
    marginBottom: 10,
  },
  mlBtnText: { color: THEME.textBright, fontWeight: "700", fontSize: 13 },
  mlResult: { gap: 8 },
  mlRow: { flexDirection: "row", gap: 8 },
  mlError: { color: THEME.loss, fontSize: 12, textAlign: "center" },
  multiSection: { marginTop: 16, padding: 14, backgroundColor: THEME.surface, borderRadius: 8, borderLeftWidth: 3, borderLeftColor: THEME.info },
  multiRow: { flexDirection: "row", alignItems: "center", paddingVertical: 5, borderBottomWidth: 1, borderBottomColor: THEME.border, gap: 8 },
  multiTicker: { color: THEME.text, fontWeight: "700", fontSize: 13, minWidth: 50 },
  multiStat: { color: THEME.textMuted, fontSize: 11, fontVariant: ["tabular-nums"] },
  chartWrap: { backgroundColor: THEME.surface, borderRadius: 6, padding: 4, marginTop: 6 },
  eqSection: { marginTop: 16, padding: 14, backgroundColor: THEME.surface, borderRadius: 8, borderLeftWidth: 3, borderLeftColor: THEME.profit },
  eqHeader: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginBottom: 8 },
  replayBtn: { paddingHorizontal: 10, paddingVertical: 5, borderRadius: 4, backgroundColor: THEME.accent },
  replayText: { color: THEME.textBright, fontWeight: "600", fontSize: 11 },
});
