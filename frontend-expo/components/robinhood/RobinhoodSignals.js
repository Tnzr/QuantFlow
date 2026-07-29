import React, { useEffect, useState, useCallback } from "react";
import { View, Text, Pressable, StyleSheet, ScrollView } from "react-native";
import THEME from "../../theme/colors";

/**
 * Comprehensive Robinhood Signals view.
 * Displays: account info, positions, signals, sentiment, options, scanner
 */
export default function RobinhoodSignals({ useRobinhood, token }) {
  const {
    status,
    portfolio,
    forecast,
    signals,
    sentiment,
    options,
    scanner,
    loading,
    error,
    fetchPortfolio,
    fetchForecast,
    fetchSignals,
    fetchSentiment,
    fetchOptions,
    fetchScanner,
    importWatchlist,
  } = useRobinhood(token);

  const [activeTab, setActiveTab] = useState("signals");
  const [refreshing, setRefreshing] = useState(false);

  const refreshAll = useCallback(async () => {
    setRefreshing(true);
    try {
      await Promise.allSettled([
        fetchPortfolio(),
        fetchForecast(),
        fetchSignals(),
        fetchSentiment(),
        fetchScanner(),
      ]);
    } finally {
      setRefreshing(false);
    }
  }, [fetchPortfolio, fetchForecast, fetchSignals, fetchSentiment, fetchScanner]);

  useEffect(() => {
    if (status?.connected) {
      refreshAll();
    }
  }, [status?.connected]);

  if (!status?.connected) {
    return (
      <View style={styles.disconnected}>
        <Text style={styles.disconnectedTitle}>Robinhood Not Connected</Text>
        <Text style={styles.disconnectedHelp}>
          Connect your Robinhood account in Settings to access personalized signals, options, and scanner.
        </Text>
      </View>
    );
  }

  const account = portfolio?.account?.content?.[0]?.text
    ? JSON.parse(portfolio.account.content[0].text)
    : null;
  const accounts = account?.data?.accounts || [];
  const positions = forecast?.forecasts || [];
  const analysisSource = forecast?.summary?.analysis_source || "watchlist";

  return (
    <View style={styles.wrap}>
      {/* Header */}
      <View style={styles.header}>
        <View>
          <Text style={styles.title}>Robinhood Signals</Text>
          <Text style={styles.subtitle}>Personalized AI analysis & trading signals</Text>
        </View>
        <Pressable style={styles.refreshBtn} onPress={refreshAll} disabled={refreshing}>
          <Text style={styles.refreshText}>{refreshing ? "..." : "↻"}</Text>
        </Pressable>
      </View>

      {/* Account Selector */}
      {accounts.length > 0 && (
        <View style={styles.accountsBox}>
          <Text style={styles.sectionLabel}>Your Accounts</Text>
          {accounts.map((acc, i) => {
            const isAgentic = acc.agentic_allowed;
            const isDefault = acc.is_default;
            return (
              <View key={i} style={[styles.accountRow, isDefault && styles.accountDefault]}>
                <View style={styles.accountLeft}>
                  <Text style={styles.accountType}>
                    {acc.brokerage_account_type?.toUpperCase() || "Account"}
                    {isDefault ? " (Default)" : ""}
                    {isAgentic ? " 🤖" : ""}
                  </Text>
                  <Text style={styles.accountNum}>••••{acc.account_number?.slice(-4)}</Text>
                </View>
                <View style={styles.accountRight}>
                  {isAgentic && <Text style={styles.agenticTag}>AGENTIC</Text>}
                  {acc.option_level && acc.option_level !== "option_level_0" && (
                    <Text style={styles.optionTag}>{acc.option_level.replace("option_level_", "L")}</Text>
                  )}
                </View>
              </View>
            );
          })}
          {accounts.find(a => !a.agentic_allowed && a.option_level && a.option_level !== "option_level_0") && (
            <Text style={styles.accountNote}>
              Note: Your default margin account (option level 2) is NOT accessible to the agent.
              To use AI signals for your investments, either move positions to your Agentic account,
              or upgrade agent access for your default account.
            </Text>
          )}
        </View>
      )}

      {/* Forecast Summary */}
      {forecast && forecast.summary && (
        <View style={styles.forecastBox}>
          <View style={styles.forecastHeader}>
            <Text style={styles.forecastLabel}>
              {analysisSource === "position" ? "Position Forecast" : "Watchlist Forecast"}
            </Text>
            <Text style={[
              styles.forecastValue,
              { color: forecast.projected_change >= 0 ? THEME.profit : THEME.loss }
            ]}>
              {forecast.projected_change >= 0 ? "+" : ""}${forecast.projected_change?.toFixed(2)}
              <Text style={styles.forecastPct}>
                {" "}({forecast.projected_change_pct >= 0 ? "+" : ""}{forecast.projected_change_pct?.toFixed(2)}%)
              </Text>
            </Text>
          </View>
          {forecast.summary.recommendation && (
            <Text style={styles.recommendation}>{forecast.summary.recommendation}</Text>
          )}
          <View style={styles.forecastStats}>
            <Text style={styles.statText}>↑ {forecast.summary.bullish_count} bullish</Text>
            <Text style={styles.statText}>↓ {forecast.summary.bearish_count} bearish</Text>
          </View>
        </View>
      )}

      {/* Tab Navigation */}
      <View style={styles.tabRow}>
        {[
          { key: "signals", label: "Signals" },
          { key: "sentiment", label: "Sentiment" },
          { key: "options", label: "Options" },
          { key: "scanner", label: "Scanner" },
        ].map(tab => (
          <Pressable
            key={tab.key}
            style={[styles.tab, activeTab === tab.key && styles.tabActive]}
            onPress={() => setActiveTab(tab.key)}
          >
            <Text style={[styles.tabText, activeTab === tab.key && styles.tabTextActive]}>
              {tab.label}
            </Text>
          </Pressable>
        ))}
      </View>

      {/* Tab Content */}
      {activeTab === "signals" && <SignalsTab signals={signals} loading={loading} />}
      {activeTab === "sentiment" && <SentimentTab sentiment={sentiment} loading={loading} />}
      {activeTab === "options" && <OptionsTab options={options} loading={loading} onFetch={fetchOptions} />}
      {activeTab === "scanner" && <ScannerTab scanner={scanner} loading={loading} />}
    </View>
  );
}

function SignalsTab({ signals, loading }) {
  if (loading && !signals) return <Text style={styles.loading}>Loading signals...</Text>;
  if (!signals || !signals.signals) return <Text style={styles.noData}>No signals available</Text>;

  const sigList = signals.signals || [];
  if (sigList.length === 0) return <Text style={styles.noData}>No signals available</Text>;

  return (
    <View>
      {sigList.map((sig, i) => {
        const signalColor =
          sig.signal === "STRONG BUY" || sig.signal === "BUY" ? THEME.profit :
          sig.signal === "STRONG SELL" || sig.signal === "SELL" ? THEME.loss :
          sig.signal === "WATCH BUY" ? THEME.profitDim :
          sig.signal === "WATCH SELL" ? THEME.lossDim :
          THEME.textMuted;
        return (
          <View key={i} style={styles.signalRow}>
            <View style={styles.signalLeft}>
              <Text style={styles.signalTicker}>{sig.ticker}</Text>
              <Text style={styles.signalPrice}>${sig.current_price}</Text>
            </View>
            <View style={styles.signalMid}>
              <View style={[styles.signalBadge, { backgroundColor: signalColor, borderColor: signalColor }]}>
                <Text style={styles.signalBadgeText}>{sig.signal}</Text>
              </View>
              <Text style={styles.signalReason}>{sig.reason}</Text>
            </View>
            <View style={styles.signalRight}>
              <Text style={styles.signalRSI}>RSI: {sig.rsi}</Text>
              <Text style={styles.signalML}>ML: {sig.ml_confidence * 100}%</Text>
            </View>
          </View>
        );
      })}
    </View>
  );
}

function SentimentTab({ sentiment, loading }) {
  if (loading && !sentiment) return <Text style={styles.loading}>Loading sentiment...</Text>;
  if (!sentiment || !sentiment.sentiment_analysis) return <Text style={styles.noData}>No sentiment data</Text>;

  const analysis = sentiment.sentiment_analysis || [];
  const summary = sentiment.summary || {};

  return (
    <View>
      <View style={styles.sentimentSummary}>
        <Text style={styles.sentimentLabel}>Overall Market Sentiment</Text>
        <Text style={[
          styles.sentimentValue,
          { color: summary.overall_sentiment === "BULLISH" ? THEME.profit : summary.overall_sentiment === "BEARISH" ? THEME.loss : THEME.textMuted }
        ]}>
          {summary.overall_sentiment} ({summary.avg_sentiment_score?.toFixed(2)})
        </Text>
        {summary.market_warnings?.map((w, i) => (
          <Text key={i} style={styles.warning}>⚠ {w}</Text>
        ))}
      </View>

      {analysis.map((sa, i) => (
        <View key={i} style={styles.sentimentRow}>
          <View style={styles.sentimentLeft}>
            <Text style={styles.sentimentTicker}>{sa.ticker}</Text>
            <Text style={styles.sentimentPrice}>${sa.current_price}</Text>
          </View>
          <View style={styles.sentimentMid}>
            <View style={[
              styles.sentimentBadge,
              { borderColor: sa.ml_direction === "BUY" ? THEME.profit : sa.ml_direction === "SELL" ? THEME.loss : THEME.border }
            ]}>
              <Text style={[
                styles.sentimentBadgeText,
                { color: sa.ml_direction === "BUY" ? THEME.profit : sa.ml_direction === "SELL" ? THEME.loss : THEME.textMuted }
              ]}>
                {sa.ml_direction}
              </Text>
            </View>
            <Text style={styles.sentimentPlay}>{sa.market_play?.replace("_", " ")}</Text>
            {sa.warnings?.map((w, j) => (
              <Text key={j} style={styles.sentimentWarning}>⚠ {w}</Text>
            ))}
            {sa.rules_of_thumb?.map((r, j) => (
              <Text key={j} style={styles.sentimentRule}>→ {r}</Text>
            ))}
          </View>
          <View style={styles.sentimentRight}>
            <Text style={styles.sentimentRSI}>RSI: {sa.rsi}</Text>
            {sa.pe_ratio && <Text style={styles.sentimentP}>P/E: {sa.pe_ratio}</Text>}
            {sa.next_earnings && <Text style={styles.sentimentEarnings}>📅 {sa.next_earnings}</Text>}
          </View>
        </View>
      ))}
    </View>
  );
}

function OptionsTab({ options, loading, onFetch }) {
  const [strategy, setStrategy] = useState("long");
  const [expiryDays, setExpiryDays] = useState(30);
  const [balance, setBalance] = useState(10000);

  useEffect(() => {
    if (options) {
      onFetch({ strategy, expiry_days: expiryDays, balance });
    }
  }, [strategy, expiryDays]);

  if (loading && !options) return <Text style={styles.loading}>Loading options...</Text>;

  return (
    <View>
      <View style={styles.optionsControls}>
        <Text style={styles.controlLabel}>Strategy</Text>
        <View style={styles.strategyRow}>
          {["long", "short"].map(s => (
            <Pressable
              key={s}
              style={[styles.strategyBtn, strategy === s && styles.strategyBtnActive]}
              onPress={() => setStrategy(s)}
            >
              <Text style={[styles.strategyText, strategy === s && styles.strategyTextActive]}>
                {s.toUpperCase()}
              </Text>
            </Pressable>
          ))}
        </View>
        <Text style={styles.controlLabel}>Expiry (days)</Text>
        <View style={styles.expiryRow}>
          {[7, 14, 30, 60].map(d => (
            <Pressable
              key={d}
              style={[styles.expiryBtn, expiryDays === d && styles.expiryBtnActive]}
              onPress={() => setExpiryDays(d)}
            >
              <Text style={[styles.expiryText, expiryDays === d && styles.expiryTextActive]}>
                {d}D
              </Text>
            </Pressable>
          ))}
        </View>
        <View style={styles.balanceRow}>
          <Text style={styles.controlLabel}>Balance: ${balance.toLocaleString()}</Text>
        </View>
      </View>

      {options && options.recommendations && options.recommendations.length > 0 ? (
        <View>
          {options.recommendations.map((rec, i) => (
            <View key={i} style={styles.optionRow}>
              <View style={styles.optionLeft}>
                <Text style={styles.optionTicker}>{rec.ticker}</Text>
                <Text style={styles.optionStrategy}>{rec.strategy}</Text>
              </View>
              <View style={styles.optionMid}>
                <Text style={styles.optionStrike}>${rec.strike} {rec.expiry}</Text>
                <Text style={styles.optionDetails}>
                  {rec.contracts} contracts × ${rec.entry_price} = ${rec.total_cost?.toLocaleString()}
                </Text>
                <Text style={styles.optionDetails}>
                  Δ={rec.delta} | IV={rec.implied_vol} | {rec.days_to_expiry}d
                </Text>
                <Text style={styles.optionReason}>{rec.reason}</Text>
              </View>
              <View style={styles.optionRight}>
                <Text style={styles.optionScore}>Score: {rec.score}</Text>
                <View style={[
                  styles.riskBadge,
                  { backgroundColor: rec.risk_level === "HIGH" ? THEME.loss : rec.risk_level === "MEDIUM" ? THEME.warn : THEME.profit }
                ]}>
                  <Text style={styles.riskText}>{rec.risk_level}</Text>
                </View>
              </View>
            </View>
          ))}
        </View>
      ) : (
        <Text style={styles.noData}>
          No options recommendations. Try {strategy === "long" ? "short" : "long"} strategy or increase expiry window.
        </Text>
      )}
    </View>
  );
}

function ScannerTab({ scanner, loading }) {
  if (loading && !scanner) return <Text style={styles.loading}>Running scanner...</Text>;
  if (!scanner || !scanner.scan_results) return <Text style={styles.noData}>No scan results</Text>;

  return (
    <View>
      <View style={styles.scannerSummary}>
        <Text style={styles.scannerLabel}>Scanner Results</Text>
        <View style={styles.scannerStats}>
          <Text style={styles.scannerStat}>↑ {scanner.summary?.strong_buy || 0} strong buy</Text>
          <Text style={styles.scannerStat}>↗ {scanner.summary?.buy || 0} buy</Text>
          <Text style={styles.scannerStat}>→ {scanner.summary?.hold || 0} hold</Text>
          <Text style={styles.scannerStat}>↘ {scanner.summary?.sell || 0} sell</Text>
          <Text style={styles.scannerStat}>↓ {scanner.summary?.strong_sell || 0} strong sell</Text>
        </View>
        {scanner.summary?.top_pick && (
          <Text style={styles.topPick}>Top pick: {scanner.summary.top_pick}</Text>
        )}
      </View>

      {scanner.scan_results.map((r, i) => {
        const signalColor =
          r.signal === "STRONG BUY" || r.signal === "BUY" ? THEME.profit :
          r.signal === "STRONG SELL" || r.signal === "SELL" ? THEME.loss :
          THEME.textMuted;
        return (
          <View key={i} style={styles.scannerRow}>
            <View style={styles.scannerLeft}>
              <Text style={styles.scannerTicker}>{r.ticker}</Text>
              <Text style={styles.scannerPrice}>${r.price}</Text>
            </View>
            <View style={styles.scannerMid}>
              <View style={[styles.scannerBadge, { backgroundColor: signalColor, borderColor: signalColor }]}>
                <Text style={styles.scannerBadgeText}>{r.signal}</Text>
              </View>
              <Text style={styles.scannerReason}>{r.reason}</Text>
              <Text style={styles.scannerMeta}>
                Vol: {(r.volume / 1e6).toFixed(1)}M | RSI: {r.rsi}
              </Text>
            </View>
          </View>
        );
      })}
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: { marginBottom: 16 },
  header: { flexDirection: "row", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 8 },
  title: { color: THEME.text, fontSize: 14, fontWeight: "700" },
  subtitle: { color: THEME.textMuted, fontSize: 11, marginTop: 2 },
  refreshBtn: { paddingHorizontal: 8, paddingVertical: 4, borderRadius: 4, backgroundColor: THEME.surfaceLight, borderWidth: 1, borderColor: THEME.border },
  refreshText: { color: THEME.accent, fontSize: 14, fontWeight: "700" },
  disconnected: { marginBottom: 16, padding: 14, backgroundColor: THEME.surface, borderRadius: 8, borderLeftWidth: 3, borderLeftColor: THEME.textMuted },
  disconnectedTitle: { color: THEME.text, fontSize: 13, fontWeight: "700" },
  disconnectedHelp: { color: THEME.textMuted, fontSize: 11, marginTop: 4 },
  accountsBox: { backgroundColor: THEME.surface, borderRadius: 8, padding: 10, marginBottom: 8, borderLeftWidth: 3, borderLeftColor: THEME.accent },
  sectionLabel: { color: THEME.textMuted, fontSize: 10, fontWeight: "700", textTransform: "uppercase", marginBottom: 6 },
  accountRow: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", paddingVertical: 4, paddingHorizontal: 8, backgroundColor: THEME.surfaceLight, borderRadius: 4, marginBottom: 4 },
  accountDefault: { borderLeftWidth: 3, borderLeftColor: THEME.accent },
  accountLeft: { flex: 1 },
  accountType: { color: THEME.text, fontSize: 11, fontWeight: "600" },
  accountNum: { color: THEME.textMuted, fontSize: 10, fontFamily: "monospace" },
  accountRight: { flexDirection: "row", gap: 4 },
  agenticTag: { backgroundColor: THEME.profit, color: THEME.textBright, fontSize: 8, fontWeight: "800", paddingHorizontal: 4, paddingVertical: 1, borderRadius: 2 },
  optionTag: { backgroundColor: THEME.surface, color: THEME.text, fontSize: 8, fontWeight: "700", paddingHorizontal: 4, paddingVertical: 1, borderRadius: 2, borderWidth: 1, borderColor: THEME.border },
  accountNote: { color: THEME.warn, fontSize: 10, marginTop: 6, fontStyle: "italic" },
  forecastBox: { backgroundColor: THEME.surface, borderRadius: 8, padding: 10, marginBottom: 8, borderLeftWidth: 3, borderLeftColor: THEME.accent },
  forecastHeader: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginBottom: 4 },
  forecastLabel: { color: THEME.textMuted, fontSize: 10, fontWeight: "700", textTransform: "uppercase" },
  forecastValue: { fontSize: 16, fontWeight: "800", fontVariant: ["tabular-nums"] },
  forecastPct: { fontSize: 11, fontWeight: "600" },
  recommendation: { color: THEME.text, fontSize: 11, marginTop: 4, fontStyle: "italic" },
  forecastStats: { flexDirection: "row", gap: 12, marginTop: 6 },
  statText: { color: THEME.textMuted, fontSize: 10, fontWeight: "600" },
  tabRow: { flexDirection: "row", backgroundColor: THEME.surface, borderRadius: 6, padding: 2, marginBottom: 8 },
  tab: { flex: 1, paddingVertical: 6, alignItems: "center", borderRadius: 4 },
  tabActive: { backgroundColor: THEME.accentGlow },
  tabText: { color: THEME.textMuted, fontSize: 10, fontWeight: "700" },
  tabTextActive: { color: THEME.accent },
  loading: { color: THEME.textMuted, fontSize: 11, fontStyle: "italic", textAlign: "center", padding: 20 },
  noData: { color: THEME.textMuted, fontSize: 11, textAlign: "center", padding: 20, fontStyle: "italic" },
  signalRow: { flexDirection: "row", backgroundColor: THEME.surface, borderRadius: 6, padding: 8, marginBottom: 4, alignItems: "center" },
  signalLeft: { flex: 1, alignItems: "center" },
  signalTicker: { color: THEME.text, fontSize: 13, fontWeight: "800" },
  signalPrice: { color: THEME.textMuted, fontSize: 10, fontVariant: ["tabular-nums"] },
  signalMid: { flex: 2, paddingHorizontal: 8 },
  signalBadge: { paddingHorizontal: 6, paddingVertical: 2, borderRadius: 3, borderWidth: 1, alignSelf: "flex-start", marginBottom: 2 },
  signalBadgeText: { color: THEME.textBright, fontSize: 9, fontWeight: "800" },
  signalReason: { color: THEME.textMuted, fontSize: 10, lineHeight: 13 },
  signalRight: { flex: 1, alignItems: "flex-end" },
  signalRSI: { color: THEME.textMuted, fontSize: 10, fontWeight: "600" },
  signalML: { color: THEME.textMuted, fontSize: 9, marginTop: 1 },
  sentimentSummary: { backgroundColor: THEME.surface, borderRadius: 6, padding: 10, marginBottom: 8 },
  sentimentLabel: { color: THEME.textMuted, fontSize: 10, fontWeight: "700", textTransform: "uppercase" },
  sentimentValue: { fontSize: 16, fontWeight: "800", marginTop: 4 },
  warning: { color: THEME.warn, fontSize: 10, marginTop: 4 },
  sentimentRow: { flexDirection: "row", backgroundColor: THEME.surface, borderRadius: 6, padding: 8, marginBottom: 4, alignItems: "flex-start" },
  sentimentLeft: { flex: 1, alignItems: "center" },
  sentimentTicker: { color: THEME.text, fontSize: 13, fontWeight: "800" },
  sentimentPrice: { color: THEME.textMuted, fontSize: 10, fontVariant: ["tabular-nums"] },
  sentimentMid: { flex: 2, paddingHorizontal: 8 },
  sentimentBadge: { paddingHorizontal: 6, paddingVertical: 2, borderRadius: 3, borderWidth: 1, alignSelf: "flex-start", marginBottom: 2 },
  sentimentBadgeText: { fontSize: 9, fontWeight: "800" },
  sentimentPlay: { color: THEME.textMuted, fontSize: 9, fontStyle: "italic", marginTop: 2 },
  sentimentWarning: { color: THEME.warn, fontSize: 9, marginTop: 1 },
  sentimentRule: { color: THEME.textMuted, fontSize: 9, marginTop: 1 },
  sentimentRight: { flex: 1, alignItems: "flex-end" },
  sentimentRSI: { color: THEME.textMuted, fontSize: 10, fontWeight: "600" },
  sentimentP: { color: THEME.textMuted, fontSize: 9, marginTop: 1 },
  sentimentEarnings: { color: THEME.accent, fontSize: 9, marginTop: 1 },
  optionsControls: { backgroundColor: THEME.surface, borderRadius: 6, padding: 10, marginBottom: 8 },
  controlLabel: { color: THEME.textMuted, fontSize: 10, fontWeight: "700", textTransform: "uppercase", marginBottom: 4 },
  strategyRow: { flexDirection: "row", gap: 4, marginBottom: 8 },
  strategyBtn: { flex: 1, paddingVertical: 6, borderRadius: 4, backgroundColor: THEME.bg, borderWidth: 1, borderColor: THEME.border, alignItems: "center" },
  strategyBtnActive: { backgroundColor: THEME.accentGlow, borderColor: THEME.accent },
  strategyText: { color: THEME.textMuted, fontSize: 10, fontWeight: "700" },
  strategyTextActive: { color: THEME.accent },
  expiryRow: { flexDirection: "row", gap: 4, marginBottom: 8 },
  expiryBtn: { flex: 1, paddingVertical: 6, borderRadius: 4, backgroundColor: THEME.bg, borderWidth: 1, borderColor: THEME.border, alignItems: "center" },
  expiryBtnActive: { backgroundColor: THEME.accentGlow, borderColor: THEME.accent },
  expiryText: { color: THEME.textMuted, fontSize: 10, fontWeight: "700" },
  expiryTextActive: { color: THEME.accent },
  balanceRow: { paddingTop: 4, borderTopWidth: 1, borderTopColor: THEME.border },
  optionRow: { flexDirection: "row", backgroundColor: THEME.surface, borderRadius: 6, padding: 8, marginBottom: 4 },
  optionLeft: { flex: 1, alignItems: "center" },
  optionTicker: { color: THEME.text, fontSize: 13, fontWeight: "800" },
  optionStrategy: { color: THEME.textMuted, fontSize: 9, fontWeight: "600" },
  optionMid: { flex: 2, paddingHorizontal: 8 },
  optionStrike: { color: THEME.text, fontSize: 12, fontWeight: "700" },
  optionDetails: { color: THEME.textMuted, fontSize: 9, fontVariant: ["tabular-nums"], marginTop: 1 },
  optionReason: { color: THEME.textMuted, fontSize: 9, marginTop: 2, fontStyle: "italic" },
  optionRight: { flex: 1, alignItems: "flex-end" },
  optionScore: { color: THEME.accent, fontSize: 11, fontWeight: "700", marginBottom: 4 },
  riskBadge: { paddingHorizontal: 6, paddingVertical: 2, borderRadius: 3 },
  riskText: { color: THEME.textBright, fontSize: 8, fontWeight: "800" },
  scannerSummary: { backgroundColor: THEME.surface, borderRadius: 6, padding: 10, marginBottom: 8 },
  scannerLabel: { color: THEME.textMuted, fontSize: 10, fontWeight: "700", textTransform: "uppercase" },
  scannerStats: { flexDirection: "row", flexWrap: "wrap", gap: 8, marginTop: 6 },
  scannerStat: { color: THEME.textMuted, fontSize: 10, fontWeight: "600" },
  topPick: { color: THEME.accent, fontSize: 11, fontWeight: "700", marginTop: 6 },
  scannerRow: { flexDirection: "row", backgroundColor: THEME.surface, borderRadius: 6, padding: 8, marginBottom: 4 },
  scannerLeft: { flex: 1, alignItems: "center" },
  scannerTicker: { color: THEME.text, fontSize: 13, fontWeight: "800" },
  scannerPrice: { color: THEME.textMuted, fontSize: 10, fontVariant: ["tabular-nums"] },
  scannerMid: { flex: 2, paddingHorizontal: 8 },
  scannerBadge: { paddingHorizontal: 6, paddingVertical: 2, borderRadius: 3, borderWidth: 1, alignSelf: "flex-start", marginBottom: 2 },
  scannerBadgeText: { color: THEME.textBright, fontSize: 9, fontWeight: "800" },
  scannerReason: { color: THEME.textMuted, fontSize: 9, marginTop: 1 },
  scannerMeta: { color: THEME.textMuted, fontSize: 8, marginTop: 2 },
});
