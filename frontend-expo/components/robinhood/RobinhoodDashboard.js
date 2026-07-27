import React, { useEffect, useState, useCallback } from "react";
import { View, Text, Pressable, StyleSheet, ScrollView } from "react-native";
import THEME from "../../theme/colors";

/**
 * Robinhood portfolio panel for the Dashboard.
 * Shows: account summary, positions with personalized ML forecasts,
 * importable watchlist, and AI-powered action recommendations.
 */
export default function RobinhoodDashboard({ useRobinhood, token, useWatchlist }) {
  const {
    status,
    portfolio,
    forecast,
    loading,
    error,
    fetchPortfolio,
    fetchForecast,
    importWatchlist,
    addTicker,
  } = useRobinhood(token);
  const { addTicker: addLocalTicker } = useWatchlist ? useWatchlist() : { addTicker: () => {} };
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    if (status?.connected) {
      fetchPortfolio();
      fetchForecast();
    }
  }, [status?.connected]);

  if (!status?.connected) {
    return (
      <View style={styles.disconnected}>
        <Text style={styles.disconnectedTitle}>Robinhood Not Connected</Text>
        <Text style={styles.disconnectedHelp}>
          Connect your Robinhood account in Settings to get personalized AI forecasts for your portfolio.
        </Text>
      </View>
    );
  }

  const handleImport = async () => {
    const result = await importWatchlist();
    if (result?.imported) {
      result.imported.forEach(t => addLocalTicker(t));
    }
  };

  const account = portfolio?.account || {};
  const positions = portfolio?.positions || [];
  const fcs = forecast?.forecasts || [];
  const fcMap = {};
  fcs.forEach(f => { if (f.ticker) fcMap[f.ticker] = f; });

  return (
    <View style={styles.wrap}>
      <View style={styles.header}>
        <View>
          <Text style={styles.title}>Robinhood Portfolio</Text>
          <Text style={styles.subtitle}>Personalized AI analysis of your positions</Text>
        </View>
        <Pressable
          style={styles.refreshBtn}
          onPress={() => { fetchPortfolio(); fetchForecast(); }}
        >
          <Text style={styles.refreshText}>↻</Text>
        </Pressable>
      </View>

      {loading && <Text style={styles.loading}>Syncing...</Text>}

      {/* Account summary */}
      {account.equity != null && (
        <View style={styles.accountRow}>
          <View style={styles.accountBox}>
            <Text style={styles.accountLabel}>Equity</Text>
            <Text style={styles.accountValue}>${Number(account.equity).toFixed(2)}</Text>
          </View>
          <View style={styles.accountBox}>
            <Text style={styles.accountLabel}>Cash</Text>
            <Text style={styles.accountValue}>${Number(account.cash || 0).toFixed(2)}</Text>
          </View>
          <View style={styles.accountBox}>
            <Text style={styles.accountLabel}>Buying Power</Text>
            <Text style={styles.accountValue}>${Number(account.buying_power || 0).toFixed(2)}</Text>
          </View>
        </View>
      )}

      {/* Forecast summary */}
      {forecast && (
        <View style={styles.forecastBox}>
          <View style={styles.forecastHeader}>
            <Text style={styles.forecastLabel}>AI Portfolio Forecast</Text>
            <Text style={[
              styles.forecastValue,
              { color: forecast.projected_change >= 0 ? THEME.profit : THEME.loss }
            ]}>
              {forecast.projected_change >= 0 ? "+" : ""}${forecast.projected_change?.toFixed(2) || "0.00"}
              <Text style={styles.forecastPct}>
                {" "}({forecast.projected_change_pct >= 0 ? "+" : ""}{forecast.projected_change_pct?.toFixed(2) || "0.00"}%)
              </Text>
            </Text>
          </View>
          {forecast.summary?.recommendation && (
            <Text style={styles.recommendation}>{forecast.summary.recommendation}</Text>
          )}
        </View>
      )}

      {/* Positions with AI actions */}
      {positions.length > 0 && (
        <>
          <Text style={styles.sectionTitle}>Positions ({positions.length})</Text>
          {positions.map((pos, i) => {
            const ticker = pos.symbol || pos.ticker;
            const fc = fcMap[ticker] || {};
            const pl = pos.unrealized_pl || 0;
            const plPct = pos.unrealized_plpc || 0;
            const action = fc.action || "—";
            const actionColor =
              action.startsWith("STRONG BUY") || action === "BUY" ? THEME.profit :
              action.startsWith("STRONG SELL") || action === "SELL" ? THEME.loss :
              THEME.textMuted;
            return (
              <View key={i} style={styles.positionRow}>
                <View style={styles.positionLeft}>
                  <Text style={styles.positionTicker}>{ticker}</Text>
                  <Text style={styles.positionShares}>
                    {pos.quantity || pos.shares} shares · ${Number(pos.current_price || 0).toFixed(2)}
                  </Text>
                </View>
                <View style={styles.positionMid}>
                  <View style={[styles.actionBadge, { borderColor: actionColor }]}>
                    <Text style={[styles.actionText, { color: actionColor }]}>
                      {action}
                    </Text>
                  </View>
                  {fc.confidence != null && (
                    <Text style={styles.confidence}>{(fc.confidence * 100).toFixed(0)}%</Text>
                  )}
                </View>
                <View style={styles.positionRight}>
                  <Text style={[
                    styles.positionPl,
                    { color: pl >= 0 ? THEME.profit : THEME.loss }
                  ]}>
                    {pl >= 0 ? "+" : ""}${pl.toFixed(2)}
                  </Text>
                  <Text style={[
                    styles.positionPlPct,
                    { color: plPct >= 0 ? THEME.profit : THEME.loss }
                  ]}>
                    {plPct >= 0 ? "+" : ""}{(plPct * 100).toFixed(2)}%
                  </Text>
                </View>
              </View>
            );
          })}
        </>
      )}

      {/* Watchlist import */}
      <Pressable style={styles.importBtn} onPress={handleImport}>
        <Text style={styles.importBtnText}>Import Robinhood Watchlist → QuantFlow</Text>
      </Pressable>
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
  disconnected: {
    marginBottom: 16,
    padding: 14,
    backgroundColor: THEME.surface,
    borderRadius: 8,
    borderLeftWidth: 3,
    borderLeftColor: THEME.textMuted,
  },
  disconnectedTitle: { color: THEME.text, fontSize: 13, fontWeight: "700" },
  disconnectedHelp: { color: THEME.textMuted, fontSize: 11, marginTop: 4 },
  header: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "flex-start",
    marginBottom: 10,
  },
  title: { color: THEME.text, fontSize: 14, fontWeight: "700" },
  subtitle: { color: THEME.textMuted, fontSize: 11, marginTop: 2 },
  refreshBtn: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
    backgroundColor: THEME.surfaceLight,
    borderWidth: 1,
    borderColor: THEME.border,
  },
  refreshText: { color: THEME.accent, fontSize: 14, fontWeight: "700" },
  loading: { color: THEME.textMuted, fontSize: 11, fontStyle: "italic", marginBottom: 6 },
  accountRow: {
    flexDirection: "row",
    gap: 6,
    marginBottom: 10,
  },
  accountBox: {
    flex: 1,
    backgroundColor: THEME.surfaceLight,
    borderRadius: 6,
    padding: 8,
    alignItems: "center",
  },
  accountLabel: { color: THEME.textMuted, fontSize: 9, textTransform: "uppercase" },
  accountValue: { color: THEME.text, fontSize: 13, fontWeight: "700", fontVariant: ["tabular-nums"], marginTop: 2 },
  forecastBox: {
    backgroundColor: THEME.bg,
    borderRadius: 6,
    padding: 10,
    marginBottom: 10,
    borderLeftWidth: 3,
    borderLeftColor: THEME.accent,
  },
  forecastHeader: { flexDirection: "row", justifyContent: "space-between", alignItems: "center" },
  forecastLabel: { color: THEME.textMuted, fontSize: 11, fontWeight: "600" },
  forecastValue: { fontSize: 15, fontWeight: "800", fontVariant: ["tabular-nums"] },
  forecastPct: { fontSize: 11, fontWeight: "600" },
  recommendation: { color: THEME.text, fontSize: 11, marginTop: 6, fontStyle: "italic" },
  sectionTitle: { color: THEME.textMuted, fontSize: 11, fontWeight: "700", textTransform: "uppercase", marginBottom: 6, marginTop: 4 },
  positionRow: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: THEME.surfaceLight,
    borderRadius: 6,
    paddingHorizontal: 10,
    paddingVertical: 8,
    marginBottom: 4,
  },
  positionLeft: { flex: 1.2 },
  positionTicker: { color: THEME.text, fontWeight: "700", fontSize: 12 },
  positionShares: { color: THEME.textMuted, fontSize: 10, marginTop: 1 },
  positionMid: { flex: 1.2, alignItems: "center" },
  actionBadge: {
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 3,
    borderWidth: 1,
  },
  actionText: { fontSize: 9, fontWeight: "800" },
  confidence: { color: THEME.textMuted, fontSize: 9, marginTop: 1 },
  positionRight: { flex: 1, alignItems: "flex-end" },
  positionPl: { fontWeight: "700", fontSize: 12, fontVariant: ["tabular-nums"] },
  positionPlPct: { fontSize: 10, fontVariant: ["tabular-nums"] },
  importBtn: {
    paddingVertical: 8,
    borderRadius: 6,
    backgroundColor: THEME.surfaceLight,
    borderWidth: 1,
    borderColor: THEME.border,
    alignItems: "center",
    marginTop: 10,
  },
  importBtnText: { color: THEME.text, fontWeight: "600", fontSize: 11 },
});
