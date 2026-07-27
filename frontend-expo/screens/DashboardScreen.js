import React, { useState, useCallback } from "react";
import { ScrollView, RefreshControl, View, Text, Pressable, StyleSheet } from "react-native";
import THEME from "../theme/colors";
import usePortfolio from "../hooks/usePortfolio";
import { apiGet, apiPost } from "../services/api";
import AccountSummary from "../components/dashboard/AccountSummary";
import EquityCurve from "../components/charts/EquityCurve";
import PositionList from "../components/dashboard/PositionList";
import SignalCard from "../components/dashboard/SignalCard";
import ErrorBanner from "../components/shared/ErrorBanner";
import LoadingSpinner from "../components/shared/LoadingSpinner";
import MetricCard from "../components/shared/MetricCard";
import ScreenTitle from "../components/shared/ScreenTitle";
import WatchlistPanel from "../components/dashboard/WatchlistPanel";
import AllocationAdvisor from "../components/dashboard/AllocationAdvisor";
import PortfolioManager from "../components/dashboard/PortfolioManager";

export default function DashboardScreen({ token }) {
  const { account, positions, signals, equityCurve, loading, error, refresh } = usePortfolio(token);
  const [recommendations, setRecommendations] = useState(null);
  const [recLoading, setRecLoading] = useState(false);

  const loadRecommendations = useCallback(async () => {
    setRecLoading(true);
    try {
      const data = await apiGet("/recommend/latest", token);
      setRecommendations(data?.items || []);
    } catch {
      setRecommendations([]);
    } finally {
      setRecLoading(false);
    }
  }, [token]);

  return (
    <ScrollView
      style={styles.wrap}
      contentContainerStyle={styles.content}
      refreshControl={<RefreshControl refreshing={loading} onRefresh={refresh} tintColor={THEME.accent} />}
    >
      <ScreenTitle title="Dashboard" subtitle="Account, watchlist, and portfolio overview" icon="D" />
      {error && <ErrorBanner message={error} onRetry={refresh} />}
      {loading && !account && <LoadingSpinner />}

      <AccountSummary account={account} />
      <EquityCurve data={equityCurve} />

      <WatchlistPanel token={token} />
      <AllocationAdvisor token={token} />
      <PortfolioManager token={token} />

      <PositionList positions={positions} />
      <SignalCard signals={signals} />

      <View style={styles.section}>
        <View style={styles.sectionHeader}>
          <Text style={styles.sectionTitle}>ML Recommendations</Text>
          <Pressable style={styles.recBtn} onPress={loadRecommendations} disabled={recLoading}>
            <Text style={styles.recBtnText}>{recLoading ? "Loading..." : recommendations ? "Refresh" : "Load"}</Text>
          </Pressable>
        </View>
        {recommendations && recommendations.length > 0 ? (
          (() => {
            // Deduplicate by ticker, keeping the most recent (first in list)
            const seen = new Set();
            const unique = recommendations.filter(r => {
              const t = r.ticker || r.symbol;
              if (!t) return false;
              if (seen.has(t)) return false;
              seen.add(t);
              return true;
            });
            return unique.slice(0, 5).map((r, i) => {
              const action = (r.action || r.signal || r.bias || "HOLD").toUpperCase();
              return (
                <View key={i} style={styles.recRow}>
                  <Text style={styles.recTicker}>{r.ticker || r.symbol || "???"}</Text>
                  <Text style={[styles.recAction, {
                    color: action === "BUY" || action === "LONG" ? THEME.profit :
                           action === "SELL" || action === "SHORT" ? THEME.loss : THEME.neutral
                  }]}>
                    {action}
                  </Text>
                  <Text style={styles.recScore}>
                    {r.score != null ? `${(r.score * 100).toFixed(0)}%` :
                     r.confidence != null ? `${(r.confidence * 100).toFixed(0)}%` : "—"}
                  </Text>
                </View>
              );
            });
          })()
        ) : recommendations && recommendations.length === 0 ? (
          <Text style={styles.noData}>No recommendations available</Text>
        ) : null}
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Portfolio Risk</Text>
        {account ? (
          <View style={styles.riskRow}>
            <MetricCard label="Max Position" value={`${((positions.reduce((s, p) => s + (p.market_value || 0), 0) / (account.equity || 1)) * 100).toFixed(1)}%`} />
            <MetricCard label="Cash Ratio" value={`${((account.cash || 0) / (account.equity || 1) * 100).toFixed(1)}%`} />
          </View>
        ) : (
          <Text style={styles.noData}>Loading account data...</Text>
        )}
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  wrap: { flex: 1, backgroundColor: THEME.bg },
  content: { padding: 16, paddingBottom: 32 },
  section: {
    marginBottom: 16,
    padding: 14,
    backgroundColor: THEME.surface,
    borderRadius: 8,
    borderLeftWidth: 3,
    borderLeftColor: THEME.accent,
  },
  sectionHeader: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginBottom: 8 },
  sectionTitle: { color: THEME.text, fontSize: 14, fontWeight: "700" },
  recBtn: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 4,
    backgroundColor: THEME.accent,
  },
  recBtnText: { color: THEME.textBright, fontSize: 11, fontWeight: "600" },
  recRow: {
    flexDirection: "row",
    alignItems: "center",
    paddingVertical: 6,
    borderBottomWidth: 1,
    borderBottomColor: THEME.border,
  },
  recTicker: { color: THEME.text, fontWeight: "700", fontSize: 13, flex: 1 },
  recAction: { fontWeight: "800", fontSize: 12, minWidth: 40, textAlign: "center" },
  recScore: { color: THEME.textMuted, fontSize: 12, minWidth: 40, textAlign: "right", fontVariant: ["tabular-nums"] },
  noData: { color: THEME.textMuted, fontSize: 12, marginTop: 4 },
  riskRow: { flexDirection: "row", gap: 8, marginTop: 8 },
});
