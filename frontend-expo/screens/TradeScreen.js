import React, { useState, useEffect, useCallback } from "react";
import { ScrollView, RefreshControl, View, Text, Pressable, StyleSheet } from "react-native";
import { apiGet, apiPost } from "../services/api";
import useWatchlist from "../hooks/useWatchlist";
import THEME from "../theme/colors";
import OrderTicket from "../components/trade/OrderTicket";
import PositionRow from "../components/trade/PositionRow";
import OrderHistory from "../components/trade/OrderHistory";
import Sparkline from "../components/charts/Sparkline";
import ErrorBanner from "../components/shared/ErrorBanner";
import LoadingSpinner from "../components/shared/LoadingSpinner";
import ScreenTitle from "../components/shared/ScreenTitle";

export default function TradeScreen({ token }) {
  const [portfolio, setPortfolio] = useState(null);
  const [orders, setOrders] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [recommendations, setRecommendations] = useState(null);
  const [marketSignals, setMarketSignals] = useState([]);
  const { watchlist } = useWatchlist();

  const fetchData = useCallback(async () => {
    try {
      setError(null);
      const [portData, orderData, recData] = await Promise.allSettled([
        apiGet("/portfolio/positions", token),
        apiGet("/execution/intents", token),
        apiGet("/recommend/latest", token),
      ]);

      if (portData.status === "fulfilled") setPortfolio(portData.value);
      if (orderData.status === "fulfilled") setOrders(orderData.value);
      if (recData.status === "fulfilled") setRecommendations(recData.value?.items || []);
      if (Array.isArray(watchlist) && watchlist.length > 0) {
        const signals = await Promise.allSettled(watchlist.map(async (t) => {
          try {
            const fc = await apiGet(`/charts/forecast?ticker=${encodeURIComponent(t)}`, token);
            return { ticker: t, direction: fc.direction, forecast: fc.forecast?.slice(0, 5) || [] };
          } catch { return { ticker: t, error: true }; }
        }));
        setMarketSignals(signals.filter(s => s.status === "fulfilled").map(s => s.value));
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [token, watchlist]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const positions = portfolio?.positions || [];
  const ordersList = orders?.items || (Array.isArray(orders) ? orders : orders?.intents || orders?.orders || []);

  return (
    <ScrollView
      style={styles.wrap}
      contentContainerStyle={styles.content}
      refreshControl={<RefreshControl refreshing={loading} onRefresh={fetchData} tintColor={THEME.accent} />}
    >
      <ScreenTitle title="Trade" subtitle="Place orders, view positions and history" icon="T" />
      {error && <ErrorBanner message={error} onRetry={fetchData} />}

      <OrderTicket token={token} portfolio={portfolio} onOrderPlaced={fetchData} />


      {marketSignals.length > 0 && (
        <View style={styles.recSection}>
          <Text style={styles.recHeading}>Market Overview</Text>
          {marketSignals.map((s) => (
            <View key={s.ticker} style={styles.marketRow}>
              <Text style={styles.marketTicker}>{s.ticker}</Text>
              <Text style={[styles.marketDir, { color: s.direction === "BUY" ? THEME.profit : s.direction === "SELL" ? THEME.loss : THEME.neutral }]}>
                {s.direction || "—"}
              </Text>
              <View style={styles.marketSpark}>
                <Sparkline data={s.forecast?.map(f => f.price) || []} width={60} height={22} />
              </View>
            </View>
          ))}
        </View>
      )}

      {recommendations && recommendations.length > 0 && (
        <View style={styles.recSection}>
          <Text style={styles.recHeading}>Top Recommendations</Text>
          {recommendations.slice(0, 3).map((r, i) => (
            <View key={i} style={styles.recRow}>
              <Text style={styles.recTicker}>{r.ticker || r.symbol}</Text>
              <Text style={[styles.recAction, {
                color: (r.action || r.signal || "").toUpperCase() === "BUY" ? THEME.profit : THEME.loss
              }]}>
                {(r.action || r.signal || "HOLD").toUpperCase()}
              </Text>
              <Text style={styles.recScore}>
                {r.score != null ? `${(r.score * 100).toFixed(0)}%` : "—"}
              </Text>
            </View>
          ))}
        </View>
      )}

      <Text style={styles.section}>Active Positions</Text>
      {loading && !positions.length ? (
        <LoadingSpinner />
      ) : positions.length > 0 ? (
        positions.map((p, i) => <PositionRow key={p.symbol || p.asset_id || i} item={p} />)
      ) : (
        <OrderHistory orders={[]} />
      )}

      <Text style={styles.section}>Order History</Text>
      {loading && !ordersList.length ? (
        <LoadingSpinner />
      ) : (
        <OrderHistory orders={ordersList} />
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  wrap: { flex: 1, backgroundColor: THEME.bg },
  content: { padding: 16, paddingBottom: 32 },
  section: { color: THEME.text, fontSize: 14, fontWeight: "700", marginTop: 16, marginBottom: 8 },
  recSection: {
    marginBottom: 16,
    padding: 12,
    backgroundColor: THEME.surface,
    borderRadius: 8,
    borderLeftWidth: 3,
    borderLeftColor: THEME.accent,
  },
  recHeading: { color: THEME.text, fontSize: 13, fontWeight: "700", marginBottom: 6 },
  recRow: {
    flexDirection: "row",
    alignItems: "center",
    paddingVertical: 4,
  },
  recTicker: { color: THEME.text, fontWeight: "700", fontSize: 13, flex: 1 },
  recAction: { fontWeight: "800", fontSize: 12, minWidth: 40, textAlign: "center" },

  marketRow: { flexDirection: "row", alignItems: "center", paddingVertical: 3, gap: 8 },
  marketTicker: { color: THEME.text, fontWeight: "700", fontSize: 12, minWidth: 50 },
  marketDir: { fontWeight: "800", fontSize: 11, minWidth: 40 },
  marketSpark: {},

  recScore: { color: THEME.textMuted, fontSize: 12, minWidth: 40, textAlign: "right" },
});
