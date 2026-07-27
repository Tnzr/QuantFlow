import React from "react";
import { View, Text, FlatList, StyleSheet } from "react-native";
import THEME from "../../theme/colors";
import EmptyState from "../shared/EmptyState";

function PositionRow({ item }) {
  const pnl = item.unrealizedPnl || item.unrealized_pl || item.unrealized_intraday_pl || 0;
  const pnlPct = item.unrealizedPnlPct || item.unrealized_plpc || item.unrealized_intraday_plpc || 0;
  const isPositive = pnl >= 0;

  return (
    <View style={styles.row}>
      <View style={styles.left}>
        <Text style={styles.ticker}>{item.symbol || item.asset_id}</Text>
        <Text style={styles.shares}>{item.qty || item.shares} shares</Text>
      </View>
      <View style={styles.right}>
        <Text style={[styles.pnl, { color: isPositive ? THEME.profit : THEME.loss }]}>
          {isPositive ? "+" : ""}${pnl.toFixed(2)}
        </Text>
        <Text style={[styles.pnlPct, { color: isPositive ? THEME.profit : THEME.loss }]}>
          ({isPositive ? "+" : ""}{pnlPct.toFixed(2)}%)
        </Text>
      </View>
    </View>
  );
}

export default function PositionList({ positions }) {
  if (!positions || positions.length === 0) {
    return (
      <EmptyState
        title="No positions"
        message="Place a paper trade to get started."
      />
    );
  }

  return (
    <View style={styles.wrap}>
      <Text style={styles.heading}>Positions ({positions.length})</Text>
      <FlatList
        data={positions}
        keyExtractor={(item, i) => item.symbol || item.asset_id || String(i)}
        renderItem={({ item }) => <PositionRow item={item} />}
        scrollEnabled={false}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: { marginBottom: 16 },
  heading: { color: THEME.text, fontSize: 14, fontWeight: "700", marginBottom: 8 },
  row: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    backgroundColor: THEME.surface,
    paddingHorizontal: 12,
    paddingVertical: 10,
    borderRadius: 6,
    marginBottom: 4,
  },
  left: {},
  right: { alignItems: "flex-end" },
  ticker: { color: THEME.text, fontWeight: "700", fontSize: 14 },
  shares: { color: THEME.textMuted, fontSize: 12 },
  pnl: { fontWeight: "700", fontSize: 14, fontVariant: ["tabular-nums"] },
  pnlPct: { fontSize: 11, fontVariant: ["tabular-nums"] },
});
