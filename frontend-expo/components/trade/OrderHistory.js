import React from "react";
import { View, Text, FlatList, StyleSheet } from "react-native";
import THEME from "../../theme/colors";
import EmptyState from "../shared/EmptyState";

const SIDE_COLORS = { buy: THEME.profit, sell: THEME.loss };

export default function OrderHistory({ orders }) {
  if (!orders || orders.length === 0) {
    return (
      <EmptyState
        title="No orders"
        message="Paper trades will appear here after placement."
      />
    );
  }

  return (
    <View style={styles.wrap}>
      <Text style={styles.heading}>Order History ({orders.length})</Text>
      <FlatList
        data={orders}
        keyExtractor={(item, i) => item.id || item.order_id || String(i)}
        renderItem={({ item }) => {
          const side = (item.side || item.action || "").toLowerCase();
          const sideColor = SIDE_COLORS[side] || THEME.textMuted;
          return (
            <View style={styles.row}>
              <View style={styles.left}>
                <Text style={[styles.side, { color: sideColor }]}>
                  {side.toUpperCase()}
                </Text>
                <Text style={styles.ticker}>{item.symbol || item.ticker}</Text>
              </View>
              <Text style={styles.qty}>{item.qty || item.filled_qty || "—"}</Text>
              <Text style={styles.status}>{item.status || "—"}</Text>
              <Text style={styles.price}>
                {item.filled_avg_price ? `$${Number(item.filled_avg_price).toFixed(2)}` : "—"}
              </Text>
            </View>
          );
        }}
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
    alignItems: "center",
    backgroundColor: THEME.surface,
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 6,
    marginBottom: 3,
  },
  left: { flexDirection: "row", alignItems: "center", gap: 8, flex: 2 },
  side: { fontWeight: "800", fontSize: 12, minWidth: 36 },
  ticker: { color: THEME.text, fontWeight: "600", fontSize: 13 },
  qty: { color: THEME.textMuted, fontSize: 12, flex: 1, textAlign: "center", fontVariant: ["tabular-nums"] },
  status: { color: THEME.textMuted, fontSize: 11, flex: 1, textAlign: "center" },
  price: { color: THEME.text, fontSize: 12, flex: 1, textAlign: "right", fontVariant: ["tabular-nums"] },
});
