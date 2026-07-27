import React from "react";
import { View, Text, StyleSheet } from "react-native";
import THEME from "../../theme/colors";

export default function PositionRow({ item }) {
  const pnl = item.unrealized_pl || item.unrealized_intraday_pl || 0;
  const pnlPct = item.unrealized_plpc || item.unrealized_intraday_plpc || 0;
  const isPositive = pnl >= 0;
  const currentPrice =
    item.current_price != null
      ? item.current_price
      : (item.market_value != null && item.qty ? item.market_value / item.qty : null);
  const avgPrice =
    item.avg_entry_price != null
      ? item.avg_entry_price
      : (item.cost_basis != null && item.qty ? item.cost_basis / item.qty : null);

  return (
    <View style={styles.row}>
      <View style={styles.left}>
        <Text style={styles.ticker}>{item.symbol || item.asset_id}</Text>
        <Text style={styles.detail}>
          {item.qty || item.shares} sh @ ${avgPrice ? avgPrice.toFixed(2) : "—"}
        </Text>
      </View>
      <View style={styles.center}>
        {currentPrice && <Text style={styles.price}>${currentPrice.toFixed(2)}</Text>}
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

const styles = StyleSheet.create({
  row: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: THEME.surface,
    paddingHorizontal: 12,
    paddingVertical: 10,
    borderRadius: 6,
    marginBottom: 4,
  },
  left: { flex: 2 },
  center: { flex: 1, alignItems: "center" },
  right: { flex: 1, alignItems: "flex-end" },
  ticker: { color: THEME.text, fontWeight: "700", fontSize: 14 },
  detail: { color: THEME.textMuted, fontSize: 11 },
  price: { color: THEME.textMuted, fontSize: 12, fontVariant: ["tabular-nums"] },
  pnl: { fontWeight: "700", fontSize: 13, fontVariant: ["tabular-nums"] },
  pnlPct: { fontSize: 11, fontVariant: ["tabular-nums"] },
});
