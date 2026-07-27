import React from "react";
import { View, Text, StyleSheet } from "react-native";
import MetricCard from "../shared/MetricCard";
import THEME from "../../theme/colors";

export default function AccountSummary({ account }) {
  if (!account) {
    return (
      <View style={styles.wrap}>
        <Text style={styles.na}>No account data</Text>
      </View>
    );
  }

  const equity = account.equity || account.portfolio_value || 0;
  const cash = account.cash || account.buying_power || 0;
  const buyingPower = account.buying_power || account.multiplier * (account.cash || 0) || 0;
  const dayPnL = account.day_pnl || account.unrealized_intraday_pl || 0;
  const dayPnLPct = account.day_pnl_pct || (equity ? (dayPnL / (equity - dayPnL)) * 100 : 0);

  return (
    <View style={styles.wrap}>
      <Text style={styles.heading}>Account</Text>
      <View style={styles.row}>
        <MetricCard label="Equity" value={equity} prefix="$" large />
        <MetricCard label="Cash" value={cash} prefix="$" />
      </View>
      <View style={styles.row}>
        <MetricCard label="Buying Power" value={buyingPower} prefix="$" />
        <MetricCard
          label="Day P&L"
          value={dayPnL}
          prefix="$"
          suffix={dayPnLPct ? ` (${dayPnLPct.toFixed(2)}%)` : ""}
        />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: { marginBottom: 16 },
  heading: { color: THEME.text, fontSize: 14, fontWeight: "700", marginBottom: 8 },
  row: { flexDirection: "row", gap: 8, marginBottom: 8 },
  na: { color: THEME.textMuted, fontSize: 13, textAlign: "center", paddingVertical: 16 },
});
