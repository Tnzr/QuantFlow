import React from "react";
import { View, Text, StyleSheet } from "react-native";
import MetricCard from "../shared/MetricCard";
import THEME from "../../theme/colors";
import { getTimezoneAbbr } from "../../utils/dateFormat";

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
      <View style={styles.headerRow}>
        <Text style={styles.heading}>Account</Text>
        <Text style={styles.tz}>Times in {getTimezoneAbbr()}</Text>
      </View>
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
  headerRow: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginBottom: 8 },
  heading: { color: THEME.text, fontSize: 14, fontWeight: "700" },
  tz: { color: THEME.textMuted, fontSize: 9, fontStyle: "italic" },
  row: { flexDirection: "row", gap: 8, marginBottom: 8 },
  na: { color: THEME.textMuted, fontSize: 13, textAlign: "center", paddingVertical: 16 },
});
