import React from "react";
import { View, Text, Pressable, StyleSheet } from "react-native";
import MetricCard from "../shared/MetricCard";
import EmptyState from "../shared/EmptyState";
import THEME from "../../theme/colors";

export default function BacktestResults({ summary, sigmaBuckets, trades, onRerun }) {
  if (!summary) {
    return (
      <EmptyState
        title="No results"
        message="Configure and run a backtest to see results here."
        actionLabel="Scroll up to configure"
      />
    );
  }

  const kpis = [
    { label: "Avg Return", value: summary.avg_ret != null ? `${(Number(summary.avg_ret) * 100).toFixed(2)}%` : "—" },
    { label: "Sharpe", value: summary.sharpe != null ? Number(summary.sharpe).toFixed(2) : "—" },
    { label: "Max DD", value: summary.max_dd != null ? `${(Number(summary.max_dd) * 100).toFixed(2)}%` : "—" },
    { label: "Win Rate", value: summary.win_rate != null ? `${(Number(summary.win_rate) * 100).toFixed(1)}%` : "—" },
    { label: "Trades", value: summary.n_trades ?? "—" },
    { label: "Profit Factor", value: summary.profit_factor != null ? Number(summary.profit_factor).toFixed(2) : "—" },
    { label: "Sortino", value: summary.sortino != null ? Number(summary.sortino).toFixed(2) : "—" },
    { label: "CAGR", value: summary.cagr != null ? `${(Number(summary.cagr) * 100).toFixed(2)}%` : "—" },
  ];

  const buyHoldReturn = (trades && trades.length > 0 && trades[0]?.entry_price > 0)
    ? ((trades[trades.length - 1]?.exit_price - trades[0]?.entry_price) / trades[0]?.entry_price)
    : null;

  const handleExport = () => {
    if (!trades || trades.length === 0) return;
    const headers = "ticker,entry_date,exit_date,entry_price,exit_price,return,pnl\n";
    const rows = trades
      .map(
        (t) =>
          `${t.symbol || t.ticker || ""},${t.entry_date || t.entry_time || ""},${t.exit_date || t.exit_time || ""},${t.entry_price || ""},${t.exit_price || ""},${t.return_pct || t.return_ || ""},${t.pnl || ""}`
      )
      .join("\n");

    const blob = new Blob([headers + rows], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "backtest_results.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <View style={styles.wrap}>
      <View style={styles.headerRow}>
        <Text style={styles.heading}>Results</Text>
        <View style={styles.actions}>
          {trades && trades.length > 0 && (
            <Pressable style={styles.exportBtn} onPress={handleExport}>
              <Text style={styles.exportText}>Export CSV</Text>
            </Pressable>
          )}
          {onRerun && (
            <Pressable style={styles.rerunBtn} onPress={onRerun}>
              <Text style={styles.rerunText}>Re-run</Text>
            </Pressable>
          )}
        </View>
      </View>

      <View style={styles.kpiGrid}>
        {kpis.map((k) => (
          <View key={k.label} style={styles.kpiWrap}>
            <MetricCard label={k.label} value={k.value} />
          </View>
        ))}
      </View>

      {sigmaBuckets && (Array.isArray(sigmaBuckets) ? sigmaBuckets : Object.entries(sigmaBuckets || {})).length > 0 && (
        <View style={styles.section}>
          <Text style={styles.subheading}>Sigma Buckets</Text>
          <View style={styles.table}>
            <View style={styles.tableHeader}>
              <Text style={[styles.th, styles.colName]}>Bucket</Text>
              <Text style={[styles.th, styles.colNum]}>Trades</Text>
              <Text style={[styles.th, styles.colNum]}>Win Rate</Text>
              <Text style={[styles.th, styles.colNum]}>Avg Return</Text>
            </View>
            {(Array.isArray(sigmaBuckets) ? sigmaBuckets : Object.entries(sigmaBuckets || {})).map(
              (item, i) => {
                const name = item.name || item[0] || `Bucket ${i}`;
                const data = item[1] || item;
                const trades = data.trades ?? data.count ?? "—";
                const winRate = data.win_rate != null ? Number(data.win_rate).toFixed(1) : "—";
                const avgReturn = data.avg_return != null ? Number(data.avg_return).toFixed(2) : "—";
                return (
                  <View key={i} style={styles.tableRow}>
                    <Text style={[styles.td, styles.colName]}>{name}</Text>
                    <Text style={[styles.td, styles.colNum]}>{trades}</Text>
                    <Text style={[styles.td, styles.colNum]}>{winRate}%</Text>
                    <Text style={[styles.td, styles.colNum]}>{avgReturn}%</Text>
                  </View>
                );
              }
            )}
          </View>
        </View>
      )}

      {trades && trades.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.subheading}>Trade Log ({trades.length})</Text>
          <View style={styles.table}>
            <View style={styles.tableHeader}>
              <Text style={[styles.th, styles.colName]}>Ticker</Text>
              <Text style={[styles.th, styles.colNum]}>Entry</Text>
              <Text style={[styles.th, styles.colNum]}>Exit</Text>
              <Text style={[styles.th, styles.colNum]}>P&L</Text>
            </View>
            {trades.slice(0, 50).map((t, i) => {
              const pnl = (t.pnl != null) ? Number(t.pnl) : ((t.return_pct != null && t.entry_price != null) ? Number(t.entry_price) * Number(t.return_pct) : 0);
              const isWin = pnl >= 0;
              return (
                <View key={i} style={styles.tableRow}>
                  <Text style={[styles.td, styles.colName]}>{t.symbol || t.ticker || "—"}</Text>
                  <Text style={[styles.td, styles.colNum]}>
                    {t.entry_price != null ? `$${Number(t.entry_price).toFixed(2)}` : "—"}
                  </Text>
                  <Text style={[styles.td, styles.colNum]}>
                    {t.exit_price != null ? `$${Number(t.exit_price).toFixed(2)}` : "—"}
                  </Text>
                  <Text style={[styles.td, styles.colNum, { color: isWin ? THEME.profit : THEME.loss }]}>
                    {isWin ? "+" : ""}${pnl.toFixed(2)} ({t.return_pct != null ? `${(Number(t.return_pct) * 100).toFixed(1)}%` : "—"})
                  </Text>
                </View>
              );
            })}
            {trades.length > 50 && (
              <Text style={styles.more}>+ {trades.length - 50} more trades</Text>
            )}
          </View>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: { marginBottom: 16 },
  headerRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 10,
  },
  heading: { color: THEME.text, fontSize: 14, fontWeight: "700" },
  actions: { flexDirection: "row", gap: 8 },
  exportBtn: {
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 4,
    backgroundColor: THEME.surface,
    borderWidth: 1,
    borderColor: THEME.border,
  },
  exportText: { color: THEME.textMuted, fontSize: 11 },
  rerunBtn: {
    paddingHorizontal: 12,
    paddingVertical: 5,
    borderRadius: 4,
    backgroundColor: THEME.accent,
  },
  rerunText: { color: THEME.textBright, fontWeight: "600", fontSize: 11 },
  kpiGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 8,
    marginBottom: 16,
  },
  kpiWrap: { width: "48%", flexGrow: 1 },
  section: { marginBottom: 16 },
  subheading: { color: THEME.textMuted, fontSize: 12, fontWeight: "600", marginBottom: 6, textTransform: "uppercase" },
  table: {
    backgroundColor: THEME.surface,
    borderRadius: 6,
    overflow: "hidden",
  },
  tableHeader: {
    flexDirection: "row",
    backgroundColor: THEME.surfaceLight,
    paddingHorizontal: 10,
    paddingVertical: 6,
  },
  tableRow: {
    flexDirection: "row",
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderTopWidth: 1,
    borderTopColor: THEME.border,
  },
  th: { color: THEME.textMuted, fontSize: 11, fontWeight: "700" },
  td: { color: THEME.text, fontSize: 12 },
  colName: { flex: 2 },
  colNum: { flex: 1, textAlign: "right", fontVariant: ["tabular-nums"] },
  more: { color: THEME.textMuted, fontSize: 11, textAlign: "center", paddingVertical: 6 },
  compareBar: { flexDirection: "row", alignItems: "center", gap: 8, padding: 8, backgroundColor: THEME.surfaceLight, borderRadius: 6, marginBottom: 10 },
  compareLabel: { color: THEME.textMuted, fontSize: 11 },
  compareVal: { fontSize: 12, fontWeight: "700", fontVariant: ["tabular-nums"] },
  compareWinner: { fontSize: 10, fontWeight: "600", marginLeft: "auto" },
});
