import React from "react";
import { View, Text, StyleSheet } from "react-native";
import THEME from "../../theme/colors";
import EmptyState from "../shared/EmptyState";

const SIGNAL_COLORS = {
  BUY: THEME.profit,
  SELL: THEME.loss,
  HOLD: THEME.neutral,
};

export default function SignalCard({ signals }) {
  if (!signals || signals.length === 0) {
    return (
      <EmptyState
        title="No signals"
        message="AI signals will appear here when available."
      />
    );
  }

  return (
    <View style={styles.wrap}>
      <Text style={styles.heading}>AI Signals</Text>
      {signals.map((s, i) => {
        const action = (s.action || s.signal || "HOLD").toUpperCase();
        const color = SIGNAL_COLORS[action] || THEME.neutral;
        const ticker = s.ticker || s.symbol || "???";
        const confidence = s.confidence != null ? `${(s.confidence * 100).toFixed(0)}%` : "—";
        return (
          <View key={s.ticker || i} style={styles.card}>
            <View style={[styles.badge, { backgroundColor: color }]}>
              <Text style={styles.badgeText}>{action}</Text>
            </View>
            <Text style={styles.ticker}>{ticker}</Text>
            <Text style={styles.conf}>{confidence}</Text>
          </View>
        );
      })}
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: { marginBottom: 16 },
  heading: { color: THEME.text, fontSize: 14, fontWeight: "700", marginBottom: 8 },
  card: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: THEME.surface,
    paddingHorizontal: 12,
    paddingVertical: 10,
    borderRadius: 6,
    marginBottom: 4,
    gap: 10,
  },
  badge: {
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 4,
  },
  badgeText: { color: THEME.bg, fontWeight: "800", fontSize: 11 },
  ticker: { color: THEME.text, fontWeight: "700", fontSize: 14, flex: 1 },
  conf: { color: THEME.textMuted, fontSize: 12 },
});
