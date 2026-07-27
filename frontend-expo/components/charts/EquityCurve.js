import React from "react";
import { View, Text, StyleSheet } from "react-native";
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import THEME from "../../theme/colors";

export default function EquityCurve({ data }) {
  if (!data || data.length === 0) {
    return (
      <View style={styles.empty}>
        <Text style={styles.emptyText}>No equity data</Text>
      </View>
    );
  }

  const chartData = data.map((d, i) => ({
    i,
    label: d.date || d.time || String(i),
    value: d.equity || d.value || d.total || 0,
  }));

  const isUp = chartData.length >= 2 && chartData[chartData.length - 1].value >= chartData[0].value;

  return (
    <View style={styles.wrap}>
      <Text style={styles.heading}>Equity Curve</Text>
      <View style={styles.chart}>
        <ResponsiveContainer width="100%" height={180}>
          <AreaChart data={chartData}>
            <defs>
              <linearGradient id="eqGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={isUp ? THEME.profit : THEME.loss} stopOpacity={0.3} />
                <stop offset="100%" stopColor={isUp ? THEME.profit : THEME.loss} stopOpacity={0} />
              </linearGradient>
            </defs>
            <XAxis dataKey="label" hide />
            <YAxis hide domain={['dataMin - 1', 'dataMax + 1']} />
            <Tooltip
              contentStyle={{ backgroundColor: THEME.surface, border: `1px solid ${THEME.border}`, borderRadius: 4 }}
              labelStyle={{ color: THEME.textMuted, fontSize: 11 }}
              formatter={(val) => [`$${Number(val).toFixed(2)}`]}
            />
            <Area
              type="monotone"
              dataKey="value"
              stroke={isUp ? THEME.profit : THEME.loss}
              strokeWidth={2}
              fill="url(#eqGrad)"
            />
          </AreaChart>
        </ResponsiveContainer>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: { marginBottom: 16 },
  heading: { color: THEME.text, fontSize: 14, fontWeight: "700", marginBottom: 8 },
  chart: { backgroundColor: THEME.surface, borderRadius: 8, padding: 8 },
  empty: { backgroundColor: THEME.surface, borderRadius: 8, padding: 32, alignItems: "center" },
  emptyText: { color: THEME.textMuted, fontSize: 13 },
});
