import React from "react";
import { View, Text, StyleSheet } from "react-native";
import { LineChart, Line, XAxis, YAxis, Tooltip, ReferenceLine, ResponsiveContainer } from "recharts";
import THEME from "../../theme/colors";
import { formatChartDate } from "../../utils/dateFormat";

export default function RsiChart({ data }) {
  if (!data || data.length === 0) {
    return null;
  }

  const items = Array.isArray(data) ? data : (data?.items || []);
  const chartData = items
    .filter((d) => d.rsi14 != null)
    .map((v, i) => ({
      idx: i,
      label: v.date || v.time || String(i),
      rsi: Number(v.rsi14),
    }));

  if (chartData.length === 0) {
    return (
      <View style={styles.empty}>
        <Text style={styles.emptyText}>No RSI data</Text>
      </View>
    );
  }

  const formatDate = (d) => formatChartDate(d, "short");

  const step = Math.max(1, Math.floor(chartData.length / 5));

  return (
    <View style={styles.wrap}>
      <Text style={styles.heading}>RSI (14)</Text>
      <View style={styles.chart}>
        <ResponsiveContainer width="100%" height={140}>
          <LineChart data={chartData} margin={{ top: 5, right: 5, bottom: 20, left: 0 }}>
            <XAxis
              dataKey="idx"
              tickFormatter={(i) => (i % step === 0 ? formatDate(chartData[i]?.label) : "")}
              tick={{ fill: THEME.textMuted, fontSize: 10 }}
              axisLine={{ stroke: THEME.border }}
              tickLine={false}
            />
            <YAxis domain={[0, 100]} tick={{ fill: THEME.textMuted, fontSize: 10 }} width={35} axisLine={{ stroke: THEME.border }} />
            <Tooltip
              contentStyle={{ backgroundColor: THEME.surface, border: `1px solid ${THEME.border}`, borderRadius: 4 }}
              labelStyle={{ color: THEME.textMuted, fontSize: 11 }}
              formatter={(val) => [Number(val).toFixed(1)]}
            />
            <ReferenceLine y={70} stroke={THEME.loss} strokeDasharray="4 4" strokeOpacity={0.5} />
            <ReferenceLine y={30} stroke={THEME.profit} strokeDasharray="4 4" strokeOpacity={0.5} />
            <Line
              type="monotone"
              dataKey="rsi"
              stroke={THEME.accent}
              strokeWidth={2}
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: { marginBottom: 4 },
  heading: { color: THEME.text, fontSize: 13, fontWeight: "700", marginBottom: 4 },
  chart: { backgroundColor: THEME.surface, borderRadius: 8, padding: 4 },
  empty: { backgroundColor: THEME.surface, borderRadius: 8, padding: 32, alignItems: "center" },
  emptyText: { color: THEME.textMuted, fontSize: 13 },
});
