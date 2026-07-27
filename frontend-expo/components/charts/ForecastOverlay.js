import React from "react";
import { View, Text, StyleSheet } from "react-native";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Area, ComposedChart } from "recharts";
import THEME from "../../theme/colors";

export default function ForecastOverlay({ data, visible }) {
  if (!visible || !data) return null;

  const forecasts = data.forecast || data.predictions || data.path || data || [];
  const list = Array.isArray(forecasts) ? forecasts : [];

  if (list.length === 0) return null;

  const chartData = list.map((v, i) => ({
    idx: i,
    label: v.date || `+${i + 1}`,
    pred: typeof v === "number" ? v : (v.price || v.mean || v.value || v.prediction || 0),
    upper: typeof v === "object" ? (v.upper || v.high) : null,
    lower: typeof v === "object" ? (v.lower || v.low) : null,
  }));

  return (
    <View style={styles.wrap}>
      <Text style={styles.heading}>Forecast ({list.length} steps)</Text>
      <View style={styles.chart}>
        <ResponsiveContainer width="100%" height={120}>
          <ComposedChart data={chartData}>
            <XAxis dataKey="label" hide />
            <YAxis hide domain={["auto", "auto"]} />
            <Tooltip
              contentStyle={{ backgroundColor: THEME.surface, border: `1px solid ${THEME.border}`, borderRadius: 4 }}
              labelStyle={{ color: THEME.textMuted, fontSize: 11 }}
              formatter={(val) => [val.toFixed(4)]}
            />
            {chartData[0]?.upper != null && (
              <>
                <Area
                  type="monotone"
                  dataKey="upper"
                  stroke="none"
                  fill={THEME.accentGlow}
                  fillOpacity={0.2}
                />
                <Area
                  type="monotone"
                  dataKey="lower"
                  stroke="none"
                  fill={THEME.bg}
                  fillOpacity={0}
                />
              </>
            )}
            <Line
              type="monotone"
              dataKey="pred"
              stroke={THEME.accent}
              strokeWidth={2}
              strokeDasharray="5 5"
              dot={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: { marginBottom: 12 },
  heading: { color: THEME.text, fontSize: 13, fontWeight: "700", marginBottom: 6 },
  chart: { backgroundColor: THEME.surface, borderRadius: 8, padding: 4 },
});
