import React from "react";
import { View, Text, StyleSheet } from "react-native";
import { LineChart, Line, ResponsiveContainer } from "recharts";
import THEME from "../../theme/colors";

export default function Sparkline({ data, width = 80, height = 30 }) {
  if (!data || data.length < 2) {
    return <View style={[styles.wrap, { width, height }]} />;
  }

  const values = data.map((v, i) => ({ i, v: typeof v === "number" ? v : v.close || v.value || 0 }));
  const first = values[0]?.v || 0;
  const last = values[values.length - 1]?.v || 0;
  const isUp = last >= first;

  return (
    <View style={[styles.wrap, { width, height }]}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={values}>
          <Line
            type="monotone"
            dataKey="v"
            stroke={isUp ? THEME.profit : THEME.loss}
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {},
});
