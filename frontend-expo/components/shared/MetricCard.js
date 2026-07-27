import React, { useEffect, useRef } from "react";
import { View, Text, Animated, StyleSheet } from "react-native";
import THEME from "../../theme/colors";

export default function MetricCard({
  label,
  value,
  prefix,
  suffix,
  color,
  large,
  subtitle,
}) {
  const anim = useRef(new Animated.Value(0)).current;
  const displayColor = color || THEME.text;

  useEffect(() => {
    Animated.timing(anim, {
      toValue: 1,
      duration: 800,
      useNativeDriver: true,
    }).start();
  }, [anim, value]);

  const isPositive = typeof value === "number" && value > 0;
  const isNegative = typeof value === "number" && value < 0;
  const valColor =
    color || (isPositive ? THEME.profit : isNegative ? THEME.loss : THEME.text);

  const formatVal = (v) => {
    if (typeof v === "number") {
      return v % 1 === 0 ? v.toLocaleString() : v.toFixed(2);
    }
    if (typeof v === "object" && v !== null) {
      return JSON.stringify(v);
    }
    return String(v || "");
  };

  return (
    <Animated.View style={[styles.card, { opacity: anim, transform: [{ translateY: anim.interpolate({ inputRange: [0, 1], outputRange: [12, 0] }) }] }]}>
      <Text style={styles.label}>{label}</Text>
      <Text style={[styles.value, { color: valColor }, large && styles.valueLarge]}>
        {prefix || ""}
        {formatVal(value)}
        {suffix || ""}
      </Text>
      {subtitle && <Text style={styles.sub}>{subtitle}</Text>}
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: THEME.surface,
    borderRadius: 8,
    padding: 14,
    minWidth: 100,
    flex: 1,
  },
  label: { color: THEME.textMuted, fontSize: 11, fontWeight: "600", textTransform: "uppercase", marginBottom: 4 },
  value: { fontSize: 20, fontWeight: "700", fontVariant: ["tabular-nums"] },
  valueLarge: { fontSize: 28 },
  sub: { color: THEME.neutral, fontSize: 11, marginTop: 2 },
});
