import React from "react";
import { View, Text, StyleSheet } from "react-native";
import THEME from "../../theme/colors";

export default function StatusBar({ apiStatus, authMode, modelStatus }) {
  const dotColor =
    apiStatus === "connected"
      ? THEME.profit
      : apiStatus === "checking"
        ? THEME.warn
        : THEME.loss;

  return (
    <View style={styles.bar}>
      <View style={styles.left}>
        <View style={[styles.dot, { backgroundColor: dotColor }]} />
        <Text style={styles.text}>
          API: {apiStatus || "checking..."}
        </Text>
        {authMode && authMode !== "disabled" && (
          <Text style={styles.pill}>Auth: {authMode}</Text>
        )}
      </View>
      {modelStatus && (
        <Text style={styles.text}>Model: {modelStatus}</Text>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  bar: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    backgroundColor: THEME.surface,
    paddingHorizontal: 14,
    paddingVertical: 6,
    borderBottomWidth: 1,
    borderBottomColor: THEME.border,
  },
  left: { flexDirection: "row", alignItems: "center", gap: 8 },
  dot: { width: 8, height: 8, borderRadius: 4 },
  text: { color: THEME.textMuted, fontSize: 11 },
  pill: {
    color: THEME.bg,
    backgroundColor: THEME.accent,
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 3,
    fontSize: 10,
    fontWeight: "600",
    overflow: "hidden",
  },
});
