import React from "react";
import { View, Text, StyleSheet } from "react-native";
import THEME from "../../theme/colors";

export default function ScreenTitle({ title, subtitle, icon }) {
  return (
    <View style={styles.wrap}>
      <View style={styles.row}>
        {icon && <Text style={styles.icon}>{icon}</Text>}
        <Text style={styles.title}>{title}</Text>
      </View>
      {subtitle && <Text style={styles.subtitle}>{subtitle}</Text>}
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    paddingHorizontal: 16,
    paddingTop: 14,
    paddingBottom: 10,
    borderBottomWidth: 1,
    borderBottomColor: THEME.border,
    backgroundColor: THEME.bg,
  },
  row: { flexDirection: "row", alignItems: "center", gap: 8 },
  icon: {
    fontSize: 20,
    fontWeight: "800",
    color: THEME.accent,
    backgroundColor: THEME.accentGlow,
    width: 32,
    height: 32,
    borderRadius: 6,
    textAlign: "center",
    lineHeight: 32,
  },
  title: { color: THEME.text, fontSize: 20, fontWeight: "800" },
  subtitle: { color: THEME.textMuted, fontSize: 11, marginTop: 4, marginLeft: 40 },
});
