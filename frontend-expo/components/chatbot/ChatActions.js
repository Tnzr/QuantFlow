import React from "react";
import { View, Text, Pressable, StyleSheet } from "react-native";
import THEME from "../../theme/colors";

export default function ChatActions({ actions, onExecute }) {
  if (!actions || actions.length === 0) return null;

  return (
    <View style={styles.wrap}>
      {actions.map((a, i) => (
        <Pressable key={i} style={styles.action} onPress={() => onExecute?.(a)}>
          <Text style={styles.label}>{a.label || a.name || a.tool || `Action ${i + 1}`}</Text>
          {a.description && <Text style={styles.desc}>{a.description}</Text>}
          {a.params && <Text style={styles.params}>{JSON.stringify(a.params)}</Text>}
        </Pressable>
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: { paddingHorizontal: 10, marginTop: 6 },
  action: { backgroundColor: THEME.accentGlow, borderWidth: 1, borderColor: THEME.accent, borderRadius: 6, paddingHorizontal: 10, paddingVertical: 7, marginBottom: 4 },
  label: { color: THEME.accent, fontWeight: "600", fontSize: 12 },
  desc: { color: THEME.textMuted, fontSize: 11, marginTop: 2 },
  params: { color: THEME.neutral, fontSize: 10, fontFamily: "monospace", marginTop: 2 },
});
