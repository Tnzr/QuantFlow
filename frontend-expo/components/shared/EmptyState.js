import React from "react";
import { View, Text, Pressable, StyleSheet } from "react-native";
import THEME from "../../theme/colors";

export default function EmptyState({ title, message, actionLabel, onAction }) {
  return (
    <View style={styles.wrap}>
      <Text style={styles.title}>{title || "Nothing here yet"}</Text>
      {message && <Text style={styles.msg}>{message}</Text>}
      {actionLabel && onAction && (
        <Pressable style={styles.btn} onPress={onAction}>
          <Text style={styles.btnText}>{actionLabel}</Text>
        </Pressable>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    paddingHorizontal: 32,
    paddingVertical: 48,
  },
  title: { color: THEME.textMuted, fontSize: 16, fontWeight: "600", marginBottom: 8 },
  msg: { color: THEME.neutral, fontSize: 13, textAlign: "center", marginBottom: 16 },
  btn: {
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 6,
    backgroundColor: THEME.accent,
  },
  btnText: { color: THEME.textBright, fontWeight: "600", fontSize: 14 },
});
