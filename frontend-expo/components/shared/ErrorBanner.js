import React from "react";
import { View, Text, Pressable, StyleSheet } from "react-native";
import THEME from "../../theme/colors";

export default function ErrorBanner({ message, onRetry }) {
  if (!message) return null;

  return (
    <View style={styles.banner}>
      <Text style={styles.icon}>!</Text>
      <Text style={styles.text} numberOfLines={3}>
        {message}
      </Text>
      {onRetry && (
        <Pressable style={styles.btn} onPress={onRetry}>
          <Text style={styles.btnText}>Retry</Text>
        </Pressable>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  banner: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: THEME.lossDim,
    borderLeftWidth: 3,
    borderLeftColor: THEME.loss,
    paddingHorizontal: 12,
    paddingVertical: 10,
    borderRadius: 6,
    marginBottom: 12,
  },
  icon: {
    color: THEME.loss,
    fontWeight: "700",
    fontSize: 16,
    marginRight: 8,
  },
  text: {
    color: THEME.loss,
    fontSize: 13,
    flex: 1,
  },
  btn: {
    marginLeft: 10,
    paddingHorizontal: 12,
    paddingVertical: 5,
    borderRadius: 4,
    backgroundColor: THEME.loss,
  },
  btnText: { color: THEME.bg, fontWeight: "600", fontSize: 12 },
});
