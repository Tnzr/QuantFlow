import React from "react";
import { View, Text, StyleSheet } from "react-native";
import THEME from "../../theme/colors";

export default function ChatBubble({ message }) {
  const isUser = message.role === "user";
  const isError = message.error;

  return (
    <View style={[styles.wrap, isUser ? styles.userWrap : styles.assistantWrap]}>
      <View
        style={[
          styles.bubble,
          isUser ? styles.userBubble : styles.assistantBubble,
          isError && styles.errorBubble,
        ]}
      >
        <Text
          style={[
            styles.text,
            isUser ? styles.userText : styles.assistantText,
            isError && styles.errorText,
          ]}
        >
          {message.text}
        </Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: { marginBottom: 10, paddingHorizontal: 4 },
  userWrap: { alignItems: "flex-end" },
  assistantWrap: { alignItems: "flex-start" },
  bubble: {
    maxWidth: "85%",
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 10,
  },
  userBubble: { backgroundColor: THEME.accent },
  assistantBubble: { backgroundColor: THEME.surfaceLight },
  errorBubble: { backgroundColor: THEME.lossDim, borderWidth: 1, borderColor: THEME.loss },
  text: { fontSize: 13, lineHeight: 18 },
  userText: { color: THEME.textBright },
  assistantText: { color: THEME.text },
  errorText: { color: THEME.loss },
});
