import React, { useRef, useEffect } from "react";
import { View, TextInput, Pressable, Text, StyleSheet } from "react-native";
import THEME from "../../theme/colors";

export default function ChatInput({ value, onChange, onSend, sending, autoFocus }) {
  const inputRef = useRef(null);

  useEffect(() => {
    if (autoFocus && inputRef.current) {
      setTimeout(() => inputRef.current?.focus(), 200);
    }
  }, [autoFocus]);

  return (
    <View style={styles.wrap}>
      <TextInput
        ref={inputRef}
        style={styles.input}
        placeholder="Ask QuantFlow..."
        placeholderTextColor={THEME.neutral}
        value={value}
        onChangeText={onChange}
        multiline
        maxLength={2000}
        returnKeyType="send"
        blurOnSubmit
        onSubmitEditing={() => {
          if (value.trim() && !sending) onSend();
        }}
        accessibilityLabel="Chat message"
      />
      <Pressable
        style={[styles.btn, (!value.trim() || sending) && styles.btnDisabled]}
        onPress={onSend}
        disabled={!value.trim() || sending}
      >
        <Text style={styles.btnText}>{sending ? "..." : "->"}</Text>
      </Pressable>
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    flexDirection: "row",
    alignItems: "flex-end",
    paddingHorizontal: 10,
    paddingVertical: 8,
    borderTopWidth: 1,
    borderTopColor: THEME.border,
    gap: 6,
  },
  input: {
    flex: 1,
    backgroundColor: THEME.surfaceLight,
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 8,
    color: THEME.text,
    fontSize: 13,
    maxHeight: 100,
  },
  btn: {
    width: 38,
    height: 38,
    borderRadius: 19,
    backgroundColor: THEME.accent,
    justifyContent: "center",
    alignItems: "center",
  },
  btnDisabled: { opacity: 0.4 },
  btnText: { color: THEME.textBright, fontWeight: "800", fontSize: 16 },
});
