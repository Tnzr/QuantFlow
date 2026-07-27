import React from "react";
import { View, Text, StyleSheet } from "react-native";
import THEME from "../../theme/colors";

export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, info) {
    console.error("ErrorBoundary caught:", error, info);
  }

  render() {
    if (this.state.hasError) {
      return (
        <View style={styles.wrap}>
          <Text style={styles.title}>Something went wrong</Text>
          <Text style={styles.message}>{this.state.error?.message || "Unknown error"}</Text>
        </View>
      );
    }
    return this.props.children;
  }
}

const styles = StyleSheet.create({
  wrap: { padding: 20, backgroundColor: THEME.surface, borderRadius: 8, marginBottom: 16 },
  title: { color: THEME.loss, fontSize: 14, fontWeight: "700", marginBottom: 8 },
  message: { color: THEME.textMuted, fontSize: 12 },
});
