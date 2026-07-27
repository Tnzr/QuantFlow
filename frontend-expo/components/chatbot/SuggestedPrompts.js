import React from "react";
import { View, Text, Pressable, StyleSheet } from "react-native";
import THEME from "../../theme/colors";

const PROMPTS_BY_TAB = {
  dashboard: [
    "What's my best performer?",
    "Show risk breakdown",
    "How much buying power do I have?",
  ],
  backtest: [
    "Explain these results",
    "What went wrong with this strategy?",
    "Optimize my parameters",
  ],
  market: [
    "Sentiment analysis on AAPL",
    "Upcoming events affecting NVDA",
    "Show seasonality for MSFT",
  ],
  trade: [
    "Is this a good entry?",
    "Show risk before I confirm",
    "What's the market context?",
  ],
  settings: [
    "Test my Alpaca connection",
    "What model checkpoint should I use?",
    "Explain risk parameters",
  ],
};

const DEFAULT_PROMPTS = [
  "What's my best performer?",
  "Run a backtest on AAPL",
  "Show my portfolio positions",
  "Explain this signal",
  "Analyze MSFT seasonality",
];

export default function SuggestedPrompts({ tab, onSelect }) {
  const prompts = PROMPTS_BY_TAB[tab] || DEFAULT_PROMPTS;

  return (
    <View style={styles.wrap}>
      <Text style={styles.heading}>Suggested</Text>
      {prompts.map((p, i) => (
        <Pressable key={i} style={styles.chip} onPress={() => onSelect(p)}>
          <Text style={styles.chipText}>{p}</Text>
        </Pressable>
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: { paddingHorizontal: 10, paddingVertical: 6 },
  heading: { color: THEME.textMuted, fontSize: 11, fontWeight: "600", marginBottom: 6 },
  chip: {
    backgroundColor: THEME.surfaceLight,
    borderRadius: 6,
    paddingHorizontal: 10,
    paddingVertical: 7,
    marginBottom: 4,
  },
  chipText: { color: THEME.textMuted, fontSize: 12 },
});
