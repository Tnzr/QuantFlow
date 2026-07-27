import React, { useEffect, useState, useCallback } from "react";
import { View, Text, Pressable, ScrollView, StyleSheet } from "react-native";
import THEME from "../../theme/colors";
import useAssistant from "../../hooks/useAssistant";
import useWatchlist from "../../hooks/useWatchlist";
import { apiPost } from "../../services/api";
import ChatBubble from "./ChatBubble";
import ChatInput from "./ChatInput";
import SuggestedPrompts from "./SuggestedPrompts";
import ChatActions from "./ChatActions";
import LoadingSpinner from "../shared/LoadingSpinner";
import ScreenTitle from "../shared/ScreenTitle";

export default function ChatSidebar({ token, onClose, activeTab }) {
  const { messages, sending, error, send, clear, loadHistory } = useAssistant(token);
  const { addTicker } = useWatchlist();
  const [input, setInput] = useState("");
  const [actionResult, setActionResult] = useState(null);

  useEffect(() => {
    loadHistory();
  }, [loadHistory]);

  const executeAction = useCallback(
    async (action) => {
      const tool = (action.tool || action.name || "").toLowerCase();
      const params = action.params || {};
      try {
        let resultText = "";
        if (tool.includes("watchlist") && tool.includes("add")) {
          const sym = (params.ticker || action.ticker || "").toString().trim().toUpperCase();
          if (sym) {
            addTicker(sym);
            resultText = `Added ${sym} to watchlist.`;
          } else {
            resultText = "Could not determine ticker to add.";
          }
        } else if (tool.includes("backtest") || tool.includes("run")) {
          const ticker = params.ticker || action.ticker || "";
          resultText = `Started backtest${ticker ? ` for ${ticker}` : ""}.`;
        } else if (tool.includes("trade") || tool.includes("buy") || tool.includes("sell")) {
          const ticker = (params.ticker || "").toString().toUpperCase();
          const side = params.side || (tool.includes("sell") ? "sell" : "buy");
          const qty = parseFloat(params.qty) || 1;
          if (!ticker) {
            resultText = "Ticker required for trade.";
          } else {
            try {
              const res = await apiPost("/trade/paper", { ticker, side, qty, order_type: "market" }, token);
              resultText = `Order placed: ${ticker} ${side} x${qty} (${res.status || res.order?.status || "submitted"})`;
            } catch (e) {
              resultText = `Trade failed: ${e.message}`;
            }
          }
        } else {
          resultText = `Executed ${action.label || tool || "action"}.`;
        }
        setActionResult({ ts: Date.now(), text: resultText });
      } catch (e) {
        setActionResult({ ts: Date.now(), text: `Action failed: ${e.message}` });
      }
    },
    [addTicker, token]
  );

  const handleSend = () => {
    if (!input.trim() || sending) return;
    send(input, { tab: activeTab });
    setInput("");
  };

  const handlePrompt = (prompt) => {
    send(prompt, { tab: activeTab });
  };

  return (
    <View style={styles.wrap}>
      <View style={styles.header}>
        <Text style={styles.title}>QuantFlow Assistant</Text>
        <View style={styles.headerActions}>
          <Pressable onPress={clear}>
            <Text style={styles.clearBtn}>Clear</Text>
          </Pressable>
          <Pressable onPress={onClose}>
            <Text style={styles.closeBtn}>X</Text>
          </Pressable>
        </View>
      </View>

      <ScrollView style={styles.msgs} contentContainerStyle={styles.msgsContent}>
        {messages.length === 0 && !sending && (
          <SuggestedPrompts tab={activeTab} onSelect={handlePrompt} />
        )}
        {messages.map((m, i) => (
          <View key={i}>
            <ChatBubble message={m} />
            {m.actions && m.actions.length > 0 && (
              <ChatActions actions={m.actions} onExecute={executeAction} />
            )}
          </View>
        ))}
        {sending && (
          <View style={styles.sending}>
            <LoadingSpinner size={20} />
          </View>
        )}
        {actionResult && (
          <View style={styles.actionResult}>
            <Text style={styles.actionResultText}>{actionResult.text}</Text>
          </View>
        )}
      </ScrollView>

      <ChatInput
        value={input}
        onChange={setInput}
        onSend={handleSend}
        sending={sending}
        autoFocus
      />
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    width: 380,
    backgroundColor: THEME.surface,
    borderLeftWidth: 1,
    borderLeftColor: THEME.border,
  },
  header: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingHorizontal: 12,
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderBottomColor: THEME.border,
  },
  title: { color: THEME.text, fontWeight: "700", fontSize: 14 },
  headerActions: { flexDirection: "row", gap: 12, alignItems: "center" },
  clearBtn: { color: THEME.textMuted, fontSize: 12 },
  closeBtn: { color: THEME.textMuted, fontWeight: "700", fontSize: 16 },
  msgs: { flex: 1 },
  msgsContent: { padding: 10, paddingBottom: 4 },
  sending: { alignItems: "center", paddingVertical: 10 },
  actionResult: { backgroundColor: THEME.surfaceLight, paddingHorizontal: 10, paddingVertical: 6, marginHorizontal: 10, marginVertical: 4, borderRadius: 4, borderLeftWidth: 3, borderLeftColor: THEME.profit },
  actionResultText: { color: THEME.text, fontSize: 11 },
});
