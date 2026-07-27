import React, { useState, useEffect } from "react";
import { View, Text, Pressable, StyleSheet, Switch, Alert } from "react-native";
import THEME from "../../theme/colors";

/**
 * Robinhood MCP connection panel for Settings.
 * Handles OAuth flow, connection status, trade opt-in, watchlist import.
 */
export default function RobinhoodSettings({ useRobinhood, token }) {
  const {
    status,
    connect,
    disconnect,
    setTradeOptIn,
    importWatchlist,
    error,
  } = useRobinhood(token);

  const [tradeEnabled, setTradeEnabled] = useState(false);
  const [importStatus, setImportStatus] = useState(null);

  useEffect(() => {
    if (status?.scopes) {
      setTradeEnabled(status.scopes.includes("trade"));
    }
  }, [status]);

  const isConnected = status?.connected;
  const isDemo = status?.demo_mode;

  const handleConnect = () => {
    if (isConnected) {
      Alert.alert(
        "Disconnect Robinhood?",
        "This will revoke access to your Robinhood account data.",
        [
          { text: "Cancel", style: "cancel" },
          { text: "Disconnect", style: "destructive", onPress: disconnect },
        ]
      );
    } else {
      // For demo: directly complete OAuth
      connect();
    }
  };

  const handleTradeToggle = async (value) => {
    setTradeEnabled(value);
    await setTradeOptIn(value);
  };

  const handleImportWatchlist = async () => {
    setImportStatus("importing");
    const result = await importWatchlist();
    if (result) {
      setImportStatus(`Imported ${result.count} tickers: ${result.imported.join(", ")}`);
    } else {
      setImportStatus("import failed");
    }
    setTimeout(() => setImportStatus(null), 5000);
  };

  return (
    <View style={styles.wrap}>
      <View style={styles.header}>
        <View>
          <Text style={styles.title}>Robinhood (MCP)</Text>
          <Text style={styles.subtitle}>Connect via official Model Context Protocol</Text>
        </View>
        <View style={[
          styles.statusBadge,
          { backgroundColor: isConnected ? THEME.profitDim : THEME.surfaceLight, borderColor: isConnected ? THEME.profit : THEME.border }
        ]}>
          <Text style={[
            styles.statusText,
            { color: isConnected ? THEME.profit : THEME.textMuted }
          ]}>
            {isConnected ? "Connected" : "Not Connected"}
          </Text>
        </View>
      </View>

      {isDemo && (
        <View style={styles.demoBanner}>
          <Text style={styles.demoText}>
            ⚠ Demo Mode — using simulated portfolio data. Set ROBINHOOD_MCP_DEMO=false and configure OAuth credentials for real connections.
          </Text>
        </View>
      )}

      {isConnected && status && (
        <View style={styles.detailsBox}>
          <DetailRow label="Workspace" value={status.workspace_id} />
          <DetailRow label="Scopes" value={status.scopes.join(", ") || "none"} />
          {status.last_sync > 0 && (
            <DetailRow
              label="Last Sync"
              value={new Date(status.last_sync * 1000).toLocaleString()}
            />
          )}
          {status.last_error && (
            <DetailRow label="Last Error" value={status.last_error} error />
          )}
        </View>
      )}

      <Pressable
        style={[styles.btn, isConnected ? styles.btnDisconnect : styles.btnConnect]}
        onPress={handleConnect}
      >
        <Text style={styles.btnText}>
          {isConnected ? "Disconnect Robinhood" : "Connect Robinhood Account"}
        </Text>
      </Pressable>

      {isConnected && (
        <>
          <View style={styles.toggleRow}>
            <View style={{ flex: 1 }}>
              <Text style={styles.toggleLabel}>Enable Trade Execution</Text>
              <Text style={styles.toggleHelp}>
                Required to place orders via QuantFlow. Trades are isolated to your Agentic account.
              </Text>
            </View>
            <Switch
              value={tradeEnabled}
              onValueChange={handleTradeToggle}
              trackColor={{ false: THEME.border, true: THEME.profitDim }}
              thumbColor={tradeEnabled ? THEME.profit : THEME.textMuted}
            />
          </View>

          <Pressable style={styles.secondaryBtn} onPress={handleImportWatchlist}>
            <Text style={styles.secondaryBtnText}>Import Watchlist from Robinhood</Text>
          </Pressable>
          {importStatus && (
            <Text style={styles.importStatus}>{importStatus}</Text>
          )}
        </>
      )}

      {error && (
        <Text style={styles.error}>Error: {error}</Text>
      )}
    </View>
  );
}

function DetailRow({ label, value, error }) {
  return (
    <View style={styles.detailRow}>
      <Text style={styles.detailLabel}>{label}</Text>
      <Text style={[styles.detailValue, error && { color: THEME.loss }]} numberOfLines={1}>
        {value}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    marginBottom: 16,
    padding: 14,
    backgroundColor: THEME.surface,
    borderRadius: 8,
    borderLeftWidth: 3,
    borderLeftColor: THEME.accent,
  },
  header: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "flex-start",
    marginBottom: 10,
  },
  title: { color: THEME.text, fontSize: 14, fontWeight: "700" },
  subtitle: { color: THEME.textMuted, fontSize: 11, marginTop: 2 },
  statusBadge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
    borderWidth: 1,
  },
  statusText: { fontSize: 11, fontWeight: "700" },
  demoBanner: {
    backgroundColor: THEME.surfaceLight,
    borderRadius: 6,
    padding: 10,
    marginBottom: 10,
    borderLeftWidth: 3,
    borderLeftColor: THEME.warn,
  },
  demoText: { color: THEME.warn, fontSize: 10, lineHeight: 14 },
  detailsBox: {
    backgroundColor: THEME.bg,
    borderRadius: 6,
    padding: 10,
    marginBottom: 10,
  },
  detailRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    paddingVertical: 3,
  },
  detailLabel: { color: THEME.textMuted, fontSize: 11 },
  detailValue: { color: THEME.text, fontSize: 11, fontWeight: "600", flex: 1, textAlign: "right" },
  btn: {
    paddingVertical: 10,
    borderRadius: 6,
    alignItems: "center",
    marginBottom: 10,
  },
  btnConnect: { backgroundColor: THEME.profit },
  btnDisconnect: { backgroundColor: THEME.loss },
  btnText: { color: THEME.textBright, fontWeight: "700", fontSize: 12 },
  toggleRow: {
    flexDirection: "row",
    alignItems: "center",
    paddingVertical: 8,
    borderTopWidth: 1,
    borderTopColor: THEME.border,
    marginTop: 4,
  },
  toggleLabel: { color: THEME.text, fontSize: 12, fontWeight: "600" },
  toggleHelp: { color: THEME.textMuted, fontSize: 10, marginTop: 2 },
  secondaryBtn: {
    paddingVertical: 8,
    borderRadius: 6,
    backgroundColor: THEME.surfaceLight,
    borderWidth: 1,
    borderColor: THEME.border,
    alignItems: "center",
    marginTop: 8,
  },
  secondaryBtnText: { color: THEME.text, fontWeight: "600", fontSize: 11 },
  importStatus: { color: THEME.profit, fontSize: 10, marginTop: 6, fontStyle: "italic" },
  error: { color: THEME.loss, fontSize: 11, marginTop: 6 },
});
