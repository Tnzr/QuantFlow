import React, { useState, useEffect } from "react";
import { View, Text, Pressable, StyleSheet, Switch, TextInput } from "react-native";
import THEME from "../../theme/colors";

/**
 * Robinhood MCP connection panel for Settings.
 * Handles OAuth flow (popup), manual token entry, connection status, trade opt-in.
 */
export default function RobinhoodSettings({ useRobinhood, token }) {
  const {
    status,
    connect,
    disconnect,
    setTradeOptIn,
    importWatchlist,
    error,
    setError: setRobinhoodError,
    setManualToken,
  } = useRobinhood(token);

  const [tradeEnabled, setTradeEnabled] = useState(false);
  const [importStatus, setImportStatus] = useState(null);
  const [confirmingDisconnect, setConfirmingDisconnect] = useState(false);
  const [disconnecting, setDisconnecting] = useState(false);
  const [connecting, setConnecting] = useState(false);
  const [showManualToken, setShowManualToken] = useState(false);
  const [manualAccessToken, setManualAccessToken] = useState("");
  const [manualRefreshToken, setManualRefreshToken] = useState("");
  const [manualSubmitting, setManualSubmitting] = useState(false);

  useEffect(() => {
    if (status?.scopes) {
      setTradeEnabled(status.scopes.includes("trade"));
    }
  }, [status]);

  // Auto-reset connecting state when connection completes
  useEffect(() => {
    if (connecting && (status?.connected || status?.last_error)) {
      setConnecting(false);
    }
  }, [status, connecting]);

  const isConnected = status?.connected;
  const isDemo = status?.demo_mode;

  const handleConnect = async () => {
    if (isConnected) {
      setConfirmingDisconnect(true);
      return;
    }
    setConnecting(true);
    setRobinhoodError(null);
    try {
      await connect();
    } catch (e) {
      // error handled in hook
    }
    // Auto-reset connecting state after 3 seconds if not already reset
    setTimeout(() => setConnecting(false), 3000);
  };

  const handleManualToken = async () => {
    setManualSubmitting(true);
    setRobinhoodError(null);
    try {
      const ok = await setManualToken(
        manualAccessToken.trim(),
        manualRefreshToken.trim() || null,
        3600,
      );
      if (ok) {
        setShowManualToken(false);
        setManualAccessToken("");
        setManualRefreshToken("");
      }
    } catch (e) {
      // error handled in hook
    } finally {
      setManualSubmitting(false);
    }
  };

  const confirmDisconnect = async () => {
    setDisconnecting(true);
    try {
      await disconnect();
      setConfirmingDisconnect(false);
    } catch (e) {
      // error handled in hook
    } finally {
      setDisconnecting(false);
    }
  };

  const cancelDisconnect = () => {
    setConfirmingDisconnect(false);
  };

  const handleTradeToggle = async (value) => {
    setTradeEnabled(value);
    await setTradeOptIn(value);
  };

  const handleImportWatchlist = async () => {
    setImportStatus({ type: "loading", text: "Importing..." });
    const result = await importWatchlist();
    if (result) {
      setImportStatus({
        type: "success",
        text: `Imported ${result.count} tickers: ${result.imported.join(", ")}`,
      });
    } else {
      setImportStatus({ type: "error", text: "Import failed. Is Robinhood connected?" });
    }
    setTimeout(() => setImportStatus(null), 6000);
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
          <Text style={styles.demoTitle}>⚠ Demo Mode</Text>
          <Text style={styles.demoText}>
            Using simulated portfolio data. The "Connect" button will auto-complete OAuth with demo credentials — no real Robinhood login required.
          </Text>
          <Text style={styles.demoSetupTitle}>For real Robinhood connection:</Text>
          <Text style={styles.demoSetupText}>
            1. Register as a developer at robinhood.com/us/en/support/agentic-trading{"\n"}
            2. Add to your .env file:{"\n"}
               {"   "}ROBINHOOD_MCP_DEMO=false{"\n"}
               {"   "}ROBINHOOD_CLIENT_ID=your_client_id{"\n"}
               {"   "}ROBINHOOD_CLIENT_SECRET=your_client_secret{"\n"}
            3. Restart with: make down && make up
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

      {confirmingDisconnect ? (
        <View style={styles.confirmBox}>
          <Text style={styles.confirmTitle}>Disconnect Robinhood?</Text>
          <Text style={styles.confirmHelp}>
            This will revoke access to your Robinhood account data and clear the watchlist sync.
          </Text>
          <View style={styles.confirmRow}>
            <Pressable style={styles.confirmCancel} onPress={cancelDisconnect} disabled={disconnecting}>
              <Text style={styles.confirmCancelText}>Cancel</Text>
            </Pressable>
            <Pressable
              style={[styles.confirmDanger, disconnecting && { opacity: 0.5 }]}
              onPress={confirmDisconnect}
              disabled={disconnecting}
            >
              <Text style={styles.confirmDangerText}>
                {disconnecting ? "Disconnecting..." : "Disconnect"}
              </Text>
            </Pressable>
          </View>
        </View>
      ) : (
        <Pressable
          style={[
            styles.btn,
            isConnected ? styles.btnDisconnect : styles.btnConnect,
            connecting && { opacity: 0.5 },
          ]}
          onPress={handleConnect}
          disabled={connecting}
        >
          <Text style={styles.btnText}>
            {isConnected
              ? "Disconnect Robinhood"
              : connecting
              ? "Connecting..."
              : "Connect Robinhood Account"}
          </Text>
        </Pressable>
      )}

      {!isConnected && (
        <View style={styles.oauthSection}>
          <Pressable
            style={styles.manualTokenToggle}
            onPress={() => setShowManualToken(!showManualToken)}
          >
            <Text style={styles.manualTokenToggleText}>
              {showManualToken ? "▼ Hide Manual Token Entry" : "▶ Have an OAuth token? Enter it manually"}
            </Text>
          </Pressable>
          {showManualToken && (
            <View style={styles.manualTokenBox}>
              <Text style={styles.manualTokenTitle}>Manual OAuth Token</Text>
              <Text style={styles.manualTokenHelp}>
                Complete OAuth in your browser, then paste the access token here.
                Get your token from robinhood.com → Settings → Agentic Trading → Connect Agent.
              </Text>
              <Text style={styles.label}>Access Token</Text>
              <TextInput
                style={styles.input}
                placeholder="Paste access token here"
                placeholderTextColor={THEME.neutral}
                value={manualAccessToken}
                onChangeText={setManualAccessToken}
                secureTextEntry
                autoCapitalize="none"
                autoCorrect={false}
              />
              <Text style={styles.label}>Refresh Token (optional)</Text>
              <TextInput
                style={styles.input}
                placeholder="Paste refresh token here"
                placeholderTextColor={THEME.neutral}
                value={manualRefreshToken}
                onChangeText={setManualRefreshToken}
                secureTextEntry
                autoCapitalize="none"
                autoCorrect={false}
              />
              <Pressable
                style={[styles.manualTokenBtn, manualSubmitting && { opacity: 0.5 }]}
                onPress={handleManualToken}
                disabled={manualSubmitting || !manualAccessToken.trim()}
              >
                <Text style={styles.manualTokenBtnText}>
                  {manualSubmitting ? "Saving..." : "Save Token"}
                </Text>
              </Pressable>
            </View>
          )}
        </View>
      )}

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
            <View style={[
              styles.importStatusBox,
              importStatus.type === "error" ? styles.importStatusError : styles.importStatusOk
            ]}>
              <Text style={[
                styles.importStatusText,
                importStatus.type === "error" && { color: THEME.loss }
              ]}>
                {importStatus.text}
              </Text>
            </View>
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
  demoTitle: { color: THEME.warn, fontSize: 12, fontWeight: "700", marginBottom: 4 },
  demoText: { color: THEME.textMuted, fontSize: 10, lineHeight: 14, marginBottom: 6 },
  demoSetupTitle: { color: THEME.text, fontSize: 10, fontWeight: "700", marginBottom: 2 },
  demoSetupText: { color: THEME.textMuted, fontSize: 9, lineHeight: 13, fontFamily: "monospace" },
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
  importStatusBox: {
    padding: 8,
    borderRadius: 4,
    marginTop: 6,
    borderWidth: 1,
  },
  importStatusOk: {
    backgroundColor: THEME.profitDim,
    borderColor: THEME.profit,
  },
  importStatusError: {
    backgroundColor: THEME.lossDim,
    borderColor: THEME.loss,
  },
  importStatusText: { fontSize: 10 },
  confirmBox: {
    backgroundColor: THEME.surfaceLight,
    borderRadius: 6,
    padding: 12,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: THEME.loss,
  },
  confirmTitle: { color: THEME.loss, fontSize: 13, fontWeight: "700" },
  confirmHelp: { color: THEME.textMuted, fontSize: 11, marginTop: 4, marginBottom: 8 },
  confirmRow: { flexDirection: "row", gap: 8 },
  confirmCancel: {
    flex: 1,
    paddingVertical: 8,
    borderRadius: 4,
    backgroundColor: THEME.bg,
    borderWidth: 1,
    borderColor: THEME.border,
    alignItems: "center",
  },
  confirmCancelText: { color: THEME.text, fontWeight: "600", fontSize: 12 },
  confirmDanger: {
    flex: 1,
    paddingVertical: 8,
    borderRadius: 4,
    backgroundColor: THEME.loss,
    alignItems: "center",
  },
  confirmDangerText: { color: THEME.textBright, fontWeight: "700", fontSize: 12 },
  error: { color: THEME.loss, fontSize: 11, marginTop: 6 },
  oauthSection: {
    marginTop: 10,
    paddingTop: 10,
    borderTopWidth: 1,
    borderTopColor: THEME.border,
  },
  manualTokenToggle: {
    paddingVertical: 8,
    alignItems: "center",
  },
  manualTokenToggleText: {
    color: THEME.accent,
    fontSize: 11,
    fontWeight: "600",
  },
  manualTokenBox: {
    backgroundColor: THEME.surfaceLight,
    borderRadius: 6,
    padding: 12,
    marginTop: 8,
    borderWidth: 1,
    borderColor: THEME.border,
  },
  manualTokenTitle: {
    color: THEME.text,
    fontSize: 13,
    fontWeight: "700",
    marginBottom: 4,
  },
  manualTokenHelp: {
    color: THEME.textMuted,
    fontSize: 10,
    marginBottom: 10,
    lineHeight: 14,
  },
  label: {
    color: THEME.textMuted,
    fontSize: 11,
    fontWeight: "600",
    marginBottom: 4,
  },
  input: {
    backgroundColor: THEME.bg,
    borderWidth: 1,
    borderColor: THEME.border,
    borderRadius: 6,
    paddingHorizontal: 10,
    paddingVertical: 8,
    color: THEME.text,
    fontSize: 12,
    marginBottom: 10,
  },
  manualTokenBtn: {
    backgroundColor: THEME.accent,
    borderRadius: 6,
    paddingVertical: 8,
    alignItems: "center",
  },
  manualTokenBtnText: {
    color: THEME.textBright,
    fontWeight: "700",
    fontSize: 12,
  },
});
