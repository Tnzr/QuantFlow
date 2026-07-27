import React, { useEffect, useState, useCallback } from "react";
import {
  View,
  Text,
  TextInput,
  Pressable,
  StyleSheet,
  SafeAreaView,
  KeyboardAvoidingView,
  Platform,
} from "react-native";
import { StatusBar } from "expo-status-bar";

import THEME from "./theme/colors";
import useAuth from "./hooks/useAuth";
import { apiGet, getApiBase } from "./services/api";
import { getMlEngineUrl } from "./services/config";
import StatusBarComponent from "./components/shared/StatusBar";
import DashboardScreen from "./screens/DashboardScreen";
import TradeScreen from "./screens/TradeScreen";
import MarketScreen from "./screens/MarketScreen";
import BacktestScreen from "./screens/BacktestScreen";
import SettingsScreen from "./screens/SettingsScreen";
import ChatSidebar from "./components/chatbot/ChatSidebar";

const TABS = [
  { key: "dashboard", label: "Dashboard", icon: "D" },
  { key: "trade", label: "Trade", icon: "T" },
  { key: "market", label: "Market", icon: "M" },
  { key: "backtest", label: "Backtest", icon: "B" },
  { key: "settings", label: "Settings", icon: "S" },
];

export default function App() {
  const { token, unlocked, loading: authLoading, error: authError, unlock } = useAuth();
  const [activeTab, setActiveTab] = useState("dashboard");
  const [apiStatus, setApiStatus] = useState("checking");
  const [modelStatus, setModelStatus] = useState(null);
  const [authMode, setAuthMode] = useState(null);
  const [chatOpen, setChatOpen] = useState(false);

  const [gateKey, setGateKey] = useState("");
  const [gateSubmitting, setGateSubmitting] = useState(false);
  const [gateError, setGateError] = useState(null);

  const checkHealth = useCallback(async () => {
    try {
      setApiStatus("checking");
      const data = await apiGet("/health");
      setApiStatus("connected");
      setAuthMode(data.auth_mode || null);
      return true;
    } catch {
      setApiStatus("disconnected");
      return false;
    }
  }, []);

  const checkModel = useCallback(async () => {
    try {
      const data = await apiGet(`${getMlEngineUrl()}/health`);
      setModelStatus(data?.model_loaded ? "loaded" : "loading");
    } catch {
      setModelStatus("offline");
    }
  }, []);

  useEffect(() => {
    checkHealth();
    checkModel();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, [checkHealth, checkModel]);

  const handleUnlock = async () => {
    if (!gateKey.trim()) {
      setGateError("Enter an API key or token");
      return;
    }
    setGateSubmitting(true);
    setGateError(null);
    const ok = await unlock(gateKey.trim());
    if (!ok) setGateError(authError || "Authentication failed");
    setGateSubmitting(false);
  };

  if (authLoading) {
    return (
      <SafeAreaView style={styles.root}>
        <StatusBar style="light" />
        <View style={styles.loadingWrap}>
          <Text style={styles.loadingText}>Loading...</Text>
        </View>
      </SafeAreaView>
    );
  }

  if (!unlocked) {
    return (
      <SafeAreaView style={styles.root}>
        <StatusBar style="light" />
        <KeyboardAvoidingView
          behavior={Platform.OS === "ios" ? "padding" : "height"}
          style={styles.gateWrap}
        >
          <View style={styles.gateCard}>
            <Text style={styles.gateTitle}>QuantFlow</Text>
            <Text style={styles.gateSubtitle}>Algorithmic Trading & Portfolio Intelligence</Text>
            <Text style={styles.gateSub}>Enter your API key or Bearer token</Text>
            <TextInput
              style={styles.gateInput}
              placeholder="API Key / Bearer Token"
              placeholderTextColor={THEME.neutral}
              value={gateKey}
              onChangeText={setGateKey}
              onSubmitEditing={handleUnlock}
              secureTextEntry
              autoFocus
              returnKeyType="go"
            />
            {gateError && <Text style={styles.gateError}>{gateError}</Text>}
            <Pressable
              style={[styles.gateBtn, gateSubmitting && styles.gateBtnDisabled]}
              onPress={handleUnlock}
              disabled={gateSubmitting}
            >
              <Text style={styles.gateBtnText}>
                {gateSubmitting ? "Verifying..." : "Unlock Workspace"}
              </Text>
            </Pressable>
            {apiStatus === "disconnected" && (
              <Text style={styles.gateWarn}>
                Cannot reach API at {getApiBase()}
              </Text>
            )}
          </View>
        </KeyboardAvoidingView>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.root}>
      <StatusBar style="light" />
      <StatusBarComponent
        apiStatus={apiStatus}
        authMode={authMode}
        modelStatus={modelStatus}
      />

      <View style={styles.header}>
        <Text style={styles.brand}>QuantFlow</Text>
        <Text style={styles.tagline}>Algorithmic Trading & Portfolio Intelligence</Text>
      </View>

      <View style={styles.body}>
        <View style={styles.sidebar}>
          {TABS.map((t) => (
            <Pressable
              key={t.key}
              style={[styles.navItem, activeTab === t.key && styles.navActive]}
              onPress={() => setActiveTab(t.key)}
            >
              <View style={[styles.navIcon, activeTab === t.key && styles.navIconActive]}>
                <Text style={[styles.navIconText, activeTab === t.key && styles.navIconTextActive]}>
                  {t.icon}
                </Text>
              </View>
              <Text style={[styles.navLabel, activeTab === t.key && styles.navLabelActive]}>
                {t.label}
              </Text>
            </Pressable>
          ))}
          <View style={styles.sidebarSpacer} />
          <Pressable
            style={[styles.navItem, styles.chatNavItem, chatOpen && styles.navActive]}
            onPress={() => setChatOpen((v) => !v)}
          >
            <View style={[styles.navIcon, chatOpen && styles.navIconActive]}>
              <Text style={[styles.navIconText, chatOpen && styles.navIconTextActive]}>C</Text>
            </View>
            <Text style={[styles.navLabel, chatOpen && styles.navLabelActive]}>Chat</Text>
          </Pressable>
        </View>

        <View style={styles.screenWrap}>
          <View style={[styles.screenPanel, activeTab === "dashboard" ? styles.screenShow : styles.screenHide]}>
            <DashboardScreen token={token} />
          </View>
          <View style={[styles.screenPanel, activeTab === "trade" ? styles.screenShow : styles.screenHide]}>
            <TradeScreen token={token} />
          </View>
          <View style={[styles.screenPanel, activeTab === "market" ? styles.screenShow : styles.screenHide]}>
            <MarketScreen token={token} />
          </View>
          <View style={[styles.screenPanel, activeTab === "backtest" ? styles.screenShow : styles.screenHide]}>
            <BacktestScreen token={token} />
          </View>
          <View style={[styles.screenPanel, activeTab === "settings" ? styles.screenShow : styles.screenHide]}>
            <SettingsScreen token={token} />
          </View>
        </View>

        {chatOpen && (
          <ChatSidebar token={token} onClose={() => setChatOpen(false)} activeTab={activeTab} />
        )}
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: THEME.bg,
  },
  loadingWrap: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
  },
  loadingText: { color: THEME.textMuted, fontSize: 16 },
  gateWrap: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    paddingHorizontal: 32,
  },
  gateCard: { width: "100%", maxWidth: 400 },
  gateTitle: { color: THEME.accent, fontSize: 36, fontWeight: "800", textAlign: "center", marginBottom: 4 },
  gateSubtitle: { color: THEME.textMuted, fontSize: 13, textAlign: "center", marginBottom: 24 },
  gateSub: { color: THEME.textMuted, fontSize: 14, textAlign: "center", marginBottom: 24 },
  gateInput: {
    backgroundColor: THEME.surface,
    borderWidth: 1,
    borderColor: THEME.border,
    borderRadius: 8,
    paddingHorizontal: 14,
    paddingVertical: 12,
    color: THEME.text,
    fontSize: 14,
    marginBottom: 8,
  },
  gateError: { color: THEME.loss, fontSize: 13, marginBottom: 8 },
  gateWarn: { color: THEME.warn, fontSize: 12, textAlign: "center", marginTop: 16 },
  gateBtn: { backgroundColor: THEME.accent, borderRadius: 8, paddingVertical: 13, alignItems: "center", marginTop: 4 },
  gateBtnDisabled: { opacity: 0.6 },
  gateBtnText: { color: THEME.textBright, fontWeight: "700", fontSize: 15 },
  header: {
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderBottomColor: THEME.border,
    backgroundColor: THEME.surface,
  },
  brand: { color: THEME.accent, fontSize: 18, fontWeight: "800" },
  tagline: { color: THEME.textMuted, fontSize: 11, marginTop: 2 },
  body: { flex: 1, flexDirection: "row" },
  sidebar: {
    width: 72,
    backgroundColor: THEME.surface,
    borderRightWidth: 1,
    borderRightColor: THEME.border,
    paddingTop: 8,
    alignItems: "center",
  },
  sidebarSpacer: { flex: 1 },
  navItem: {
    width: 64,
    paddingVertical: 12,
    alignItems: "center",
    borderRadius: 8,
    marginBottom: 2,
  },
  navActive: {
    backgroundColor: THEME.accentGlow,
  },
  navIcon: {
    width: 36,
    height: 36,
    borderRadius: 8,
    backgroundColor: THEME.bg,
    justifyContent: "center",
    alignItems: "center",
    marginBottom: 4,
  },
  navIconActive: {
    backgroundColor: THEME.accent,
  },
  navIconText: {
    color: THEME.textMuted,
    fontWeight: "800",
    fontSize: 14,
  },
  navIconTextActive: {
    color: THEME.textBright,
  },
  navLabel: {
    color: THEME.textMuted,
    fontSize: 9,
    fontWeight: "600",
    textAlign: "center",
  },
  navLabelActive: {
    color: THEME.accent,
  },
  chatNavItem: {
    marginTop: 4,
    marginBottom: 8,
  },
  screenWrap: { flex: 1, position: "relative" },
  screenPanel: {
    position: "absolute",
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
  },
  screenShow: { display: "flex" },
  screenHide: { display: "none" },
});
