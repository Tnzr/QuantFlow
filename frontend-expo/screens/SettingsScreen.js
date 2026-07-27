import React, { useState, useEffect } from "react";
import { View, Text, TextInput, Pressable, ScrollView, StyleSheet } from "react-native";
import { apiGet, apiPost, apiPut, getApiBase } from "../services/api";
import { getMlEngineUrl } from "../services/config";
import { saveAlpacaKeys, getAlpacaKeys } from "../services/auth";
import { storageSet, storageGet } from "../services/storage";
import ScreenTitle from "../components/shared/ScreenTitle";
import THEME from "../theme/colors";

export default function SettingsScreen({ token }) {
  const [apiKey, setApiKey] = useState("");
  const [apiSecret, setApiSecret] = useState("");
  const [llmEndpoint, setLlmEndpoint] = useState("");
  const [llmKey, setLlmKey] = useState("");
  const [riskStopLoss, setRiskStopLoss] = useState("7");
  const [dailyLossLimit, setDailyLossLimit] = useState("5000");
  const [maxPositionPct, setMaxPositionPct] = useState("20");
  const [testStatus, setTestStatus] = useState(null);
  const [saveStatus, setSaveStatus] = useState(null);
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [backendAlpacaConfigured, setBackendAlpacaConfigured] = useState(null);

  useEffect(() => {
    getAlpacaKeys().then((keys) => {
      if (keys) {
        if (keys.apiKey) setApiKey(keys.apiKey);
        if (keys.secret) setApiSecret(keys.secret);
      }
    });
    storageGet("llm_config").then((cfg) => {
      if (cfg) {
        setLlmEndpoint(cfg.endpoint || "");
        setLlmKey(cfg.key || "");
      }
    });
    storageGet("risk_params").then((params) => {
      if (params) {
        setRiskStopLoss(String(params.stop_loss_pct ?? 7));
        setDailyLossLimit(String(params.daily_loss_limit ?? 5000));
        setMaxPositionPct(String(params.max_position_pct ?? 20));
      }
    });
    storageGet("selected_model").then((m) => {
      if (m) setSelectedModel(m);
    });

    // Check if backend has Alpaca configured
    apiGet("/market/alpaca/quote/AAPL", token)
      .then((data) => {
        const configured = data && data.quote && data.quote.bid != null;
        setBackendAlpacaConfigured(configured);
      })
      .catch(() => setBackendAlpacaConfigured(false));

    apiGet(`${getMlEngineUrl()}/models`)
      .then((data) => {
        const list = data.models || data || [];
        const arr = Array.isArray(list) ? list : [];
        setModels(arr);
        // Auto-select first model if none selected
        if (arr.length > 0) {
          const firstName = arr[0].model_id || arr[0].name || arr[0].id || String(0);
          if (!selectedModel) setSelectedModel(firstName);
        }
      })
      .catch(() => {});
  }, []);

  const handleTestAlpaca = async () => {
    setTestStatus("testing");
    try {
      await apiPost("/settings/test-alpaca", {
        api_key: apiKey,
        api_secret: apiSecret,
      }, token);
      setTestStatus("ok");
    } catch (err) {
      setTestStatus(err.message);
    }
  };

  const handleSave = async () => {
    setSaveStatus("saving");
    try {
      await apiPut("/settings/save", {
        risk_stop_loss_pct: parseFloat(riskStopLoss) || 7,
        daily_loss_limit: parseFloat(dailyLossLimit) || 5000,
        max_position_pct: parseFloat(maxPositionPct) || 20,
        model_checkpoint: selectedModel || null,
      }, token);

      await saveAlpacaKeys(apiKey, apiSecret);
      await storageSet("llm_config", { endpoint: llmEndpoint, key: llmKey });
      await storageSet("risk_params", {
        stop_loss_pct: parseFloat(riskStopLoss) || 7,
        daily_loss_limit: parseFloat(dailyLossLimit) || 5000,
        max_position_pct: parseFloat(maxPositionPct) || 20,
      });
      if (selectedModel) await storageSet("selected_model", selectedModel);

      setSaveStatus("ok");
      setTimeout(() => setSaveStatus(null), 3000);
    } catch (err) {
      setSaveStatus(err.message);
    }
  };

  return (
    <ScrollView style={styles.wrap} contentContainerStyle={styles.content}>
      <ScreenTitle title="Settings" subtitle="API keys, risk params, and model selection" icon="S" />
      <Text style={styles.heading}>Alpaca Connection</Text>
      {backendAlpacaConfigured !== null && (
        <View style={[styles.statusBanner, { backgroundColor: backendAlpacaConfigured ? THEME.profitDim : THEME.lossDim, borderColor: backendAlpacaConfigured ? THEME.profit : THEME.loss }]}>
          <Text style={[styles.statusBannerText, { color: backendAlpacaConfigured ? THEME.profit : THEME.loss }]}>
            {backendAlpacaConfigured ? "✓ Backend has Alpaca configured (live data active)" : "⚠ Backend Alpaca not configured"}
          </Text>
        </View>
      )}
      <Text style={styles.label}>Alpaca API Key (optional — overrides server .env)</Text>
      <TextInput
        style={styles.input}
        placeholder="e.g. PKHCSTZB6SAY..."
        placeholderTextColor={THEME.neutral}
        value={apiKey}
        onChangeText={setApiKey}
        secureTextEntry
        accessibilityLabel="Alpaca API key"
      />
      <Text style={styles.label}>Alpaca API Secret (optional — overrides server .env)</Text>
      <TextInput
        style={styles.input}
        placeholder="e.g. 68LYFArk38ce..."
        placeholderTextColor={THEME.neutral}
        value={apiSecret}
        onChangeText={setApiSecret}
        secureTextEntry
        accessibilityLabel="Alpaca API secret"
      />
      <Pressable style={styles.btn} onPress={handleTestAlpaca}>
        <Text style={styles.btnText}>
          {testStatus === "testing" ? "Testing..." : "Test Connection"}
        </Text>
      </Pressable>
      {testStatus && testStatus !== "testing" && (
        <Text
          style={[
            styles.status,
            { color: testStatus === "ok" ? THEME.profit : THEME.loss },
          ]}
        >
          {testStatus === "ok" ? "Connection OK" : testStatus}
        </Text>
      )}

      <Text style={styles.heading}>LLM Configuration</Text>
      <Text style={styles.label}>LLM Endpoint URL</Text>
      <TextInput
        style={styles.input}
        placeholder="e.g. https://api.deepseek.com"
        placeholderTextColor={THEME.neutral}
        value={llmEndpoint}
        onChangeText={setLlmEndpoint}
        accessibilityLabel="LLM endpoint URL"
      />
      <Text style={styles.label}>LLM API Key</Text>
      <TextInput
        style={styles.input}
        placeholder="e.g. sk-..."
        placeholderTextColor={THEME.neutral}
        value={llmKey}
        onChangeText={setLlmKey}
        secureTextEntry
        accessibilityLabel="LLM API key"
      />

      <Text style={styles.heading}>Risk Parameters</Text>
      <View style={styles.row}>
        <View style={styles.half}>
          <Text style={styles.label}>Stop Loss %</Text>
          <TextInput
            style={styles.input}
            placeholder="e.g. 7"
            placeholderTextColor={THEME.neutral}
            value={riskStopLoss}
            onChangeText={setRiskStopLoss}
            keyboardType="decimal-pad"
            accessibilityLabel="Stop loss percent"
          />
        </View>
        <View style={styles.half}>
          <Text style={styles.label}>Daily Loss Limit ($)</Text>
          <TextInput
            style={styles.input}
            placeholder="e.g. 5000"
            placeholderTextColor={THEME.neutral}
            value={dailyLossLimit}
            onChangeText={setDailyLossLimit}
            keyboardType="decimal-pad"
            accessibilityLabel="Daily loss limit"
          />
        </View>
      </View>
      <Text style={styles.label}>Max Position %</Text>
      <TextInput
        style={styles.input}
        placeholder="e.g. 20"
        placeholderTextColor={THEME.neutral}
        value={maxPositionPct}
        onChangeText={setMaxPositionPct}
        keyboardType="decimal-pad"
        accessibilityLabel="Max position percent"
      />

      <Text style={styles.heading}>Model Checkpoint</Text>
      {models.length > 0 ? (
        <View style={styles.modelList}>
          {models.map((m, i) => {
            const name = m.model_id || m.name || m.id || String(i);
            return (
              <Pressable
                key={i}
                style={[styles.modelItem, selectedModel === name && styles.modelActive]}
                onPress={() => setSelectedModel(name)}
              >
                <Text
                  style={[
                    styles.modelText,
                    selectedModel === name && styles.modelTextActive,
                  ]}
                >
                  {name}
                </Text>
              </Pressable>
            );
          })}
        </View>
      ) : (
        <Text style={styles.na}>No models loaded (ML engine offline)</Text>
      )}

      <View style={styles.info}>
        <Text style={styles.infoLabel}>Backend URL</Text>
        <Text style={styles.infoValue}>{getApiBase()}</Text>
      </View>

      <Pressable style={[styles.btn, styles.saveBtn]} onPress={handleSave}>
        <Text style={styles.btnText}>
          {saveStatus === "saving" ? "Saving..." : "Save Settings"}
        </Text>
      </Pressable>
      {saveStatus && saveStatus !== "saving" && (
        <Text
          style={[
            styles.status,
            { color: saveStatus === "ok" ? THEME.profit : THEME.loss },
          ]}
        >
          {saveStatus === "ok" ? "Settings saved" : saveStatus}
        </Text>
      )}

      <Text style={styles.warning}>
        API keys are stored client-side in AsyncStorage. Not suitable for production use.
      </Text>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  wrap: { flex: 1, backgroundColor: THEME.bg },
  content: { padding: 16, paddingBottom: 32 },
  heading: {
    color: THEME.text,
    fontSize: 14,
    fontWeight: "700",
    marginTop: 8,
    marginBottom: 8,
  },
  input: {
    backgroundColor: THEME.surface,
    borderWidth: 1,
    borderColor: THEME.border,
    borderRadius: 6,
    paddingHorizontal: 12,
    paddingVertical: 10,
    color: THEME.text,
    fontSize: 13,
    marginBottom: 8,
  },
  half: { flex: 1 },
  row: { flexDirection: "row", gap: 8 },
  label: { color: THEME.textMuted, fontSize: 11, fontWeight: "600", marginBottom: 4 },
  statusBanner: { borderWidth: 1, borderRadius: 6, paddingHorizontal: 10, paddingVertical: 8, marginBottom: 10 },
  statusBannerText: { fontSize: 11, fontWeight: "600" },
  btn: {
    backgroundColor: THEME.accent,
    borderRadius: 6,
    paddingVertical: 10,
    alignItems: "center",
    marginBottom: 8,
  },
  saveBtn: { backgroundColor: THEME.profit, marginTop: 16 },
  btnText: { color: THEME.textBright, fontWeight: "700", fontSize: 13 },
  status: { fontSize: 12, marginBottom: 8, textAlign: "center" },
  modelList: { marginBottom: 8 },
  modelItem: {
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 4,
    backgroundColor: THEME.surface,
    marginBottom: 4,
    borderWidth: 1,
    borderColor: THEME.border,
  },
  modelActive: { backgroundColor: THEME.accentGlow, borderColor: THEME.accent },
  modelText: { color: THEME.textMuted, fontSize: 12 },
  modelTextActive: { color: THEME.accent, fontWeight: "600" },
  na: { color: THEME.textMuted, fontSize: 12, marginBottom: 8 },
  info: {
    backgroundColor: THEME.surface,
    padding: 10,
    borderRadius: 6,
    marginBottom: 8,
  },
  infoLabel: { color: THEME.textMuted, fontSize: 11 },
  infoValue: { color: THEME.text, fontSize: 13, fontFamily: "monospace" },
  warning: {
    color: THEME.warn,
    fontSize: 11,
    textAlign: "center",
    marginTop: 20,
    paddingHorizontal: 16,
  },
});
