import React, { useEffect, useMemo, useRef, useState } from "react";
import { SafeAreaView, ScrollView, Text, View, TextInput, Pressable, StyleSheet } from "react-native";
import { StatusBar } from "expo-status-bar";
import AsyncStorage from "@react-native-async-storage/async-storage";

function resolveApiBase() {
  const envBase = String(process.env.EXPO_PUBLIC_API_BASE_URL || "").trim();
  if (envBase) return envBase;
  if (typeof window !== "undefined") {
    const protocol = window.location.protocol || "http:";
    const host = window.location.hostname || "127.0.0.1";
    return `${protocol}//${host}:3000`;
  }
  return "http://127.0.0.1:3000";
}

const API_BASE = resolveApiBase();
const API_KEY = process.env.EXPO_PUBLIC_API_KEY || "";

const FIREBASE_WEB_CONFIG = {
  apiKey: process.env.EXPO_PUBLIC_FIREBASE_API_KEY || "",
  authDomain: process.env.EXPO_PUBLIC_FIREBASE_AUTH_DOMAIN || "",
  projectId: process.env.EXPO_PUBLIC_FIREBASE_PROJECT_ID || "",
  appId: process.env.EXPO_PUBLIC_FIREBASE_APP_ID || "",
};

const SURFACES = ["Public", "Workspace"];
const SCREENS = ["Overview", "Portfolio", "Trade", "Scanner", "Recommend", "Analytics", "Charts", "Options", "Backtest", "Market", "Execution", "Assistant", "Presets", "Auth", "Settings"];

const THEME = {
  bg: "#f3efe5",
  panel: "#fffdf8",
  panelAlt: "#f8f2e8",
  text: "#231f1a",
  textMuted: "#6b5f53",
  accent: "#0f766e",
  accentDeep: "#115e59",
  border: "#dbcdb8",
  warn: "#9a3412",
  ok: "#166534",
};

const ASSISTANT_SUGGESTED_PROMPTS = [
  "What's my best performer?",
  "Run a backtest on AAPL",
  "Show my portfolio positions",
  "Explain this signal",
  "Analyze MSFT seasonality",
  "Check MCP connection status",
];

function compactJson(value, cap = 600) {
  try {
    const s = JSON.stringify(value);
    return s.length > cap ? `${s.slice(0, cap)}...` : s;
  } catch {
    return String(value);
  }
}

function parseNumber(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function parseTickers(text) {
  const arr = String(text || "")
    .split(",")
    .map((x) => x.trim().toUpperCase())
    .filter(Boolean);
  return arr.length ? arr : null;
}

function parseCsv(text) {
  return String(text || "")
    .split(",")
    .map((x) => x.trim())
    .filter(Boolean);
}

function fmtPct(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return `${(Number(value) * 100).toFixed(2)}%`;
}

function fmtNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(digits);
}

function fmtTimestamp(value) {
  if (!value) return "-";
  try {
    const d = new Date(value);
    if (!Number.isFinite(d.getTime())) return "-";
    return d.toLocaleString();
  } catch {
    return "-";
  }
}

function isAbortError(error) {
  const msg = String(error?.message || "").toLowerCase();
  return error?.name === "AbortError" || msg.includes("abort") || msg.includes("cancel");
}

function fmtDurationSeconds(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  const total = Math.max(0, Math.round(Number(value)));
  const m = Math.floor(total / 60);
  const s = total % 60;
  if (m <= 0) return `${s}s`;
  return `${m}m ${s}s`;
}

function buildLinePointCloud(points, stepPx = 3) {
  const cloud = [];
  for (let idx = 1; idx < points.length; idx += 1) {
    const a = points[idx - 1];
    const b = points[idx];
    if (!a || !b) continue;
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const steps = Math.max(1, Math.ceil(Math.max(Math.abs(dx), Math.abs(dy)) / stepPx));
    for (let s = 0; s <= steps; s += 1) {
      const t = s / steps;
      cloud.push({ x: a.x + dx * t, y: a.y + dy * t });
    }
  }
  return cloud;
}

function dayOfYearLabel(row, fallback) {
  const doy = Number(row?.doy ?? row?.day_of_year ?? NaN);
  if (Number.isFinite(doy) && doy >= 1 && doy <= 366) {
    return String(Math.round(doy));
  }
  return String(fallback);
}

function MiniSeriesChart({ series, height = 120, xLabels = [] }) {
  const [chartWidth, setChartWidth] = useState(0);
  const cleaned = (series || []).map((row) => ({
    ...row,
    type: row.type || "line",
    values: (row.values || []).map((value) => {
      const n = Number(value);
      return Number.isFinite(n) ? n : null;
    }),
  })).filter((row) => row.values.some((value) => value !== null));

  if (!cleaned.length) {
    return <Text style={styles.item}>No chart data loaded.</Text>;
  }

  const maxLen = Math.max(...cleaned.map((row) => row.values.length));
  const flat = cleaned.flatMap((row) => row.values).filter((value) => value !== null);
  const min = Math.min(...flat);
  const max = Math.max(...flat);
  const span = max - min || 1;
  const innerHeight = Math.max(20, height - 12);
  const innerWidth = Math.max(20, chartWidth - 16);
  const yTicks = [max, min + span / 2, min];
  const yFmt = (value) => {
    const abs = Math.abs(Number(value));
    if (abs >= 1000) return fmtNumber(value, 0);
    if (abs >= 100) return fmtNumber(value, 1);
    if (abs >= 1) return fmtNumber(value, 2);
    return fmtNumber(value, 4);
  };

  const xStart = xLabels.length ? String(xLabels[0] || "") : "1";
  const xEnd = xLabels.length ? String(xLabels[Math.max(0, xLabels.length - 1)] || "") : String(maxLen);

  return (
    <View style={styles.chartOuter}>
      <View style={styles.chartBodyRow}>
        <View style={styles.chartYAxis}>
          {yTicks.map((tick, idx) => (
            <Text key={`y-${String(idx)}`} style={styles.chartAxisText}>{yFmt(tick)}</Text>
          ))}
        </View>
        <View style={[styles.chartFrame, { height }]}> 
          <View
            style={styles.chartMeasureLayer}
            onLayout={(event) => setChartWidth(Number(event?.nativeEvent?.layout?.width || 0))}
          />
          {[0, 0.5, 1].map((ratio) => (
            <View key={`grid-${String(ratio)}`} style={[styles.chartGridLine, { bottom: `${ratio * 100}%` }]} />
          ))}
          {cleaned.map((row, seriesIdx) => (
            <View key={`${row.label}-${String(seriesIdx)}`} style={styles.chartLayer}>
              {(() => {
                const points = row.values
                  .map((value, idx) => {
                    if (value === null) return null;
                    const x = maxLen > 1 ? (idx / (maxLen - 1)) * innerWidth : 0;
                    const y = innerHeight - ((value - min) / span) * innerHeight;
                    return { x, y };
                  })
                  .filter(Boolean);

                if (row.type === "bar") {
                  return row.values.map((value, idx) => {
                    if (value === null) return null;
                    const x = maxLen > 1 ? (idx / (maxLen - 1)) * innerWidth : 0;
                    const y = innerHeight - ((value - min) / span) * innerHeight;
                    return (
                      <View
                        key={`${row.label}-bar-${String(idx)}`}
                        style={[
                          styles.chartPoint,
                          {
                            left: x,
                            height: Math.max(2, innerHeight - y + 2),
                            backgroundColor: row.color,
                            width: row.width || 3,
                            opacity: row.opacity || 0.85,
                          },
                        ]}
                      />
                    );
                  });
                }

                const dotSize = Math.max(2, Number(row.width || 2));
                const cloud = buildLinePointCloud(points, 3);
                return cloud.map((pt, idx) => (
                  <View
                    key={`${row.label}-line-${String(idx)}`}
                    style={[
                      styles.chartLineDot,
                      {
                        left: pt.x - dotSize / 2,
                        top: pt.y - dotSize / 2,
                        width: dotSize,
                        height: dotSize,
                        borderRadius: dotSize,
                        backgroundColor: row.color,
                        opacity: row.opacity || 0.95,
                      },
                    ]}
                  />
                ));
              })()}
            </View>
          ))}
        </View>
      </View>
      <View style={styles.chartXAxis}>
        <Text style={styles.chartAxisText}>{xStart}</Text>
        <Text style={styles.chartAxisText}>{xEnd}</Text>
      </View>
    </View>
  );
}

function HorizontalDistribution({ items, color = THEME.accent }) {
  const rows = (items || []).slice(0, 16);
  if (!rows.length) return <Text style={styles.item}>No distribution loaded.</Text>;
  const maxWeight = Math.max(...rows.map((row) => Number(row.weighted_density || row.density || 0)), 0.0001);
  return (
    <View style={styles.distributionList}>
      {rows.map((row, idx) => {
        const density = Number(row.weighted_density || row.density || 0);
        return (
          <View key={`dist-${String(idx)}`} style={styles.distributionRow}>
            <Text style={styles.distributionLabel}>{fmtNumber(row.midpoint)}</Text>
            <View style={styles.distributionTrack}>
              <View style={[styles.distributionFill, { width: `${(density / maxWeight) * 100}%`, backgroundColor: color }]} />
            </View>
            <Text style={styles.distributionValue}>{fmtPct(density)}</Text>
          </View>
        );
      })}
    </View>
  );
}

function ScoreBars({ breakdown }) {
  const entries = Object.entries(breakdown || {});
  if (!entries.length) return <Text style={styles.item}>No score breakdown loaded.</Text>;
  return (
    <View style={styles.scoreList}>
      {entries.map(([label, value]) => (
        <View key={label} style={styles.scoreRow}>
          <Text style={styles.scoreLabel}>{label.replace(/_/g, " ")}</Text>
          <View style={styles.scoreTrack}>
            <View style={[styles.scoreFill, { width: `${Math.max(0, Math.min(100, Number(value) * 100))}%` }]} />
          </View>
          <Text style={styles.scoreValue}>{fmtNumber(value)}</Text>
        </View>
      ))}
    </View>
  );
}

function NewsTimelineChart({ timeline }) {
  const priceRows = timeline?.price || [];
  const markers = timeline?.markers || [];
  if (!priceRows.length) return <Text style={styles.item}>No timeline data loaded.</Text>;
  const priceValues = priceRows.map((row) => Number(row.close ?? row["adj close"] ?? row.price ?? 0));
  const priceDates = priceRows.map((row) => String(row.date || ""));
  const maxLen = Math.max(priceValues.length, 1);
  const flat = priceValues.filter((value) => Number.isFinite(value));
  const min = Math.min(...flat);
  const max = Math.max(...flat);
  const span = max - min || 1;
  const markerColor = (label) => {
    if (label === "bullish") return "#166534";
    if (label === "bearish") return "#b91c1c";
    return "#a16207";
  };
  const findMarkerIndex = (date) => {
    const idx = priceDates.indexOf(String(date));
    if (idx >= 0) return idx;
    let best = 0, bestDistance = Number.POSITIVE_INFINITY;
    const target = new Date(String(date)).getTime();
    priceDates.forEach((d, index) => {
      const distance = Math.abs(new Date(d).getTime() - target);
      if (Number.isFinite(distance) && distance < bestDistance) { bestDistance = distance; best = index; }
    });
    return best;
  };
  return (
    <View style={styles.timelineFrame}>
      <View style={styles.timelineChartWrap}>
        {(() => {
          const width = 1000, height = 126;
          const points = priceValues.map((value, idx) => {
            if (!Number.isFinite(value)) return null;
            const x = maxLen > 1 ? (idx / (maxLen - 1)) * width : 0;
            const y = height - ((value - min) / span) * height;
            return { x, y };
          }).filter(Boolean);
          return buildLinePointCloud(points, 4).map((pt, idx) => (
            <View key={`tl-${idx}`} style={[styles.chartLineDot, { left: `${(pt.x / width) * 100}%`, top: pt.y, width: 2, height: 2, borderRadius: 2, backgroundColor: THEME.text, opacity: 0.95 }]} />
          ));
        })()}
        {markers.slice(0, 18).map((marker, idx) => {
          const left = maxLen > 1 ? `${(findMarkerIndex(marker.date) / (maxLen - 1)) * 100}%` : "0%";
          const color = markerColor(marker.sentiment_label);
          return <View key={`mk-${idx}`} style={[styles.timelineMarker, { left }]}><View style={[styles.timelineDot, { backgroundColor: color }]} /><Text style={[styles.timelineMarkerText, { color }]}>{marker.sentiment_label?.slice(0, 1)?.toUpperCase() || "N"}</Text></View>;
        })}
      </View>
    </View>
  );
}

function NewsSummaryList({ summary }) {
  const groups = summary?.items || [];
  if (!groups.length) return <Text style={styles.item}>No article summary loaded.</Text>;
  return <View style={styles.newsSummaryList}>{groups.map((group) => <View key={group.date} style={styles.newsSummaryCard}><Text style={styles.analyticsHeadline}>{group.date} | {group.count} articles | avg sentiment {fmtNumber(group.avg_sentiment)}</Text><Text style={styles.item}>{group.summary || "No summary text available."}</Text></View>)}</View>;
}

function ProgressBar({ progress, label, compact = false }) {
  const pct = Math.max(6, Math.min(98, Number(progress) || 0));
  return (
    <View style={compact ? styles.progressCompactWrap : styles.progressWrap}>
      {!!label && <Text style={styles.progressLabel}>{label}</Text>}
      <View style={styles.progressTrack}><View style={[styles.progressFill, { width: `${pct}%` }]} /></View>
      <Text style={styles.progressValue}>{Math.round(pct)}%</Text>
    </View>
  );
}

function authHeaders({ apiKey, bearerToken }) {
  const headers = {};
  if (apiKey) headers["x-api-key"] = apiKey;
  if (bearerToken) headers.Authorization = `Bearer ${bearerToken}`;
  return headers;
}

async function withRetry(fn, maxRetries = 3, baseDelayMs = 500) {
  let lastError;
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try { return await fn(); } catch (e) {
      lastError = e;
      const msg = String(e?.message || "");
      if ((e.status < 500 && !msg.includes("Failed to fetch") && !msg.includes("Network request failed")) || attempt === maxRetries - 1) {
        if (msg.includes("Failed to fetch")) { const err = new Error(`Failed to fetch from ${API_BASE}. Ensure the API is running and EXPO_PUBLIC_API_BASE_URL is correct.`); err.status = e.status; throw err; }
        throw e;
      }
      await new Promise(resolve => setTimeout(resolve, baseDelayMs * Math.pow(2, attempt) + Math.random() * 100));
    }
  }
  throw lastError;
}

async function apiGet(path, auth, options = {}) {
  return withRetry(async () => {
    const res = await fetch(`${API_BASE}${path}`, { headers: { ...authHeaders(auth) }, signal: options.signal });
    if (!res.ok) { const err = new Error(`HTTP ${res.status}`); err.status = res.status; throw err; }
    return res.json();
  });
}

async function apiPost(path, payload, auth, options = {}) {
  return withRetry(async () => {
    const res = await fetch(`${API_BASE}${path}`, { method: "POST", headers: { "Content-Type": "application/json", ...authHeaders(auth) }, body: JSON.stringify(payload), signal: options.signal });
    if (!res.ok) { const err = new Error(`HTTP ${res.status}`); err.status = res.status; throw err; }
    return res.json();
  });
}

async function apiPut(path, payload, auth, options = {}) {
  return withRetry(async () => {
    const res = await fetch(`${API_BASE}${path}`, { method: "PUT", headers: { "Content-Type": "application/json", ...authHeaders(auth) }, body: JSON.stringify(payload), signal: options.signal });
    if (!res.ok) { const err = new Error(`HTTP ${res.status}`); err.status = res.status; throw err; }
    return res.json();
  });
}

async function apiDelete(path, auth, options = {}) {
  return withRetry(async () => {
    const res = await fetch(`${API_BASE}${path}`, { method: "DELETE", headers: { ...authHeaders(auth) }, signal: options.signal });
    if (!res.ok) { const err = new Error(`HTTP ${res.status}`); err.status = res.status; throw err; }
    return res.json();
  });
}

export default function App() {
  // ---- Core state ----
  const [surface, setSurface] = useState("Public");
  const [workspaceUnlocked, setWorkspaceUnlocked] = useState(false);
  const [screen, setScreen] = useState("Overview");
  const [status, setStatus] = useState("idle");
  const [apiKeyInput, setApiKeyInput] = useState(API_KEY);
  const [bearerToken, setBearerToken] = useState("");

  const [health, setHealth] = useState(null);
  const [opsReport, setOpsReport] = useState(null);
  const [envValidation, setEnvValidation] = useState(null);
  const [presets, setPresets] = useState([]);
  const [latestUniverse, setLatestUniverse] = useState([]);
  const [latestRecs, setLatestRecs] = useState([]);
  const [intents, setIntents] = useState([]);
  const [optionIdeas, setOptionIdeas] = useState([]);
  const [backtestSummary, setBacktestSummary] = useState(null);
  const [backtestEquity, setBacktestEquity] = useState([]);
  const [mcpStatus, setMcpStatus] = useState(null);
  const [mcpReadiness, setMcpReadiness] = useState(null);
  const [mcpLoginResult, setMcpLoginResult] = useState(null);
  const [mcpRuntimeConfig, setMcpRuntimeConfig] = useState(null);
  const [mcpBearerInput, setMcpBearerInput] = useState("");
  const [mcpHeaderInput, setMcpHeaderInput] = useState("Authorization");
  const [mcpConfigStatus, setMcpConfigStatus] = useState("idle");
  const [mcpOAuthStatus, setMcpOAuthStatus] = useState("idle");
  const [mcpOAuthStartResult, setMcpOAuthStartResult] = useState(null);
  const [mcpSignals, setMcpSignals] = useState(null);
  const [mcpSignalsError, setMcpSignalsError] = useState("");
  const [mcpRunbookReport, setMcpRunbookReport] = useState(null);
  const [assistantInput, setAssistantInput] = useState("");
  const [assistantResponse, setAssistantResponse] = useState(null);
  const [assistantConversation, setAssistantConversation] = useState([]);
  const [assistantActionResult, setAssistantActionResult] = useState(null);
  const [assistantAutoRunGet, setAssistantAutoRunGet] = useState(true);
  const [assistantAutoResults, setAssistantAutoResults] = useState([]);
  const [assistantAutoSummary, setAssistantAutoSummary] = useState(null);
  const [assistantDryRun, setAssistantDryRun] = useState(true);
  const [pendingAction, setPendingAction] = useState(null);
  const [firebaseAuthCheck, setFirebaseAuthCheck] = useState(null);
  const [googleSignInStatus, setGoogleSignInStatus] = useState("idle");
  const [userSettingsText, setUserSettingsText] = useState('{"watchlist": ["AAPL", "MSFT"], "risk_profile": "balanced"}');
  const [userSettingsData, setUserSettingsData] = useState(null);
  const [settingsSaveStatus, setSettingsSaveStatus] = useState("idle");

  const [ticker, setTicker] = useState("AAPL");
  const [notional, setNotional] = useState("250");
  const [horizon, setHorizon] = useState("1m");
  const [dbPath, setDbPath] = useState("sqlite:///quantflow.db");

  // ---- Scanner state ----
  const [scanParallel, setScanParallel] = useState(true);
  const [scanWorkers, setScanWorkers] = useState("4");
  const [scanResult, setScanResult] = useState(null);
  const [scanJobId, setScanJobId] = useState("");
  const [scanJobStatus, setScanJobStatus] = useState("idle");
  const [scanProgressPct, setScanProgressPct] = useState(0);
  const [scanEtaSeconds, setScanEtaSeconds] = useState(null);
  const [scanElapsedSeconds, setScanElapsedSeconds] = useState(null);
  const [scanCurrentPreset, setScanCurrentPreset] = useState("");
  const [scanFailedPresets, setScanFailedPresets] = useState([]);
  const [scanPresetsCompleted, setScanPresetsCompleted] = useState(0);
  const [scanPresetCount, setScanPresetCount] = useState(0);
  const [scanHistory, setScanHistory] = useState([]);
  const [scanHistoryFilter, setScanHistoryFilter] = useState("all");
  const [scanStartedAt, setScanStartedAt] = useState("");
  const [scanFinishedAt, setScanFinishedAt] = useState("");
  const [scanUniverseRefreshedAt, setScanUniverseRefreshedAt] = useState("");
  const [scanLastMessage, setScanLastMessage] = useState("");
  const [presetDetails, setPresetDetails] = useState(null);
  const [selectedPresetsText, setSelectedPresetsText] = useState("weekly_momo,weekly_bear");
  const [presetSearch, setPresetSearch] = useState("");
  const [universePresetFilter, setUniversePresetFilter] = useState("");
  const [universeLimit, setUniverseLimit] = useState("24");
  const [presetEditName, setPresetEditName] = useState("");
  const [presetEditContent, setPresetEditContent] = useState("");
  const [presetSaveStatus, setPresetSaveStatus] = useState("idle");
  const [rememberCredentials, setRememberCredentials] = useState(false);

  const [recommendTickersText, setRecommendTickersText] = useState("AAPL,MSFT");
  const [recommendSave, setRecommendSave] = useState(true);
  const [recommendResult, setRecommendResult] = useState(null);
  const [optionsBias, setOptionsBias] = useState("long");
  const [optionsBudget, setOptionsBudget] = useState("300");
  const [optionsResultMeta, setOptionsResultMeta] = useState(null);

  // ---- Backtest state ----
  const [backtestStart, setBacktestStart] = useState("2020-01-01");
  const [backtestEnd, setBacktestEnd] = useState("");
  const [backtestEntryRsi, setBacktestEntryRsi] = useState("50");
  const [backtestMaxHoldDays, setBacktestMaxHoldDays] = useState("7");
  const [backtestStopLossPct, setBacktestStopLossPct] = useState("0");
  const [backtestMaFilter, setBacktestMaFilter] = useState("sma20");
  const [backtestTakeProfitPct, setBacktestTakeProfitPct] = useState("0");
  const [backtestMaTrendFilter, setBacktestMaTrendFilter] = useState("none");
  const [backtestStep, setBacktestStep] = useState(-1);

  // ---- Analytics state ----
  const [analyticsTickersText, setAnalyticsTickersText] = useState("AAPL,MSFT,NVDA,AMZN");
  const [analyticsPeriod, setAnalyticsPeriod] = useState("1y");
  const [analyticsInterval, setAnalyticsInterval] = useState("1d");
  const [analyticsBins, setAnalyticsBins] = useState("24");
  const [analyticsHorizon, setAnalyticsHorizon] = useState("30");
  const [compareBase, setCompareBase] = useState("100");
  const [seasonalityYears, setSeasonalityYears] = useState("10");
  const [seasonalitySectorOverride, setSeasonalitySectorOverride] = useState("");
  const [seasonalityEtfOverride, setSeasonalityEtfOverride] = useState("");
  const [analyticsRanked, setAnalyticsRanked] = useState([]);
  const [analyticsLatest, setAnalyticsLatest] = useState([]);
  const [analyticsBatchId, setAnalyticsBatchId] = useState("");
  const [analyticsSaveResult, setAnalyticsSaveResult] = useState(null);
  const [signalsData, setSignalsData] = useState(null);
  const [signalsLoading, setSignalsLoading] = useState(false);
  const [signalsTimeframe, setSignalsTimeframe] = useState("5Min");
  const [trainingFeatures, setTrainingFeatures] = useState([]);
  const [newsSummary, setNewsSummary] = useState(null);
  const [newsTimeline, setNewsTimeline] = useState(null);
  const [newsDays, setNewsDays] = useState("30");
  const [newsSentiment, setNewsSentiment] = useState("all");
  const [indicatorChart, setIndicatorChart] = useState([]);
  const [seasonalityChart, setSeasonalityChart] = useState([]);
  const [distributionChart, setDistributionChart] = useState(null);
  const [forecastChart, setForecastChart] = useState(null);
  const [performanceCompareChart, setPerformanceCompareChart] = useState(null);
  const [seasonalityCompareChart, setSeasonalityCompareChart] = useState(null);
  const [marketSeriesCache, setMarketSeriesCache] = useState(null);

  // ---- Execution state ----
  const [executionAction, setExecutionAction] = useState("buy");
  const [executionQty, setExecutionQty] = useState("1");
  const [executionOrdersToday, setExecutionOrdersToday] = useState("0");
  const [executionPositionAfter, setExecutionPositionAfter] = useState("250");
  const [executionBrokerMode, setExecutionBrokerMode] = useState("robinhood_mcp");
  const [executionAuto, setExecutionAuto] = useState(false);
  const [executionMaxDaily, setExecutionMaxDaily] = useState("5000");
  const [executionMaxOrders, setExecutionMaxOrders] = useState("20");
  const [executionMaxPosition, setExecutionMaxPosition] = useState("2000");
  const [executionResult, setExecutionResult] = useState(null);

  // ---- Portfolio tab ----
  const [portfolioPositions, setPortfolioPositions] = useState([]);
  const [portfolioEquity, setPortfolioEquity] = useState([]);
  const [portfolioAccount, setPortfolioAccount] = useState(null);
  const [portfolioMetrics, setPortfolioMetrics] = useState(null);

  // ---- Trade tab ----
  const [tradeTicker, setTradeTicker] = useState("AAPL");
  const [tradeAction, setTradeAction] = useState("buy");
  const [tradeQty, setTradeQty] = useState("10");
  const [tradeStopLoss, setTradeStopLoss] = useState("");
  const [tradeTakeProfit, setTradeTakeProfit] = useState("");
  const [tradeOrderHistory, setTradeOrderHistory] = useState([]);
  const [tradeActivePositions, setTradeActivePositions] = useState([]);

  // ---- Market tab ----
  const [marketTicker, setMarketTicker] = useState("AAPL");
  const [marketTimeframe, setMarketTimeframe] = useState("1d");
  const [marketIndicators, setMarketIndicators] = useState([]);
  const [marketForecast, setMarketForecast] = useState(null);

  // ---- Settings tab ----
  const [settingsAlpacaKey, setSettingsAlpacaKey] = useState("");
  const [settingsAlpacaSecret, setSettingsAlpacaSecret] = useState("");
  const [settingsAlpacaStatus, setSettingsAlpacaStatus] = useState("idle");
  const [settingsLlmKey, setSettingsLlmKey] = useState("");
  const [settingsLlmEndpoint, setSettingsLlmEndpoint] = useState("");
  const [settingsRiskStopLoss, setSettingsRiskStopLoss] = useState("5");
  const [settingsRiskMaxPos, setSettingsRiskMaxPos] = useState("20");
  const [settingsDailyLossLimit, setSettingsDailyLossLimit] = useState("2000");
  const [settingsSaveMsg, setSettingsSaveMsg] = useState("");

  // ---- Assistant ----
  const [assistantPromptsExpanded, setAssistantPromptsExpanded] = useState(false);

  const [inFlightOps, setInFlightOps] = useState(0);
  const [activeTaskLabel, setActiveTaskLabel] = useState("");
  const [progressTick, setProgressTick] = useState(Date.now());

  const authState = useMemo(() => ({ apiKey: apiKeyInput.trim(), bearerToken: bearerToken.trim() }), [apiKeyInput, bearerToken]);
  const baseLabel = useMemo(() => `API: ${API_BASE}`, []);
  const selectedPresets = useMemo(() => parseCsv(selectedPresetsText), [selectedPresetsText]);
  const availablePresets = useMemo(() => {
    const items = presetDetails?.items || presets || [];
    const q = presetSearch.trim().toLowerCase();
    return q ? items.filter((name) => String(name).toLowerCase().includes(q)) : items;
  }, [presetDetails, presets, presetSearch]);
  const filteredUniverse = useMemo(() => {
    const q = universePresetFilter.trim().toLowerCase();
    const lim = Math.max(1, Math.min(200, parseNumber(universeLimit, 24)));
    const rows = q ? (latestUniverse || []).filter((r) => String(r.preset || "").toLowerCase().includes(q)) : (latestUniverse || []);
    return [...rows].sort((a, b) => String(b.created_at || "").localeCompare(String(a.created_at || ""))).slice(0, lim);
  }, [latestUniverse, universePresetFilter, universeLimit]);
  const indicatorTail = useMemo(() => (indicatorChart || []).slice(-90), [indicatorChart]);
  const forecastHistory = useMemo(() => (forecastChart?.history || []).slice(-30), [forecastChart]);
  const forecastForward = useMemo(() => forecastChart?.forecast || [], [forecastChart]);
  const seasonalityTail = useMemo(() => {
    const rows = (seasonalityChart || []).map((row, idx) => ({ ...row, _idx: idx }));
    rows.sort((a, b) => { const ad = Number(a?.doy ?? a?.day_of_year ?? NaN); const bd = Number(b?.doy ?? b?.day_of_year ?? NaN); if (Number.isFinite(ad) && Number.isFinite(bd)) return ad - bd; if (Number.isFinite(ad)) return -1; if (Number.isFinite(bd)) return 1; return Number(a._idx || 0) - Number(b._idx || 0); });
    return rows;
  }, [seasonalityChart]);
  const scanHistoryRowsSeries = useMemo(() => (scanHistory || []).slice(0, 20).map((row) => Number(row.rows_saved || 0)).reverse(), [scanHistory]);
  const scanHistoryDurationSeries = useMemo(() => (scanHistory || []).slice(0, 20).map((row) => Number(row.elapsed_seconds || 0)).reverse(), [scanHistory]);
  const scanHistoryFiltered = useMemo(() => {
    return scanHistoryFilter === "all" ? (scanHistory || []) : (scanHistory || []).filter((row) => String(row.status || "").toLowerCase() === scanHistoryFilter);
  }, [scanHistory, scanHistoryFilter]);

  const isWeb = typeof window !== "undefined";
  const scanPollRef = useRef(null);
  const progressStartedAtRef = useRef(0);
  const assistantRunRef = useRef(0);

  // ---- Progress tracking ----
  useEffect(() => { if (inFlightOps > 0 && !progressStartedAtRef.current) progressStartedAtRef.current = Date.now(); if (!inFlightOps) progressStartedAtRef.current = 0; }, [inFlightOps]);
  useEffect(() => { if (!inFlightOps) return; const id = setInterval(() => setProgressTick(Date.now()), 250); return () => clearInterval(id); }, [inFlightOps]);
  const progressValue = useMemo(() => { if (!inFlightOps) return 0; const elapsedMs = Math.max(0, progressTick - (progressStartedAtRef.current || progressTick)); return Math.min(92, 12 + 80 * (1 - Math.exp(-elapsedMs / 2200))); }, [inFlightOps, progressTick]);

  const scannerBusy = inFlightOps > 0 && status.includes("scan");
  const recommendBusy = inFlightOps > 0 && status.includes("recommend");
  const analyticsBusy = inFlightOps > 0 && status.includes("analytics");
  const optionsBusy = inFlightOps > 0 && status.includes("options");
  const backtestBusy = inFlightOps > 0 && status.includes("backtest");
  const executionBusy = inFlightOps > 0 && status.includes("propos");
  const assistantBusy = inFlightOps > 0 && (status.includes("assistant") || status.includes("run "));
  const authBusy = inFlightOps > 0 && (settingsSaveStatus === "loading" || settingsSaveStatus === "saving" || status.includes("auth") || status.includes("identity"));
  const scanIsActive = ["queued", "running", "cancel_requested"].includes(scanJobStatus);

  const withProgress = async (label, action) => { setActiveTaskLabel(label); setInFlightOps((v) => v + 1); try { return await action(); } finally { setInFlightOps((v) => Math.max(0, v - 1)); } };

  const stopScanPolling = () => { if (scanPollRef.current) { clearInterval(scanPollRef.current); scanPollRef.current = null; } };

  const applyScanStatus = (snapshot) => {
    if (!snapshot) return;
    setScanJobId(snapshot.job_id || ""); setScanJobStatus(String(snapshot.status || "idle")); setScanProgressPct(Number(snapshot.progress_pct || 0));
    setScanEtaSeconds(snapshot.eta_seconds); setScanElapsedSeconds(snapshot.elapsed_seconds); setScanCurrentPreset(snapshot.current_preset || "");
    setScanFailedPresets(snapshot.failed_presets || []); setScanPresetsCompleted(Number(snapshot.presets_completed || 0)); setScanPresetCount(Number(snapshot.preset_count || 0));
    setScanStartedAt(snapshot.started_at || ""); setScanFinishedAt(snapshot.finished_at || ""); setScanLastMessage(snapshot.message || "");
    if (["completed", "failed", "canceled"].includes(String(snapshot.status || ""))) { stopScanPolling(); if (snapshot.status === "completed" || snapshot.status === "canceled") loadOverview(); loadScanHistory(scanHistoryFilter);
      if (snapshot.status === "completed") { setStatus("scan completed"); setScanResult({ selected_presets: snapshot.selected_presets || [], preset_count: Number(snapshot.preset_count || 0), rows_saved: Number(snapshot.rows_saved || 0), failed_presets: snapshot.failed_presets || [] }); }
      else if (snapshot.status === "canceled") setStatus("scan canceled"); else setStatus("scan failed"); }
  };

  const startScanPolling = (jobId) => { stopScanPolling(); scanPollRef.current = setInterval(async () => { try { applyScanStatus(await apiGet(`/pipeline/scan/status/${encodeURIComponent(jobId)}`, authState)); } catch (e) { setScanLastMessage(`Status polling error: ${e.message}`); } }, 1200); };

  // ---- URL / localStorage sync ----
  const syncUrlState = (nextSurface, nextScreen) => { if (!isWeb) return; try { const params = new URLSearchParams(window.location.search || ""); params.set("view", String(nextSurface).toLowerCase()); if (nextSurface === "Workspace") params.set("tab", String(nextScreen)); else params.delete("tab"); window.history.replaceState({}, "", `${window.location.pathname}${params.toString() ? `?${params}` : ""}`); } catch (e) {} };

  useEffect(() => {
    if (isWeb) { try { const params = new URLSearchParams(window.location.search || ""); const view = params.get("view"), tab = params.get("tab"), oauthResult = params.get("mcp_oauth"); if (view === "public") setSurface("Public"); else if (view === "workspace") setSurface("Workspace"); if (tab && SCREENS.includes(tab)) setScreen(tab); if (oauthResult === "success") { setMcpOAuthStatus("completed"); setStatus("mcp oauth completed"); loadMcpStatus(); params.delete("mcp_oauth"); window.history.replaceState({}, "", `${window.location.pathname}${params.toString() ? `?${params}` : ""}`); } } catch (e) {} }
    (async () => {
      try { const stored = await AsyncStorage.getItem("quantflow_credentials"); if (stored) { const { apiKey, bearerToken: token, rememberMe } = JSON.parse(stored); if (apiKey) setApiKeyInput(apiKey); if (token) setBearerToken(token); setRememberCredentials(rememberMe || false); } } catch (e) {}
      // Restore assistant conversation from localStorage
      try { const savedChat = await AsyncStorage.getItem("quantflow_chat"); if (savedChat) { const parsed = JSON.parse(savedChat); setAssistantConversation(Array.isArray(parsed) ? parsed.slice(-20) : []); } } catch (e) {}
    })();
    loadOverview();
    return () => stopScanPolling();
  }, []);

  useEffect(() => { syncUrlState(surface, screen); }, [surface, screen]);
  useEffect(() => { loadScanHistory(scanHistoryFilter); }, [scanHistoryFilter]);

  // Persist assistant conversation to localStorage when it changes
  useEffect(() => {
    try { AsyncStorage.setItem("quantflow_chat", JSON.stringify(assistantConversation.slice(-20))); } catch (e) {}
  }, [assistantConversation]);

  const saveCredentials = async () => { try { if (rememberCredentials) await AsyncStorage.setItem("quantflow_credentials", JSON.stringify({ apiKey: apiKeyInput.trim(), bearerToken: bearerToken.trim(), rememberMe: true })); else await AsyncStorage.removeItem("quantflow_credentials"); } catch (e) {} };
  const logoutWorkspace = async () => { await AsyncStorage.removeItem("quantflow_credentials"); setApiKeyInput(API_KEY); setBearerToken(""); setRememberCredentials(false); setWorkspaceUnlocked(false); setFirebaseAuthCheck(null); setSurface("Public"); setScreen("Overview"); setStatus("logged out"); };

  // ====== API CALLS (existing, kept intact) ======

  const loadMcpStatus = async (options = {}) => { await withProgress("Checking MCP", async () => { try { const [data, readiness, signals] = await Promise.all([apiGet("/broker/mcp/status", authState, options), apiGet("/broker/mcp/readiness", authState, options), apiGet("/portfolio/signals?broker_mode=robinhood_mcp", authState, options).catch((e) => ({ error: e.message }))]); setMcpStatus(data || null); setMcpReadiness(readiness || null); if (signals?.error) { setMcpSignals(null); setMcpSignalsError(String(signals.error)); } else { setMcpSignals(signals || null); setMcpSignalsError(""); } } catch (e) { setMcpStatus({ connected: false, authenticated: false, error: e.message }); setMcpReadiness(null); setMcpSignals(null); setMcpSignalsError(String(e.message)); } }); };

  const connectMcp = async () => { setStatus("connecting mcp"); await withProgress("Connecting Robinhood MCP", async () => { try { const [statusData, readinessData, signalsData] = await Promise.all([apiGet("/broker/mcp/status", authState), apiGet("/broker/mcp/readiness", authState), apiGet("/portfolio/signals?broker_mode=robinhood_mcp", authState).catch((e) => ({ error: e.message }))]); setMcpStatus(statusData || null); setMcpReadiness(readinessData || null); if (signalsData?.error) { setMcpSignals(null); setMcpSignalsError(String(signalsData.error)); } else { setMcpSignals(signalsData || null); setMcpSignalsError(""); } setMcpLoginResult({ ok: Boolean(statusData?.authenticated), message: statusData?.authenticated ? "Connected." : statusData?.connected ? "Transport reachable. Complete Robinhood auth." : "MCP not reachable." }); setStatus("ready"); } catch (e) { setMcpLoginResult({ ok: false, error: e.message }); setStatus(`error: ${e.message}`); } }); };

  const loadMcpRuntimeConfig = async () => { setMcpConfigStatus("loading"); await withProgress("Loading MCP runtime config", async () => { try { const data = await apiGet("/broker/mcp/config", authState); setMcpRuntimeConfig(data || null); setMcpHeaderInput(String(data?.runtime?.auth_header || "Authorization")); setMcpConfigStatus("ready"); } catch (e) { setMcpConfigStatus(`error: ${e.message}`); } }); };

  const saveMcpRuntimeConfig = async () => { setMcpConfigStatus("saving"); await withProgress("Saving MCP runtime config", async () => { try { const tokenRaw = mcpBearerInput.trim(); if (!tokenRaw) throw new Error("MCP bearer token is required"); const bearer = tokenRaw.toLowerCase().startsWith("bearer ") ? tokenRaw : `Bearer ${tokenRaw}`; await apiPut("/broker/mcp/config", { auth_header: (mcpHeaderInput || "Authorization").trim() || "Authorization", bearer_token: bearer }, authState); setMcpConfigStatus("saved"); setMcpBearerInput(""); await loadMcpStatus(); } catch (e) { setMcpConfigStatus(`error: ${e.message}`); } }); };

  const clearMcpRuntimeConfig = async () => { setMcpConfigStatus("clearing"); await withProgress("Clearing MCP runtime config", async () => { try { await apiDelete("/broker/mcp/config", authState); setMcpRuntimeConfig(null); setMcpConfigStatus("cleared"); setMcpBearerInput(""); await loadMcpStatus(); } catch (e) { setMcpConfigStatus(`error: ${e.message}`); } }); };

  const startMcpBackendOAuth = async () => { setMcpOAuthStatus("starting"); await withProgress("Starting backend MCP OAuth", async () => { try { const continueUrl = isWeb ? `${window.location.origin}${window.location.pathname}?view=workspace&tab=Auth` : ""; const data = await apiPost("/broker/mcp/oauth/start", { continue_url: continueUrl }, authState); setMcpOAuthStartResult(data || null); setMcpOAuthStatus("ready"); const authUrl = String(data?.authorization_url || "").trim(); if (authUrl && isWeb) { window.open(authUrl, "_blank", "noopener,noreferrer"); setStatus("mcp oauth window opened"); } else if (authUrl) setStatus("open authorization_url from response to continue oauth"); } catch (e) { setMcpOAuthStatus(`error: ${e.message}`); } }); };

  const loadScanHistory = async (statusFilter = "all") => { try { const q = statusFilter !== "all" ? `&status=${encodeURIComponent(statusFilter)}` : ""; setScanHistory((await apiGet(`/pipeline/scan/history?limit=24${q}`, authState)).items || []); } catch (e) { setScanLastMessage(`Scan history unavailable: ${e.message}`); } };

  const loadOverview = async (options = {}) => { setStatus("loading"); await withProgress("Loading overview", async () => { try { const [h, p, pd, i, u, r, o, envv, history] = await Promise.all([apiGet("/health", authState, options), apiGet("/scanner/presets", authState, options), apiGet("/scanner/presets/details", authState, options).catch(() => null), apiGet("/execution/intents?limit=10", authState, options), apiGet("/scanner/latest-universe", authState, options), apiGet("/recommend/latest", authState, options), apiGet("/ops/latest-report", authState, options).catch(() => null), apiGet("/ops/env/validate", authState, options).catch(() => null), apiGet(`/pipeline/scan/history?limit=24${scanHistoryFilter !== "all" ? `&status=${encodeURIComponent(scanHistoryFilter)}` : ""}`, authState, options).catch(() => ({ items: [] }))]); setHealth(h); setOpsReport(o); setEnvValidation(envv); setPresets(p.items || []); setPresetDetails(pd || null); setIntents(i.items || []); setLatestUniverse(u.items || []); setScanUniverseRefreshedAt(new Date().toISOString()); setLatestRecs(r.items || []); setScanHistory(history?.items || []); await loadMcpStatus(options); setStatus("ready"); } catch (e) { setStatus(isAbortError(e) ? "canceled" : `error: ${e.message}`); } }); };

  const togglePreset = (name) => { const set = new Set(selectedPresets); if (set.has(name)) set.delete(name); else set.add(name); setSelectedPresetsText(Array.from(set).join(",")); };
  const selectPresetCategory = (name) => setSelectedPresetsText((presetDetails?.categories?.[name] || []).join(","));

  const signInWithGoogleWeb = async () => { if (!isWeb) { setGoogleSignInStatus("Google popup login is available on web builds only."); return; } setGoogleSignInStatus("starting"); await withProgress("Signing in", async () => { try { if (!FIREBASE_WEB_CONFIG.apiKey || !FIREBASE_WEB_CONFIG.authDomain || !FIREBASE_WEB_CONFIG.projectId || !FIREBASE_WEB_CONFIG.appId) throw new Error("Missing EXPO_PUBLIC_FIREBASE_* config variables for web login"); const [{ initializeApp, getApps }, { getAuth, GoogleAuthProvider, signInWithPopup }] = await Promise.all([import("firebase/app"), import("firebase/auth")]); const app = getApps().length ? getApps()[0] : initializeApp(FIREBASE_WEB_CONFIG); const result = await signInWithPopup(getAuth(app), new GoogleAuthProvider()); const idToken = await result.user.getIdToken(); setBearerToken(idToken); setGoogleSignInStatus(`signed in as ${result.user.email || result.user.uid}`); await checkFirebaseIdentity(); } catch (e) { setGoogleSignInStatus(`error: ${e.message}`); } }); };

  const checkFirebaseIdentity = async () => { setStatus("checking identity"); await withProgress("Checking identity", async () => { try { setFirebaseAuthCheck(await apiGet("/auth/me", authState)); setStatus("ready"); } catch (e) { setFirebaseAuthCheck({ authenticated: false, error: e.message }); setStatus(`error: ${e.message}`); } }); };

  const unlockWorkspace = async () => { setStatus("auth check"); await withProgress("Unlocking workspace", async () => { try { setFirebaseAuthCheck(await apiGet("/auth/me", authState)); setWorkspaceUnlocked(true); setSurface("Workspace"); await saveCredentials(); setStatus("ready"); } catch (e) { setFirebaseAuthCheck({ authenticated: false, error: e.message }); setWorkspaceUnlocked(false); setStatus(`error: ${e.message}`); } }); };

  const unlockWorkspaceLocal = async () => { setFirebaseAuthCheck({ mode: "local-dev", authenticated: true, note: "Workspace unlocked without token because API auth is disabled." }); setWorkspaceUnlocked(true); setSurface("Workspace"); await saveCredentials(); setStatus("ready"); };

  const loadUserSettings = async () => { setSettingsSaveStatus("loading"); await withProgress("Loading user settings", async () => { try { const data = await apiGet("/user/settings", authState); setUserSettingsData(data); setUserSettingsText(JSON.stringify(data.settings || {}, null, 2)); setSettingsSaveStatus("ready"); } catch (e) { setUserSettingsData({ ok: false, error: e.message }); setSettingsSaveStatus(`error: ${e.message}`); } }); };

  const saveUserSettings = async () => { setSettingsSaveStatus("saving"); await withProgress("Saving user settings", async () => { try { const data = await apiPut("/user/settings", { settings: JSON.parse(userSettingsText || "{}") }, authState); setUserSettingsData(data); setSettingsSaveStatus("saved"); } catch (e) { setSettingsSaveStatus(`error: ${e.message}`); } }); };

  const runScanPipeline = async () => { stopScanPolling(); setScanProgressPct(0); setScanEtaSeconds(null); setScanElapsedSeconds(null); setScanCurrentPreset(""); setScanFailedPresets([]); setScanPresetsCompleted(0); setScanPresetCount(0); setScanFinishedAt(""); setScanLastMessage("Scan queued"); setStatus("scan running"); await withProgress("Running scan", async () => { try { const start = await apiPost("/pipeline/scan/start", { db: dbPath, parallel: scanParallel, max_workers: Math.max(1, Math.min(16, parseNumber(scanWorkers, 4))), presets: selectedPresets.length ? selectedPresets : null }, authState); applyScanStatus(start); startScanPolling(start.job_id); setStatus("scan running"); } catch (e) { stopScanPolling(); setScanLastMessage(`Scan failed: ${e.message}`); setScanFinishedAt(new Date().toISOString()); setStatus(`error: ${e.message}`); } }); };

  const cancelScanPipeline = () => { if (!scanJobId || !scanIsActive) return; setStatus("scan cancel requested"); apiPost(`/pipeline/scan/cancel/${encodeURIComponent(scanJobId)}`, {}, authState).then((snap) => applyScanStatus(snap)).catch((e) => setScanLastMessage(`Cancel failed: ${e.message}`)); };

  const runRecommendPipeline = async () => { setStatus("recommend running"); await withProgress("Running recommendations", async () => { try { setRecommendResult(await apiPost("/pipeline/recommend", { db: dbPath, tickers: parseTickers(recommendTickersText), save: recommendSave }, authState) || null); await loadOverview(); } catch (e) { setStatus(`error: ${e.message}`); } }); };

  const loadBacktest = async () => { setStatus("running backtest"); setBacktestStep(-1); await withProgress("Running backtest", async () => { try { const endQ = backtestEnd.trim() ? `&end=${encodeURIComponent(backtestEnd.trim())}` : ""; const tickerSafe = encodeURIComponent(ticker); const periodQuery = encodeURIComponent(analyticsPeriod || "2y"); const intervalQuery = encodeURIComponent(analyticsInterval || "1d"); const [data, indicators, seasonality, summary, timeline] = await Promise.all([apiGet(`/backtest/short-term?ticker=${tickerSafe}&start=${encodeURIComponent(backtestStart)}${endQ}&entry_rsi_threshold=${encodeURIComponent(parseNumber(backtestEntryRsi, 50))}&max_hold_days=${encodeURIComponent(Math.max(1, parseNumber(backtestMaxHoldDays, 7)))}&stop_loss_pct=${encodeURIComponent(Math.max(0, parseNumber(backtestStopLossPct, 0)))}&ma_filter=${encodeURIComponent((backtestMaFilter || "sma20").trim().toLowerCase())}&take_profit_pct=${encodeURIComponent(Math.max(0, parseNumber(backtestTakeProfitPct, 0)))}&ma_trend_filter=${encodeURIComponent((backtestMaTrendFilter || "none").trim().toLowerCase())}`, authState), apiGet(`/charts/indicators?ticker=${tickerSafe}&period=${periodQuery}&interval=${intervalQuery}`, authState).catch(() => null), apiGet(`/charts/seasonality?ticker=${tickerSafe}&years=10`, authState).catch(() => null), apiGet(`/news/summary?ticker=${tickerSafe}&db=${encodeURIComponent(dbPath)}&days=${encodeURIComponent(parseNumber(newsDays, 30))}&sentiment=${encodeURIComponent(newsSentiment)}&max_groups=8`, authState).catch(() => null), apiGet(`/news/timeline?ticker=${tickerSafe}&db=${encodeURIComponent(dbPath)}&days=${encodeURIComponent(parseNumber(newsDays, 30))}&period=${periodQuery}&interval=${intervalQuery}&sentiment=${encodeURIComponent(newsSentiment)}&max_articles=24`, authState).catch(() => null)]); setBacktestSummary(data.summary || null); setBacktestEquity(data.equity || []); if (indicators?.items) setIndicatorChart(indicators.items); if (seasonality?.items) setSeasonalityChart(seasonality.items); if (summary) setNewsSummary(summary); if (timeline) setNewsTimeline(timeline); setStatus("ready"); } catch (e) { setStatus(`error: ${e.message}`); } }); };

  const loadPortfolio = async () => { setStatus("loading portfolio"); await withProgress("Loading portfolio", async () => { try { const [statusData, signalsData, positionData] = await Promise.all([apiGet("/broker/mcp/status", authState).catch(() => null), apiGet("/portfolio/signals?broker_mode=robinhood_mcp", authState).catch(() => null), apiGet("/portfolio/positions", authState).catch(() => null)]); if (statusData?.account) setPortfolioAccount(statusData.account); if (signalsData?.signals) { setMcpSignals(signalsData); const metrics = calcPortfolioMetrics(signalsData); setPortfolioMetrics(metrics); } if (positionData?.positions) setPortfolioPositions(positionData.positions); if (positionData?.equity_curve) setPortfolioEquity(positionData.equity_curve); setStatus("ready"); } catch (e) { setStatus(`error: ${e.message}`); } }); };

  const loadTradeData = async () => { setStatus("loading trade data"); await withProgress("Loading trade data", async () => { try { const [orders, positions] = await Promise.all([apiGet("/execution/intents?limit=30", authState).catch(() => ({ items: [] })), apiGet("/portfolio/positions", authState).catch(() => ({ positions: [] }))]); setTradeOrderHistory(orders.items || []); setTradeActivePositions(positions.positions || []); setStatus("ready"); } catch (e) { setStatus(`error: ${e.message}`); } }); };

  const loadMarketData = async () => { setStatus("loading market data"); await withProgress("Loading market data", async () => { try { const tickerSafe = encodeURIComponent(marketTicker); const [indicators, forecast] = await Promise.all([apiGet(`/charts/indicators?ticker=${tickerSafe}&period=1y&interval=${encodeURIComponent(marketTimeframe)}`, authState).catch(() => null), apiGet(`/charts/forecast?ticker=${tickerSafe}&period=1y&interval=${encodeURIComponent(marketTimeframe)}&horizon=21`, authState).catch(() => null)]); if (indicators?.items) setMarketIndicators(indicators.items); if (forecast) setMarketForecast(forecast); setStatus("ready"); } catch (e) { setStatus(`error: ${e.message}`); } }); };

  const testAlpacaConnection = async () => { setSettingsAlpacaStatus("testing"); await withProgress("Testing Alpaca", async () => { try { const data = await apiPost("/settings/test-alpaca", { key: settingsAlpacaKey, secret: settingsAlpacaSecret }, authState); setSettingsAlpacaStatus(data?.ok ? "connected" : `failed: ${data?.error || "unknown"}`); } catch (e) { setSettingsAlpacaStatus(`error: ${e.message}`); } }); };

  const saveAppSettings = async () => { setSettingsSaveMsg("saving..."); await withProgress("Saving settings", async () => { try { await apiPut("/settings/save", { risk_stop_loss_pct: parseNumber(settingsRiskStopLoss, 5), risk_max_position_pct: parseNumber(settingsRiskMaxPos, 20), daily_loss_limit: parseNumber(settingsDailyLossLimit, 2000), llm_key: settingsLlmKey, llm_endpoint: settingsLlmEndpoint }, authState); setSettingsSaveMsg("Settings saved successfully"); setTimeout(() => setSettingsSaveMsg(""), 3000); } catch (e) { setSettingsSaveMsg(`Error: ${e.message}`); } }); };

  const executePaperTrade = async () => { setStatus("placing paper trade"); await withProgress("Paper trade", async () => { try { const payload = { ticker: tradeTicker.toUpperCase(), action: tradeAction, qty: parseNumber(tradeQty, 10) }; if (tradeStopLoss.trim()) payload.stop_loss = parseNumber(tradeStopLoss, 0); if (tradeTakeProfit.trim()) payload.take_profit = parseNumber(tradeTakeProfit, 0); await apiPost("/trade/paper", payload, authState); await loadTradeData(); setStatus("trade placed"); } catch (e) { setStatus(`error: ${e.message}`); } }); };

  // ====== Existing API calls continued ======
  const loadOptions = async () => { setStatus("loading options"); await withProgress("Loading options", async () => { try { const data = await apiGet(`/options/ideas?ticker=${encodeURIComponent(ticker)}&horizon=${encodeURIComponent(horizon)}&bias=${encodeURIComponent(optionsBias)}&budget=${encodeURIComponent(parseNumber(optionsBudget, 300))}`, authState); setOptionIdeas(data.items || []); setOptionsResultMeta({ count: data.count || 0, horizon, bias: optionsBias, budget: parseNumber(optionsBudget, 300) }); setStatus("ready"); } catch (e) { setStatus(`error: ${e.message}`); } }); };

  const cacheMarketSeries = async () => { setStatus("caching market series"); await withProgress("Caching yfinance series", async () => { try { const data = await apiPost(`/market/series/cache?ticker=${encodeURIComponent(ticker)}&period=${encodeURIComponent(analyticsPeriod)}&interval=${encodeURIComponent(analyticsInterval)}`, {}, authState); setMarketSeriesCache(data || null); if (data?.items?.length) setIndicatorChart(data.items); setStatus("ready"); } catch (e) { setStatus(`error: ${e.message}`); } }); };

  const loadAnalytics = async () => { setStatus("loading analytics"); await withProgress("Loading analytics", async () => { try { const tickerList = parseTickers(analyticsTickersText) || [ticker]; const rankQuery = encodeURIComponent(tickerList.join(",")); const periodQuery = encodeURIComponent(analyticsPeriod); const intervalQuery = encodeURIComponent(analyticsInterval); const [ranked, indicators, seasonality, distribution, forecast, summary, timeline, performanceCompare, seasonalityCompare] = await Promise.all([apiGet(`/recommend/analyze?tickers=${rankQuery}&period=${periodQuery}&interval=${intervalQuery}&limit=${Math.min(12, tickerList.length)}`, authState), apiGet(`/charts/indicators?ticker=${encodeURIComponent(ticker)}&period=${periodQuery}&interval=${intervalQuery}`, authState), apiGet(`/charts/seasonality?ticker=${encodeURIComponent(ticker)}&years=10`, authState), apiGet(`/charts/price-distribution?ticker=${encodeURIComponent(ticker)}&period=${periodQuery}&interval=${intervalQuery}&bins=${encodeURIComponent(parseNumber(analyticsBins, 24))}`, authState), apiGet(`/charts/forecast?ticker=${encodeURIComponent(ticker)}&period=${periodQuery}&interval=${intervalQuery}&horizon=${encodeURIComponent(parseNumber(analyticsHorizon, 30))}`, authState), apiGet(`/news/summary?ticker=${encodeURIComponent(ticker)}&db=${encodeURIComponent(dbPath)}&days=${encodeURIComponent(parseNumber(newsDays, 30))}&sentiment=${encodeURIComponent(newsSentiment)}&max_groups=8`, authState), apiGet(`/news/timeline?ticker=${encodeURIComponent(ticker)}&db=${encodeURIComponent(dbPath)}&days=${encodeURIComponent(parseNumber(newsDays, 30))}&period=${periodQuery}&interval=${intervalQuery}&sentiment=${encodeURIComponent(newsSentiment)}&max_articles=24`, authState), apiGet(`/charts/performance-compare?tickers=${rankQuery}&period=${periodQuery}&interval=${intervalQuery}&base=${encodeURIComponent(parseNumber(compareBase, 100))}`, authState), apiGet(`/charts/seasonality-compare?ticker=${encodeURIComponent(ticker)}&years=${encodeURIComponent(Math.max(3, Math.min(20, parseNumber(seasonalityYears, 10))))}&sector=${encodeURIComponent(seasonalitySectorOverride || "")}&sector_etf=${encodeURIComponent(seasonalityEtfOverride || "")}`, authState)]); setAnalyticsRanked(ranked.items || []); setAnalyticsSaveResult(null); setIndicatorChart(indicators.items || []); setSeasonalityChart(seasonality.items || []); setDistributionChart(distribution || null); setForecastChart(forecast || null); setNewsSummary(summary || null); setNewsTimeline(timeline || null); setPerformanceCompareChart(performanceCompare || null); setSeasonalityCompareChart(seasonalityCompare || null); setStatus("ready"); } catch (e) { setStatus(`error: ${e.message}`); } }); };

  const loadChartsFast = async () => { setStatus("loading charts"); await withProgress("Loading core charts", async () => { try { const periodQuery = encodeURIComponent(analyticsPeriod); const intervalQuery = encodeURIComponent(analyticsInterval); const years = encodeURIComponent(Math.max(3, Math.min(20, parseNumber(seasonalityYears, 10)))); const [indicators, seasonality, seasonalityCompare] = await Promise.all([apiGet(`/charts/indicators?ticker=${encodeURIComponent(ticker)}&period=${periodQuery}&interval=${intervalQuery}`, authState), apiGet(`/charts/seasonality?ticker=${encodeURIComponent(ticker)}&years=${years}`, authState), apiGet(`/charts/seasonality-compare?ticker=${encodeURIComponent(ticker)}&years=${years}&sector=${encodeURIComponent(seasonalitySectorOverride || "")}&sector_etf=${encodeURIComponent(seasonalityEtfOverride || "")}`, authState)]); setIndicatorChart(indicators.items || []); setSeasonalityChart(seasonality.items || []); setSeasonalityCompareChart(seasonalityCompare || null); setStatus("ready"); } catch (e) { setStatus(`error: ${e.message}`); } }); };

  const saveAnalyticsLeaderboard = async () => { setStatus("saving analytics leaderboard"); await withProgress("Saving leaderboard", async () => { try { const tickerList = parseTickers(analyticsTickersText) || [ticker]; const data = await apiPost("/analytics/leaderboard", { db: dbPath, tickers: tickerList, period: analyticsPeriod, interval: analyticsInterval, limit: Math.min(12, tickerList.length), save: true, forecast_horizon: Math.max(5, Math.min(90, parseNumber(analyticsHorizon, 30))) }, authState); setAnalyticsRanked(data.items || []); setAnalyticsLatest(data.items || []); setAnalyticsBatchId(data.batch_id || ""); setAnalyticsSaveResult(data || null); setStatus("ready"); } catch (e) { setStatus(`error: ${e.message}`); } }); };

  const loadLatestAnalyticsLeaderboard = async () => { setStatus("loading latest leaderboard"); await withProgress("Loading latest leaderboard", async () => { try { const data = await apiGet(`/analytics/leaderboard/latest?db=${encodeURIComponent(dbPath)}`, authState); setAnalyticsLatest(data.items || []); setAnalyticsBatchId(data.batch_id || ""); setStatus("ready"); } catch (e) { setStatus(`error: ${e.message}`); } }); };

  const loadSignals = async () => { setSignalsLoading(true); setStatus("loading signals"); await withProgress("Loading signals", async () => { try { setSignalsData(await apiGet(`/signals/generate?ticker=${encodeURIComponent(ticker)}&timeframe=${encodeURIComponent(signalsTimeframe)}&lookback_bars=120&detectors=breakout,rsi,macd,pullback`, authState) || null); setStatus("ready"); } catch (e) { setStatus(`error: ${e.message}`); } finally { setSignalsLoading(false); } }); };

  const loadTrainingFeatures = async () => { setStatus("loading training features"); await withProgress("Loading training features", async () => { try { const tickerList = parseTickers(analyticsTickersText) || [ticker]; setTrainingFeatures((await apiGet(`/analytics/training-features?tickers=${encodeURIComponent(tickerList.join(","))}&period=${encodeURIComponent(analyticsPeriod)}&interval=${encodeURIComponent(analyticsInterval)}&forecast_horizon=${encodeURIComponent(parseNumber(analyticsHorizon, 30))}&limit=${encodeURIComponent(Math.min(12, tickerList.length))}`, authState)).items || []); setStatus("ready"); } catch (e) { setStatus(`error: ${e.message}`); } }); };

  const proposeOrder = async () => { setStatus("proposing"); await withProgress("Proposing order", async () => { try { setExecutionResult(await apiPost("/execution/propose", { ticker, action: executionAction, qty: parseNumber(executionQty, 1), notional: parseNumber(notional, 0), orders_today: parseNumber(executionOrdersToday, 0), position_notional_after: parseNumber(executionPositionAfter, parseNumber(notional, 0)), broker_mode: executionBrokerMode, auto: executionAuto, max_daily_notional: parseNumber(executionMaxDaily, 5000), max_orders_per_day: parseNumber(executionMaxOrders, 20), max_position_notional: parseNumber(executionMaxPosition, 2000), db: dbPath }, authState) || null); await loadOverview(); } catch (e) { setStatus(`error: ${e.message}`); } }); };

  // ====== Assistant with state context injection ======
  const buildAssistantContext = () => ({ portfolio: { account: portfolioAccount, signals: mcpSignals?.signals?.length || 0, positions: portfolioPositions.length }, signals: (mcpSignals?.signals || []).slice(0, 5).map(s => ({ ticker: s.ticker, action: s.action || s.signal, strength: s.strength || s.score })), activeTab: screen, backtestState: backtestSummary ? { ticker, n_trades: backtestSummary.n_trades, win_rate: backtestSummary.win_rate, sharpe: backtestSummary.sharpe } : null });

  const askAssistant = async () => {
    const userMessage = assistantInput.trim();
    if (!userMessage) return;
    const runId = Date.now(); assistantRunRef.current = runId;
    const userTurn = { role: "user", content: userMessage, ts: new Date().toISOString() };
    setAssistantConversation((prev) => [...prev, userTurn]); setAssistantInput(""); setAssistantAutoResults([]); setAssistantAutoSummary(null);
    setStatus("assistant query");
    await withProgress("Running assistant query", async () => {
      try {
        const ctx = buildAssistantContext();
        const data = await apiPost("/assistant/query", { message: `[Context: ${JSON.stringify(ctx)}] ${userMessage}`, db: "sqlite:///quantflow.db" }, authState);
        let enhanced = data || null;
        const userLower = userMessage.toLowerCase(), answerText = String(data?.answer || "").toLowerCase();
        const isSmallTalk = /^(hi|hello|hey|yo|thanks|thank you|ok|okay|cool)$/.test(userLower);
        const shouldSuppressAutoRun = isSmallTalk;

        // Auto-enhance ranking queries
        if (/rank|best|top|opportunit|technical|seasonal|sentiment/.test(userLower) && !isSmallTalk) {
          try {
            const latest = await apiGet("/recommend/latest?limit=30", authState).catch(() => ({ items: [] }));
            const rows = (latest?.items || []).map((row) => { const entry = Number(row.entry || 0), target = Number(row.target || 0), confidence = Number(row.confidence || 0), bias = String(row.bias || "neutral").toLowerCase(), edge = entry > 0 ? (target - entry) / entry : 0; return { ticker: String(row.ticker || "").toUpperCase(), horizon: String(row.horizon || ""), bias, confidence, edge, conviction: confidence * 0.7 + Math.max(-1, Math.min(1, edge)) * 0.3 }; }).filter((row) => row.ticker && row.bias !== "neutral");
            const best = [], seen = new Set(); rows.sort((a, b) => b.conviction - a.conviction); for (const row of rows) { if (seen.has(row.ticker)) continue; seen.add(row.ticker); best.push(row); if (best.length >= 5) break; }
            if (best.length) { const top = best.slice(0, 5); enhanced = { ...(data || {}), answer: `Top ranked: ${top.map(r => `${r.ticker} (${r.bias}, conf ${fmtNumber(r.confidence, 2)}, edge ${fmtPct(r.edge)})`).join("; ")}`, opportunities: top }; }
          } catch (e) {}
        }

        setAssistantResponse(enhanced || null); setAssistantActionResult(null);
        const assistantTurn = { role: "assistant", content: enhanced?.answer || (enhanced ? compactJson(enhanced, 800) : "No response."), suggested_tool_calls: enhanced?.suggested_tool_calls || [], summary: enhanced?.summary || null, ts: new Date().toISOString() };
        setAssistantConversation((prev) => [...prev, assistantTurn]);

        if (assistantAutoRunGet && !shouldSuppressAutoRun) {
          const autoCalls = (enhanced?.suggested_tool_calls || []).filter((c) => String(c?.method || "GET").toUpperCase() === "GET").slice(0, 3);
          if (autoCalls.length) { const results = []; for (const call of autoCalls) { if (assistantRunRef.current !== runId) return; try { results.push({ ok: true, call, result: await apiGet(call.path, authState) }); } catch (e) { results.push({ ok: false, call, error: e.message }); } } if (assistantRunRef.current !== runId) return; setAssistantAutoResults(results); const okCount = results.filter(r => r.ok).length; setAssistantAutoSummary({ ts: new Date().toISOString(), total: results.length, ok: okCount, failed: results.length - okCount }); }
        }
        setStatus("ready");
      } catch (e) { setAssistantConversation((prev) => [...prev, { role: "assistant", content: `Error: ${e.message}`, ts: new Date().toISOString() }]); setStatus(`error: ${e.message}`); }
    });
  };

  const executeSuggestedToolCall = async (call) => { setStatus(`run ${call.name || "action"}`); await withProgress(`Running ${call.name || "action"}`, async () => { try { const method = (call.method || "GET").toUpperCase(); if (method === "POST" && assistantDryRun) { setAssistantActionResult({ ok: true, dryRun: true, call, result: { message: "Dry-run mode: POST action not executed.", would_call: call.path, payload: call.payload || {} } }); setStatus("ready"); return; } const result = method === "POST" ? await apiPost(call.path, call.payload || {}, authState) : await apiGet(call.path, authState); setAssistantActionResult({ ok: true, call, result }); await loadOverview(); setStatus("ready"); } catch (e) { setAssistantActionResult({ ok: false, call, error: e.message }); setStatus(`error: ${e.message}`); } }); };
  const requestExecuteAction = (call) => { if ((call.method || "GET").toUpperCase() === "POST") { setPendingAction(call); return; } executeSuggestedToolCall(call); };
  const confirmPendingAction = async () => { if (!pendingAction) return; const call = pendingAction; setPendingAction(null); await executeSuggestedToolCall(call); };
  const cancelPendingAction = () => { setPendingAction(null); setStatus("ready"); };

  const runMcpOnboardingRunbook = async () => { setStatus("assistant mcp runbook"); await withProgress("Running MCP onboarding runbook", async () => { const steps = [], stamp = new Date().toISOString(); const runStep = async (name, path) => { try { steps.push({ name, path, ok: true }); return await apiGet(path, authState); } catch (e) { steps.push({ name, path, ok: false, error: e.message }); return { error: e.message }; } }; const statusData = await runStep("Check MCP status", "/broker/mcp/status"); const readinessData = await runStep("Check MCP readiness", "/broker/mcp/readiness"); const signalsData = await runStep("Load portfolio signals", "/portfolio/signals?broker_mode=robinhood_mcp"); if (!statusData?.error) setMcpStatus(statusData || null); if (!readinessData?.error) setMcpReadiness(readinessData || null); if (!signalsData?.error) { setMcpSignals(signalsData || null); setMcpSignalsError(""); } else { setMcpSignals(null); setMcpSignalsError(String(signalsData.error || "unknown")); } const blockers = (readinessData?.checks || []).filter(r => r.state === "fail"), warnings = (readinessData?.checks || []).filter(r => r.state === "warn"), remediation = [...(readinessData?.next_steps || [])]; if (statusData?.connected && !statusData?.authenticated) remediation.unshift("Complete Robinhood auth in your MCP client session."); if (!statusData?.connected) remediation.unshift("Verify MCP endpoint connectivity and configuration."); const report = { ran_at: stamp, transport_connected: Boolean(statusData?.connected), authenticated: Boolean(statusData?.authenticated), account_available: Boolean(statusData?.account_available), positions_count: Number(statusData?.positions_count || 0), signals_count: Number(signalsData?.count || 0), overall_ready: Boolean(readinessData?.overall_ready), blockers, warnings, remediation, steps }; setMcpRunbookReport(report); setAssistantConversation((prev) => [...prev, { role: "assistant", content: `MCP runbook complete. Ready: ${report.overall_ready ? "yes" : "no"}. Blockers: ${report.blockers.length}. Signals: ${report.signals_count}.`, suggested_tool_calls: [{ name: "check_mcp_status", method: "GET", path: "/broker/mcp/status" }, { name: "check_mcp_readiness", method: "GET", path: "/broker/mcp/readiness" }, { name: "load_portfolio_signals", method: "GET", path: "/portfolio/signals?broker_mode=robinhood_mcp" }], ts: stamp }]); setStatus("ready"); }); };

  function calcPortfolioMetrics(signalsData) {
    const sigs = signalsData?.signals || [];
    const longCount = sigs.filter(s => (s.action || s.signal) === "buy" || s.direction === "long").length;
    const shortCount = sigs.filter(s => (s.action || s.signal) === "sell" || s.direction === "short").length;
    const avgStrength = sigs.length ? sigs.reduce((sum, s) => sum + (Number(s.strength || s.score || s.confidence || 0)), 0) / sigs.length : 0;
    return { total_signals: sigs.length, long_count: longCount, short_count: shortCount, avg_strength: avgStrength };
  }

  // ==========================================
  // RENDER FUNCTIONS
  // ==========================================

  const renderLanding = () => (
    <><View style={styles.hero}><Text style={styles.heroEyebrow}>QuantFlow</Text><Text style={styles.heroTitle}>From quant workflows to a product-ready public landing and operator console</Text><Text style={styles.heroBody}>Unified scanner, recommender, options picker, policy-gated execution, MCP readiness, and auth-backed user settings.</Text><View style={styles.inlineRowWrap}><Pressable style={styles.button} onPress={loadOverview}><Text style={styles.buttonText}>Refresh Public Metrics</Text></Pressable><Pressable style={styles.secondaryButton} onPress={() => setSurface("Workspace")}><Text style={styles.secondaryButtonText}>Go To Operator Workspace</Text></Pressable></View></View>
    <View style={styles.cardGrid}><View style={styles.card}><Text style={styles.cardTitle}>Data Acquisition</Text><Text style={styles.item}>Finviz scanner with proxy/API transport fallback.</Text><Text style={styles.item}>Latest presets: {presets.length}</Text><Text style={styles.item}>Universe rows: {latestUniverse.length}</Text></View><View style={styles.card}><Text style={styles.cardTitle}>Decision Intelligence</Text><Text style={styles.item}>Rule-based recommendations plus options filtering.</Text><Text style={styles.item}>Latest recommendations: {latestRecs.length}</Text><Text style={styles.item}>Recent intents: {intents.length}</Text></View><View style={styles.card}><Text style={styles.cardTitle}>Trust & Operations</Text><Text style={styles.item}>Policy-gated execution, MCP readiness checks.</Text><Text style={styles.item}>Auth enabled: {health?.auth_enabled ? "yes" : "no"}</Text></View></View></>
  );

  const renderWorkspaceGate = () => (
    <View style={styles.card}><Text style={styles.cardTitle}>Operator Workspace Gate</Text><Text style={styles.item}>Authenticate before enabling workflow tabs.</Text>
    {!workspaceUnlocked ? (<><Text style={styles.fieldLabel}>API Key</Text><TextInput value={apiKeyInput} onChangeText={setApiKeyInput} style={styles.input} placeholder="x-api-key (optional)" autoCapitalize="none" /><Text style={styles.fieldLabel}>Bearer Token</Text><TextInput value={bearerToken} onChangeText={setBearerToken} style={[styles.input, styles.tokenInput]} placeholder="Firebase ID token (Bearer)" autoCapitalize="none" multiline /><View style={styles.inlineRow}><Pressable onPress={() => setRememberCredentials(!rememberCredentials)} style={styles.inlineRow}><View style={[styles.checkbox, rememberCredentials && styles.checkboxChecked]} /><Text style={styles.item}>Remember credentials</Text></Pressable></View><View style={styles.inlineRowWrap}><Pressable style={styles.button} onPress={unlockWorkspace}><Text style={styles.buttonText}>Unlock Workspace</Text></Pressable>{!health?.auth_enabled && <Pressable style={styles.secondaryButton} onPress={unlockWorkspaceLocal}><Text style={styles.secondaryButtonText}>Enter Workspace (Local Dev)</Text></Pressable>}<Pressable style={styles.secondaryButton} onPress={() => setSurface("Public")}><Text style={styles.secondaryButtonText}>Back To Public Landing</Text></Pressable></View></>) : (<><Text style={styles.item}>✓ Workspace unlocked</Text><View style={styles.inlineRowWrap}><Pressable style={styles.cancelButton} onPress={logoutWorkspace}><Text style={styles.buttonText}>Logout</Text></Pressable></View></>)}</View>
  );

  const renderOverview = () => (
    <><View style={styles.row}><Pressable style={styles.button} onPress={loadOverview}><Text style={styles.buttonText}>Refresh</Text></Pressable></View>
    <View style={styles.card}><Text style={styles.cardTitle}>Health</Text><Text style={styles.item}>{health ? compactJson(health) : "not loaded"}</Text></View>
    <View style={styles.card}><Text style={styles.cardTitle}>Robinhood MCP</Text><Text>Connected: {mcpStatus ? (mcpStatus.connected ? "yes" : "no") : "unknown"}</Text><Text>Authenticated: {mcpStatus ? (mcpStatus.authenticated ? "yes" : "no") : "unknown"}</Text><Text>Positions: {mcpStatus ? String(mcpStatus.positions_count || 0) : "0"}</Text><Text>Signals: {mcpSignals ? String(mcpSignals.count || 0) : "0"}</Text><Pressable style={styles.button} onPress={loadMcpStatus}><Text style={styles.buttonText}>Check MCP</Text></Pressable><Pressable style={styles.secondaryButton} onPress={connectMcp}><Text style={styles.secondaryButtonText}>Connect Robinhood</Text></Pressable></View>
    <View style={styles.card}><Text style={styles.cardTitle}>Presets ({presets.length})</Text>{(presets || []).slice(0, 8).map((p) => <Text key={p} style={styles.item}>{p}</Text>)}</View>
    <View style={styles.card}><Text style={styles.cardTitle}>Recent Intents ({intents.length})</Text>{(intents || []).slice(0, 8).map((i) => <Text key={String(i.id)} style={styles.item}>#{i.id} {i.ticker} {i.action} ${i.notional} [{i.status}]</Text>)}</View></>
  );

  const renderScanner = () => (
    <><View style={styles.card}><Text style={styles.cardTitle}>Scanner Pipeline Controls</Text>{(scanIsActive || scannerBusy) && <ProgressBar progress={scanIsActive ? scanProgressPct : progressValue} label={scanIsActive ? `Scan ${scanProgressPct.toFixed(1)}%` : "Running scanner pipeline"} compact />}<Text style={styles.item}>DB path</Text><TextInput value={dbPath} onChangeText={setDbPath} style={styles.input} placeholder="sqlite:///quantflow.db" autoCapitalize="none" /><Text style={styles.item}>Preset profiles</Text><TextInput value={selectedPresetsText} onChangeText={setSelectedPresetsText} style={styles.input} placeholder="weekly_momo,weekly_bear" autoCapitalize="none" /><View style={styles.inlineRowWrap}><Pressable style={styles.smallButton} onPress={() => selectPresetCategory("weekly")}><Text style={styles.buttonText}>Weekly</Text></Pressable><Pressable style={styles.smallButton} onPress={() => selectPresetCategory("monthly")}><Text style={styles.buttonText}>Monthly</Text></Pressable><Pressable style={styles.smallButton} onPress={() => setSelectedPresetsText((presetDetails?.items || presets || []).join(","))}><Text style={styles.buttonText}>All</Text></Pressable></View><Text style={styles.fieldLabel}>Preset Search</Text><TextInput value={presetSearch} onChangeText={setPresetSearch} style={styles.input} placeholder="Search preset name" autoCapitalize="none" /><View style={styles.inlineRowWrap}>{availablePresets.slice(0, 12).map((name) => { const active = selectedPresets.includes(name); return <Pressable key={name} style={active ? styles.smallButton : styles.mutedButton} onPress={() => togglePreset(name)}><Text style={styles.buttonText}>{name}</Text></Pressable>; })}</View><View style={styles.inlineRowWrap}><Pressable style={styles.button} onPress={runScanPipeline}><Text style={styles.buttonText}>Run Scan</Text></Pressable>{scanIsActive && <Pressable style={styles.cancelButton} onPress={cancelScanPipeline}><Text style={styles.buttonText}>Cancel Scan</Text></Pressable>}<Pressable style={styles.secondaryButton} onPress={() => loadScanHistory(scanHistoryFilter)}><Text style={styles.secondaryButtonText}>Refresh History</Text></Pressable></View></View>
    <View style={styles.panelSplit}><View style={[styles.card, styles.panelCol]}><Text style={styles.cardTitle}>Scan Status</Text><Text style={styles.item}>Job: {scanJobId || "none"} | state: {scanJobStatus}</Text><Text style={styles.item}>Progress: {scanPresetsCompleted}/{scanPresetCount || "-"} presets</Text><Text style={styles.cardTitle}>History</Text><View style={styles.inlineRowWrap}>{[["all", "All"], ["completed", "Completed"], ["failed", "Failed"]].map(([value, label]) => <Pressable key={value} style={scanHistoryFilter === value ? styles.smallButton : styles.mutedButton} onPress={() => setScanHistoryFilter(value)}><Text style={styles.buttonText}>{label}</Text></Pressable>)}</View>{(scanHistoryFiltered || []).slice(0, 10).map((row) => <View key={row.job_id} style={styles.analyticsItemCard}><Text style={styles.analyticsHeadline}>{row.status} | {fmtTimestamp(row.created_at)}</Text><Text style={styles.item}>Job {row.job_id} | rows {row.rows_saved} | elapsed {fmtDurationSeconds(row.elapsed_seconds)}</Text></View>)}</View><View style={[styles.card, styles.panelCol]}><Text style={styles.cardTitle}>Universe Viewer</Text><Text style={styles.fieldLabel}>Preset Filter</Text><TextInput value={universePresetFilter} onChangeText={setUniversePresetFilter} style={styles.input} placeholder="Filter by preset name" autoCapitalize="none" /><Text style={styles.fieldLabel}>Rows To Show</Text><TextInput value={universeLimit} onChangeText={setUniverseLimit} style={styles.input} placeholder="Rows to show" keyboardType="numeric" /><Text style={styles.cardTitle}>Latest Universe ({filteredUniverse.length} shown)</Text>{(filteredUniverse || []).map((row, idx) => { const price = row.price ?? row.close ?? row.last ?? "-"; const rsi = row.rsi ?? row.rsi14 ?? "-"; return <Text key={String(idx)} style={styles.item}>{row.ticker || "?"} | {row.preset || "-"} | price {price} | rsi {rsi}</Text>; })}</View></View></>
  );

  const renderRecommend = () => (
    <View style={styles.card}><Text style={styles.cardTitle}>Recommendations</Text>{recommendBusy && <ProgressBar progress={progressValue} label="Computing recommendations" compact />}<Text style={styles.item}>DB path</Text><TextInput value={dbPath} onChangeText={setDbPath} style={styles.input} placeholder="sqlite:///quantflow.db" autoCapitalize="none" /><Text style={styles.item}>Tickers (comma-separated)</Text><TextInput value={recommendTickersText} onChangeText={setRecommendTickersText} style={styles.input} placeholder="AAPL,MSFT,NVDA" autoCapitalize="characters" /><View style={styles.inlineRowWrap}><Text style={styles.item}>Save to DB:</Text><Pressable style={recommendSave ? styles.smallButton : styles.mutedButton} onPress={() => setRecommendSave(true)}><Text style={styles.buttonText}>YES</Text></Pressable><Pressable style={!recommendSave ? styles.smallButton : styles.mutedButton} onPress={() => setRecommendSave(false)}><Text style={styles.buttonText}>NO</Text></Pressable></View><Pressable style={styles.button} onPress={runRecommendPipeline}><Text style={styles.buttonText}>Run Recommend</Text></Pressable>{!!recommendResult && <Text style={styles.item}>Run result: tickers {recommendResult.ticker_count}, recs {recommendResult.recommendation_count}, saved {recommendResult.saved ? "yes" : "no"}</Text>}<Text style={styles.cardTitle}>Latest Recs ({latestRecs.length})</Text>{(latestRecs || []).slice(0, 12).map((row, idx) => <Text key={String(idx)} style={styles.item}>{row.ticker} {row.bias || "-"} ({row.horizon || "?"}) conf {Number(row.confidence || 0).toFixed(2)} entry {Number(row.entry || 0).toFixed(2)} stop {Number(row.stop || 0).toFixed(2)} target {Number(row.target || 0).toFixed(2)}</Text>)}</View>
  );

  const renderOptions = () => (
    <View style={styles.card}><Text style={styles.cardTitle}>Options Ideas</Text><Text style={styles.fieldLabel}>Ticker</Text><TextInput value={ticker} onChangeText={setTicker} style={styles.input} placeholder="Ticker" /><Text style={styles.fieldLabel}>Horizon</Text><TextInput value={horizon} onChangeText={setHorizon} style={styles.input} placeholder="Horizon: 1w/1m/3m" /><Text style={styles.fieldLabel}>Bias</Text><TextInput value={optionsBias} onChangeText={setOptionsBias} style={styles.input} placeholder="Bias: long/short" /><Text style={styles.fieldLabel}>Budget</Text><TextInput value={optionsBudget} onChangeText={setOptionsBudget} style={styles.input} placeholder="Budget" keyboardType="numeric" /><Pressable style={styles.button} onPress={loadOptions}><Text style={styles.buttonText}>Load Ideas</Text></Pressable>{!!optionsResultMeta && <Text style={styles.item}>Result: {optionsResultMeta.count} ideas | {optionsResultMeta.horizon} | {optionsResultMeta.bias} | budget ${optionsResultMeta.budget}</Text>}{(optionIdeas || []).slice(0, 12).map((row, idx) => <Text key={String(idx)} style={styles.item}>{row.expiry || "?"} {row.right || ""} {row.strike ? Number(row.strike).toFixed(2) : "-"} | mid {row.mid ? Number(row.mid).toFixed(2) : "-"}</Text>)}</View>
  );

  // ====== PORTFOLIO TAB ======
  const renderPortfolio = () => {
    const equityValues = portfolioEquity.map((row) => Number(row.equity || row.value || 0));
    const equityDates = portfolioEquity.map((row) => String(row.date || row.timestamp || ""));
    const signalItems = mcpSignals?.signals || [];

    return (
      <>
        <View style={styles.card}>
          <Text style={styles.cardTitle}>Portfolio Dashboard</Text>
          <View style={styles.inlineRowWrap}>
            <Pressable style={styles.button} onPress={loadPortfolio}><Text style={styles.buttonText}>Refresh Portfolio</Text></Pressable>
            <Pressable style={styles.secondaryButton} onPress={loadMcpStatus}><Text style={styles.secondaryButtonText}>Check MCP</Text></Pressable>
          </View>
        </View>

        <View style={styles.panelSplit}>
          <View style={[styles.card, styles.panelCol]}>
            <Text style={styles.cardTitle}>Account</Text>
            {portfolioAccount ? (
              <>
                <View style={styles.metricRow}><Text style={styles.metricLabel}>Equity</Text><Text style={styles.metricValue}>${fmtNumber(portfolioAccount.equity, 2)}</Text></View>
                <View style={styles.metricRow}><Text style={styles.metricLabel}>Cash</Text><Text style={styles.metricValue}>${fmtNumber(portfolioAccount.cash, 2)}</Text></View>
                <View style={styles.metricRow}><Text style={styles.metricLabel}>Buying Power</Text><Text style={styles.metricValue}>${fmtNumber(portfolioAccount.buying_power, 2)}</Text></View>
              </>
            ) : (
              <Text style={styles.mutedText}>Connect Robinhood MCP to load account data.</Text>
            )}
          </View>

          <View style={[styles.card, styles.panelCol]}>
            <Text style={styles.cardTitle}>Signal Metrics</Text>
            {portfolioMetrics ? (
              <>
                <View style={styles.metricRow}><Text style={styles.metricLabel}>Total Signals</Text><Text style={styles.metricValue}>{portfolioMetrics.total_signals}</Text></View>
                <View style={styles.metricRow}><Text style={styles.metricLabel}>Long / Short</Text><Text style={styles.metricValue}>{portfolioMetrics.long_count} / {portfolioMetrics.short_count}</Text></View>
                <View style={styles.metricRow}><Text style={styles.metricLabel}>Avg Strength</Text><Text style={styles.metricValue}>{fmtNumber(portfolioMetrics.avg_strength, 3)}</Text></View>
              </>
            ) : (
              <Text style={styles.mutedText}>Load portfolio signals to see metrics.</Text>
            )}
          </View>
        </View>

        {equityValues.length > 0 && (
          <View style={styles.card}>
            <Text style={styles.cardTitle}>Equity Curve</Text>
            <MiniSeriesChart series={[{ label: "equity", values: equityValues, color: THEME.accentDeep, width: 3 }]} xLabels={equityDates} height={160} />
          </View>
        )}

        <View style={styles.card}>
          <Text style={styles.cardTitle}>Active Positions ({portfolioPositions.length})</Text>
          {portfolioPositions.length === 0 ? (
            <Text style={styles.mutedText}>No active positions loaded.</Text>
          ) : (
            portfolioPositions.map((pos, idx) => {
              const pnl = Number(pos.unrealized_pl || pos.pnl || 0);
              const pnlColor = pnl >= 0 ? THEME.ok : THEME.warn;
              return (
                <View key={`pos-${idx}`} style={[styles.analyticsItemCard, { borderLeftWidth: 4, borderLeftColor: pnlColor }]}>
                  <Text style={styles.analyticsHeadline}>{pos.ticker || pos.symbol} — {pos.qty || pos.quantity} shares @ ${fmtNumber(pos.avg_entry_price || pos.avg_price)}</Text>
                  <Text style={styles.item}>Current: ${fmtNumber(pos.current_price || pos.price)} | Market value: ${fmtNumber(pos.market_value, 2)}</Text>
                  <Text style={[styles.item, { color: pnlColor, fontWeight: "700" }]}>P&L: {pnl >= 0 ? "+" : ""}{fmtPct(pos.unrealized_plpc || 0)} (${fmtNumber(pnl, 2)})</Text>
                </View>
              );
            })
          )}
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>AI Signals ({signalItems.length})</Text>
          {signalItems.length === 0 ? (
            <Text style={styles.mutedText}>No AI signals loaded. Run Check MCP or Connect Robinhood.</Text>
          ) : (
            signalItems.map((sig, idx) => {
              const dir = sig.action || sig.signal || "hold";
              const dirColor = dir === "buy" || dir === "long" ? THEME.ok : dir === "sell" || dir === "short" ? THEME.warn : THEME.textMuted;
              const conf = Number(sig.strength || sig.score || sig.confidence || 0);
              return (
                <View key={`sig-${idx}`} style={[styles.analyticsItemCard, { borderLeftWidth: 4, borderLeftColor: dirColor }]}>
                  <View style={styles.inlineRow}>
                    <Text style={[styles.analyticsHeadline, { color: dirColor }]}>{dir.toUpperCase()}</Text>
                    <Text style={styles.analyticsHeadline}>{sig.ticker}</Text>
                    <View style={{ flex: 1 }} />
                    <Text style={styles.item}>Confidence {(conf * 100).toFixed(0)}%</Text>
                  </View>
                  {sig.reason && <Text style={styles.item}>{String(sig.reason).slice(0, 120)}</Text>}
                </View>
              );
            })
          )}
        </View>
      </>
    );
  };

  // ====== TRADE TAB ======
  const renderTrade = () => (
    <>
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Paper Trade</Text>
        <Text style={styles.mutedText}>Simulated paper trading. No real orders are placed.</Text>
        <View style={styles.inlineRowWrap}>
          <Pressable style={styles.button} onPress={loadTradeData}><Text style={styles.buttonText}>Refresh Trade Data</Text></Pressable>
        </View>

        <Text style={styles.fieldLabel}>Ticker</Text>
        <TextInput value={tradeTicker} onChangeText={setTradeTicker} style={styles.input} placeholder="AAPL" autoCapitalize="characters" />
        <View style={styles.inlineRowWrap}>
          <Text style={[styles.fieldLabel, styles.compactLabel]}>Action</Text>
          <Text style={[styles.fieldLabel, styles.compactLabel]}>Quantity</Text>
        </View>
        <View style={styles.inlineRowWrap}>
          <TextInput value={tradeAction} onChangeText={setTradeAction} style={[styles.input, styles.compactInput]} placeholder="buy/sell" autoCapitalize="none" />
          <TextInput value={tradeQty} onChangeText={setTradeQty} style={[styles.input, styles.compactInput]} placeholder="10" keyboardType="numeric" />
        </View>
        <View style={styles.inlineRowWrap}>
          <Text style={[styles.fieldLabel, styles.compactLabel]}>Stop Loss %</Text>
          <Text style={[styles.fieldLabel, styles.compactLabel]}>Take Profit %</Text>
        </View>
        <View style={styles.inlineRowWrap}>
          <TextInput value={tradeStopLoss} onChangeText={setTradeStopLoss} style={[styles.input, styles.compactInput]} placeholder="Optional stop loss %" keyboardType="numeric" />
          <TextInput value={tradeTakeProfit} onChangeText={setTradeTakeProfit} style={[styles.input, styles.compactInput]} placeholder="Optional take profit %" keyboardType="numeric" />
        </View>
        <Pressable style={styles.button} onPress={executePaperTrade}><Text style={styles.buttonText}>Place Paper Trade</Text></Pressable>
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Active Positions ({tradeActivePositions.length})</Text>
        {tradeActivePositions.length === 0 ? (
          <Text style={styles.mutedText}>No active paper positions.</Text>
        ) : (
          tradeActivePositions.map((pos, idx) => {
            const pnl = Number(pos.unrealized_pl || pos.pnl || 0);
            return (
              <View key={`tpos-${idx}`} style={[styles.analyticsItemCard, { borderLeftWidth: 4, borderLeftColor: pnl >= 0 ? THEME.ok : THEME.warn }]}>
                <Text style={styles.analyticsHeadline}>{pos.ticker} — {pos.qty} shares @ ${fmtNumber(pos.avg_entry_price || pos.avg_price)}</Text>
                <Text style={[styles.item, { color: pnl >= 0 ? THEME.ok : THEME.warn, fontWeight: "700" }]}>P&L: {pnl >= 0 ? "+" : ""}{fmtPct(pos.unrealized_plpc || 0)} (${fmtNumber(pnl, 2)})</Text>
              </View>
            );
          })
        )}
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Order History ({tradeOrderHistory.length})</Text>
        {tradeOrderHistory.length === 0 ? (
          <Text style={styles.mutedText}>No paper trade history yet.</Text>
        ) : (
          tradeOrderHistory.slice(0, 20).map((order, idx) => (
            <View key={`ord-${idx}`} style={styles.analyticsItemCard}>
              <Text style={styles.analyticsHeadline}>{order.ticker} — {order.action} x{order.qty || order.quantity} [{(order.status || "pending").toUpperCase()}]</Text>
              <Text style={styles.item}>Notional: ${fmtNumber(order.notional, 2)} | {fmtTimestamp(order.created_at || order.ts)}</Text>
            </View>
          ))
        )}
      </View>
    </>
  );

  // ====== ANALYTICS TAB ======
  const renderAnalytics = () => {
    const priceSeries = indicatorTail.map((row) => Number(row["adj close"] || row.close || 0));
    const sma20Series = indicatorTail.map((row) => Number(row.sma20 || 0));
    const sma50Series = indicatorTail.map((row) => Number(row.sma50 || 0));
    const rsiSeries = indicatorTail.map((row) => Number(row.rsi14 || 0));
    const volumeSeries = indicatorTail.map((row) => Number(row.volume || 0));
    const seasonalityAvg = seasonalityTail.map((row) => Number(row.avg_ret || 0));
    const seasonalityStd = seasonalityTail.map((row) => Number(row.std_ret || 0));
    const forecastPriceSeries = forecastForward.map((row) => Number(row.price || 0));
    const forecastUpperSeries = forecastForward.map((row) => Number(row.upper || 0));
    const forecastLowerSeries = forecastForward.map((row) => Number(row.lower || 0));
    const historySeries = forecastHistory.map((row) => Number(row.price || 0));
    const compareRows = performanceCompareChart?.items || [];
    const compareTickers = performanceCompareChart?.tickers || [];
    const compareSeries = compareTickers.slice(0, 6).map((name, idx) => {
      const palette = [THEME.text, THEME.accentDeep, THEME.warn, "#0ea5e9", "#b45309", "#475569"];
      return { label: name, values: compareRows.map((row) => Number(row[name] || 0)), color: palette[idx % palette.length], width: 2 };
    });
    const seasonalityCompareRows = seasonalityCompareChart?.items || [];
    const seasonalityStockAvg = seasonalityCompareRows.map((row) => Number(row.stock_avg_ret || 0));
    const seasonalitySectorAvg = seasonalityCompareRows.map((row) => Number(row.sector_avg_ret || 0));
    const seasonalitySpread = seasonalityCompareRows.map((row) => Number(row.spread || 0));
    const seasonalityStockCum = seasonalityCompareRows.map((row) => Number(row.stock_cum || 0));
    const seasonalitySectorCum = seasonalityCompareRows.map((row) => Number(row.sector_cum || 0));

    return (
      <><View style={styles.card}><Text style={styles.cardTitle}>Analytics Workbench</Text>{analyticsBusy && <ProgressBar progress={progressValue} label="Loading analytics datasets" compact />}<Text style={styles.fieldLabel}>Primary Ticker</Text><TextInput value={ticker} onChangeText={setTicker} style={styles.input} placeholder="Ticker" autoCapitalize="characters" /><Text style={styles.fieldLabel}>Ranking Basket</Text><TextInput value={analyticsTickersText} onChangeText={setAnalyticsTickersText} style={styles.input} placeholder="AAPL,MSFT,NVDA" autoCapitalize="characters" /><View style={styles.inlineRowWrap}><TextInput value={analyticsPeriod} onChangeText={setAnalyticsPeriod} style={[styles.input, styles.compactInput]} placeholder="Period" autoCapitalize="none" /><TextInput value={analyticsInterval} onChangeText={setAnalyticsInterval} style={[styles.input, styles.compactInput]} placeholder="Interval" autoCapitalize="none" /><TextInput value={analyticsBins} onChangeText={setAnalyticsBins} style={[styles.input, styles.compactInput]} placeholder="Bins" keyboardType="numeric" /><TextInput value={analyticsHorizon} onChangeText={setAnalyticsHorizon} style={[styles.input, styles.compactInput]} placeholder="Forecast days" keyboardType="numeric" /></View><View style={styles.inlineRowWrap}><Pressable style={styles.button} onPress={loadAnalytics}><Text style={styles.buttonText}>Load Analytics</Text></Pressable><Pressable style={styles.secondaryButton} onPress={loadChartsFast}><Text style={styles.secondaryButtonText}>Load Core Charts Fast</Text></Pressable></View></View>
      <View style={styles.card}><Text style={styles.cardTitle}>Ranked Setups ({analyticsRanked.length})</Text>{(analyticsRanked || []).slice(0, 6).map((row) => (<View key={row.ticker} style={styles.analyticsItemCard}><Text style={styles.analyticsHeadline}>{row.ticker} | {row.bias} | {row.primary_horizon} | composite {fmtNumber(row.composite_score)}</Text><Text style={styles.item}>Price {fmtNumber(row.price)} | support {fmtNumber(row.support_resistance?.nearest_support)} | resistance {fmtNumber(row.support_resistance?.nearest_resistance)}</Text><ScoreBars breakdown={row.score_breakdown} /></View>))}</View>
      <View style={styles.card}><Text style={styles.cardTitle}>Price + SMA</Text><MiniSeriesChart series={[{ label: "price", values: priceSeries, color: THEME.text, width: 3, type: "line" }, { label: "sma20", values: sma20Series, color: THEME.accent, width: 2, type: "line" }, { label: "sma50", values: sma50Series, color: THEME.warn, width: 2, type: "line" }]} xLabels={indicatorTail.map((row) => String(row.date || ""))} height={140} />
      <Text style={styles.item}>RSI 14</Text><MiniSeriesChart series={[{ label: "rsi", values: rsiSeries, color: THEME.accentDeep, width: 3, type: "line" }]} xLabels={indicatorTail.map((row) => String(row.date || ""))} height={90} />
      <Text style={styles.item}>Volume</Text><MiniSeriesChart series={[{ label: "volume", values: volumeSeries, color: "#7c6d5a", width: 3, opacity: 0.75, type: "bar" }]} xLabels={indicatorTail.map((row) => String(row.date || ""))} height={90} /></View>
      {distributionChart && (<View style={styles.card}><Text style={styles.cardTitle}>Price Distribution</Text><Text style={styles.item}>Current {fmtNumber(distributionChart?.current_price)} | range {fmtPct(distributionChart?.range_pct)}</Text><HorizontalDistribution items={distributionChart?.bins || []} color={THEME.accent} /></View>)}
      {forecastChart && (<View style={styles.card}><Text style={styles.cardTitle}>Forecast Envelope</Text><Text style={styles.item}>Daily trend {fmtPct(forecastChart?.daily_trend_pct)} | projected return {fmtPct(forecastChart?.forecast_return_pct)}</Text><MiniSeriesChart series={[{ label: "history", values: historySeries, color: THEME.text, width: 3 }]} height={120} /><MiniSeriesChart series={[{ label: "fc-lower", values: forecastLowerSeries, color: "#d97706", width: 2, opacity: 0.5 }, { label: "fc-center", values: forecastPriceSeries, color: THEME.accentDeep, width: 3 }, { label: "fc-upper", values: forecastUpperSeries, color: "#16a34a", width: 2, opacity: 0.5 }]} height={120} /></View>)}
      {seasonalityChart.length > 0 && (<View style={styles.card}><Text style={styles.cardTitle}>Seasonality</Text><MiniSeriesChart series={[{ label: "avg_ret", values: seasonalityAvg, color: THEME.accent, width: 3 }, { label: "std_ret", values: seasonalityStd, color: THEME.warn, width: 2, opacity: 0.7 }]} xLabels={seasonalityTail.map((row, idx) => dayOfYearLabel(row, idx + 1))} height={120} /></View>)}
      {performanceCompareChart && (<View style={styles.card}><Text style={styles.cardTitle}>Performance Comparison</Text><MiniSeriesChart series={compareSeries} height={130} />{(performanceCompareChart?.summary || []).slice(0, 8).map((row) => <Text key={`cmp-${row.ticker}`} style={styles.item}>{row.ticker} | return {fmtPct(row.return_pct)} | vol {fmtPct(row.volatility)}</Text>)}</View>)}
      {trainingFeatures.length > 0 && (<View style={styles.card}><Text style={styles.cardTitle}>Training Features ({trainingFeatures.length})</Text>{(trainingFeatures || []).slice(0, 8).map((row) => (<View key={`feat-${row.ticker}`} style={styles.analyticsItemCard}><Text style={styles.analyticsHeadline}>{row.ticker} | composite {fmtNumber(row.composite_score)}</Text><Text style={styles.item}>RSI {fmtNumber(row.rsi14)} | ATR {fmtNumber(row.atr14)} | Vol20 {fmtNumber(row.vol20, 3)}</Text></View>))}</View>)}</>
    );
  };

  // ====== CHARTS TAB ======
  const renderCharts = () => {
    const priceSeries = indicatorTail.map((row) => Number(row["adj close"] || row.close || 0));
    const sma20Series = indicatorTail.map((row) => Number(row.sma20 || 0));
    const sma50Series = indicatorTail.map((row) => Number(row.sma50 || 0));
    const rsiSeries = indicatorTail.map((row) => Number(row.rsi14 || 0));
    const volumeSeries = indicatorTail.map((row) => Number(row.volume || 0));
    const seasonalityAvg = seasonalityTail.map((row) => Number(row.avg_ret || 0));
    const seasonalityStd = seasonalityTail.map((row) => Number(row.std_ret || 0));
    const hasData = priceSeries.some((v) => v > 0);
    return (
      <><View style={styles.card}><Text style={styles.cardTitle}>Charts — {ticker}</Text><View style={styles.inlineRowWrap}><TextInput value={ticker} onChangeText={setTicker} style={[styles.input, styles.compactInput]} placeholder="Ticker" autoCapitalize="characters" /><TextInput value={analyticsPeriod} onChangeText={setAnalyticsPeriod} style={[styles.input, styles.compactInput]} placeholder="Period" autoCapitalize="none" /><TextInput value={analyticsInterval} onChangeText={setAnalyticsInterval} style={[styles.input, styles.compactInput]} placeholder="Interval" autoCapitalize="none" /></View><View style={styles.inlineRowWrap}><Pressable style={styles.button} onPress={loadAnalytics}><Text style={styles.buttonText}>Load Charts</Text></Pressable><Pressable style={styles.secondaryButton} onPress={loadChartsFast}><Text style={styles.secondaryButtonText}>Load Fast</Text></Pressable></View></View>
      {hasData && (<><View style={styles.card}><Text style={styles.cardTitle}>Price — {ticker}</Text><MiniSeriesChart series={[{ label: "price", values: priceSeries, color: THEME.text, width: 3, type: "line" }, { label: "sma20", values: sma20Series, color: THEME.accent, width: 2, type: "line" }, { label: "sma50", values: sma50Series, color: THEME.warn, width: 2, type: "line" }]} xLabels={indicatorTail.map((row) => String(row.date || ""))} height={160} /></View>
      <View style={styles.card}><Text style={styles.cardTitle}>RSI 14</Text><MiniSeriesChart series={[{ label: "rsi14", values: rsiSeries, color: THEME.accentDeep, width: 3, type: "line" }]} xLabels={indicatorTail.map((row) => String(row.date || ""))} height={120} /></View>
      <View style={styles.card}><Text style={styles.cardTitle}>Volume</Text><MiniSeriesChart series={[{ label: "volume", values: volumeSeries, color: "#7c6d5a", width: 3, opacity: 0.75, type: "bar" }]} xLabels={indicatorTail.map((row) => String(row.date || ""))} height={100} /></View>
      <View style={styles.card}><Text style={styles.cardTitle}>Seasonality</Text><MiniSeriesChart series={[{ label: "avg_ret", values: seasonalityAvg, color: THEME.accent, width: 3 }, { label: "std_ret", values: seasonalityStd, color: THEME.warn, width: 2, opacity: 0.7 }]} xLabels={seasonalityTail.map((row, idx) => dayOfYearLabel(row, idx + 1))} height={130} /></View></>)}</>
    );
  };

  // ====== EXECUTION TAB ======
  const renderExecution = () => (
    <View style={styles.card}><Text style={styles.cardTitle}>Propose Order</Text>{executionBusy && <ProgressBar progress={progressValue} label="Evaluating policy" compact />}<Text style={styles.fieldLabel}>Ticker</Text><TextInput value={ticker} onChangeText={setTicker} style={styles.input} placeholder="Ticker" /><Text style={styles.fieldLabel}>Action</Text><TextInput value={executionAction} onChangeText={setExecutionAction} style={styles.input} placeholder="Action: buy/sell" autoCapitalize="none" /><Text style={styles.fieldLabel}>Quantity</Text><TextInput value={executionQty} onChangeText={setExecutionQty} style={styles.input} placeholder="Qty" keyboardType="numeric" /><Text style={styles.fieldLabel}>Notional</Text><TextInput value={notional} onChangeText={setNotional} style={styles.input} placeholder="Notional" keyboardType="numeric" /><Text style={styles.fieldLabel}>Broker Mode</Text><TextInput value={executionBrokerMode} onChangeText={setExecutionBrokerMode} style={styles.input} placeholder="robinhood_mcp" autoCapitalize="none" /><Text style={styles.item}>Policy limits</Text><Text style={styles.fieldLabel}>Max Daily Notional</Text><TextInput value={executionMaxDaily} onChangeText={setExecutionMaxDaily} style={styles.input} placeholder="5000" keyboardType="numeric" /><Text style={styles.fieldLabel}>Max Position Notional</Text><TextInput value={executionMaxPosition} onChangeText={setExecutionMaxPosition} style={styles.input} placeholder="2000" keyboardType="numeric" /><Pressable style={styles.button} onPress={proposeOrder}><Text style={styles.buttonText}>Create Intent</Text></Pressable>{!!executionResult && <Text style={styles.item}>Decision: {executionResult.allow ? "APPROVED" : "BLOCKED"} | intent #{executionResult.intent_id} | reason: {executionResult.reason}</Text>}<Text style={styles.cardTitle}>Recent Intents ({intents.length})</Text>{(intents || []).slice(0, 12).map((i) => <Text key={String(i.id)} style={styles.item}>#{i.id} {i.ticker} {i.action} ${i.notional} [{i.status}]</Text>)}</View>
  );

  // ====== BACKTEST TAB (enhanced) ======
  const renderBacktest = () => {
    const equityValues = backtestEquity.map((row) => Number(row.equity || 0));
    const equityDates = backtestEquity.map((row) => String(row.date || row.timestamp || ""));
    const priceTail = indicatorTail.map((row) => Number(row["adj close"] || row.close || 0));
    const sma20 = indicatorTail.map((row) => Number(row.sma20 || 0));
    const bbUpper = indicatorTail.map((row) => Number(row.bb_upper || 0));
    const bbLower = indicatorTail.map((row) => Number(row.bb_lower || 0));
    const rsi14 = indicatorTail.map((row) => Number(row.rsi14 || 0));
    const volSeries = indicatorTail.map((row) => Number(row.volume || 0));
    const indicatorDates = indicatorTail.map((row) => String(row.date || ""));
    const tradeMarkers = backtestSummary?.trades || [];
    const stepIdx = backtestStep >= 0 && backtestStep < equityValues.length ? backtestStep : -1;

    return (
      <>
        <View style={styles.card}>
          <Text style={styles.cardTitle}>Backtest</Text>
          {backtestBusy && <ProgressBar progress={progressValue} label="Backtest in progress" compact />}
          <Text style={styles.fieldLabel}>Ticker</Text>
          <TextInput value={ticker} onChangeText={setTicker} style={styles.input} placeholder="Ticker" />
          <View style={styles.inlineRowWrap}>
            <Text style={[styles.fieldLabel, styles.compactLabel]}>Start Date</Text>
            <Text style={[styles.fieldLabel, styles.compactLabel]}>End Date</Text>
          </View>
          <View style={styles.inlineRowWrap}>
            <TextInput value={backtestStart} onChangeText={setBacktestStart} style={[styles.input, styles.compactInput]} placeholder="YYYY-MM-DD" autoCapitalize="none" />
            <TextInput value={backtestEnd} onChangeText={setBacktestEnd} style={[styles.input, styles.compactInput]} placeholder="Optional YYYY-MM-DD" autoCapitalize="none" />
          </View>
          <View style={styles.inlineRowWrap}>
            <Text style={[styles.fieldLabel, styles.compactLabel]}>Entry RSI</Text>
            <Text style={[styles.fieldLabel, styles.compactLabel]}>Max Hold Days</Text>
            <Text style={[styles.fieldLabel, styles.compactLabel]}>Stop Loss %</Text>
            <Text style={[styles.fieldLabel, styles.compactLabel]}>Take Profit %</Text>
          </View>
          <View style={styles.inlineRowWrap}>
            <TextInput value={backtestEntryRsi} onChangeText={setBacktestEntryRsi} style={[styles.input, styles.compactInput]} placeholder="50" keyboardType="numeric" />
            <TextInput value={backtestMaxHoldDays} onChangeText={setBacktestMaxHoldDays} style={[styles.input, styles.compactInput]} placeholder="7" keyboardType="numeric" />
            <TextInput value={backtestStopLossPct} onChangeText={setBacktestStopLossPct} style={[styles.input, styles.compactInput]} placeholder="0" keyboardType="numeric" />
            <TextInput value={backtestTakeProfitPct} onChangeText={setBacktestTakeProfitPct} style={[styles.input, styles.compactInput]} placeholder="0" keyboardType="numeric" />
          </View>
          <View style={styles.inlineRowWrap}>
            <Text style={[styles.fieldLabel, styles.compactLabel]}>MA Filter</Text>
            <Text style={[styles.fieldLabel, styles.compactLabel]}>MA Trend</Text>
          </View>
          <View style={styles.inlineRowWrap}>
            <TextInput value={backtestMaFilter} onChangeText={setBacktestMaFilter} style={[styles.input, styles.compactInput]} placeholder="sma20" autoCapitalize="none" />
            <TextInput value={backtestMaTrendFilter} onChangeText={setBacktestMaTrendFilter} style={[styles.input, styles.compactInput]} placeholder="none" autoCapitalize="none" />
          </View>
          <Pressable style={styles.button} onPress={loadBacktest}><Text style={styles.buttonText}>Run Short-Term Backtest</Text></Pressable>
        </View>

        {backtestSummary ? (
          <>
            <View style={styles.card}>
              <Text style={styles.cardTitle}>Performance Metrics</Text>
              <View style={styles.panelSplit}>
                <View style={[styles.card, styles.panelCol, { padding: 8 }]}>
                  <Text style={styles.item}>Trades: {backtestSummary.n_trades}</Text>
                  <Text style={styles.item}>Win Rate: {fmtPct(backtestSummary.win_rate)}</Text>
                  <Text style={styles.item}>Avg Return: {fmtPct(backtestSummary.avg_ret)}</Text>
                  <Text style={styles.item}>Max DD: {fmtPct(backtestSummary.max_dd)}</Text>
                </View>
                <View style={[styles.card, styles.panelCol, { padding: 8 }]}>
                  <Text style={styles.item}>Sharpe: {fmtNumber(backtestSummary.sharpe)}</Text>
                  <Text style={styles.item}>Sortino: {fmtNumber(backtestSummary.sortino)}</Text>
                  <Text style={styles.item}>CAGR: {fmtPct(backtestSummary.cagr)}</Text>
                  <Text style={styles.item}>Profit Factor: {fmtNumber(backtestSummary.profit_factor)}</Text>
                </View>
              </View>
            </View>

            <View style={styles.card}>
              <Text style={styles.cardTitle}>Equity Curve ({equityValues.length} points)</Text>
              {stepIdx >= 0 && <Text style={styles.item}>Step {stepIdx + 1} / {equityValues.length} | Equity: ${fmtNumber(equityValues[stepIdx], 2)}</Text>}
              <MiniSeriesChart series={[{ label: "equity", values: equityValues, color: THEME.accentDeep, width: 3 }]} xLabels={equityDates} height={140} />
              {equityValues.length > 1 && (
                <View style={styles.inlineRowWrap}>
                  <Pressable style={styles.smallButton} onPress={() => setBacktestStep(0)}><Text style={styles.buttonText}>◀◀</Text></Pressable>
                  <Pressable style={styles.smallButton} onPress={() => setBacktestStep(Math.max(0, backtestStep - 1))}><Text style={styles.buttonText}>◀</Text></Pressable>
                  <Text style={[styles.item, { minWidth: 100, textAlign: "center" }]}>{backtestStep < 0 ? "Full" : `${backtestStep + 1}/${equityValues.length}`}</Text>
                  <Pressable style={styles.smallButton} onPress={() => setBacktestStep(Math.min(equityValues.length - 1, backtestStep + 1))}><Text style={styles.buttonText}>▶</Text></Pressable>
                  <Pressable style={styles.smallButton} onPress={() => setBacktestStep(equityValues.length - 1)}><Text style={styles.buttonText}>▶▶</Text></Pressable>
                  <Pressable style={styles.mutedButton} onPress={() => setBacktestStep(-1)}><Text style={styles.buttonText}>Reset</Text></Pressable>
                </View>
              )}
            </View>

            {priceTail.length > 0 && (
              <View style={styles.card}>
                <Text style={styles.cardTitle}>Price + Bollinger Bands (20,2)</Text>
                <MiniSeriesChart
                  series={[
                    { label: "price", values: priceTail, color: THEME.text, width: 3, type: "line" },
                    { label: "sma20", values: sma20, color: THEME.accent, width: 2, type: "line" },
                    { label: "bb_upper", values: bbUpper.length ? bbUpper : sma20.map(v => v * 1.02), color: "#16a34a", width: 1, opacity: 0.5, type: "line" },
                    { label: "bb_lower", values: bbLower.length ? bbLower : sma20.map(v => v * 0.98), color: "#dc2626", width: 1, opacity: 0.5, type: "line" },
                  ]}
                  xLabels={indicatorDates} height={140}
                />
              </View>
            )}

            <View style={styles.panelSplit}>
              {rsi14.length > 0 && (
                <View style={[styles.card, styles.panelCol]}>
                  <Text style={styles.cardTitle}>RSI (14)</Text>
                  <MiniSeriesChart series={[{ label: "rsi", values: rsi14, color: THEME.accentDeep, width: 3, type: "line" }]} xLabels={indicatorDates} height={90} />
                  <View style={styles.inlineRow}>
                    <Text style={[styles.item, { color: THEME.warn, fontSize: 10 }]}>— 70 overbought</Text>
                    <Text style={[styles.item, { color: THEME.ok, fontSize: 10 }]}>— 30 oversold</Text>
                  </View>
                </View>
              )}
              {volSeries.length > 0 && (
                <View style={[styles.card, styles.panelCol]}>
                  <Text style={styles.cardTitle}>Volume</Text>
                  <MiniSeriesChart series={[{ label: "volume", values: volSeries, color: "#7c6d5a", width: 3, opacity: 0.75, type: "bar" }]} xLabels={indicatorDates} height={90} />
                </View>
              )}
            </View>

            {tradeMarkers.length > 0 && (
              <View style={styles.card}>
                <Text style={styles.cardTitle}>Trade Markers ({tradeMarkers.length} trades)</Text>
                {tradeMarkers.slice(0, 12).map((trade, idx) => (
                  <View key={`trade-${idx}`} style={[styles.analyticsItemCard, { borderLeftWidth: 4, borderLeftColor: Number(trade.return_pct || 0) >= 0 ? THEME.ok : THEME.warn }]}>
                    <Text style={styles.analyticsHeadline}>
                      {trade.action || "buy"} {trade.ticker || ticker} @ ${fmtNumber(trade.entry_price)} → {trade.exit_reason || "close"} @ ${fmtNumber(trade.exit_price)} | Return: {fmtPct(trade.return_pct)}
                    </Text>
                    <Text style={styles.item}>
                      Entry: {fmtTimestamp(trade.entry_date)} | Exit: {fmtTimestamp(trade.exit_date)} | Hold: {trade.hold_days || "-"}d
                      {trade.model_sigma !== undefined ? ` | σ: ${fmtNumber(trade.model_sigma, 4)}` : ""}
                    </Text>
                  </View>
                ))}
              </View>
            )}
          </>
        ) : (
          <View style={styles.card}><Text style={styles.mutedText}>No backtest loaded. Configure parameters and run.</Text></View>
        )}
      </>
    );
  };

  // ====== MARKET TAB ======
  const renderMarket = () => {
    const marketPrice = marketIndicators.map((row) => Number(row["adj close"] || row.close || 0));
    const marketSma20 = marketIndicators.map((row) => Number(row.sma20 || 0));
    const marketSma50 = marketIndicators.map((row) => Number(row.sma50 || 0));
    const marketRsi = marketIndicators.map((row) => Number(row.rsi14 || 0));
    const marketVol = marketIndicators.map((row) => Number(row.volume || 0));
    const marketDates = marketIndicators.map((row) => String(row.date || ""));

    const fcPrice = (marketForecast?.forecast || []).map((row) => Number(row.price || 0));
    const fcUpper = (marketForecast?.forecast || []).map((row) => Number(row.upper || 0));
    const fcLower = (marketForecast?.forecast || []).map((row) => Number(row.lower || 0));
    const fcHistory = (marketForecast?.history || []).map((row) => Number(row.price || 0));

    return (
      <>
        <View style={styles.card}>
          <Text style={styles.cardTitle}>Market — {marketTicker}</Text>
          <View style={styles.inlineRowWrap}>
            <TextInput value={marketTicker} onChangeText={setMarketTicker} style={[styles.input, styles.compactInput]} placeholder="Ticker" autoCapitalize="characters" />
            <View style={styles.rowWrap}>
              {["1m", "5m", "15m", "1H", "1d"].map((tf) => (
                <Pressable key={tf} style={[styles.chip, marketTimeframe === tf && styles.chipActive]} onPress={() => setMarketTimeframe(tf)}>
                  <Text style={[styles.chipText, marketTimeframe === tf && styles.chipActiveText]}>{tf}</Text>
                </Pressable>
              ))}
            </View>
            <Pressable style={styles.button} onPress={loadMarketData}><Text style={styles.buttonText}>Load Data</Text></Pressable>
          </View>
        </View>

        {marketPrice.length > 0 ? (
          <>
            <View style={styles.card}>
              <Text style={styles.cardTitle}>Price — {marketTicker}</Text>
              <MiniSeriesChart series={[
                { label: "price", values: marketPrice, color: THEME.text, width: 3, type: "line" },
                { label: "sma20", values: marketSma20, color: THEME.accent, width: 2, type: "line" },
                { label: "sma50", values: marketSma50, color: THEME.warn, width: 2, type: "line" },
              ]} xLabels={marketDates} height={160} />
            </View>

            <View style={styles.panelSplit}>
              <View style={[styles.card, styles.panelCol]}>
                <Text style={styles.cardTitle}>RSI (14)</Text>
                <MiniSeriesChart series={[{ label: "rsi", values: marketRsi, color: THEME.accentDeep, width: 3, type: "line" }]} xLabels={marketDates} height={100} />
              </View>
              <View style={[styles.card, styles.panelCol]}>
                <Text style={styles.cardTitle}>Volume</Text>
                <MiniSeriesChart series={[{ label: "vol", values: marketVol, color: "#7c6d5a", width: 3, opacity: 0.75, type: "bar" }]} xLabels={marketDates} height={100} />
              </View>
            </View>

            {fcPrice.length > 0 && (
              <View style={styles.card}>
                <Text style={styles.cardTitle}>AI Forecast Overlay — {marketTicker}</Text>
                <Text style={styles.item}>Projected return: {fmtPct(marketForecast?.forecast_return_pct)} | Confidence band: {fmtPct(marketForecast?.confidence_band_pct)}</Text>
                <Text style={styles.item}>Recent history ({fcHistory.length} bars)</Text>
                <MiniSeriesChart series={[{ label: "history", values: fcHistory, color: THEME.text, width: 3 }]} height={100} />
                <Text style={styles.item}>21-step forecast path (dashed overlay)</Text>
                <MiniSeriesChart series={[
                  { label: "fc-lower", values: fcLower, color: "#d97706", width: 1, opacity: 0.5 },
                  { label: "fc-center", values: fcPrice, color: THEME.accentDeep, width: 3 },
                  { label: "fc-upper", values: fcUpper, color: "#16a34a", width: 1, opacity: 0.5 },
                ]} height={120} />
              </View>
            )}
          </>
        ) : (
          <View style={styles.card}><Text style={styles.mutedText}>Enter a ticker, select timeframe, and load data to view charts.</Text></View>
        )}
      </>
    );
  };

  // ====== SETTINGS TAB ======
  const renderSettings = () => (
    <>
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Configuration</Text>
        <Text style={styles.item}>Backend URL: {API_BASE}</Text>
        <Text style={styles.item}>Auth enabled: {health?.auth_enabled ? "yes" : "no"}</Text>
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Alpaca API</Text>
        <Text style={styles.mutedText}>Set these to use Alpaca for real-time market data and trading.</Text>
        <Text style={styles.fieldLabel}>API Key</Text>
        <TextInput value={settingsAlpacaKey} onChangeText={setSettingsAlpacaKey} style={styles.input} placeholder="PK..." autoCapitalize="none" secureTextEntry />
        <Text style={styles.fieldLabel}>Secret Key</Text>
        <TextInput value={settingsAlpacaSecret} onChangeText={setSettingsAlpacaSecret} style={styles.input} placeholder="SK..." autoCapitalize="none" secureTextEntry />
        <View style={styles.inlineRowWrap}>
          <Pressable style={styles.button} onPress={testAlpacaConnection}><Text style={styles.buttonText}>Test Connection</Text></Pressable>
          <Text style={styles.item}>Status: {settingsAlpacaStatus}</Text>
        </View>
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>LLM API</Text>
        <Text style={styles.mutedText}>Configure your LLM endpoint for the AI Assistant.</Text>
        <Text style={styles.fieldLabel}>API Key</Text>
        <TextInput value={settingsLlmKey} onChangeText={setSettingsLlmKey} style={styles.input} placeholder="sk-..." autoCapitalize="none" secureTextEntry />
        <Text style={styles.fieldLabel}>Endpoint URL</Text>
        <TextInput value={settingsLlmEndpoint} onChangeText={setSettingsLlmEndpoint} style={styles.input} placeholder="https://api.openai.com/v1" autoCapitalize="none" />
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Risk Parameters</Text>
        <View style={styles.inlineRowWrap}>
          <Text style={[styles.fieldLabel, styles.compactLabel]}>Stop Loss %</Text>
          <Text style={[styles.fieldLabel, styles.compactLabel]}>Max Position %</Text>
          <Text style={[styles.fieldLabel, styles.compactLabel]}>Daily Loss Limit $</Text>
        </View>
        <View style={styles.inlineRowWrap}>
          <TextInput value={settingsRiskStopLoss} onChangeText={setSettingsRiskStopLoss} style={[styles.input, styles.compactInput]} placeholder="5" keyboardType="numeric" />
          <TextInput value={settingsRiskMaxPos} onChangeText={setSettingsRiskMaxPos} style={[styles.input, styles.compactInput]} placeholder="20" keyboardType="numeric" />
          <TextInput value={settingsDailyLossLimit} onChangeText={setSettingsDailyLossLimit} style={[styles.input, styles.compactInput]} placeholder="2000" keyboardType="numeric" />
        </View>
        <Pressable style={styles.button} onPress={saveAppSettings}><Text style={styles.buttonText}>Save Settings</Text></Pressable>
        {!!settingsSaveMsg && <Text style={[styles.item, { color: settingsSaveMsg.includes("Error") ? THEME.warn : THEME.ok }]}>{settingsSaveMsg}</Text>}
      </View>
    </>
  );

  // ====== ASSISTANT (with suggested prompts + localStorage) ======
  const renderAssistant = (compact = false) => (
    <View style={styles.card}>
      <Text style={styles.cardTitle}>QuantFlow Assistant</Text>
      {assistantBusy && <ProgressBar progress={progressValue} label="Assistant processing" compact />}
      <View style={styles.inlineRow}>
        <Text style={[styles.label, { flex: 1 }]}>Queries local QuantFlow data and suggests API automation workflows.</Text>
        <View style={styles.inlineRow}>
          <Text style={styles.mutedText}>Dry-run </Text>
          <Pressable style={assistantDryRun ? styles.smallButton : styles.mutedButton} onPress={() => setAssistantDryRun(true)}><Text style={styles.buttonText}>ON</Text></Pressable>
          <Pressable style={!assistantDryRun ? styles.smallButton : styles.mutedButton} onPress={() => setAssistantDryRun(false)}><Text style={styles.buttonText}>OFF</Text></Pressable>
        </View>
      </View>
      {!compact && (
        <View style={styles.inlineRowWrap}>
          <Pressable style={styles.secondaryButton} onPress={runMcpOnboardingRunbook}><Text style={styles.secondaryButtonText}>MCP Runbook</Text></Pressable>
        </View>
      )}

      {/* Suggested prompts */}
      {!compact && (
        <View>
          <Pressable style={styles.mutedButton} onPress={() => setAssistantPromptsExpanded(!assistantPromptsExpanded)}>
            <Text style={styles.buttonText}>{assistantPromptsExpanded ? "Hide" : "Show"} Suggested Prompts</Text>
          </Pressable>
          {assistantPromptsExpanded && (
            <View style={styles.rowWrap}>
              {ASSISTANT_SUGGESTED_PROMPTS.map((prompt) => (
                <Pressable key={prompt} style={styles.chip} onPress={() => { setAssistantInput(prompt); }}>
                  <Text style={styles.chipText}>{prompt}</Text>
                </Pressable>
              ))}
            </View>
          )}
        </View>
      )}

      {assistantConversation.length === 0 && (
        <View style={styles.chatEmptyState}><Text style={styles.mutedText}>No messages yet. Ask anything about your portfolio, scanner results, or recommended setups. Conversation persists in browser storage.</Text></View>
      )}
      {assistantConversation.map((turn, idx) => (
        <View key={String(idx)} style={turn.role === "user" ? styles.chatBubbleUser : styles.chatBubbleAssistant}>
          <Text style={turn.role === "user" ? styles.chatLabelUser : styles.chatLabelAssistant}>{turn.role === "user" ? "You" : "Assistant"}</Text>
          <Text style={turn.role === "user" ? styles.chatTextUser : styles.chatTextAssistant}>{turn.content}</Text>
          {!!turn.summary && <Text style={styles.chatMeta}>Summary: {compactJson(turn.summary)}</Text>}
          {!!(!compact && turn.suggested_tool_calls?.length) && (
            <View style={{ marginTop: 8 }}>
              <Text style={styles.chatMeta}>Suggested actions:</Text>
              {turn.suggested_tool_calls.slice(0, 6).map((c, ci) => (
                <View key={String(ci)} style={styles.chatToolCallRow}>
                  <Text style={styles.chatToolCallText}>{c.method} {c.path}{c.name ? ` — ${c.name}` : ""}</Text>
                  <Pressable style={styles.smallButton} onPress={() => requestExecuteAction(c)}><Text style={styles.buttonText}>Run</Text></Pressable>
                </View>
              ))}
            </View>
          )}
        </View>
      ))}

      {!!(!compact && pendingAction) && (
        <View style={[styles.card, { marginTop: 8 }]}><Text style={styles.cardTitle}>Confirm Action</Text><Text style={styles.item}>{(pendingAction.method || "POST").toUpperCase()} {pendingAction.path}</Text><View style={styles.inlineRow}><Pressable style={styles.smallButton} onPress={confirmPendingAction}><Text style={styles.buttonText}>Confirm</Text></Pressable><Pressable style={styles.mutedButton} onPress={cancelPendingAction}><Text style={styles.buttonText}>Cancel</Text></Pressable></View></View>
      )}
      {!!assistantActionResult && (
        <View style={styles.chatBubbleAssistant}><Text style={styles.chatLabelAssistant}>Action Result</Text><Text style={styles.chatTextAssistant}>{assistantActionResult.ok ? (assistantActionResult.dryRun ? "Dry-run — action not executed." : "Success.") : `Error: ${assistantActionResult.error}`}</Text></View>
      )}
      {!!assistantAutoResults.length && (
        <View style={styles.chatBubbleAssistant}><Text style={styles.chatLabelAssistant}>Auto-Run Results</Text>{!!assistantAutoSummary && <Text style={styles.chatMeta}>{assistantAutoSummary.ok}/{assistantAutoSummary.total} ok</Text>}{assistantAutoResults.slice(0, 5).map((row, idx) => <Text key={`auto-${idx}`} style={styles.chatMeta}>{row.ok ? "OK" : "ERR"} | {row.call?.method} {row.call?.path}</Text>)}</View>
      )}
      {!!mcpRunbookReport && (
        <View style={styles.chatBubbleAssistant}><Text style={styles.chatLabelAssistant}>MCP Onboarding Report</Text><Text style={styles.chatMeta}>Ready: {mcpRunbookReport.overall_ready ? "yes" : "no"} | Blockers: {mcpRunbookReport.blockers.length} | Signals: {mcpRunbookReport.signals_count}</Text></View>
      )}

      <View style={[styles.chatInputWrap, { marginTop: 10 }]}>
        <Text style={styles.fieldLabel}>Message</Text>
        <TextInput value={assistantInput} onChangeText={setAssistantInput} style={[styles.input, styles.chatInput]} placeholder="Ask about setups, backtest results, portfolio…" multiline scrollEnabled />
        <Pressable style={[styles.button, { marginBottom: 0, alignSelf: "flex-end" }]} onPress={askAssistant}><Text style={styles.buttonText}>Send</Text></Pressable>
      </View>
      {assistantConversation.length > 0 && (
        <Pressable style={[styles.mutedButton, { marginTop: 6 }]} onPress={() => { setAssistantConversation([]); setAssistantActionResult(null); AsyncStorage.removeItem("quantflow_chat").catch(() => {}); }}>
          <Text style={styles.buttonText}>Clear Chat</Text>
        </Pressable>
      )}
      {!!compact && <Pressable style={[styles.secondaryButton, { marginTop: 8 }]} onPress={() => setScreen("Assistant")}><Text style={styles.secondaryButtonText}>Open Full Assistant</Text></Pressable>}
    </View>
  );

  const renderAuth = () => (
    <><View style={styles.card}><Text style={styles.cardTitle}>Connection & Auth Controls</Text>{authBusy && <ProgressBar progress={progressValue} label="Auth action in progress" compact />}<Text style={styles.fieldLabel}>API Key</Text><TextInput value={apiKeyInput} onChangeText={setApiKeyInput} style={styles.input} placeholder="x-api-key (optional)" autoCapitalize="none" /><Text style={styles.fieldLabel}>Bearer Token</Text><TextInput value={bearerToken} onChangeText={setBearerToken} style={[styles.input, styles.tokenInput]} placeholder="Firebase ID token (Bearer)" autoCapitalize="none" multiline /><View style={styles.inlineRowWrap}><Pressable style={styles.button} onPress={signInWithGoogleWeb}><Text style={styles.buttonText}>Sign In With Google (Web)</Text></Pressable><Pressable style={styles.button} onPress={checkFirebaseIdentity}><Text style={styles.buttonText}>Check /auth/me</Text></Pressable></View><Text style={styles.item}>Google login status: {googleSignInStatus}</Text></View>
     <View style={styles.card}><Text style={styles.cardTitle}>Robinhood MCP Runtime Auth</Text><Text style={styles.warnText}>Never store real broker tokens in client storage.</Text><View style={styles.inlineRowWrap}><Pressable style={styles.button} onPress={startMcpBackendOAuth}><Text style={styles.buttonText}>Robinhood Auth (Backend OAuth)</Text></Pressable></View><Text style={styles.fieldLabel}>MCP Bearer Token</Text><TextInput value={mcpBearerInput} onChangeText={setMcpBearerInput} style={[styles.input, styles.tokenInput]} placeholder="Bearer eyJ..." autoCapitalize="none" multiline /><View style={styles.inlineRowWrap}><Pressable style={styles.button} onPress={saveMcpRuntimeConfig}><Text style={styles.buttonText}>Save MCP Config</Text></Pressable><Pressable style={styles.cancelButton} onPress={clearMcpRuntimeConfig}><Text style={styles.buttonText}>Clear MCP Config</Text></Pressable></View><Text style={styles.item}>MCP config status: {mcpConfigStatus}</Text></View>
     <View style={styles.card}><Text style={styles.cardTitle}>Firebase User Settings</Text>{settingsSaveStatus === "saving" && <ProgressBar progress={progressValue} label="Saving user settings" compact />}<View style={styles.inlineRowWrap}><Pressable style={styles.button} onPress={loadUserSettings}><Text style={styles.buttonText}>Load Settings</Text></Pressable><Pressable style={styles.secondaryButton} onPress={saveUserSettings}><Text style={styles.secondaryButtonText}>Save Settings</Text></Pressable></View><Text style={styles.fieldLabel}>User Settings JSON</Text><TextInput value={userSettingsText} onChangeText={setUserSettingsText} style={[styles.input, styles.jsonInput]} placeholder='{"watchlist": ["AAPL"]}' multiline /></View></>
  );

  const renderPresetsManager = () => (
    <><View style={styles.card}><Text style={styles.cardTitle}>Preset Management</Text><Text style={styles.item}>View, create, and edit finviz screener presets.</Text></View>
    <View style={styles.card}><Text style={styles.cardTitle}>Available Presets ({presets.length})</Text><ScrollView horizontal style={styles.inlineRow}>{presets.map((name) => (<Pressable key={name} onPress={() => { setPresetEditName(name); const profile = presetDetails?.profiles?.[name] || {}; setPresetEditContent(JSON.stringify(profile, null, 2)); }} style={[styles.smallButton, presetEditName === name && { backgroundColor: THEME.accentDeep }]}><Text style={styles.buttonText}>{name}</Text></Pressable>))}</ScrollView></View>
    <View style={styles.card}><Text style={styles.cardTitle}>Create / Edit Preset</Text><Text style={styles.fieldLabel}>Preset Name</Text><TextInput value={presetEditName} onChangeText={setPresetEditName} style={styles.input} placeholder="Preset name" autoCapitalize="none" /><Text style={styles.fieldLabel}>Preset Filter JSON</Text><TextInput value={presetEditContent} onChangeText={setPresetEditContent} style={[styles.input, styles.jsonInput]} placeholder='{"Average Volume": "Over 300K"}' multiline /><View style={styles.inlineRowWrap}><Pressable style={styles.button} onPress={async () => { if (!presetEditName.trim()) { setPresetSaveStatus("error: preset name required"); return; } try { setPresetSaveStatus("saving"); await apiPost("/scanner/presets/update", { name: presetEditName.trim(), filters: JSON.parse(presetEditContent) }, authState); setPresetSaveStatus("saved"); setPresets([...new Set([...presets, presetEditName.trim()])]); await loadOverview(); } catch (e) { setPresetSaveStatus(`error: ${e.message}`); } }}><Text style={styles.buttonText}>Save Preset</Text></Pressable></View></View></>
  );

  // ====== APP SHELL ======
  return (
    <SafeAreaView style={styles.container}>
      <StatusBar style="dark" />
      <ScrollView contentContainerStyle={styles.content}>
        <Text style={styles.title}>QuantFlow</Text>
        <Text style={styles.meta}>{baseLabel} | Status: {status}</Text>
        {inFlightOps > 0 && <ProgressBar progress={progressValue} label={activeTaskLabel || "Loading"} />}

        <View style={styles.tabRow}>
          {SURFACES.map((name) => (
            <Pressable key={name} style={[styles.tab, surface === name ? styles.tabActive : null]} onPress={() => setSurface(name)}>
              <Text style={[styles.tabText, surface === name ? styles.tabTextActive : null]}>{name}</Text>
            </Pressable>
          ))}
        </View>

        {surface === "Public" && renderLanding()}
        {surface === "Workspace" && !workspaceUnlocked && renderWorkspaceGate()}

        {surface === "Workspace" && workspaceUnlocked && (
          <View style={styles.workspaceLayout}>
            <View style={styles.workspaceMain}>
              <View style={styles.tabRow}>
                {SCREENS.map((name) => (
                  <Pressable key={name} style={[styles.tab, screen === name ? styles.tabActive : null]} onPress={() => setScreen(name)}>
                    <Text style={[styles.tabText, screen === name ? styles.tabTextActive : null]}>{name}</Text>
                  </Pressable>
                ))}
              </View>

              {screen === "Overview" && renderOverview()}
              {screen === "Portfolio" && renderPortfolio()}
              {screen === "Trade" && renderTrade()}
              {screen === "Scanner" && renderScanner()}
              {screen === "Recommend" && renderRecommend()}
              {screen === "Analytics" && renderAnalytics()}
              {screen === "Charts" && renderCharts()}
              {screen === "Options" && renderOptions()}
              {screen === "Backtest" && renderBacktest()}
              {screen === "Market" && renderMarket()}
              {screen === "Execution" && renderExecution()}
              {screen === "Assistant" && renderAssistant()}
              {screen === "Presets" && renderPresetsManager()}
              {screen === "Auth" && renderAuth()}
              {screen === "Settings" && renderSettings()}
            </View>
            <View style={styles.workspaceSidePanel}>
              {renderAssistant(true)}
            </View>
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: THEME.bg },
  content: { padding: 16, gap: 12, maxWidth: 980, width: "100%", alignSelf: "center" },
  title: { fontSize: 30, fontWeight: "800", color: THEME.text, letterSpacing: 0.4 },
  meta: { color: THEME.textMuted },
  row: { flexDirection: "row", gap: 8 },
  tabRow: { flexDirection: "row", gap: 8, flexWrap: "wrap", marginTop: 4 },
  tab: { borderWidth: 1, borderColor: THEME.border, borderRadius: 999, paddingVertical: 7, paddingHorizontal: 12, backgroundColor: THEME.panelAlt },
  tabActive: { backgroundColor: "#d7efe9", borderColor: THEME.accent },
  tabText: { color: THEME.textMuted, fontWeight: "700" },
  tabTextActive: { color: THEME.accentDeep },
  workspaceLayout: { flexDirection: "row", alignItems: "flex-start", gap: 12, flexWrap: "wrap" },
  workspaceMain: { flex: 1, minWidth: 320, maxWidth: 760 },
  workspaceSidePanel: { width: 300, minWidth: 280 },
  hero: { backgroundColor: THEME.panel, borderRadius: 18, borderWidth: 1, borderColor: THEME.border, padding: 18, gap: 10 },
  heroEyebrow: { color: THEME.accentDeep, fontWeight: "800", letterSpacing: 1.2, textTransform: "uppercase", fontSize: 12 },
  heroTitle: { fontSize: 28, color: THEME.text, fontWeight: "800", lineHeight: 34 },
  heroBody: { color: THEME.textMuted, lineHeight: 21, fontSize: 15 },
  cardGrid: { gap: 12 },
  card: { backgroundColor: THEME.panel, borderRadius: 14, padding: 12, gap: 8, borderWidth: 1, borderColor: THEME.border },
  cardTitle: { fontSize: 17, fontWeight: "800", color: THEME.text },
  item: { color: THEME.text, lineHeight: 20 },
  warnText: { color: THEME.warn },
  mutedText: { color: THEME.textMuted, fontSize: 13, lineHeight: 18 },
  fieldLabel: { color: THEME.textMuted, fontSize: 12, fontWeight: "700", marginTop: 2 },
  input: { borderWidth: 1, borderColor: THEME.border, borderRadius: 10, padding: 10, backgroundColor: "#fff" },
  compactInput: { minWidth: 110, flexGrow: 1 },
  compactLabel: { minWidth: 110, flexGrow: 1 },
  tokenInput: { minHeight: 88, textAlignVertical: "top" },
  jsonInput: { minHeight: 180, textAlignVertical: "top", fontFamily: "monospace" },
  button: { backgroundColor: THEME.accent, paddingVertical: 10, paddingHorizontal: 14, borderRadius: 10, alignSelf: "flex-start" },
  secondaryButton: { backgroundColor: "#d7efe9", paddingVertical: 10, paddingHorizontal: 14, borderRadius: 10, alignSelf: "flex-start" },
  secondaryButtonText: { color: THEME.accentDeep, fontWeight: "800" },
  smallButton: { backgroundColor: THEME.accentDeep, paddingVertical: 6, paddingHorizontal: 10, borderRadius: 8, alignSelf: "flex-start" },
  mutedButton: { backgroundColor: "#7c6d5a", paddingVertical: 6, paddingHorizontal: 10, borderRadius: 8, alignSelf: "flex-start" },
  cancelButton: { backgroundColor: "#9a3412", paddingVertical: 10, paddingHorizontal: 14, borderRadius: 10, alignSelf: "flex-start" },
  buttonText: { color: "white", fontWeight: "700" },
  checkbox: { width: 20, height: 20, borderWidth: 2, borderColor: THEME.border, borderRadius: 4, marginRight: 8 },
  checkboxChecked: { backgroundColor: THEME.accent, borderColor: THEME.accent },
  inlineRow: { gap: 8, flexDirection: "row", alignItems: "center" },
  inlineRowWrap: { gap: 8, flexDirection: "row", alignItems: "center", flexWrap: "wrap" },
  rowWrap: { flexDirection: "row", flexWrap: "wrap", gap: 6, marginBottom: 8 },
  chip: { paddingVertical: 5, paddingHorizontal: 10, borderRadius: 20, borderWidth: 1, borderColor: THEME.border, backgroundColor: THEME.panelAlt },
  chipActive: { borderColor: THEME.accent, backgroundColor: THEME.accent },
  chipText: { color: THEME.text, fontSize: 12 },
  chipActiveText: { color: "#fff", fontWeight: "700" },
  buttonDisabled: { opacity: 0.5 },
  panelSplit: { flexDirection: "row", gap: 12, flexWrap: "wrap" },
  panelCol: { flex: 1, minWidth: 320 },
  chartOuter: { gap: 4 },
  chartBodyRow: { flexDirection: "row", gap: 8, alignItems: "stretch" },
  chartYAxis: { width: 58, justifyContent: "space-between", paddingVertical: 2 },
  chartXAxis: { flexDirection: "row", justifyContent: "space-between", marginLeft: 66 },
  chartAxisText: { color: THEME.textMuted, fontSize: 11 },
  chartFrame: { flex: 1, borderWidth: 1, borderColor: THEME.border, borderRadius: 12, paddingHorizontal: 8, paddingBottom: 6, paddingTop: 6, backgroundColor: THEME.panelAlt, overflow: "hidden", position: "relative" },
  chartMeasureLayer: { position: "absolute", left: 0, right: 0, top: 0, bottom: 0 },
  progressWrap: { gap: 6, padding: 10, borderWidth: 1, borderColor: THEME.border, borderRadius: 10, backgroundColor: "#fff" },
  progressCompactWrap: { gap: 6, padding: 8, borderWidth: 1, borderColor: THEME.border, borderRadius: 10, backgroundColor: "#fff" },
  progressLabel: { color: THEME.text, fontWeight: "700" },
  progressTrack: { height: 10, borderRadius: 999, backgroundColor: "#efe5d3", overflow: "hidden" },
  progressFill: { height: "100%", borderRadius: 999, backgroundColor: THEME.accentDeep },
  progressValue: { color: THEME.textMuted, fontSize: 12 },
  chartLayer: { position: "absolute", left: 8, right: 8, top: 6, bottom: 6 },
  chartPoint: { position: "absolute", bottom: 0, borderRadius: 999 },
  chartLineDot: { position: "absolute" },
  chartGridLine: { position: "absolute", left: 8, right: 8, borderTopWidth: 1, borderTopColor: "rgba(35, 31, 26, 0.12)" },
  distributionList: { gap: 6 },
  distributionRow: { flexDirection: "row", alignItems: "center", gap: 8 },
  distributionLabel: { width: 64, color: THEME.text, fontSize: 12 },
  distributionTrack: { flex: 1, backgroundColor: "#efe5d3", borderRadius: 999, height: 10, overflow: "hidden" },
  distributionFill: { height: "100%", borderRadius: 999 },
  distributionValue: { width: 56, color: THEME.textMuted, fontSize: 12, textAlign: "right" },
  scoreList: { gap: 6 },
  scoreRow: { flexDirection: "row", alignItems: "center", gap: 8 },
  scoreLabel: { width: 120, color: THEME.text, fontSize: 12, textTransform: "capitalize" },
  scoreTrack: { flex: 1, backgroundColor: "#efe5d3", borderRadius: 999, height: 10, overflow: "hidden" },
  scoreFill: { height: "100%", borderRadius: 999, backgroundColor: THEME.accentDeep },
  scoreValue: { width: 44, color: THEME.textMuted, fontSize: 12, textAlign: "right" },
  analyticsItemCard: { borderWidth: 1, borderColor: THEME.border, borderRadius: 12, padding: 10, gap: 6, backgroundColor: THEME.panelAlt },
  analyticsHeadline: { color: THEME.text, fontWeight: "800" },
  timelineFrame: { borderWidth: 1, borderColor: THEME.border, borderRadius: 12, padding: 10, backgroundColor: THEME.panelAlt, gap: 8 },
  timelineChartWrap: { height: 140, borderWidth: 1, borderColor: THEME.border, borderRadius: 12, backgroundColor: "#fff", overflow: "hidden", position: "relative" },
  timelineMarker: { position: "absolute", bottom: 0, alignItems: "center", marginLeft: -8, width: 16 },
  timelineDot: { width: 8, height: 8, borderRadius: 999, marginBottom: 2 },
  timelineMarkerText: { fontSize: 10, fontWeight: "800" },
  newsSummaryList: { gap: 8 },
  newsSummaryCard: { borderWidth: 1, borderColor: THEME.border, borderRadius: 12, padding: 10, backgroundColor: "#fff" },
  chatEmptyState: { padding: 16, borderRadius: 12, backgroundColor: THEME.panelAlt, marginBottom: 8 },
  chatBubbleUser: { alignSelf: "flex-end", backgroundColor: THEME.accentDeep, borderRadius: 16, borderBottomRightRadius: 4, padding: 12, marginVertical: 4, maxWidth: "85%" },
  chatBubbleAssistant: { alignSelf: "flex-start", backgroundColor: THEME.panelAlt, borderWidth: 1, borderColor: THEME.border, borderRadius: 16, borderBottomLeftRadius: 4, padding: 12, marginVertical: 4, maxWidth: "92%" },
  chatLabelUser: { fontSize: 11, fontWeight: "700", color: "rgba(255,255,255,0.8)", marginBottom: 4 },
  chatLabelAssistant: { fontSize: 11, fontWeight: "700", color: THEME.textMuted, marginBottom: 4 },
  chatTextUser: { color: "#fff", fontSize: 14, lineHeight: 20 },
  chatTextAssistant: { color: THEME.text, fontSize: 14, lineHeight: 20 },
  chatMeta: { color: THEME.textMuted, fontSize: 12, marginTop: 6, lineHeight: 18 },
  chatToolCallRow: { flexDirection: "row", alignItems: "center", justifyContent: "space-between", gap: 8, marginTop: 6 },
  chatToolCallText: { flex: 1, color: THEME.text, fontSize: 12, fontFamily: "monospace" },
  chatInputWrap: { gap: 8, width: "100%" },
  chatInput: { flex: 1, width: "100%", minHeight: 48, maxHeight: 120, textAlignVertical: "top" },
  label: { color: THEME.textMuted, fontSize: 13 },
  metricRow: { flexDirection: "row", justifyContent: "space-between", paddingVertical: 4 },
  metricLabel: { color: THEME.textMuted, fontSize: 13 },
  metricValue: { color: THEME.text, fontSize: 15, fontWeight: "700" },
});