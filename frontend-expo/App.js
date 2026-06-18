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
    return `${protocol}//${host}:8100`;
  }
  return "http://127.0.0.1:8100";
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
const SCREENS = ["Overview", "Scanner", "Recommend", "Analytics", "Charts", "Options", "Backtest", "Execution", "Assistant", "Presets", "Auth"];

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
  if (!rows.length) {
    return <Text style={styles.item}>No distribution loaded.</Text>;
  }
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
  if (!entries.length) {
    return <Text style={styles.item}>No score breakdown loaded.</Text>;
  }
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
  if (!priceRows.length) {
    return <Text style={styles.item}>No timeline data loaded.</Text>;
  }

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
    let best = 0;
    let bestDistance = Number.POSITIVE_INFINITY;
    const target = new Date(String(date)).getTime();
    priceDates.forEach((d, index) => {
      const current = new Date(d).getTime();
      const distance = Math.abs(current - target);
      if (Number.isFinite(distance) && distance < bestDistance) {
        bestDistance = distance;
        best = index;
      }
    });
    return best;
  };

  return (
    <View style={styles.timelineFrame}>
      <View style={styles.timelineChartWrap}>
        {(() => {
          const width = 1000;
          const height = 126;
          const points = priceValues
            .map((value, idx) => {
              if (!Number.isFinite(value)) return null;
              const x = maxLen > 1 ? (idx / (maxLen - 1)) * width : 0;
              const y = height - ((value - min) / span) * height;
              return { x, y };
            })
            .filter(Boolean);
          const cloud = buildLinePointCloud(points, 4);
          return cloud.map((pt, idx) => (
            <View
              key={`timeline-line-${String(idx)}`}
              style={[
                styles.chartLineDot,
                {
                  left: `${(pt.x / width) * 100}%`,
                  top: pt.y,
                  width: 2,
                  height: 2,
                  borderRadius: 2,
                  backgroundColor: THEME.text,
                  opacity: 0.95,
                },
              ]}
            />
          ));
        })()}
        {markers.slice(0, 18).map((marker, idx) => {
          const markerIndex = findMarkerIndex(marker.date);
          const left = maxLen > 1 ? `${(markerIndex / (maxLen - 1)) * 100}%` : "0%";
          const color = markerColor(marker.sentiment_label);
          return (
            <View key={`marker-${String(idx)}-${marker.date}`} style={[styles.timelineMarker, { left }]}> 
              <View style={[styles.timelineDot, { backgroundColor: color }]} />
              <Text style={[styles.timelineMarkerText, { color }]}>{marker.sentiment_label?.slice(0, 1)?.toUpperCase() || "N"}</Text>
            </View>
          );
        })}
      </View>
      <View style={styles.timelineLegendRow}>
        <Text style={styles.item}>Price line with news markers: B bullish, N neutral, R bearish.</Text>
      </View>
    </View>
  );
}

function NewsSummaryList({ summary }) {
  const groups = summary?.items || [];
  if (!groups.length) {
    return <Text style={styles.item}>No article summary loaded.</Text>;
  }
  return (
    <View style={styles.newsSummaryList}>
      {groups.map((group) => (
        <View key={group.date} style={styles.newsSummaryCard}>
          <Text style={styles.analyticsHeadline}>{group.date} | {group.count} articles | avg sentiment {fmtNumber(group.avg_sentiment)}</Text>
          <Text style={styles.item}>{group.summary || "No summary text available."}</Text>
        </View>
      ))}
    </View>
  );
}

function ProgressBar({ progress, label, compact = false }) {
  const pct = Math.max(6, Math.min(98, Number(progress) || 0));
  return (
    <View style={compact ? styles.progressCompactWrap : styles.progressWrap}>
      {!!label && <Text style={styles.progressLabel}>{label}</Text>}
      <View style={styles.progressTrack}>
        <View style={[styles.progressFill, { width: `${pct}%` }]} />
      </View>
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
    try {
      return await fn();
    } catch (e) {
      lastError = e;
      const message = String(e?.message || "");
      const isNetworkError = message.includes("Failed to fetch") || message.includes("Network request failed");
      const isRetryable = e.status >= 500 || message.includes("aborted") || isNetworkError;
      if (!isRetryable || attempt === maxRetries - 1) {
        if (isNetworkError) {
          const err = new Error(`Failed to fetch from ${API_BASE}. Ensure the API is running and EXPO_PUBLIC_API_BASE_URL is correct.`);
          err.status = e.status;
          throw err;
        }
        throw e;
      }
      const delayMs = baseDelayMs * Math.pow(2, attempt) + Math.random() * 100;
      await new Promise(resolve => setTimeout(resolve, delayMs));
    }
  }
  throw lastError;
}

async function apiGet(path, auth, options = {}) {
  return withRetry(async () => {
    const res = await fetch(`${API_BASE}${path}`, { headers: { ...authHeaders(auth) }, signal: options.signal });
    if (!res.ok) {
      const err = new Error(`HTTP ${res.status}`);
      err.status = res.status;
      throw err;
    }
    return res.json();
  });
}

async function apiPost(path, payload, auth, options = {}) {
  return withRetry(async () => {
    const res = await fetch(`${API_BASE}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeaders(auth) },
      body: JSON.stringify(payload),
      signal: options.signal,
    });
    if (!res.ok) {
      const err = new Error(`HTTP ${res.status}`);
      err.status = res.status;
      throw err;
    }
    return res.json();
  });
}

async function apiPut(path, payload, auth, options = {}) {
  return withRetry(async () => {
    const res = await fetch(`${API_BASE}${path}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json", ...authHeaders(auth) },
      body: JSON.stringify(payload),
      signal: options.signal,
    });
    if (!res.ok) {
      const err = new Error(`HTTP ${res.status}`);
      err.status = res.status;
      throw err;
    }
    return res.json();
  });
}

export default function App() {
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

  const [backtestStart, setBacktestStart] = useState("2020-01-01");
  const [backtestEnd, setBacktestEnd] = useState("");
  const [backtestEntryRsi, setBacktestEntryRsi] = useState("50");
  const [backtestMaxHoldDays, setBacktestMaxHoldDays] = useState("7");
  const [backtestStopLossPct, setBacktestStopLossPct] = useState("0");
  const [backtestMaFilter, setBacktestMaFilter] = useState("sma20");
  const [backtestTakeProfitPct, setBacktestTakeProfitPct] = useState("0");
  const [backtestMaTrendFilter, setBacktestMaTrendFilter] = useState("none");

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
  const [inFlightOps, setInFlightOps] = useState(0);
  const [activeTaskLabel, setActiveTaskLabel] = useState("");
  const [progressTick, setProgressTick] = useState(Date.now());

  const authState = useMemo(
    () => ({ apiKey: apiKeyInput.trim(), bearerToken: bearerToken.trim() }),
    [apiKeyInput, bearerToken]
  );

  const baseLabel = useMemo(() => `API: ${API_BASE}`, []);

  const selectedPresets = useMemo(() => parseCsv(selectedPresetsText), [selectedPresetsText]);

  const availablePresets = useMemo(() => {
    const items = presetDetails?.items || presets || [];
    const q = presetSearch.trim().toLowerCase();
    if (!q) return items;
    return items.filter((name) => String(name).toLowerCase().includes(q));
  }, [presetDetails, presets, presetSearch]);

  const filteredUniverse = useMemo(() => {
    const q = universePresetFilter.trim().toLowerCase();
    const lim = Math.max(1, Math.min(200, parseNumber(universeLimit, 24)));
    const rows = q
      ? (latestUniverse || []).filter((r) => String(r.preset || "").toLowerCase().includes(q))
      : (latestUniverse || []);

    const sorted = [...rows].sort((a, b) => String(b.created_at || "").localeCompare(String(a.created_at || "")));
    return sorted.slice(0, lim);
  }, [latestUniverse, universePresetFilter, universeLimit]);

  const indicatorTail = useMemo(() => (indicatorChart || []).slice(-90), [indicatorChart]);
  const forecastHistory = useMemo(() => (forecastChart?.history || []).slice(-30), [forecastChart]);
  const forecastForward = useMemo(() => forecastChart?.forecast || [], [forecastChart]);
  const seasonalityTail = useMemo(() => {
    const rows = (seasonalityChart || []).map((row, idx) => ({ ...row, _idx: idx }));
    rows.sort((a, b) => {
      const ad = Number(a?.doy ?? a?.day_of_year ?? NaN);
      const bd = Number(b?.doy ?? b?.day_of_year ?? NaN);
      if (Number.isFinite(ad) && Number.isFinite(bd)) return ad - bd;
      if (Number.isFinite(ad)) return -1;
      if (Number.isFinite(bd)) return 1;
      return Number(a._idx || 0) - Number(b._idx || 0);
    });
    return rows;
  }, [seasonalityChart]);
  const scanHistoryRowsSeries = useMemo(
    () => (scanHistory || []).slice(0, 20).map((row) => Number(row.rows_saved || 0)).reverse(),
    [scanHistory]
  );
  const scanHistoryDurationSeries = useMemo(
    () => (scanHistory || []).slice(0, 20).map((row) => Number(row.elapsed_seconds || 0)).reverse(),
    [scanHistory]
  );
  const scanHistoryFiltered = useMemo(() => {
    const rows = scanHistory || [];
    if (scanHistoryFilter === "all") return rows;
    return rows.filter((row) => String(row.status || "").toLowerCase() === scanHistoryFilter);
  }, [scanHistory, scanHistoryFilter]);

  const isWeb = typeof window !== "undefined";
  const scanPollRef = useRef(null);
  const progressStartedAtRef = useRef(0);
  const assistantRunRef = useRef(0);

  useEffect(() => {
    if (inFlightOps > 0 && !progressStartedAtRef.current) {
      progressStartedAtRef.current = Date.now();
    }
    if (!inFlightOps) {
      progressStartedAtRef.current = 0;
    }
  }, [inFlightOps]);

  useEffect(() => {
    if (!inFlightOps) return;
    const id = setInterval(() => setProgressTick(Date.now()), 250);
    return () => clearInterval(id);
  }, [inFlightOps]);

  const loadStoredCredentials = async () => {
    try {
      const stored = await AsyncStorage.getItem("quantflow_credentials");
      if (stored) {
        const { apiKey, bearerToken: token, rememberMe } = JSON.parse(stored);
        if (apiKey) setApiKeyInput(apiKey);
        if (token) setBearerToken(token);
        setRememberCredentials(rememberMe || false);
      }
    } catch (e) {
      console.error("Failed to load stored credentials:", e);
    }
  };

  const saveCredentials = async () => {
    try {
      if (rememberCredentials) {
        await AsyncStorage.setItem("quantflow_credentials", JSON.stringify({
          apiKey: apiKeyInput.trim(),
          bearerToken: bearerToken.trim(),
          rememberMe: true,
        }));
      } else {
        await AsyncStorage.removeItem("quantflow_credentials");
      }
    } catch (e) {
      console.error("Failed to save credentials:", e);
    }
  };

  const logoutWorkspace = async () => {
    await AsyncStorage.removeItem("quantflow_credentials");
    setApiKeyInput(API_KEY);
    setBearerToken("");
    setRememberCredentials(false);
    setWorkspaceUnlocked(false);
    setFirebaseAuthCheck(null);
    setSurface("Public");
    setScreen("Overview");
    setStatus("logged out");
  };

  const progressValue = useMemo(() => {
    if (!inFlightOps) return 0;
    const start = progressStartedAtRef.current || progressTick;
    const elapsedMs = Math.max(0, progressTick - start);
    const asymptoticPct = 12 + 80 * (1 - Math.exp(-elapsedMs / 2200));
    return Math.min(92, asymptoticPct);
  }, [inFlightOps, progressTick]);

  const withProgress = async (label, action) => {
    setActiveTaskLabel(label);
    setInFlightOps((value) => value + 1);
    try {
      return await action();
    } finally {
      setInFlightOps((value) => Math.max(0, value - 1));
    }
  };

  const scannerBusy = inFlightOps > 0 && status.includes("scan");
  const recommendBusy = inFlightOps > 0 && status.includes("recommend");
  const analyticsBusy = inFlightOps > 0 && status.includes("analytics");
  const optionsBusy = inFlightOps > 0 && status.includes("options");
  const backtestBusy = inFlightOps > 0 && status.includes("backtest");
  const executionBusy = inFlightOps > 0 && status.includes("propos");
  const assistantBusy = inFlightOps > 0 && (status.includes("assistant") || status.includes("run "));
  const authBusy = inFlightOps > 0 && (settingsSaveStatus === "loading" || settingsSaveStatus === "saving" || status.includes("auth") || status.includes("identity"));
  const scanIsActive = ["queued", "running", "cancel_requested"].includes(scanJobStatus);

  const stopScanPolling = () => {
    if (scanPollRef.current) {
      clearInterval(scanPollRef.current);
      scanPollRef.current = null;
    }
  };

  const applyScanStatus = (snapshot) => {
    if (!snapshot) return;
    setScanJobId(snapshot.job_id || "");
    setScanJobStatus(String(snapshot.status || "idle"));
    setScanProgressPct(Number(snapshot.progress_pct || 0));
    setScanEtaSeconds(snapshot.eta_seconds);
    setScanElapsedSeconds(snapshot.elapsed_seconds);
    setScanCurrentPreset(snapshot.current_preset || "");
    setScanFailedPresets(snapshot.failed_presets || []);
    setScanPresetsCompleted(Number(snapshot.presets_completed || 0));
    setScanPresetCount(Number(snapshot.preset_count || 0));
    setScanStartedAt(snapshot.started_at || "");
    setScanFinishedAt(snapshot.finished_at || "");
    setScanLastMessage(snapshot.message || "");

    const terminalStates = ["completed", "failed", "canceled"];
    if (terminalStates.includes(String(snapshot.status || ""))) {
      stopScanPolling();
      if (snapshot.status === "completed" || snapshot.status === "canceled") {
        loadOverview();
      }
      loadScanHistory(scanHistoryFilter);
      if (snapshot.status === "completed") {
        setStatus("scan completed");
        setScanResult({
          selected_presets: snapshot.selected_presets || [],
          preset_count: Number(snapshot.preset_count || 0),
          rows_saved: Number(snapshot.rows_saved || 0),
          failed_presets: snapshot.failed_presets || [],
        });
      } else if (snapshot.status === "canceled") {
        setStatus("scan canceled");
      } else {
        setStatus("scan failed");
      }
    }
  };

  const startScanPolling = (jobId) => {
    stopScanPolling();
    scanPollRef.current = setInterval(async () => {
      try {
        const snap = await apiGet(`/pipeline/scan/status/${encodeURIComponent(jobId)}`, authState);
        applyScanStatus(snap);
      } catch (e) {
        setScanLastMessage(`Status polling error: ${e.message}`);
      }
    }, 1200);
  };

  const parseUrlState = () => {
    if (!isWeb) return;
    try {
      const params = new URLSearchParams(window.location.search || "");
      const view = params.get("view");
      const tab = params.get("tab");

      if (view === "public") {
        setSurface("Public");
      } else if (view === "workspace") {
        setSurface("Workspace");
      }

      if (tab && SCREENS.includes(tab)) {
        setScreen(tab);
      }
    } catch (e) {
      // Ignore malformed URL state and continue with defaults.
    }
  };

  const syncUrlState = (nextSurface, nextScreen) => {
    if (!isWeb) return;
    try {
      const params = new URLSearchParams(window.location.search || "");
      params.set("view", String(nextSurface).toLowerCase());
      if (nextSurface === "Workspace") {
        params.set("tab", String(nextScreen));
      } else {
        params.delete("tab");
      }
      const qs = params.toString();
      const nextUrl = `${window.location.pathname}${qs ? `?${qs}` : ""}`;
      window.history.replaceState({}, "", nextUrl);
    } catch (e) {
      // URL sync is best-effort for web demo deep linking.
    }
  };

  useEffect(() => {
    parseUrlState();
    loadStoredCredentials();
    loadOverview();
    return () => {
      stopScanPolling();
    };
  }, []);

  useEffect(() => {
    syncUrlState(surface, screen);
  }, [surface, screen]);

  useEffect(() => {
    loadScanHistory(scanHistoryFilter);
  }, [scanHistoryFilter]);

  const loadMcpStatus = async (options = {}) => {
    await withProgress("Checking MCP", async () => {
      try {
        const [data, readiness, signals] = await Promise.all([
          apiGet("/broker/mcp/status", authState, options),
          apiGet("/broker/mcp/readiness", authState, options),
          apiGet("/portfolio/signals?broker_mode=robinhood_mcp", authState, options).catch((e) => ({ error: e.message })),
        ]);
        setMcpStatus(data || null);
        setMcpReadiness(readiness || null);
        if (signals?.error) {
          setMcpSignals(null);
          setMcpSignalsError(String(signals.error));
        } else {
          setMcpSignals(signals || null);
          setMcpSignalsError("");
        }
      } catch (e) {
        setMcpStatus({ connected: false, authenticated: false, error: e.message });
        setMcpReadiness(null);
        setMcpSignals(null);
        setMcpSignalsError(String(e.message));
      }
    });
  };

  const connectMcp = async () => {
    setStatus("connecting mcp");
    await withProgress("Connecting Robinhood MCP", async () => {
      try {
        const [statusData, readinessData, signalsData] = await Promise.all([
          apiGet("/broker/mcp/status", authState),
          apiGet("/broker/mcp/readiness", authState),
          apiGet("/portfolio/signals?broker_mode=robinhood_mcp", authState).catch((e) => ({ error: e.message })),
        ]);
        setMcpStatus(statusData || null);
        setMcpReadiness(readinessData || null);
        if (signalsData?.error) {
          setMcpSignals(null);
          setMcpSignalsError(String(signalsData.error));
        } else {
          setMcpSignals(signalsData || null);
          setMcpSignalsError("");
        }
        setMcpLoginResult({
          ok: Boolean(statusData?.authenticated),
          message: statusData?.authenticated
            ? "Connected. Robinhood MCP account data is available to QuantFlow."
            : statusData?.connected
              ? "Transport is reachable. Complete Robinhood account auth in your MCP client, then re-check."
              : "MCP transport is not reachable yet. Check endpoint/config and retry.",
        });
        setStatus("ready");
      } catch (e) {
        setMcpLoginResult({ ok: false, error: e.message });
        setStatus(`error: ${e.message}`);
      }
    });
  };

  const loadScanHistory = async (statusFilter = "all") => {
    try {
      const q = statusFilter && statusFilter !== "all" ? `&status=${encodeURIComponent(statusFilter)}` : "";
      const data = await apiGet(`/pipeline/scan/history?limit=24${q}`, authState);
      setScanHistory(data.items || []);
    } catch (e) {
      setScanLastMessage(`Scan history unavailable: ${e.message}`);
    }
  };

  const loadOverview = async (options = {}) => {
    setStatus("loading");
    await withProgress("Loading overview", async () => {
      try {
        const [h, p, pd, i, u, r, o, envv, history] = await Promise.all([
          apiGet("/health", authState, options),
          apiGet("/scanner/presets", authState, options),
          apiGet("/scanner/presets/details", authState, options).catch(() => null),
          apiGet("/execution/intents?limit=10", authState, options),
          apiGet("/scanner/latest-universe", authState, options),
          apiGet("/recommend/latest", authState, options),
          apiGet("/ops/latest-report", authState, options).catch(() => null),
          apiGet("/ops/env/validate", authState, options).catch(() => null),
          apiGet(`/pipeline/scan/history?limit=24${scanHistoryFilter !== "all" ? `&status=${encodeURIComponent(scanHistoryFilter)}` : ""}`, authState, options).catch(() => ({ items: [] })),
        ]);
        setHealth(h);
        setOpsReport(o);
        setEnvValidation(envv);
        setPresets(p.items || []);
        setPresetDetails(pd || null);
        setIntents(i.items || []);
        setLatestUniverse(u.items || []);
        setScanUniverseRefreshedAt(new Date().toISOString());
        setLatestRecs(r.items || []);
        setScanHistory(history?.items || []);
        await loadMcpStatus(options);
        setStatus("ready");
      } catch (e) {
        if (isAbortError(e)) {
          setStatus("canceled");
          return;
        }
        setStatus(`error: ${e.message}`);
      }
    });
  };

  const togglePreset = (name) => {
    const set = new Set(selectedPresets);
    if (set.has(name)) {
      set.delete(name);
    } else {
      set.add(name);
    }
    setSelectedPresetsText(Array.from(set).join(","));
  };

  const selectPresetCategory = (categoryName) => {
    const items = presetDetails?.categories?.[categoryName] || [];
    setSelectedPresetsText(items.join(","));
  };

  const signInWithGoogleWeb = async () => {
    if (!isWeb) {
      setGoogleSignInStatus("Google popup login is available on web builds only.");
      return;
    }
    setGoogleSignInStatus("starting");
    await withProgress("Signing in", async () => {
      try {
      if (!FIREBASE_WEB_CONFIG.apiKey || !FIREBASE_WEB_CONFIG.authDomain || !FIREBASE_WEB_CONFIG.projectId || !FIREBASE_WEB_CONFIG.appId) {
        throw new Error("Missing EXPO_PUBLIC_FIREBASE_* config variables for web login");
      }

      const [{ initializeApp, getApps }, { getAuth, GoogleAuthProvider, signInWithPopup }] = await Promise.all([
        import("firebase/app"),
        import("firebase/auth"),
      ]);

      const app = getApps().length ? getApps()[0] : initializeApp(FIREBASE_WEB_CONFIG);
      const auth = getAuth(app);
      const provider = new GoogleAuthProvider();
      const result = await signInWithPopup(auth, provider);
      const idToken = await result.user.getIdToken();
      setBearerToken(idToken);
      setGoogleSignInStatus(`signed in as ${result.user.email || result.user.uid}`);
      await checkFirebaseIdentity();
      } catch (e) {
        setGoogleSignInStatus(`error: ${e.message}`);
      }
    });
  };

  const askAssistant = async () => {
    const userMessage = assistantInput.trim();
    if (!userMessage) return;
    const runId = Date.now();
    assistantRunRef.current = runId;
    const userTurn = { role: "user", content: userMessage, ts: new Date().toISOString() };
    setAssistantConversation((prev) => [...prev, userTurn]);
    setAssistantInput("");
    setAssistantAutoResults([]);
    setAssistantAutoSummary(null);
    setStatus("assistant query");
    await withProgress("Running assistant query", async () => {
      try {
        const data = await apiPost("/assistant/query", { message: userMessage, db: "sqlite:///quantflow.db" }, authState);
        let enhanced = data || null;
        const userLower = userMessage.toLowerCase();
        const answerText = String(data?.answer || "").toLowerCase();
        const isGeneric = answerText.includes("i can answer from offline quantflow data") || answerText.includes("ask for mcp status");
        const asksRanking = /rank|best|high conviction|top|opportunit/.test(userLower);
        const asksMultiFactor = /technical|seasonal|seasonality|temporal|sentiment/.test(userLower);
        const asksMcp = /mcp|robinhood|broker|account auth|login/.test(userLower);
        const isSmallTalk = /^[\W_]+$/.test(userLower) || /^(hi|hello|hey|yo|how are you|how r you|thanks|thank you|ok|okay|cool)$/.test(userLower);
        const asksCapabilities = /what else can you do|what can you do|what do you do|\bhelp\b|\bcapabilities\b/.test(userLower);
        const shouldAutoRankEnhance = (asksRanking || asksMultiFactor) && !isSmallTalk;
        const shouldSuppressAutoRun = isSmallTalk || asksCapabilities;

        if (asksMcp) {
          const [statusData, readinessData, signalsData] = await Promise.all([
            apiGet("/broker/mcp/status", authState).catch((e) => ({ error: e.message })),
            apiGet("/broker/mcp/readiness", authState).catch((e) => ({ error: e.message })),
            apiGet("/portfolio/signals?broker_mode=robinhood_mcp", authState).catch((e) => ({ error: e.message })),
          ]);

          if (!statusData?.error) setMcpStatus(statusData || null);
          if (!readinessData?.error) setMcpReadiness(readinessData || null);
          if (!signalsData?.error) {
            setMcpSignals(signalsData || null);
            setMcpSignalsError("");
          } else {
            setMcpSignalsError(String(signalsData.error || ""));
          }

          const readyState = statusData?.authenticated
            ? "authenticated"
            : statusData?.connected
              ? "transport connected but account auth incomplete"
              : "not connected";
          const signalCount = Number(signalsData?.count || 0);
          const nextSteps = (readinessData?.next_steps || []).slice(0, 3).join(" ");

          enhanced = {
            ...(enhanced || {}),
            answer: `Robinhood MCP status is ${readyState}. Portfolio signal rows available: ${signalCount}. ${nextSteps || "Use Check MCP and Connect Robinhood, then complete account auth in your MCP client session and re-check."}`,
            suggested_tool_calls: [
              { name: "check_mcp_status", method: "GET", path: "/broker/mcp/status" },
              { name: "check_mcp_readiness", method: "GET", path: "/broker/mcp/readiness" },
              { name: "load_portfolio_signals", method: "GET", path: "/portfolio/signals?broker_mode=robinhood_mcp" },
            ],
          };
        }

        if (shouldAutoRankEnhance || (isGeneric && !isSmallTalk && (asksRanking || asksMultiFactor))) {
          const latest = await apiGet("/recommend/latest?limit=30", authState).catch(() => ({ items: [] }));
          const rows = (latest?.items || []).map((row) => {
            const entry = Number(row.entry || 0);
            const target = Number(row.target || 0);
            const confidence = Number(row.confidence || 0);
            const bias = String(row.bias || "neutral").toLowerCase();
            const edge = entry > 0 ? (target - entry) / entry : 0;
            const conviction = confidence * 0.7 + Math.max(-1, Math.min(1, edge)) * 0.3;
            return {
              ticker: String(row.ticker || "").toUpperCase(),
              horizon: String(row.horizon || ""),
              bias,
              confidence,
              edge,
              conviction,
            };
          }).filter((row) => row.ticker && row.bias !== "neutral");

          const bestByTicker = [];
          const seen = new Set();
          rows.sort((a, b) => b.conviction - a.conviction || b.confidence - a.confidence);
          for (const row of rows) {
            if (seen.has(row.ticker)) continue;
            seen.add(row.ticker);
            bestByTicker.push(row);
            if (bestByTicker.length >= 6) break;
          }

          if (bestByTicker.length) {
            const top = bestByTicker.slice(0, 5);
            const primary = top[0].ticker;
            const bullet = top.map((r) => `${r.ticker} (${r.bias}, conf ${fmtNumber(r.confidence, 2)}, edge ${fmtPct(r.edge)})`).join("; ");
            enhanced = {
              ...(data || {}),
              answer: `Top ranked opportunities from your latest local recommendations: ${bullet}. Use the actions below to run technical, seasonality/temporal, and sentiment analysis immediately.`,
              opportunities: top,
              suggested_tool_calls: [
                { name: "load_recommend_latest", method: "GET", path: "/recommend/latest" },
                { name: "analyze_top_basket", method: "GET", path: `/recommend/analyze?tickers=${encodeURIComponent(top.map((r) => r.ticker).join(","))}&period=${encodeURIComponent(analyticsPeriod)}&interval=${encodeURIComponent(analyticsInterval)}&limit=${top.length}` },
                { name: "indicators_primary", method: "GET", path: `/charts/indicators?ticker=${encodeURIComponent(primary)}&period=${encodeURIComponent(analyticsPeriod)}&interval=${encodeURIComponent(analyticsInterval)}` },
                { name: "seasonality_primary", method: "GET", path: `/charts/seasonality-compare?ticker=${encodeURIComponent(primary)}&years=${encodeURIComponent(Math.max(3, Math.min(20, parseNumber(seasonalityYears, 10))))}` },
                { name: "sentiment_primary", method: "GET", path: `/news/summary?ticker=${encodeURIComponent(primary)}&db=${encodeURIComponent(dbPath)}&days=${encodeURIComponent(parseNumber(newsDays, 30))}&sentiment=${encodeURIComponent(newsSentiment)}&max_groups=8` },
                { name: "check_mcp", method: "GET", path: "/broker/mcp/status" },
              ],
            };
          }
        }

        setAssistantResponse(enhanced || null);
        setAssistantActionResult(null);
        const assistantTurn = {
          role: "assistant",
          content: enhanced?.answer || (enhanced ? compactJson(enhanced, 800) : "No response."),
          suggested_tool_calls: enhanced?.suggested_tool_calls || [],
          summary: enhanced?.summary || null,
          ts: new Date().toISOString(),
        };
        setAssistantConversation((prev) => [...prev, assistantTurn]);

        if (assistantAutoRunGet && !shouldSuppressAutoRun) {
          const autoCalls = (enhanced?.suggested_tool_calls || [])
            .filter((c) => String(c?.method || "GET").toUpperCase() === "GET")
            .slice(0, 3);
          if (autoCalls.length) {
            const results = [];
            for (const call of autoCalls) {
              if (assistantRunRef.current !== runId) return;
              try {
                const result = await apiGet(call.path, authState);
                results.push({ ok: true, call, result });
                if (call.path.includes("/broker/mcp/status")) setMcpStatus(result || null);
                if (call.path.includes("/broker/mcp/readiness")) setMcpReadiness(result || null);
                if (call.path.includes("/portfolio/signals")) {
                  setMcpSignals(result || null);
                  setMcpSignalsError("");
                }
              } catch (e) {
                results.push({ ok: false, call, error: e.message });
              }
            }
            if (assistantRunRef.current !== runId) return;
            setAssistantAutoResults(results);
            const okCount = results.filter((r) => r.ok).length;
            setAssistantAutoSummary({
              ts: new Date().toISOString(),
              total: results.length,
              ok: okCount,
              failed: results.length - okCount,
            });
          }
        }
        setStatus("ready");
      } catch (e) {
        setAssistantConversation((prev) => [...prev, { role: "assistant", content: `Error: ${e.message}`, ts: new Date().toISOString() }]);
        setStatus(`error: ${e.message}`);
      }
    });
  };

  const executeSuggestedToolCall = async (call) => {
    setStatus(`run ${call.name || "action"}`);
    await withProgress(`Running ${call.name || "action"}`, async () => {
      try {
      const method = (call.method || "GET").toUpperCase();
      if (method === "POST" && assistantDryRun) {
        setAssistantActionResult({
          ok: true,
          dryRun: true,
          call,
          result: {
            message: "Dry-run mode: POST action not executed.",
            would_call: call.path,
            payload: call.payload || {},
          },
        });
        setStatus("ready");
        return;
      }

      let result;
      if (method === "POST") {
        result = await apiPost(call.path, call.payload || {}, authState);
      } else {
        result = await apiGet(call.path, authState);
      }
      setAssistantActionResult({
        ok: true,
        call,
        result,
      });
      await loadOverview();
      setStatus("ready");
      } catch (e) {
        setAssistantActionResult({ ok: false, call, error: e.message });
        setStatus(`error: ${e.message}`);
      }
    });
  };

  const requestExecuteAction = (call) => {
    const method = (call.method || "GET").toUpperCase();
    if (method === "POST") {
      setPendingAction(call);
      return;
    }
    executeSuggestedToolCall(call);
  };

  const confirmPendingAction = async () => {
    if (!pendingAction) return;
    const call = pendingAction;
    setPendingAction(null);
    await executeSuggestedToolCall(call);
  };

  const cancelPendingAction = () => {
    setPendingAction(null);
    setStatus("ready");
  };

  const runMcpOnboardingRunbook = async () => {
    setStatus("assistant mcp runbook");
    setAssistantAutoResults([]);
    await withProgress("Running MCP onboarding runbook", async () => {
      const steps = [];
      const stamp = new Date().toISOString();

      const runStep = async (name, path) => {
        try {
          const result = await apiGet(path, authState);
          steps.push({ name, path, ok: true });
          return result;
        } catch (e) {
          steps.push({ name, path, ok: false, error: e.message });
          return { error: e.message };
        }
      };

      const statusData = await runStep("Check MCP status", "/broker/mcp/status");
      const readinessData = await runStep("Check MCP readiness", "/broker/mcp/readiness");
      const signalsData = await runStep("Load portfolio signals", "/portfolio/signals?broker_mode=robinhood_mcp");

      if (!statusData?.error) setMcpStatus(statusData || null);
      if (!readinessData?.error) setMcpReadiness(readinessData || null);
      if (!signalsData?.error) {
        setMcpSignals(signalsData || null);
        setMcpSignalsError("");
      } else {
        setMcpSignals(null);
        setMcpSignalsError(String(signalsData.error || "unknown error"));
      }

      const checks = readinessData?.checks || [];
      const blockers = checks.filter((row) => row.state === "fail");
      const warnings = checks.filter((row) => row.state === "warn");
      const remediation = [
        ...(readinessData?.next_steps || []),
      ];

      if (statusData?.connected && !statusData?.authenticated) {
        remediation.unshift("Complete Robinhood agentic account authentication in your MCP-capable client session, then rerun this runbook.");
      }
      if (!statusData?.connected) {
        remediation.unshift("Verify MCP endpoint connectivity and Robinhood MCP transport configuration.");
      }

      const report = {
        ran_at: stamp,
        transport_connected: Boolean(statusData?.connected),
        authenticated: Boolean(statusData?.authenticated),
        account_available: Boolean(statusData?.account_available),
        positions_available: Boolean(statusData?.positions_available),
        positions_count: Number(statusData?.positions_count || 0),
        signals_count: Number(signalsData?.count || 0),
        overall_ready: Boolean(readinessData?.overall_ready),
        blockers,
        warnings,
        remediation,
        steps,
      };

      setMcpRunbookReport(report);
      setAssistantConversation((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `MCP onboarding runbook complete. Ready: ${report.overall_ready ? "yes" : "no"}. Blockers: ${report.blockers.length}. Warnings: ${report.warnings.length}. Signals: ${report.signals_count}.`,
          summary: {
            ready: report.overall_ready,
            blockers: report.blockers.length,
            warnings: report.warnings.length,
            signals: report.signals_count,
          },
          suggested_tool_calls: [
            { name: "check_mcp_status", method: "GET", path: "/broker/mcp/status" },
            { name: "check_mcp_readiness", method: "GET", path: "/broker/mcp/readiness" },
            { name: "load_portfolio_signals", method: "GET", path: "/portfolio/signals?broker_mode=robinhood_mcp" },
          ],
          ts: stamp,
        },
      ]);
      setStatus("ready");
    });
  };

  const runScanPipeline = async () => {
    stopScanPolling();
    setScanProgressPct(0);
    setScanEtaSeconds(null);
    setScanElapsedSeconds(null);
    setScanCurrentPreset("");
    setScanFailedPresets([]);
    setScanPresetsCompleted(0);
    setScanPresetCount(0);
    setScanFinishedAt("");
    setScanLastMessage("Scan queued");
    setStatus("scan running");
    await withProgress("Running scan", async () => {
      try {
        const payload = {
          db: dbPath,
          parallel: scanParallel,
          max_workers: Math.max(1, Math.min(16, parseNumber(scanWorkers, 4))),
          presets: selectedPresets.length ? selectedPresets : null,
        };
        const start = await apiPost("/pipeline/scan/start", payload, authState);
        applyScanStatus(start);
        startScanPolling(start.job_id);
        setStatus("scan running");
      } catch (e) {
        stopScanPolling();
        setScanLastMessage(`Scan failed: ${e.message}`);
        setScanFinishedAt(new Date().toISOString());
        setStatus(`error: ${e.message}`);
      }
    });
  };

  const cancelScanPipeline = () => {
    if (!scanJobId || !scanIsActive) return;
    setStatus("scan cancel requested");
    setScanLastMessage("Cancel requested");
    apiPost(`/pipeline/scan/cancel/${encodeURIComponent(scanJobId)}`, {}, authState)
      .then((snap) => applyScanStatus(snap))
      .catch((e) => setScanLastMessage(`Cancel failed: ${e.message}`));
  };

  const runRecommendPipeline = async () => {
    setStatus("recommend running");
    await withProgress("Running recommendations", async () => {
      try {
        const payload = {
          db: dbPath,
          tickers: parseTickers(recommendTickersText),
          save: recommendSave,
        };
        const result = await apiPost("/pipeline/recommend", payload, authState);
        setRecommendResult(result || null);
        await loadOverview();
      } catch (e) {
        setStatus(`error: ${e.message}`);
      }
    });
  };

  const loadOptions = async () => {
    setStatus("loading options");
    await withProgress("Loading options", async () => {
      try {
        const data = await apiGet(
          `/options/ideas?ticker=${encodeURIComponent(ticker)}&horizon=${encodeURIComponent(horizon)}&bias=${encodeURIComponent(optionsBias)}&budget=${encodeURIComponent(parseNumber(optionsBudget, 300))}`,
          authState
        );
        setOptionIdeas(data.items || []);
        setOptionsResultMeta({ count: data.count || 0, horizon, bias: optionsBias, budget: parseNumber(optionsBudget, 300) });
        setStatus("ready");
      } catch (e) {
        setStatus(`error: ${e.message}`);
      }
    });
  };

  const loadBacktest = async () => {
    setStatus("running backtest");
    await withProgress("Running backtest", async () => {
      try {
        const endQ = backtestEnd.trim() ? `&end=${encodeURIComponent(backtestEnd.trim())}` : "";
        const tickerSafe = encodeURIComponent(ticker);
        const periodQuery = encodeURIComponent(analyticsPeriod || "2y");
        const intervalQuery = encodeURIComponent(analyticsInterval || "1d");
        const [data, indicators, seasonality, summary, timeline] = await Promise.all([
          apiGet(
            `/backtest/short-term?ticker=${tickerSafe}&start=${encodeURIComponent(backtestStart)}${endQ}&entry_rsi_threshold=${encodeURIComponent(parseNumber(backtestEntryRsi, 50))}&max_hold_days=${encodeURIComponent(Math.max(1, parseNumber(backtestMaxHoldDays, 7)))}&stop_loss_pct=${encodeURIComponent(Math.max(0, parseNumber(backtestStopLossPct, 0)))}&ma_filter=${encodeURIComponent((backtestMaFilter || "sma20").trim().toLowerCase())}&take_profit_pct=${encodeURIComponent(Math.max(0, parseNumber(backtestTakeProfitPct, 0)))}&ma_trend_filter=${encodeURIComponent((backtestMaTrendFilter || "none").trim().toLowerCase())}`,
            authState
          ),
          apiGet(`/charts/indicators?ticker=${tickerSafe}&period=${periodQuery}&interval=${intervalQuery}`, authState).catch(() => null),
          apiGet(`/charts/seasonality?ticker=${tickerSafe}&years=10`, authState).catch(() => null),
          apiGet(`/news/summary?ticker=${tickerSafe}&db=${encodeURIComponent(dbPath)}&days=${encodeURIComponent(parseNumber(newsDays, 30))}&sentiment=${encodeURIComponent(newsSentiment)}&max_groups=8`, authState).catch(() => null),
          apiGet(`/news/timeline?ticker=${tickerSafe}&db=${encodeURIComponent(dbPath)}&days=${encodeURIComponent(parseNumber(newsDays, 30))}&period=${periodQuery}&interval=${intervalQuery}&sentiment=${encodeURIComponent(newsSentiment)}&max_articles=24`, authState).catch(() => null),
        ]);
        setBacktestSummary(data.summary || null);
        setBacktestEquity(data.equity || []);
        if (indicators?.items) setIndicatorChart(indicators.items);
        if (Array.isArray(seasonality?.items)) setSeasonalityChart(seasonality.items);
        if (summary) setNewsSummary(summary);
        if (timeline) setNewsTimeline(timeline);
        setStatus("ready");
      } catch (e) {
        setStatus(`error: ${e.message}`);
      }
    });
  };

  const cacheMarketSeries = async () => {
    setStatus("caching market series");
    await withProgress("Caching yfinance series", async () => {
      try {
        const periodQuery = encodeURIComponent(analyticsPeriod);
        const intervalQuery = encodeURIComponent(analyticsInterval);
        const data = await apiPost(
          `/market/series/cache?ticker=${encodeURIComponent(ticker)}&period=${periodQuery}&interval=${intervalQuery}`,
          {},
          authState
        );
        setMarketSeriesCache(data || null);
        if (data?.items?.length) {
          setIndicatorChart(data.items);
        }
        setStatus("ready");
      } catch (e) {
        setStatus(`error: ${e.message}`);
      }
    });
  };

  const loadAnalytics = async () => {
    setStatus("loading analytics");
    await withProgress("Loading analytics", async () => {
      try {
        const tickerList = parseTickers(analyticsTickersText) || [ticker];
        const rankQuery = encodeURIComponent(tickerList.join(","));
        const periodQuery = encodeURIComponent(analyticsPeriod);
        const intervalQuery = encodeURIComponent(analyticsInterval);
        const [ranked, indicators, seasonality, distribution, forecast, summary, timeline, performanceCompare, seasonalityCompare] = await Promise.all([
          apiGet(`/recommend/analyze?tickers=${rankQuery}&period=${periodQuery}&interval=${intervalQuery}&limit=${Math.min(12, tickerList.length)}`, authState),
          apiGet(`/charts/indicators?ticker=${encodeURIComponent(ticker)}&period=${periodQuery}&interval=${intervalQuery}`, authState),
          apiGet(`/charts/seasonality?ticker=${encodeURIComponent(ticker)}&years=10`, authState),
          apiGet(`/charts/price-distribution?ticker=${encodeURIComponent(ticker)}&period=${periodQuery}&interval=${intervalQuery}&bins=${encodeURIComponent(parseNumber(analyticsBins, 24))}`, authState),
          apiGet(`/charts/forecast?ticker=${encodeURIComponent(ticker)}&period=${periodQuery}&interval=${intervalQuery}&horizon=${encodeURIComponent(parseNumber(analyticsHorizon, 30))}`, authState),
          apiGet(`/news/summary?ticker=${encodeURIComponent(ticker)}&db=${encodeURIComponent(dbPath)}&days=${encodeURIComponent(parseNumber(newsDays, 30))}&sentiment=${encodeURIComponent(newsSentiment)}&max_groups=8`, authState),
          apiGet(`/news/timeline?ticker=${encodeURIComponent(ticker)}&db=${encodeURIComponent(dbPath)}&days=${encodeURIComponent(parseNumber(newsDays, 30))}&period=${periodQuery}&interval=${intervalQuery}&sentiment=${encodeURIComponent(newsSentiment)}&max_articles=24`, authState),
          apiGet(`/charts/performance-compare?tickers=${rankQuery}&period=${periodQuery}&interval=${intervalQuery}&base=${encodeURIComponent(parseNumber(compareBase, 100))}`, authState),
          apiGet(`/charts/seasonality-compare?ticker=${encodeURIComponent(ticker)}&years=${encodeURIComponent(Math.max(3, Math.min(20, parseNumber(seasonalityYears, 10))))}&sector=${encodeURIComponent(seasonalitySectorOverride || "")}&sector_etf=${encodeURIComponent(seasonalityEtfOverride || "")}`, authState),
        ]);
        setAnalyticsRanked(ranked.items || []);
        setAnalyticsSaveResult(null);
        setIndicatorChart(indicators.items || []);
        setSeasonalityChart(seasonality.items || []);
        setDistributionChart(distribution || null);
        setForecastChart(forecast || null);
        setNewsSummary(summary || null);
        setNewsTimeline(timeline || null);
        setPerformanceCompareChart(performanceCompare || null);
        setSeasonalityCompareChart(seasonalityCompare || null);
        setStatus("ready");
      } catch (e) {
        setStatus(`error: ${e.message}`);
      }
    });
  };

  const loadChartsFast = async () => {
    setStatus("loading charts");
    await withProgress("Loading core charts", async () => {
      try {
        const periodQuery = encodeURIComponent(analyticsPeriod);
        const intervalQuery = encodeURIComponent(analyticsInterval);
        const years = encodeURIComponent(Math.max(3, Math.min(20, parseNumber(seasonalityYears, 10))));
        const [indicators, seasonality, seasonalityCompare] = await Promise.all([
          apiGet(`/charts/indicators?ticker=${encodeURIComponent(ticker)}&period=${periodQuery}&interval=${intervalQuery}`, authState),
          apiGet(`/charts/seasonality?ticker=${encodeURIComponent(ticker)}&years=${years}`, authState),
          apiGet(`/charts/seasonality-compare?ticker=${encodeURIComponent(ticker)}&years=${years}&sector=${encodeURIComponent(seasonalitySectorOverride || "")}&sector_etf=${encodeURIComponent(seasonalityEtfOverride || "")}`, authState),
        ]);
        setIndicatorChart(indicators.items || []);
        setSeasonalityChart(seasonality.items || []);
        setSeasonalityCompareChart(seasonalityCompare || null);
        setStatus("ready");
      } catch (e) {
        setStatus(`error: ${e.message}`);
      }
    });
  };

  const saveAnalyticsLeaderboard = async () => {
    setStatus("saving analytics leaderboard");
    await withProgress("Saving leaderboard", async () => {
      try {
        const tickerList = parseTickers(analyticsTickersText) || [ticker];
        const payload = {
          db: dbPath,
          tickers: tickerList,
          period: analyticsPeriod,
          interval: analyticsInterval,
          limit: Math.min(12, tickerList.length),
          save: true,
          forecast_horizon: Math.max(5, Math.min(90, parseNumber(analyticsHorizon, 30))),
        };
        const data = await apiPost("/analytics/leaderboard", payload, authState);
        setAnalyticsRanked(data.items || []);
        setAnalyticsLatest(data.items || []);
        setAnalyticsBatchId(data.batch_id || "");
        setAnalyticsSaveResult(data || null);
        setStatus("ready");
      } catch (e) {
        setStatus(`error: ${e.message}`);
      }
    });
  };

  const loadLatestAnalyticsLeaderboard = async () => {
    setStatus("loading latest leaderboard");
    await withProgress("Loading latest leaderboard", async () => {
      try {
        const data = await apiGet(`/analytics/leaderboard/latest?db=${encodeURIComponent(dbPath)}`, authState);
        setAnalyticsLatest(data.items || []);
        setAnalyticsBatchId(data.batch_id || "");
        setStatus("ready");
      } catch (e) {
        setStatus(`error: ${e.message}`);
      }
    });
  };

  const loadSignals = async () => {
    setSignalsLoading(true);
    setStatus("loading signals");
    await withProgress("Loading signals", async () => {
      try {
        const data = await apiGet(
          `/signals/generate?ticker=${encodeURIComponent(ticker)}&timeframe=${encodeURIComponent(signalsTimeframe)}&lookback_bars=120&detectors=breakout,rsi,macd,pullback`,
          authState
        );
        setSignalsData(data || null);
        setStatus("ready");
      } catch (e) {
        setStatus(`error: ${e.message}`);
      } finally {
        setSignalsLoading(false);
      }
    });
  };

  const loadTrainingFeatures = async () => {
    setStatus("loading training features");
    await withProgress("Loading training features", async () => {
      try {
        const tickerList = parseTickers(analyticsTickersText) || [ticker];
        const data = await apiGet(
          `/analytics/training-features?tickers=${encodeURIComponent(tickerList.join(","))}&period=${encodeURIComponent(analyticsPeriod)}&interval=${encodeURIComponent(analyticsInterval)}&forecast_horizon=${encodeURIComponent(parseNumber(analyticsHorizon, 30))}&limit=${encodeURIComponent(Math.min(12, tickerList.length))}`,
          authState
        );
        setTrainingFeatures(data.items || []);
        setStatus("ready");
      } catch (e) {
        setStatus(`error: ${e.message}`);
      }
    });
  };

  const proposeOrder = async () => {
    setStatus("proposing");
    await withProgress("Proposing order", async () => {
      try {
        const result = await apiPost("/execution/propose", {
          ticker,
          action: executionAction,
          qty: parseNumber(executionQty, 1),
          notional: parseNumber(notional, 0),
          orders_today: parseNumber(executionOrdersToday, 0),
          position_notional_after: parseNumber(executionPositionAfter, parseNumber(notional, 0)),
          broker_mode: executionBrokerMode,
          auto: executionAuto,
          max_daily_notional: parseNumber(executionMaxDaily, 5000),
          max_orders_per_day: parseNumber(executionMaxOrders, 20),
          max_position_notional: parseNumber(executionMaxPosition, 2000),
          db: dbPath,
        }, authState);
        setExecutionResult(result || null);
        await loadOverview();
      } catch (e) {
        setStatus(`error: ${e.message}`);
      }
    });
  };

  const checkFirebaseIdentity = async () => {
    setStatus("checking identity");
    await withProgress("Checking identity", async () => {
      try {
        const data = await apiGet("/auth/me", authState);
        setFirebaseAuthCheck(data);
        setStatus("ready");
      } catch (e) {
        setFirebaseAuthCheck({ authenticated: false, error: e.message });
        setStatus(`error: ${e.message}`);
      }
    });
  };

  const unlockWorkspace = async () => {
    setStatus("auth check");
    await withProgress("Unlocking workspace", async () => {
      try {
        const data = await apiGet("/auth/me", authState);
        setFirebaseAuthCheck(data || null);
        setWorkspaceUnlocked(true);
        setSurface("Workspace");
        await saveCredentials();
        setStatus("ready");
      } catch (e) {
        setFirebaseAuthCheck({ authenticated: false, error: e.message });
        setWorkspaceUnlocked(false);
        setStatus(`error: ${e.message}`);
      }
    });
  };

  const unlockWorkspaceLocal = async () => {
    setFirebaseAuthCheck({ mode: "local-dev", authenticated: true, note: "Workspace unlocked without token because API auth is disabled." });
    setWorkspaceUnlocked(true);
    setSurface("Workspace");
    await saveCredentials();
    setStatus("ready");
  };

  const loadUserSettings = async () => {
    setSettingsSaveStatus("loading");
    await withProgress("Loading user settings", async () => {
      try {
        const data = await apiGet("/user/settings", authState);
        setUserSettingsData(data);
        setUserSettingsText(JSON.stringify(data.settings || {}, null, 2));
        setSettingsSaveStatus("ready");
      } catch (e) {
        setUserSettingsData({ ok: false, error: e.message });
        setSettingsSaveStatus(`error: ${e.message}`);
      }
    });
  };

  const saveUserSettings = async () => {
    setSettingsSaveStatus("saving");
    await withProgress("Saving user settings", async () => {
      try {
        const parsed = JSON.parse(userSettingsText || "{}");
        const data = await apiPut("/user/settings", { settings: parsed }, authState);
        setUserSettingsData(data);
        setSettingsSaveStatus("saved");
      } catch (e) {
        setSettingsSaveStatus(`error: ${e.message}`);
      }
    });
  };

  const renderLanding = () => (
    <>
      <View style={styles.hero}>
        <Text style={styles.heroEyebrow}>QuantFlow</Text>
        <Text style={styles.heroTitle}>From quant workflows to a product-ready public landing and operator console</Text>
        <Text style={styles.heroBody}>
          Unified scanner, recommender, options picker, policy-gated execution, MCP readiness, and auth-backed user settings.
          This Public surface is safe for demos and landing use, while Workspace is gated for operator actions.
        </Text>
        <View style={styles.inlineRowWrap}>
          <Pressable style={styles.button} onPress={loadOverview}>
            <Text style={styles.buttonText}>Refresh Public Metrics</Text>
          </Pressable>
          <Pressable style={styles.secondaryButton} onPress={() => setSurface("Workspace")}>
            <Text style={styles.secondaryButtonText}>Go To Operator Workspace</Text>
          </Pressable>
        </View>
      </View>

      <View style={styles.cardGrid}>
        <View style={styles.card}>
          <Text style={styles.cardTitle}>Data Acquisition</Text>
          <Text style={styles.item}>Finviz scanner with proxy/API transport fallback and runtime diagnostics.</Text>
          <Text style={styles.item}>Latest presets: {presets.length}</Text>
          <Text style={styles.item}>Universe rows: {latestUniverse.length}</Text>
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>Decision Intelligence</Text>
          <Text style={styles.item}>Rule-based recommendations plus options filtering and short-term backtesting.</Text>
          <Text style={styles.item}>Latest recommendations: {latestRecs.length}</Text>
          <Text style={styles.item}>Recent intents: {intents.length}</Text>
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>Trust & Operations</Text>
          <Text style={styles.item}>Policy-gated execution, MCP readiness checks, and ops report endpoint.</Text>
          <Text style={styles.item}>Auth enabled: {health?.auth_enabled ? "yes" : "no"}</Text>
          <Text style={styles.item}>Firebase enabled: {health?.firebase_auth_enabled ? "yes" : "no"}</Text>
        </View>
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Direct Links</Text>
        <Text style={styles.item}>Public: {isWeb ? `${window.location.origin}${window.location.pathname}?view=public` : "available on web build"}</Text>
        <Text style={styles.item}>Workspace: {isWeb ? `${window.location.origin}${window.location.pathname}?view=workspace` : "available on web build"}</Text>
      </View>
    </>
  );

  const renderWorkspaceGate = () => (
    <View style={styles.card}>
      <Text style={styles.cardTitle}>Operator Workspace Gate</Text>
      <Text style={styles.item}>Authenticate before enabling workflow tabs and execution actions.</Text>
      {!workspaceUnlocked ? (
        <>
          <Text style={styles.fieldLabel}>API Key</Text>
          <TextInput
            value={apiKeyInput}
            onChangeText={setApiKeyInput}
            style={styles.input}
            placeholder="x-api-key (optional)"
            autoCapitalize="none"
          />
          <Text style={styles.fieldLabel}>Bearer Token</Text>
          <TextInput
            value={bearerToken}
            onChangeText={setBearerToken}
            style={[styles.input, styles.tokenInput]}
            placeholder="Firebase ID token (Bearer)"
            autoCapitalize="none"
            multiline
          />
          <View style={styles.inlineRow}>
            <Pressable onPress={() => setRememberCredentials(!rememberCredentials)} style={styles.inlineRow}>
              <View style={[styles.checkbox, rememberCredentials && styles.checkboxChecked]} />
              <Text style={styles.item}>Remember credentials</Text>
            </Pressable>
          </View>
          <View style={styles.inlineRowWrap}>
            <Pressable style={styles.button} onPress={unlockWorkspace}>
              <Text style={styles.buttonText}>Unlock Workspace</Text>
            </Pressable>
            {!health?.auth_enabled && (
              <Pressable style={styles.secondaryButton} onPress={unlockWorkspaceLocal}>
                <Text style={styles.secondaryButtonText}>Enter Workspace (Local Dev)</Text>
              </Pressable>
            )}
            <Pressable style={styles.secondaryButton} onPress={() => setSurface("Public")}>
              <Text style={styles.secondaryButtonText}>Back To Public Landing</Text>
            </Pressable>
          </View>
          <Text style={styles.item}>Auth check: {firebaseAuthCheck ? compactJson(firebaseAuthCheck, 900) : "not checked"}</Text>
        </>
      ) : (
        <>
          <Text style={styles.item}>✓ Workspace unlocked</Text>
          <Text style={styles.item}>Authentication: {firebaseAuthCheck?.authenticated ? "authenticated" : "verified"}</Text>
          <View style={styles.inlineRowWrap}>
            <Pressable style={styles.cancelButton} onPress={logoutWorkspace}>
              <Text style={styles.buttonText}>Logout</Text>
            </Pressable>
          </View>
        </>
      )}
      <Text style={styles.warnText}>Client-side gate is for UX and demo safety. Backend auth still enforces write access.</Text>
      {!health?.auth_enabled && <Text style={styles.item}>Detected local mode: API auth is disabled, so you can enter workspace without token.</Text>}
    </View>
  );

  const renderOverview = () => (
    <>
      <View style={styles.row}>
        <Pressable style={styles.button} onPress={loadOverview}>
          <Text style={styles.buttonText}>Refresh</Text>
        </Pressable>
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Health</Text>
        <Text style={styles.item}>{health ? compactJson(health) : "not loaded"}</Text>
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Environment Validation</Text>
        <Text style={styles.item}>Status: {envValidation ? (envValidation.ok ? "ready" : "action needed") : "unknown"}</Text>
        <Text style={styles.item}>Provider: {envValidation?.provider || "unknown"}</Text>
        <Text style={styles.item}>Mode: {envValidation?.scraping_mode || "unknown"}</Text>
        {!!envValidation?.missing?.length && <Text style={styles.warnText}>Missing: {envValidation.missing.join(" | ")}</Text>}
        {!!envValidation?.warnings?.length && <Text style={styles.warnText}>Warnings: {envValidation.warnings.join(" | ")}</Text>}
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Ops Report</Text>
        <Text>DB status: {opsReport ? opsReport.overall_db_status : "not loaded"}</Text>
        <Text>Generated: {opsReport ? opsReport.generated_at : "unknown"}</Text>
        <Text>
          Snapshots: {opsReport?.tables?.ticker_snapshots?.count ?? 0}
          {" | "}
          Recommendations: {opsReport?.tables?.recommendations?.count ?? 0}
        </Text>
        <Text>
          Scraping mode: {opsReport?.runtime?.scraping_status?.mode || "unknown"}
          {" | "}
          Provider: {opsReport?.runtime?.scraping_status?.provider || "unknown"}
        </Text>
        {!!opsReport?.validation && (
          <>
            <Text>Snapshot rows saved: {opsReport.validation.snapshot_rows_saved}</Text>
            <Text>Recommendations saved: {opsReport.validation.recommendations_saved}</Text>
            <Text>
              Presets: {opsReport.validation.successful_presets} ok
              {" / failed: "}
              {opsReport.validation.failed_presets?.length ? opsReport.validation.failed_presets.join(", ") : "none"}
            </Text>
          </>
        )}
        {!opsReport && <Text style={styles.warnText}>Ops report unavailable. Run ops-report or ops-refresh first.</Text>}
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Robinhood MCP</Text>
        <Text>Connected: {mcpStatus ? (mcpStatus.connected ? "yes" : "no") : "unknown"}</Text>
        <Text>Authenticated: {mcpStatus ? (mcpStatus.authenticated ? "yes" : "no") : "unknown"}</Text>
        <Text>Account data: {mcpStatus ? (mcpStatus.account_available ? "yes" : "no") : "unknown"}</Text>
        <Text>Positions: {mcpStatus ? String(mcpStatus.positions_count || 0) : "0"}</Text>
        <Text>Portfolio signals: {mcpSignals ? String(mcpSignals.count || 0) : "0"}</Text>
        {!!(mcpStatus && mcpStatus.endpoint) && <Text style={styles.item}>Endpoint: {mcpStatus.endpoint}</Text>}
        {!!(mcpStatus && (mcpStatus.account_error || mcpStatus.positions_error || mcpStatus.error)) && (
          <Text style={styles.warnText}>Last MCP error: {mcpStatus.account_error || mcpStatus.positions_error || mcpStatus.error}</Text>
        )}
        {!!mcpSignalsError && <Text style={styles.warnText}>Signals error: {mcpSignalsError}</Text>}
        <Pressable style={styles.button} onPress={loadMcpStatus}>
          <Text style={styles.buttonText}>Check MCP</Text>
        </Pressable>
        <Pressable style={styles.secondaryButton} onPress={connectMcp}>
          <Text style={styles.secondaryButtonText}>Connect Robinhood</Text>
        </Pressable>
        {!!mcpLoginResult && (
          <Text style={styles.item}>
            Connect result: {mcpLoginResult.ok ? "ok" : "not ready"}
            {mcpLoginResult.message ? ` | ${mcpLoginResult.message}` : ""}
          </Text>
        )}
        {!!mcpSignals?.signals?.length && (
          <View>
            <Text style={styles.item}>Top signal snapshot:</Text>
            {(mcpSignals.signals || []).slice(0, 3).map((sig, idx) => (
              <Text key={`sig-${String(idx)}`} style={styles.item}>
                {sig.ticker || "-"} | {sig.action || sig.signal || "hold"} | strength {fmtNumber(sig.strength || sig.score || 0, 2)}
              </Text>
            ))}
          </View>
        )}
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>MCP Login Readiness</Text>
        <Text style={styles.item}>Overall ready: {mcpReadiness ? (mcpReadiness.overall_ready ? "yes" : "no") : "unknown"}</Text>
        {(mcpReadiness?.checks || []).map((c) => (
          <Text key={c.id} style={styles.item}>
            {c.state === "pass" ? "PASS" : c.state === "warn" ? "WARN" : "FAIL"} | {c.label} | {c.details}
          </Text>
        ))}
        {(mcpReadiness?.next_steps || []).slice(0, 4).map((s, idx) => (
          <Text key={`step-${String(idx)}`} style={styles.item}>Step {idx + 1}: {s}</Text>
        ))}
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Presets ({presets.length})</Text>
        {(presets || []).slice(0, 8).map((p) => (
          <Text key={p} style={styles.item}>{p}</Text>
        ))}
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Recent Intents ({intents.length})</Text>
        {(intents || []).slice(0, 8).map((i) => (
          <Text key={String(i.id)} style={styles.item}>
            #{i.id} {i.ticker} {i.action} ${i.notional} [{i.status}]
          </Text>
        ))}
      </View>
    </>
  );

  const renderScanner = () => (
    <>
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Scanner Pipeline Controls</Text>
        {(scanIsActive || scannerBusy) && <ProgressBar progress={scanIsActive ? scanProgressPct : progressValue} label={scanIsActive ? `Scan ${scanProgressPct.toFixed(1)}%` : "Running scanner pipeline"} compact />}
        <Text style={styles.item}>DB path</Text>
        <TextInput value={dbPath} onChangeText={setDbPath} style={styles.input} placeholder="sqlite:///quantflow.db" autoCapitalize="none" />
        <Text style={styles.item}>Preset profiles (comma-separated)</Text>
        <TextInput
          value={selectedPresetsText}
          onChangeText={setSelectedPresetsText}
          style={styles.input}
          placeholder="weekly_momo,weekly_bear"
          autoCapitalize="none"
        />
        <View style={styles.inlineRowWrap}>
          <Pressable style={styles.smallButton} onPress={() => selectPresetCategory("weekly")}>
            <Text style={styles.buttonText}>Weekly</Text>
          </Pressable>
          <Pressable style={styles.smallButton} onPress={() => selectPresetCategory("monthly")}>
            <Text style={styles.buttonText}>Monthly</Text>
          </Pressable>
          <Pressable style={styles.smallButton} onPress={() => setSelectedPresetsText((presetDetails?.items || presets || []).join(","))}>
            <Text style={styles.buttonText}>All</Text>
          </Pressable>
        </View>
        <Text style={styles.fieldLabel}>Preset Search</Text>
        <TextInput value={presetSearch} onChangeText={setPresetSearch} style={styles.input} placeholder="Search preset name" autoCapitalize="none" />
        <View style={styles.inlineRowWrap}>
          {availablePresets.slice(0, 12).map((name) => {
            const active = selectedPresets.includes(name);
            return (
              <Pressable key={name} style={active ? styles.smallButton : styles.mutedButton} onPress={() => togglePreset(name)}>
                <Text style={styles.buttonText}>{name}</Text>
              </Pressable>
            );
          })}
        </View>
        <View style={styles.inlineRowWrap}>
          <Text style={styles.item}>Parallel run:</Text>
          <Pressable style={scanParallel ? styles.smallButton : styles.mutedButton} onPress={() => setScanParallel(true)}>
            <Text style={styles.buttonText}>ON</Text>
          </Pressable>
          <Pressable style={!scanParallel ? styles.smallButton : styles.mutedButton} onPress={() => setScanParallel(false)}>
            <Text style={styles.buttonText}>OFF</Text>
          </Pressable>
          <Text style={styles.fieldLabel}>Workers</Text>
          <TextInput value={scanWorkers} onChangeText={setScanWorkers} style={[styles.input, styles.compactInput]} placeholder="Workers" keyboardType="numeric" />
        </View>
        <View style={styles.inlineRowWrap}>
          <Pressable style={styles.button} onPress={runScanPipeline}>
            <Text style={styles.buttonText}>Run Scan</Text>
          </Pressable>
          {scanIsActive && (
            <Pressable style={styles.cancelButton} onPress={cancelScanPipeline}>
              <Text style={styles.buttonText}>Cancel Scan</Text>
            </Pressable>
          )}
          <Pressable style={styles.secondaryButton} onPress={() => loadScanHistory(scanHistoryFilter)}>
            <Text style={styles.secondaryButtonText}>Refresh History</Text>
          </Pressable>
          <Pressable style={styles.secondaryButton} onPress={loadOverview}>
            <Text style={styles.secondaryButtonText}>Refresh Universe</Text>
          </Pressable>
        </View>
      </View>

      <View style={styles.panelSplit}>
        <View style={[styles.card, styles.panelCol]}>
          <Text style={styles.cardTitle}>Scan Status And History</Text>
          <Text style={styles.item}>Job: {scanJobId || "none"} | state: {scanJobStatus}</Text>
          <Text style={styles.item}>Progress: {scanPresetsCompleted}/{scanPresetCount || "-"} presets | ETA: {fmtDurationSeconds(scanEtaSeconds)} | elapsed: {fmtDurationSeconds(scanElapsedSeconds)}</Text>
          <Text style={styles.item}>Current preset: {scanCurrentPreset || "-"} | failed: {scanFailedPresets.length}</Text>
          <Text style={styles.item}>Started: {fmtTimestamp(scanStartedAt)} | Finished: {fmtTimestamp(scanFinishedAt)}</Text>
          <Text style={styles.item}>Universe last refreshed: {fmtTimestamp(scanUniverseRefreshedAt)}</Text>
          <Text style={styles.item}>Scan status: {scanLastMessage || "idle"}</Text>
          {!!scanResult && (
            <Text style={styles.item}>
              Last result: presets {scanResult.preset_count}, rows saved {scanResult.rows_saved}, failed {scanResult.failed_presets?.length || 0}
            </Text>
          )}

          <Text style={styles.item}>History filter</Text>
          <View style={styles.inlineRowWrap}>
            {[
              ["all", "All"],
              ["running", "Running"],
              ["completed", "Completed"],
              ["canceled", "Canceled"],
              ["failed", "Failed"],
            ].map(([value, label]) => (
              <Pressable key={value} style={scanHistoryFilter === value ? styles.smallButton : styles.mutedButton} onPress={() => setScanHistoryFilter(value)}>
                <Text style={styles.buttonText}>{label}</Text>
              </Pressable>
            ))}
          </View>

          <Text style={styles.item}>Rows saved trend (last {scanHistoryRowsSeries.length} jobs)</Text>
          <MiniSeriesChart series={[{ label: "rows", values: scanHistoryRowsSeries, color: THEME.accentDeep, width: 3 }]} height={90} />
          <Text style={styles.item}>Duration trend seconds (last {scanHistoryDurationSeries.length} jobs)</Text>
          <MiniSeriesChart series={[{ label: "duration", values: scanHistoryDurationSeries, color: THEME.warn, width: 3 }]} height={90} />

          {(scanHistoryFiltered || []).slice(0, 10).map((row) => (
            <View key={row.job_id} style={styles.analyticsItemCard}>
              <Text style={styles.analyticsHeadline}>{row.status} | {fmtTimestamp(row.created_at)}</Text>
              <Text style={styles.item}>Job {row.job_id}</Text>
              <Text style={styles.item}>Progress {fmtNumber(row.progress_pct)}% | presets {row.presets_completed}/{row.preset_count} | rows {row.rows_saved}</Text>
              <Text style={styles.item}>Elapsed {fmtDurationSeconds(row.elapsed_seconds)} | ETA {fmtDurationSeconds(row.eta_seconds)} | failed {row.failed_presets?.length || 0}</Text>
            </View>
          ))}
        </View>

        <View style={[styles.card, styles.panelCol]}>
          <Text style={styles.cardTitle}>Universe Viewer</Text>
          <Text style={styles.fieldLabel}>Preset Filter</Text>
          <TextInput value={universePresetFilter} onChangeText={setUniversePresetFilter} style={styles.input} placeholder="Filter by preset name (e.g., weekly)" autoCapitalize="none" />
          <Text style={styles.fieldLabel}>Rows To Show</Text>
          <TextInput value={universeLimit} onChangeText={setUniverseLimit} style={styles.input} placeholder="Rows to show (1-200)" keyboardType="numeric" />
          <Text style={styles.cardTitle}>Latest Universe ({filteredUniverse.length} shown of {latestUniverse.length})</Text>
          {(filteredUniverse || []).map((row, idx) => {
            const price = row.price ?? row.close ?? row.last ?? "-";
            const rsi = row.rsi ?? row.rsi14 ?? row["RSI(14)"] ?? "-";
            const relVolume = row.rel_volume ?? row.relative_volume ?? row.rv ?? row["Rel Volume"] ?? "-";
            return (
              <Text key={String(idx)} style={styles.item}>{row.ticker || row.symbol || "?"} | {row.preset || "-"} | {row.sector || "-"} | price {price} | rsi {rsi} | rv {relVolume}</Text>
            );
          })}
        </View>
      </View>
    </>
  );

  const renderRecommend = () => (
    <View style={styles.card}>
      <Text style={styles.cardTitle}>Recommendations</Text>
      {recommendBusy && <ProgressBar progress={progressValue} label="Computing recommendations" compact />}
      <Text style={styles.item}>DB path</Text>
      <TextInput value={dbPath} onChangeText={setDbPath} style={styles.input} placeholder="sqlite:///quantflow.db" autoCapitalize="none" />
      <Text style={styles.item}>Tickers (comma-separated, leave empty for full high-interest universe)</Text>
      <TextInput value={recommendTickersText} onChangeText={setRecommendTickersText} style={styles.input} placeholder="AAPL,MSFT,NVDA" autoCapitalize="characters" />
      <View style={styles.inlineRowWrap}>
        <Text style={styles.item}>Save to DB:</Text>
        <Pressable style={recommendSave ? styles.smallButton : styles.mutedButton} onPress={() => setRecommendSave(true)}>
          <Text style={styles.buttonText}>YES</Text>
        </Pressable>
        <Pressable style={!recommendSave ? styles.smallButton : styles.mutedButton} onPress={() => setRecommendSave(false)}>
          <Text style={styles.buttonText}>NO</Text>
        </Pressable>
      </View>
      <Pressable style={styles.button} onPress={runRecommendPipeline}>
        <Text style={styles.buttonText}>Run Recommend</Text>
      </Pressable>
      {!!recommendResult && (
        <Text style={styles.item}>
          Run result: tickers {recommendResult.ticker_count}, recs {recommendResult.recommendation_count}, failed {recommendResult.failed_tickers?.length || 0}, saved {recommendResult.saved ? "yes" : "no"}
        </Text>
      )}
      <Text style={styles.cardTitle}>Latest Recs ({latestRecs.length})</Text>
      {(latestRecs || []).slice(0, 12).map((row, idx) => (
        <Text key={String(idx)} style={styles.item}>
          {row.ticker} {row.bias || "-"} ({row.horizon || "?"}) conf {Number(row.confidence || 0).toFixed(2)} entry {Number(row.entry || 0).toFixed(2)} stop {Number(row.stop || 0).toFixed(2)} target {Number(row.target || 0).toFixed(2)}
        </Text>
      ))}
    </View>
  );

  const renderOptions = () => (
    <View style={styles.card}>
      <Text style={styles.cardTitle}>Options Ideas</Text>
      <Text style={styles.fieldLabel}>Ticker</Text>
      <TextInput value={ticker} onChangeText={setTicker} style={styles.input} placeholder="Ticker" />
      <Text style={styles.fieldLabel}>Horizon</Text>
      <TextInput value={horizon} onChangeText={setHorizon} style={styles.input} placeholder="Horizon: 1w/1m/3m" />
      <Text style={styles.fieldLabel}>Bias</Text>
      <TextInput value={optionsBias} onChangeText={setOptionsBias} style={styles.input} placeholder="Bias: long/short" />
      <Text style={styles.fieldLabel}>Budget</Text>
      <TextInput value={optionsBudget} onChangeText={setOptionsBudget} style={styles.input} placeholder="Budget" keyboardType="numeric" />
      <Pressable style={styles.button} onPress={loadOptions}>
        <Text style={styles.buttonText}>Load Ideas</Text>
      </Pressable>
      {!!optionsResultMeta && (
        <Text style={styles.item}>Result: {optionsResultMeta.count} ideas | {optionsResultMeta.horizon} | {optionsResultMeta.bias} | budget ${optionsResultMeta.budget}</Text>
      )}
      {(optionIdeas || []).slice(0, 12).map((row, idx) => (
        <Text key={String(idx)} style={styles.item}>
          {row.expiry || "?"} {row.right || ""} {row.strike ? Number(row.strike).toFixed(2) : "-"} | mid {row.mid ? Number(row.mid).toFixed(2) : "-"} | spread {(row.ask !== undefined && row.bid !== undefined) ? Number(row.ask - row.bid).toFixed(2) : "-"} | OI {row.open_interest ?? "-"} | Vol {row.volume ?? "-"}
        </Text>
      ))}
    </View>
  );

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
      return {
        label: name,
        values: compareRows.map((row) => Number(row[name] || 0)),
        color: palette[idx % palette.length],
        width: 2,
      };
    });

    const seasonalityCompareRows = seasonalityCompareChart?.items || [];
    const seasonalityStockAvg = seasonalityCompareRows.map((row) => Number(row.stock_avg_ret || 0));
    const seasonalitySectorAvg = seasonalityCompareRows.map((row) => Number(row.sector_avg_ret || 0));
    const seasonalitySpread = seasonalityCompareRows.map((row) => Number(row.spread || 0));
    const seasonalityStockCum = seasonalityCompareRows.map((row) => Number(row.stock_cum || 0));
    const seasonalitySectorCum = seasonalityCompareRows.map((row) => Number(row.sector_cum || 0));

    return (
      <>
        <View style={styles.card}>
          <Text style={styles.cardTitle}>Analytics Workbench</Text>
          {analyticsBusy && <ProgressBar progress={progressValue} label="Loading analytics datasets" compact />}
          <Text style={styles.item}>Load ranked setups, indicator charts, price-distribution support and resistance, seasonality, and lightweight forecasts from the same quant backend.</Text>
          <Text style={styles.item}>Analytics DB path</Text>
          <TextInput value={dbPath} onChangeText={setDbPath} style={styles.input} placeholder="sqlite:///quantflow.db" autoCapitalize="none" />
          <Text style={styles.fieldLabel}>Primary Ticker</Text>
          <TextInput value={ticker} onChangeText={setTicker} style={styles.input} placeholder="Primary ticker" autoCapitalize="characters" />
          <Text style={styles.fieldLabel}>Ranking Basket</Text>
          <TextInput value={analyticsTickersText} onChangeText={setAnalyticsTickersText} style={styles.input} placeholder="Ranking basket: AAPL,MSFT,NVDA" autoCapitalize="characters" />
          <Text style={styles.fieldLabel}>Analytics Parameters</Text>
          <View style={styles.inlineRowWrap}>
            <Text style={[styles.fieldLabel, styles.compactLabel]}>Period</Text>
            <Text style={[styles.fieldLabel, styles.compactLabel]}>Interval</Text>
            <Text style={[styles.fieldLabel, styles.compactLabel]}>Bins</Text>
            <Text style={[styles.fieldLabel, styles.compactLabel]}>Forecast Days</Text>
            <Text style={[styles.fieldLabel, styles.compactLabel]}>News Days</Text>
            <Text style={[styles.fieldLabel, styles.compactLabel]}>News Sentiment</Text>
          </View>
          <View style={styles.inlineRowWrap}>
            <TextInput value={analyticsPeriod} onChangeText={setAnalyticsPeriod} style={[styles.input, styles.compactInput]} placeholder="Period" autoCapitalize="none" />
            <TextInput value={analyticsInterval} onChangeText={setAnalyticsInterval} style={[styles.input, styles.compactInput]} placeholder="Interval" autoCapitalize="none" />
            <TextInput value={analyticsBins} onChangeText={setAnalyticsBins} style={[styles.input, styles.compactInput]} placeholder="Bins" keyboardType="numeric" />
            <TextInput value={analyticsHorizon} onChangeText={setAnalyticsHorizon} style={[styles.input, styles.compactInput]} placeholder="Forecast days" keyboardType="numeric" />
            <TextInput value={newsDays} onChangeText={setNewsDays} style={[styles.input, styles.compactInput]} placeholder="News days" keyboardType="numeric" />
            <TextInput value={newsSentiment} onChangeText={setNewsSentiment} style={[styles.input, styles.compactInput]} placeholder="News sentiment" autoCapitalize="none" />
          </View>
          <Text style={styles.fieldLabel}>Comparison And Seasonality Overrides</Text>
          <View style={styles.inlineRowWrap}>
            <Text style={[styles.fieldLabel, styles.compactLabel]}>Compare Base</Text>
            <Text style={[styles.fieldLabel, styles.compactLabel]}>Seasonality Years</Text>
            <Text style={[styles.fieldLabel, styles.compactLabel]}>Sector Override</Text>
            <Text style={[styles.fieldLabel, styles.compactLabel]}>Sector ETF Override</Text>
          </View>
          <View style={styles.inlineRowWrap}>
            <TextInput value={compareBase} onChangeText={setCompareBase} style={[styles.input, styles.compactInput]} placeholder="Compare base" keyboardType="numeric" />
            <TextInput value={seasonalityYears} onChangeText={setSeasonalityYears} style={[styles.input, styles.compactInput]} placeholder="Seasonality years" keyboardType="numeric" />
            <TextInput value={seasonalitySectorOverride} onChangeText={setSeasonalitySectorOverride} style={[styles.input, styles.compactInput]} placeholder="Sector override" autoCapitalize="words" />
            <TextInput value={seasonalityEtfOverride} onChangeText={setSeasonalityEtfOverride} style={[styles.input, styles.compactInput]} placeholder="Sector ETF override" autoCapitalize="characters" />
          </View>
          <View style={styles.inlineRowWrap}>
            <Pressable style={styles.button} onPress={loadAnalytics}>
              <Text style={styles.buttonText}>Load Analytics</Text>
            </Pressable>
            <Pressable style={styles.secondaryButton} onPress={loadChartsFast}>
              <Text style={styles.secondaryButtonText}>Load Core Charts Fast</Text>
            </Pressable>
            <Pressable style={styles.button} onPress={cacheMarketSeries}>
              <Text style={styles.buttonText}>Fetch + Cache Series</Text>
            </Pressable>
            <Pressable style={styles.button} onPress={saveAnalyticsLeaderboard}>
              <Text style={styles.buttonText}>Save Leaderboard</Text>
            </Pressable>
            <Pressable style={styles.secondaryButton} onPress={loadLatestAnalyticsLeaderboard}>
              <Text style={styles.secondaryButtonText}>Load Latest Leaderboard</Text>
            </Pressable>
            <Pressable style={styles.secondaryButton} onPress={loadTrainingFeatures}>
              <Text style={styles.secondaryButtonText}>Load Training Features</Text>
            </Pressable>
            <Pressable style={styles.secondaryButton} onPress={runRecommendPipeline}>
              <Text style={styles.secondaryButtonText}>Run Recommend Pipeline</Text>
            </Pressable>
          </View>
          <Text style={styles.item}>Saved leaderboard batch: {analyticsBatchId || "none"}</Text>
          {!!marketSeriesCache && <Text style={styles.item}>Cached series: {marketSeriesCache.count || 0} rows at {marketSeriesCache.path || "n/a"}</Text>}
          <Text style={styles.item}>Indicator rows loaded: {indicatorChart.length} | RSI points: {indicatorTail.filter((row) => Number.isFinite(Number(row.rsi14))).length}</Text>
          {!!analyticsSaveResult && <Text style={styles.item}>Save result: count {analyticsSaveResult.count || 0} | saved {analyticsSaveResult.saved ? "yes" : "no"}</Text>}
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>Live Signal Generator</Text>
          <Text style={styles.item}>
            Runs breakout, RSI cross, MACD cross, and pullback detectors on the latest {signalsTimeframe} bars for {ticker}.
            {" "}{signalsData?.data_source === "alpaca" ? "Data: Alpaca real-time." : "Data: yfinance (delayed). Set ALPACA_API_KEY for real-time."}
          </Text>
          <View style={styles.rowWrap}>
            {["1Min","5Min","15Min","30Min","1Hour","1Day"].map((tf) => (
              <Pressable
                key={tf}
                style={[styles.chip, signalsTimeframe === tf && styles.chipActive]}
                onPress={() => setSignalsTimeframe(tf)}
              >
                <Text style={[styles.chipText, signalsTimeframe === tf && styles.chipActiveText]}>{tf}</Text>
              </Pressable>
            ))}
          </View>
          <Pressable style={[styles.button, signalsLoading && styles.buttonDisabled]} onPress={loadSignals} disabled={signalsLoading}>
            <Text style={styles.buttonText}>{signalsLoading ? "Loading…" : "Run Signal Scan"}</Text>
          </Pressable>
          {!!signalsData && (
            <>
              <Text style={styles.item}>
                {signalsData.bar_count} bars loaded | generated {signalsData.generated_at?.slice(0,19) || ""}
              </Text>
              {signalsData.latest_bar && (
                <Text style={styles.item}>
                  Latest bar — O {fmtNumber(signalsData.latest_bar.open)} H {fmtNumber(signalsData.latest_bar.high)} L {fmtNumber(signalsData.latest_bar.low)} C {fmtNumber(signalsData.latest_bar.close)} Vol {fmtNumber(signalsData.latest_bar.volume, 0)}
                </Text>
              )}
              {(signalsData.signals || []).length === 0 ? (
                <Text style={[styles.item, {color: THEME.muted}]}>No signals fired on the latest bar.</Text>
              ) : (
                <>
                  <Text style={[styles.item, {fontWeight:"bold", color: THEME.accent}]}>
                    {signalsData.signal_count} signal{signalsData.signal_count !== 1 ? "s" : ""} detected:
                  </Text>
                  {(signalsData.signals || []).map((sig, idx) => (
                    <View key={`sig-${idx}`} style={[styles.analyticsItemCard, {borderLeftWidth:3, borderLeftColor: sig.direction==="long" ? "#16a34a" : "#dc2626"}]}>
                      <Text style={styles.analyticsHeadline}>
                        {sig.signal_type.toUpperCase()} · {sig.direction.toUpperCase()} · {sig.ticker} @ {fmtNumber(sig.price)}
                      </Text>
                      <Text style={styles.item}>Confidence {fmtNumber(sig.confidence, 3)} | {sig.ts?.slice(0,19) || ""}</Text>
                      <Text style={styles.item}>
                        {Object.entries(sig.details || {}).map(([k,v]) => `${k}: ${typeof v==="number" ? fmtNumber(v,3) : v}`).join(" | ")}
                      </Text>
                    </View>
                  ))}
                </>
              )}
              {/* Mini close-price chart for the signal scan window */}
              {signalsData.bar_count > 0 && (() => {
                const bars = signalsData._bars || [];
                return null; // bars not yet passed through; full chart comes from /charts/indicators
              })()}
            </>
          )}
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>Ranked Setups ({analyticsRanked.length})</Text>
          {(analyticsRanked || []).slice(0, 6).map((row) => (
            <View key={row.ticker} style={styles.analyticsItemCard}>
              <Text style={styles.analyticsHeadline}>{row.ticker} | {row.bias} | {row.primary_horizon} | composite {fmtNumber(row.composite_score)}</Text>
              <Text style={styles.item}>Price {fmtNumber(row.price)} | support {fmtNumber(row.support_resistance?.nearest_support)} | resistance {fmtNumber(row.support_resistance?.nearest_resistance)}</Text>
              <Text style={styles.item}>Upside to resistance {fmtPct(row.support_resistance?.resistance_gap_pct)} | buffer to support {fmtPct(row.support_resistance?.support_gap_pct)}</Text>
              <ScoreBars breakdown={row.score_breakdown} />
              {(row.recommendations || []).slice(0, 3).map((rec) => (
                <Text key={`${row.ticker}-${rec.horizon}`} style={styles.item}>{rec.horizon} {rec.bias} | conf {fmtNumber(rec.confidence)} | entry {fmtNumber(rec.entry)} | stop {fmtNumber(rec.stop)} | target {fmtNumber(rec.target)}</Text>
              ))}
            </View>
          ))}
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>Persisted Leaderboard ({analyticsLatest.length})</Text>
          <Text style={styles.item}>Latest saved batch: {analyticsBatchId || "not loaded"}</Text>
          {(analyticsLatest || []).slice(0, 6).map((row, idx) => (
            <Text key={`latest-${row.ticker}-${String(idx)}`} style={styles.item}>
              {row.ticker} | {row.bias} | {row.primary_horizon} | composite {fmtNumber(row.composite_score)} | support {fmtNumber(row.nearest_support)} | resistance {fmtNumber(row.nearest_resistance)}
            </Text>
          ))}
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>Price And Trend Structure</Text>
          <Text style={styles.item}>{ticker} adjusted close vs SMA20 and SMA50 over the latest {indicatorTail.length} bars.</Text>
          <MiniSeriesChart
            series={[
              { label: "price", values: priceSeries, color: THEME.text, width: 3, type: "line" },
              { label: "sma20", values: sma20Series, color: THEME.accent, width: 2, type: "line" },
              { label: "sma50", values: sma50Series, color: THEME.warn, width: 2, type: "line" },
            ]}
            xLabels={indicatorTail.map((row) => String(row.date || ""))}
            height={140}
          />
          <Text style={styles.item}>RSI 14</Text>
          <MiniSeriesChart series={[{ label: "rsi", values: rsiSeries, color: THEME.accentDeep, width: 3, type: "line" }]} xLabels={indicatorTail.map((row) => String(row.date || ""))} height={90} />
          <Text style={styles.item}>Volume</Text>
          <MiniSeriesChart series={[{ label: "volume", values: volumeSeries, color: "#7c6d5a", width: 3, opacity: 0.75, type: "bar" }]} xLabels={indicatorTail.map((row) => String(row.date || ""))} height={90} />
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>Stock Performance Comparison</Text>
          <Text style={styles.item}>Normalized comparison for selected tickers from a base of {fmtNumber(compareBase)}. This helps compare relative strength and drawdown behavior on the same scale.</Text>
          <MiniSeriesChart series={compareSeries} height={130} />
          {(performanceCompareChart?.summary || []).slice(0, 8).map((row) => (
            <Text key={`cmp-${row.ticker}`} style={styles.item}>{row.ticker} | return {fmtPct(row.return_pct)} | vol {fmtPct(row.volatility)}</Text>
          ))}
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>Price Distribution And Quantified Levels</Text>
          <Text style={styles.item}>Distribution density approximates where price spent the most time and volume, which is a practical support and resistance proxy for both execution and training features.</Text>
          <Text style={styles.item}>Current {fmtNumber(distributionChart?.current_price)} | range {fmtPct(distributionChart?.range_pct)}</Text>
          <HorizontalDistribution items={distributionChart?.bins || []} color={THEME.accent} />
          <Text style={styles.item}>Support levels: {(distributionChart?.support_levels || []).map((row) => `${fmtNumber(row.price)} (${fmtPct(row.distance_pct)})`).join(" | ") || "-"}</Text>
          <Text style={styles.item}>Resistance levels: {(distributionChart?.resistance_levels || []).map((row) => `${fmtNumber(row.price)} (${fmtPct(row.distance_pct)})`).join(" | ") || "-"}</Text>
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>News Timeline</Text>
          <Text style={styles.item}>Articles are aligned to the nearest price date so you can compare event timing against price behavior.</Text>
          <Text style={styles.item}>Archive summary: {newsSummary?.summary || "-"}</Text>
          <NewsTimelineChart timeline={newsTimeline} />
          <NewsSummaryList summary={newsSummary} />
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>Forecast Envelope</Text>
          <Text style={styles.item}>Daily trend {fmtPct(forecastChart?.daily_trend_pct)} | projected return {fmtPct(forecastChart?.forecast_return_pct)} | uncertainty band {fmtPct(forecastChart?.confidence_band_pct)}</Text>
          <Text style={styles.item}>Recent realized prices</Text>
          <MiniSeriesChart series={[{ label: "history", values: historySeries, color: THEME.text, width: 3 }]} height={120} />
          <Text style={styles.item}>Forward path with upper and lower envelope</Text>
          <MiniSeriesChart
            series={[
              { label: "forecast-lower", values: forecastLowerSeries, color: "#d97706", width: 2, opacity: 0.5 },
              { label: "forecast-center", values: forecastPriceSeries, color: THEME.accentDeep, width: 3 },
              { label: "forecast-upper", values: forecastUpperSeries, color: "#16a34a", width: 2, opacity: 0.5 },
            ]}
            height={120}
          />
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>Seasonality</Text>
          <Text style={styles.item}>Average daily return and realized deviation by day-of-year across the historical sample.</Text>
          <MiniSeriesChart
            series={[
              { label: "avg_ret", values: seasonalityAvg, color: THEME.accent, width: 3, type: "line" },
              { label: "std_ret", values: seasonalityStd, color: THEME.warn, width: 2, opacity: 0.7, type: "line" },
            ]}
            xLabels={seasonalityTail.map((row, idx) => dayOfYearLabel(row, idx + 1))}
            height={120}
          />
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>Stock Vs Sector Seasonality</Text>
          <Text style={styles.item}>{seasonalityCompareChart?.ticker || ticker} vs {seasonalityCompareChart?.sector_etf || "sector ETF"} ({seasonalityCompareChart?.sector || "auto sector"})</Text>
          <Text style={styles.item}>Alignment corr {fmtNumber(seasonalityCompareChart?.metrics?.corr, 3)} | follow-rate {fmtPct(seasonalityCompareChart?.metrics?.follow_rate)} | mean abs diff {fmtPct(seasonalityCompareChart?.metrics?.mean_abs_diff)}</Text>
          <Text style={styles.item}>Average seasonal return by day-of-year</Text>
          <MiniSeriesChart
            series={[
              { label: "stock-avg", values: seasonalityStockAvg, color: THEME.accentDeep, width: 3, type: "line" },
              { label: "sector-avg", values: seasonalitySectorAvg, color: "#0ea5e9", width: 2, type: "line" },
              { label: "spread", values: seasonalitySpread, color: THEME.warn, width: 2, opacity: 0.7, type: "line" },
            ]}
            height={120}
          />
          <Text style={styles.item}>Cumulative seasonal path</Text>
          <MiniSeriesChart
            series={[
              { label: "stock-cum", values: seasonalityStockCum, color: THEME.text, width: 3, type: "line" },
              { label: "sector-cum", values: seasonalitySectorCum, color: "#16a34a", width: 2, type: "line" },
            ]}
            height={120}
          />
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>Training Features ({trainingFeatures.length})</Text>
          <Text style={styles.item}>These rows are model-ready numeric features derived from the same recommendation and distribution pipeline, useful for training or ranking experiments.</Text>
          {(trainingFeatures || []).slice(0, 8).map((row) => (
            <View key={`feat-${row.ticker}`} style={styles.analyticsItemCard}>
              <Text style={styles.analyticsHeadline}>{row.ticker} | composite {fmtNumber(row.composite_score)} | bias {row.top_bias} | horizon {row.primary_horizon}</Text>
              <Text style={styles.item}>5d {fmtPct(row.ret_5d)} | 21d {fmtPct(row.ret_21d)} | 63d {fmtPct(row.ret_63d)}</Text>
              <Text style={styles.item}>RSI {fmtNumber(row.rsi14)} | ATR {fmtNumber(row.atr14)} | Vol20 {fmtNumber(row.vol20, 3)}</Text>
              <Text style={styles.item}>SMA20 {fmtPct(row.distance_sma20_pct)} | SMA50 {fmtPct(row.distance_sma50_pct)} | SMA200 {fmtPct(row.distance_sma200_pct)}</Text>
              <Text style={styles.item}>Forecast {fmtPct(row.forecast_return_pct)} | band {fmtPct(row.forecast_band_pct)} | trend/day {fmtPct(row.forecast_daily_trend_pct)}</Text>
            </View>
          ))}
        </View>
      </>
    );
  };

  const renderCharts = () => {
    const priceSeries = indicatorTail.map((row) => Number(row["adj close"] || row.close || 0));
    const sma20Series = indicatorTail.map((row) => Number(row.sma20 || 0));
    const sma50Series = indicatorTail.map((row) => Number(row.sma50 || 0));
    const rsiSeries = indicatorTail.map((row) => Number(row.rsi14 || 0));
    const volumeSeries = indicatorTail.map((row) => Number(row.volume || 0));
    const seasonalityAvg = seasonalityTail.map((row) => Number(row.avg_ret || 0));
    const seasonalityStd = seasonalityTail.map((row) => Number(row.std_ret || 0));
    const seasonalityCompareRows = seasonalityCompareChart?.items || [];
    const seasonalityStockAvg = seasonalityCompareRows.map((row) => Number(row.stock_avg_ret || 0));
    const seasonalitySectorAvg = seasonalityCompareRows.map((row) => Number(row.sector_avg_ret || 0));
    const seasonalitySpread = seasonalityCompareRows.map((row) => Number(row.spread || 0));
    const seasonalityStockCum = seasonalityCompareRows.map((row) => Number(row.stock_cum || 0));
    const seasonalitySectorCum = seasonalityCompareRows.map((row) => Number(row.sector_cum || 0));
    const hasData = priceSeries.some((v) => v > 0);
    return (
      <>
        <View style={styles.card}>
          <Text style={styles.cardTitle}>Charts — {ticker}</Text>
          <Text style={styles.item}>Load price, RSI, volume, and seasonality charts for any ticker. Use the Analytics tab to configure settings and fetch data, or load directly below.</Text>
          <Text style={styles.fieldLabel}>Chart Controls</Text>
          <View style={styles.inlineRowWrap}>
            <Text style={[styles.fieldLabel, styles.compactLabel]}>Ticker</Text>
            <Text style={[styles.fieldLabel, styles.compactLabel]}>Period</Text>
            <Text style={[styles.fieldLabel, styles.compactLabel]}>Interval</Text>
          </View>
          <View style={styles.inlineRowWrap}>
            <TextInput value={ticker} onChangeText={setTicker} style={[styles.input, styles.compactInput]} placeholder="Ticker" autoCapitalize="characters" />
            <TextInput value={analyticsPeriod} onChangeText={setAnalyticsPeriod} style={[styles.input, styles.compactInput]} placeholder="Period (1y)" autoCapitalize="none" />
            <TextInput value={analyticsInterval} onChangeText={setAnalyticsInterval} style={[styles.input, styles.compactInput]} placeholder="Interval (1d)" autoCapitalize="none" />
          </View>
          <View style={styles.inlineRowWrap}>
            <Pressable style={styles.button} onPress={loadAnalytics}>
              <Text style={styles.buttonText}>Load Charts</Text>
            </Pressable>
            <Pressable style={styles.secondaryButton} onPress={loadChartsFast}>
              <Text style={styles.secondaryButtonText}>Load Fast</Text>
            </Pressable>
            <Pressable style={styles.secondaryButton} onPress={cacheMarketSeries}>
              <Text style={styles.secondaryButtonText}>Fetch + Cache Series</Text>
            </Pressable>
          </View>
          {analyticsBusy && <ProgressBar progress={progressValue} label="Loading chart data" compact />}
          {!hasData && !analyticsBusy && (
            <Text style={styles.mutedText}>No data loaded. Enter a ticker and press Load Charts.</Text>
          )}
          <Text style={styles.item}>Indicator rows: {indicatorChart.length}</Text>
        </View>

        {hasData && (
          <>
            <View style={styles.card}>
              <Text style={styles.cardTitle}>Price — {ticker}</Text>
              <Text style={styles.item}>Adjusted close with SMA 20 and SMA 50 over the latest {indicatorTail.length} bars.</Text>
              <MiniSeriesChart
                series={[
                  { label: "price", values: priceSeries, color: THEME.text, width: 3, type: "line" },
                  { label: "sma20", values: sma20Series, color: THEME.accent, width: 2, type: "line" },
                  { label: "sma50", values: sma50Series, color: THEME.warn, width: 2, type: "line" },
                ]}
                xLabels={indicatorTail.map((row) => String(row.date || ""))}
                height={160}
              />
            </View>

            <View style={styles.card}>
              <Text style={styles.cardTitle}>RSI 14</Text>
              <Text style={styles.item}>Relative Strength Index. Above 70 is typically overbought; below 30 oversold.</Text>
              <MiniSeriesChart series={[{ label: "rsi14", values: rsiSeries, color: THEME.accentDeep, width: 3, type: "line" }]} xLabels={indicatorTail.map((row) => String(row.date || ""))} height={120} />
            </View>

            <View style={styles.card}>
              <Text style={styles.cardTitle}>Volume</Text>
              <MiniSeriesChart series={[{ label: "volume", values: volumeSeries, color: "#7c6d5a", width: 3, opacity: 0.75, type: "bar" }]} xLabels={indicatorTail.map((row) => String(row.date || ""))} height={100} />
            </View>

            <View style={styles.card}>
              <Text style={styles.cardTitle}>Seasonality — Average Daily Return</Text>
              <Text style={styles.item}>Average and standard deviation of daily returns by day-of-year across the historical sample. High average with low deviation indicates a reliable seasonal edge.</Text>
              <MiniSeriesChart
                series={[
                  { label: "avg_ret", values: seasonalityAvg, color: THEME.accent, width: 3, type: "line" },
                  { label: "std_ret", values: seasonalityStd, color: THEME.warn, width: 2, opacity: 0.7, type: "line" },
                ]}
                xLabels={seasonalityTail.map((row, idx) => dayOfYearLabel(row, idx + 1))}
                height={130}
              />
            </View>

            <View style={styles.card}>
              <Text style={styles.cardTitle}>Stock vs Sector Seasonality</Text>
              <Text style={styles.item}>
                {seasonalityCompareChart?.ticker || ticker} vs {seasonalityCompareChart?.sector_etf || "sector ETF"} ({seasonalityCompareChart?.sector || "auto sector"}). Alignment corr {fmtNumber(seasonalityCompareChart?.metrics?.corr, 3)} | follow-rate {fmtPct(seasonalityCompareChart?.metrics?.follow_rate)}.
              </Text>
              <Text style={styles.item}>Average seasonal return by day-of-year</Text>
              <MiniSeriesChart
                series={[
                  { label: "stock-avg", values: seasonalityStockAvg, color: THEME.accentDeep, width: 3, type: "line" },
                  { label: "sector-avg", values: seasonalitySectorAvg, color: "#0ea5e9", width: 2, type: "line" },
                  { label: "spread", values: seasonalitySpread, color: THEME.warn, width: 2, opacity: 0.7, type: "line" },
                ]}
                height={130}
              />
              <Text style={styles.item}>Cumulative seasonal path</Text>
              <MiniSeriesChart
                series={[
                  { label: "stock-cum", values: seasonalityStockCum, color: THEME.text, width: 3, type: "line" },
                  { label: "sector-cum", values: seasonalitySectorCum, color: "#16a34a", width: 2, type: "line" },
                ]}
                height={120}
              />
            </View>
          </>
        )}
      </>
    );
  };

  const renderBacktest = () => (
    <View style={styles.card}>
      <Text style={styles.cardTitle}>Backtest</Text>
      {backtestBusy && <ProgressBar progress={progressValue} label="Backtest in progress" compact />}
      <Text style={styles.fieldLabel}>Ticker</Text>
      <TextInput value={ticker} onChangeText={setTicker} style={styles.input} placeholder="Ticker" />
      <Text style={styles.fieldLabel}>Start Date</Text>
      <TextInput value={backtestStart} onChangeText={setBacktestStart} style={styles.input} placeholder="Start date (YYYY-MM-DD)" autoCapitalize="none" />
      <Text style={styles.fieldLabel}>End Date (Optional)</Text>
      <TextInput value={backtestEnd} onChangeText={setBacktestEnd} style={styles.input} placeholder="End date (optional, YYYY-MM-DD)" autoCapitalize="none" />
      <Text style={styles.fieldLabel}>Strategy Parameters</Text>
      <View style={styles.inlineRowWrap}>
        <Text style={[styles.fieldLabel, styles.compactLabel]}>Entry RSI Threshold</Text>
        <Text style={[styles.fieldLabel, styles.compactLabel]}>Max Hold Days</Text>
      </View>
      <View style={styles.inlineRowWrap}>
        <TextInput value={backtestEntryRsi} onChangeText={setBacktestEntryRsi} style={[styles.input, styles.compactInput]} placeholder="Entry RSI threshold" keyboardType="numeric" />
        <TextInput value={backtestMaxHoldDays} onChangeText={setBacktestMaxHoldDays} style={[styles.input, styles.compactInput]} placeholder="Max hold days" keyboardType="numeric" />
      </View>
      <View style={styles.inlineRowWrap}>
        <Text style={[styles.fieldLabel, styles.compactLabel]}>Stop Loss %</Text>
        <Text style={[styles.fieldLabel, styles.compactLabel]}>MA Filter (sma20/sma50/sma200)</Text>
      </View>
      <View style={styles.inlineRowWrap}>
        <TextInput value={backtestStopLossPct} onChangeText={setBacktestStopLossPct} style={[styles.input, styles.compactInput]} placeholder="Stop loss % (0 disables)" keyboardType="numeric" />
        <TextInput value={backtestMaFilter} onChangeText={setBacktestMaFilter} style={[styles.input, styles.compactInput]} placeholder="sma20" autoCapitalize="none" />
      </View>
      <View style={styles.inlineRowWrap}>
        <Text style={[styles.fieldLabel, styles.compactLabel]}>Take Profit %</Text>
        <Text style={[styles.fieldLabel, styles.compactLabel]}>MA Trend Filter</Text>
      </View>
      <View style={styles.inlineRowWrap}>
        <TextInput value={backtestTakeProfitPct} onChangeText={setBacktestTakeProfitPct} style={[styles.input, styles.compactInput]} placeholder="Take profit % (0 disables)" keyboardType="numeric" />
        <TextInput value={backtestMaTrendFilter} onChangeText={setBacktestMaTrendFilter} style={[styles.input, styles.compactInput]} placeholder="none" autoCapitalize="none" />
      </View>
      <Pressable style={styles.button} onPress={loadBacktest}>
        <Text style={styles.buttonText}>Run Short-Term Backtest</Text>
      </Pressable>
      {backtestSummary ? (
        <>
          {!!backtestSummary.params && (
            <Text style={styles.item}>
              Params: RSI gt {fmtNumber(backtestSummary.params.entry_rsi_threshold)} | Max hold {backtestSummary.params.max_hold_days}d | Stop loss {fmtNumber(backtestSummary.params.stop_loss_pct)}% | Take profit {fmtNumber(backtestSummary.params.take_profit_pct)}% | MA {String(backtestSummary.params.ma_filter || "sma20").toUpperCase()} | Trend {String(backtestSummary.params.ma_trend_filter || "none").toUpperCase()}
            </Text>
          )}
          <Text style={styles.item}>Trades: {backtestSummary.n_trades} | Win rate: {fmtPct(backtestSummary.win_rate)} | Avg hold: {Number(backtestSummary.avg_hold_days || 0).toFixed(1)}d</Text>
          <Text style={styles.item}>Avg return: {fmtPct(backtestSummary.avg_ret)} | Median return: {fmtPct(backtestSummary.median_ret)} | Max DD: {fmtPct(backtestSummary.max_dd)}</Text>
          <Text style={styles.item}>Sharpe: {Number(backtestSummary.sharpe || 0).toFixed(2)} | Sortino: {Number(backtestSummary.sortino || 0).toFixed(2)} | CAGR: {fmtPct(backtestSummary.cagr)} | PF: {Number(backtestSummary.profit_factor || 0).toFixed(2)}</Text>
          <Text style={styles.item}>Equity curve points: {backtestEquity.length}</Text>
          <MiniSeriesChart
            series={[{ label: "equity", values: backtestEquity.map((row) => Number(row.equity || 0)), color: THEME.accentDeep, width: 3 }]}
            xLabels={backtestEquity.map((row) => String(row.date || row.timestamp || ""))}
            height={120}
          />

          <Text style={styles.cardTitle}>Price Movement Context</Text>
          <Text style={styles.item}>
            Last close {fmtNumber(indicatorTail[indicatorTail.length - 1]?.["adj close"] || indicatorTail[indicatorTail.length - 1]?.close)} |
            20-bar move {indicatorTail.length > 20 ? fmtPct((Number(indicatorTail[indicatorTail.length - 1]?.["adj close"] || indicatorTail[indicatorTail.length - 1]?.close || 0) / Number(indicatorTail[indicatorTail.length - 21]?.["adj close"] || indicatorTail[indicatorTail.length - 21]?.close || 1)) - 1) : "-"} |
            RSI {fmtNumber(indicatorTail[indicatorTail.length - 1]?.rsi14)}
          </Text>
          <MiniSeriesChart
            series={[
              { label: "price", values: indicatorTail.map((row) => Number(row["adj close"] || row.close || 0)), color: THEME.text, width: 3, type: "line" },
              { label: "sma20", values: indicatorTail.map((row) => Number(row.sma20 || 0)), color: THEME.accent, width: 2, type: "line" },
              { label: "sma50", values: indicatorTail.map((row) => Number(row.sma50 || 0)), color: THEME.warn, width: 2, type: "line" },
            ]}
            xLabels={indicatorTail.map((row) => String(row.date || ""))}
            height={120}
          />

          <Text style={styles.cardTitle}>Seasonality Context</Text>
          <Text style={styles.item}>Seasonality sample points: {seasonalityTail.length} | average daily return and deviation by day-of-year.</Text>
          <MiniSeriesChart
            series={[
              { label: "avg_ret", values: seasonalityTail.map((row) => Number(row.avg_ret || 0)), color: THEME.accent, width: 3, type: "line" },
              { label: "std_ret", values: seasonalityTail.map((row) => Number(row.std_ret || 0)), color: THEME.warn, width: 2, type: "line", opacity: 0.75 },
            ]}
            xLabels={seasonalityTail.map((row, idx) => dayOfYearLabel(row, idx + 1))}
            height={100}
          />

          <Text style={styles.cardTitle}>Sentiment Context</Text>
          <Text style={styles.item}>News summary: {newsSummary?.summary || "No sentiment summary loaded."}</Text>
          {!!(newsSummary?.items?.length) && (
            <Text style={styles.item}>
              Latest aggregate sentiment: {fmtNumber(newsSummary.items[0]?.avg_sentiment)} across {newsSummary.items[0]?.count || 0} articles ({newsSummary.items[0]?.date || "-"}).
            </Text>
          )}
          <NewsTimelineChart timeline={newsTimeline} />
          <NewsSummaryList summary={newsSummary} />
        </>
      ) : (
        <Text style={styles.item}>no backtest loaded</Text>
      )}
    </View>
  );

  const renderExecution = () => (
    <View style={styles.card}>
      <Text style={styles.cardTitle}>Propose Order</Text>
      {executionBusy && <ProgressBar progress={progressValue} label="Evaluating policy and creating intent" compact />}
      <Text style={styles.item}>DB path</Text>
      <TextInput value={dbPath} onChangeText={setDbPath} style={styles.input} placeholder="sqlite:///quantflow.db" autoCapitalize="none" />
      <Text style={styles.fieldLabel}>Ticker</Text>
      <TextInput value={ticker} onChangeText={setTicker} style={styles.input} placeholder="Ticker" />
      <Text style={styles.fieldLabel}>Action</Text>
      <TextInput value={executionAction} onChangeText={setExecutionAction} style={styles.input} placeholder="Action: buy/sell" autoCapitalize="none" />
      <Text style={styles.fieldLabel}>Quantity</Text>
      <TextInput value={executionQty} onChangeText={setExecutionQty} style={styles.input} placeholder="Qty" keyboardType="numeric" />
      <Text style={styles.fieldLabel}>Notional</Text>
      <TextInput value={notional} onChangeText={setNotional} style={styles.input} placeholder="Notional" keyboardType="numeric" />
      <Text style={styles.fieldLabel}>Orders Today</Text>
      <TextInput value={executionOrdersToday} onChangeText={setExecutionOrdersToday} style={styles.input} placeholder="Orders today" keyboardType="numeric" />
      <Text style={styles.fieldLabel}>Position Notional After</Text>
      <TextInput value={executionPositionAfter} onChangeText={setExecutionPositionAfter} style={styles.input} placeholder="Position notional after" keyboardType="numeric" />
      <Text style={styles.fieldLabel}>Broker Mode</Text>
      <TextInput value={executionBrokerMode} onChangeText={setExecutionBrokerMode} style={styles.input} placeholder="Broker mode: robinhood_mcp | robinhood" autoCapitalize="none" />
      <View style={styles.inlineRowWrap}>
        <Text style={styles.item}>Auto mode:</Text>
        <Pressable style={executionAuto ? styles.smallButton : styles.mutedButton} onPress={() => setExecutionAuto(true)}>
          <Text style={styles.buttonText}>ON</Text>
        </Pressable>
        <Pressable style={!executionAuto ? styles.smallButton : styles.mutedButton} onPress={() => setExecutionAuto(false)}>
          <Text style={styles.buttonText}>OFF</Text>
        </Pressable>
      </View>
      <Text style={styles.item}>Policy limits</Text>
      <Text style={styles.fieldLabel}>Max Daily Notional</Text>
      <TextInput value={executionMaxDaily} onChangeText={setExecutionMaxDaily} style={styles.input} placeholder="Max daily notional" keyboardType="numeric" />
      <Text style={styles.fieldLabel}>Max Orders Per Day</Text>
      <TextInput value={executionMaxOrders} onChangeText={setExecutionMaxOrders} style={styles.input} placeholder="Max orders/day" keyboardType="numeric" />
      <Text style={styles.fieldLabel}>Max Position Notional</Text>
      <TextInput value={executionMaxPosition} onChangeText={setExecutionMaxPosition} style={styles.input} placeholder="Max position notional" keyboardType="numeric" />
      <Pressable style={styles.button} onPress={proposeOrder}>
        <Text style={styles.buttonText}>Create Intent</Text>
      </Pressable>
      {!!executionResult && (
        <Text style={styles.item}>Decision: {executionResult.allow ? "APPROVED" : "BLOCKED"} | intent #{executionResult.intent_id} | reason: {executionResult.reason}</Text>
      )}
      <Text style={styles.cardTitle}>Recent Intents ({intents.length})</Text>
      {(intents || []).slice(0, 12).map((i) => (
        <Text key={String(i.id)} style={styles.item}>
          #{i.id} {i.ticker} {i.action} ${i.notional} [{i.status}]
        </Text>
      ))}
    </View>
  );

  const renderAssistant = (compact = false) => (
    <View style={styles.card}>
      <Text style={styles.cardTitle}>QuantFlow Assistant</Text>
      {assistantBusy && <ProgressBar progress={progressValue} label="Assistant processing" compact />}
      <View style={styles.inlineRow}>
        <Text style={[styles.label, { flex: 1 }]}>Queries local QuantFlow data and suggests API automation workflows.</Text>
        <View style={styles.inlineRow}>
          <Text style={styles.mutedText}>POST dry-run </Text>
          <Pressable style={assistantDryRun ? styles.smallButton : styles.mutedButton} onPress={() => setAssistantDryRun(true)}>
            <Text style={styles.buttonText}>ON</Text>
          </Pressable>
          <Pressable style={!assistantDryRun ? styles.smallButton : styles.mutedButton} onPress={() => setAssistantDryRun(false)}>
            <Text style={styles.buttonText}>OFF</Text>
          </Pressable>
        </View>
      </View>
      <View style={styles.inlineRow}>
        <Text style={styles.mutedText}>Auto-run safe GET actions </Text>
        <View style={styles.inlineRow}>
          <Pressable style={assistantAutoRunGet ? styles.smallButton : styles.mutedButton} onPress={() => setAssistantAutoRunGet(true)}>
            <Text style={styles.buttonText}>ON</Text>
          </Pressable>
          <Pressable style={!assistantAutoRunGet ? styles.smallButton : styles.mutedButton} onPress={() => setAssistantAutoRunGet(false)}>
            <Text style={styles.buttonText}>OFF</Text>
          </Pressable>
        </View>
      </View>
      {!compact && (
        <View style={styles.inlineRowWrap}>
          <Pressable style={styles.secondaryButton} onPress={runMcpOnboardingRunbook}>
            <Text style={styles.secondaryButtonText}>Run MCP Onboarding Runbook</Text>
          </Pressable>
        </View>
      )}

      {/* Conversation history */}
      {assistantConversation.length === 0 && (
        <View style={styles.chatEmptyState}>
          <Text style={styles.mutedText}>No messages yet. Ask anything about your portfolio, scanner results, or recommended setups.</Text>
        </View>
      )}
      {assistantConversation.map((turn, idx) => (
        <View key={String(idx)} style={turn.role === "user" ? styles.chatBubbleUser : styles.chatBubbleAssistant}>
          <Text style={turn.role === "user" ? styles.chatLabelUser : styles.chatLabelAssistant}>
            {turn.role === "user" ? "You" : "Assistant"}
          </Text>
          <Text style={turn.role === "user" ? styles.chatTextUser : styles.chatTextAssistant}>{turn.content}</Text>
          {!!(turn.summary) && (
            <Text style={styles.chatMeta}>Summary: {compactJson(turn.summary)}</Text>
          )}
          {!!(!compact && turn.suggested_tool_calls && turn.suggested_tool_calls.length > 0) && (
            <View style={{ marginTop: 8 }}>
              <Text style={styles.chatMeta}>Suggested actions:</Text>
              {turn.suggested_tool_calls.slice(0, 6).map((c, ci) => (
                <View key={String(ci)} style={styles.chatToolCallRow}>
                  <Text style={styles.chatToolCallText}>{c.method} {c.path}{c.name ? ` — ${c.name}` : ""}</Text>
                  <Pressable style={styles.smallButton} onPress={() => requestExecuteAction(c)}>
                    <Text style={styles.buttonText}>Run</Text>
                  </Pressable>
                </View>
              ))}
            </View>
          )}
        </View>
      ))}

      {/* Confirm pending POST */}
      {!!(!compact && pendingAction) && (
        <View style={[styles.card, { marginTop: 8 }]}>
          <Text style={styles.cardTitle}>Confirm Action</Text>
          <Text style={styles.item}>{(pendingAction.method || "POST").toUpperCase()} {pendingAction.path}</Text>
          <Text style={styles.item}>{compactJson(pendingAction.payload || {}, 800)}</Text>
          <View style={styles.inlineRow}>
            <Pressable style={styles.smallButton} onPress={confirmPendingAction}>
              <Text style={styles.buttonText}>Confirm</Text>
            </Pressable>
            <Pressable style={styles.mutedButton} onPress={cancelPendingAction}>
              <Text style={styles.buttonText}>Cancel</Text>
            </Pressable>
          </View>
        </View>
      )}

      {/* Action result */}
      {!!assistantActionResult && (
        <View style={styles.chatBubbleAssistant}>
          <Text style={styles.chatLabelAssistant}>Action Result</Text>
          <Text style={styles.chatTextAssistant}>
            {assistantActionResult.ok ? (assistantActionResult.dryRun ? "Dry-run — action not executed." : "Success.") : `Error: ${assistantActionResult.error}`}
          </Text>
          {!!assistantActionResult.ok && (
            <Text style={styles.chatMeta}>{compactJson(assistantActionResult.result, 800)}</Text>
          )}
        </View>
      )}

      {!!assistantAutoResults.length && (
        <View style={styles.chatBubbleAssistant}>
          <Text style={styles.chatLabelAssistant}>Auto-Run Results</Text>
          {!!assistantAutoSummary && (
            <Text style={styles.chatMeta}>Last run {fmtTimestamp(assistantAutoSummary.ts)} | {assistantAutoSummary.total} calls | {assistantAutoSummary.ok} ok | {assistantAutoSummary.failed} failed</Text>
          )}
          {assistantAutoResults.slice(0, 5).map((row, idx) => (
            <Text key={`auto-${String(idx)}`} style={styles.chatMeta}>
              {row.ok ? "OK" : "ERR"} | {row.call?.method || "GET"} {row.call?.path} {row.ok ? "" : `| ${row.error}`}
            </Text>
          ))}
          {assistantAutoResults.length > 5 && (
            <Text style={styles.chatMeta}>...and {assistantAutoResults.length - 5} more entries.</Text>
          )}
        </View>
      )}

      {!!mcpRunbookReport && (
        <View style={styles.chatBubbleAssistant}>
          <Text style={styles.chatLabelAssistant}>MCP Onboarding Report</Text>
          <Text style={styles.chatMeta}>Ran at: {mcpRunbookReport.ran_at}</Text>
          <Text style={styles.chatMeta}>Overall ready: {mcpRunbookReport.overall_ready ? "yes" : "no"}</Text>
          <Text style={styles.chatMeta}>Transport connected: {mcpRunbookReport.transport_connected ? "yes" : "no"}</Text>
          <Text style={styles.chatMeta}>Authenticated: {mcpRunbookReport.authenticated ? "yes" : "no"}</Text>
          <Text style={styles.chatMeta}>Positions: {mcpRunbookReport.positions_count} | Signals: {mcpRunbookReport.signals_count}</Text>
          {!!mcpRunbookReport.blockers?.length && (
            <>
              <Text style={styles.warnText}>Blockers</Text>
              {mcpRunbookReport.blockers.map((row, idx) => (
                <Text key={`block-${String(idx)}`} style={styles.chatMeta}>- {row.label}: {row.details}</Text>
              ))}
            </>
          )}
          {!!mcpRunbookReport.warnings?.length && (
            <>
              <Text style={styles.warnText}>Warnings</Text>
              {mcpRunbookReport.warnings.map((row, idx) => (
                <Text key={`warn-${String(idx)}`} style={styles.chatMeta}>- {row.label}: {row.details}</Text>
              ))}
            </>
          )}
          {!!mcpRunbookReport.remediation?.length && (
            <>
              <Text style={styles.chatMeta}>Remediation steps</Text>
              {mcpRunbookReport.remediation.slice(0, 5).map((step, idx) => (
                <Text key={`fix-${String(idx)}`} style={styles.chatMeta}>{idx + 1}. {step}</Text>
              ))}
            </>
          )}
        </View>
      )}

      {/* Input row */}
      <View style={[styles.chatInputWrap, { marginTop: 10 }]}>
        <Text style={styles.fieldLabel}>Message</Text>
        <TextInput
          value={assistantInput}
          onChangeText={setAssistantInput}
          style={[styles.input, styles.chatInput]}
          placeholder="Ask about setups, universe scan, backtest results…"
          multiline
          scrollEnabled
        />
        <Pressable style={[styles.button, { marginBottom: 0, alignSelf: "flex-end" }]} onPress={askAssistant}>
          <Text style={styles.buttonText}>Send</Text>
        </Pressable>
      </View>
      {assistantConversation.length > 0 && (
        <Pressable style={[styles.mutedButton, { marginTop: 6 }]} onPress={() => { setAssistantConversation([]); setAssistantActionResult(null); }}>
          <Text style={styles.buttonText}>Clear Chat</Text>
        </Pressable>
      )}

      {!!compact && (
        <Pressable style={[styles.secondaryButton, { marginTop: 8 }]} onPress={() => setScreen("Assistant")}>
          <Text style={styles.secondaryButtonText}>Open Full Assistant</Text>
        </Pressable>
      )}
    </View>
  );

  const renderAuth = () => (
    <>
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Connection & Auth Controls</Text>
        {authBusy && <ProgressBar progress={progressValue} label="Auth action in progress" compact />}
        <Text style={styles.item}>Use API key for token mode, or a Firebase Bearer token for user settings and Google-auth workflows.</Text>
        <Text style={styles.item}>Google web sign-in requires EXPO_PUBLIC_FIREBASE_API_KEY, EXPO_PUBLIC_FIREBASE_AUTH_DOMAIN, EXPO_PUBLIC_FIREBASE_PROJECT_ID, and EXPO_PUBLIC_FIREBASE_APP_ID.</Text>
        <Text style={styles.fieldLabel}>API Key</Text>
        <TextInput
          value={apiKeyInput}
          onChangeText={setApiKeyInput}
          style={styles.input}
          placeholder="x-api-key (optional)"
          autoCapitalize="none"
        />
        <Text style={styles.fieldLabel}>Bearer Token</Text>
        <TextInput
          value={bearerToken}
          onChangeText={setBearerToken}
          style={[styles.input, styles.tokenInput]}
          placeholder="Firebase ID token (Bearer)"
          autoCapitalize="none"
          multiline
        />
        <View style={styles.inlineRowWrap}>
          <Pressable style={styles.button} onPress={signInWithGoogleWeb}>
            <Text style={styles.buttonText}>Sign In With Google (Web)</Text>
          </Pressable>
          <Pressable style={styles.button} onPress={checkFirebaseIdentity}>
            <Text style={styles.buttonText}>Check /auth/me</Text>
          </Pressable>
          <Pressable style={styles.secondaryButton} onPress={loadOverview}>
            <Text style={styles.secondaryButtonText}>Refresh Overview</Text>
          </Pressable>
        </View>
        <Text style={styles.item}>Google login status: {googleSignInStatus}</Text>
        <Text style={styles.item}>Identity: {firebaseAuthCheck ? compactJson(firebaseAuthCheck, 900) : "not checked"}</Text>
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Firebase User Settings</Text>
        {settingsSaveStatus === "loading" || settingsSaveStatus === "saving" ? <ProgressBar progress={progressValue} label="Saving user settings" compact /> : null}
        <Text style={styles.item}>Reads and writes per-user settings to Firestore via /user/settings endpoints.</Text>
        <View style={styles.inlineRowWrap}>
          <Pressable style={styles.button} onPress={loadUserSettings}>
            <Text style={styles.buttonText}>Load Settings</Text>
          </Pressable>
          <Pressable style={styles.secondaryButton} onPress={saveUserSettings}>
            <Text style={styles.secondaryButtonText}>Save Settings</Text>
          </Pressable>
        </View>
        <Text style={styles.item}>Status: {settingsSaveStatus}</Text>
        <Text style={styles.fieldLabel}>User Settings JSON</Text>
        <TextInput
          value={userSettingsText}
          onChangeText={setUserSettingsText}
          style={[styles.input, styles.jsonInput]}
          placeholder='{"watchlist": ["AAPL"], "risk_profile": "balanced"}'
          multiline
        />
        <Text style={styles.item}>Server payload: {userSettingsData ? compactJson(userSettingsData, 1200) : "none"}</Text>
      </View>
    </>
  );

  const renderPresetsManager = () => (
    <>
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Preset Management</Text>
        <Text style={styles.item}>View, create, and edit finviz screener presets.</Text>
        {presetSaveStatus === "saving" && <ProgressBar progress={progressValue} label="Saving preset" compact />}
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Available Presets</Text>
        <Text style={styles.item}>Total: {presets.length}</Text>
        <ScrollView horizontal style={styles.inlineRow}>
          {presets.map((name) => (
            <Pressable
              key={name}
              onPress={() => {
                setPresetEditName(name);
                const profile = presetDetails?.profiles?.[name] || {};
                setPresetEditContent(JSON.stringify(profile, null, 2));
              }}
              style={[styles.smallButton, presetEditName === name && { backgroundColor: THEME.accentDeep }]}
            >
              <Text style={styles.buttonText}>{name}</Text>
            </Pressable>
          ))}
        </ScrollView>
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Create / Edit Preset</Text>
        <Text style={styles.fieldLabel}>Preset Name</Text>
        <TextInput
          value={presetEditName}
          onChangeText={setPresetEditName}
          style={styles.input}
          placeholder="Preset name (e.g., weekly_momo)"
          autoCapitalize="none"
        />
        <Text style={styles.fieldLabel}>Preset Filter JSON</Text>
        <TextInput
          value={presetEditContent}
          onChangeText={setPresetEditContent}
          style={[styles.input, styles.jsonInput]}
          placeholder='{"Average Volume": "Over 300K", "Performance": "Week Up"}'
          multiline
        />
        <View style={styles.inlineRowWrap}>
          <Pressable
            style={styles.button}
            onPress={async () => {
              if (!presetEditName.trim()) {
                setPresetSaveStatus("error: preset name required");
                return;
              }
              try {
                const parsed = JSON.parse(presetEditContent);
                setPresetSaveStatus("saving");
                const result = await apiPost("/scanner/presets/update", {
                  name: presetEditName.trim(),
                  filters: parsed,
                }, authState);
                setPresetSaveStatus("saved");
                setPresets([...new Set([...presets, presetEditName.trim()])]);
                setTimeout(() => setPresetSaveStatus("idle"), 2000);
                await loadOverview();
              } catch (e) {
                setPresetSaveStatus(`error: ${e.message}`);
              }
            }}
          >
            <Text style={styles.buttonText}>Save Preset</Text>
          </Pressable>
          <Pressable
            style={styles.secondaryButton}
            onPress={() => {
              setPresetEditName("");
              setPresetEditContent("");
              setPresetSaveStatus("idle");
            }}
          >
            <Text style={styles.secondaryButtonText}>Clear</Text>
          </Pressable>
        </View>
        <Text style={styles.item}>Status: {presetSaveStatus}</Text>
      </View>
    </>
  );

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar style="dark" />
      <ScrollView contentContainerStyle={styles.content}>
        <Text style={styles.title}>QuantFlow Experience</Text>
        <Text style={styles.meta}>{baseLabel}</Text>
        <Text style={styles.meta}>Auth key configured: {authState.apiKey ? "yes" : "no"}</Text>
        <Text style={styles.meta}>Bearer configured: {authState.bearerToken ? "yes" : "no"}</Text>
        <Text style={styles.meta}>Status: {status}</Text>
        {inFlightOps > 0 && <ProgressBar progress={progressValue} label={activeTaskLabel || "Loading"} />}
        <View style={styles.tabRow}>
          {SURFACES.map((name) => (
            <Pressable
              key={name}
              style={[styles.tab, surface === name ? styles.tabActive : null]}
              onPress={() => setSurface(name)}
            >
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
                  <Pressable
                    key={name}
                    style={[styles.tab, screen === name ? styles.tabActive : null]}
                    onPress={() => setScreen(name)}
                  >
                    <Text style={[styles.tabText, screen === name ? styles.tabTextActive : null]}>{name}</Text>
                  </Pressable>
                ))}
              </View>

              {screen === "Overview" && renderOverview()}
              {screen === "Scanner" && renderScanner()}
              {screen === "Recommend" && renderRecommend()}
              {screen === "Analytics" && renderAnalytics()}
              {screen === "Charts" && renderCharts()}
              {screen === "Options" && renderOptions()}
              {screen === "Backtest" && renderBacktest()}
              {screen === "Execution" && renderExecution()}
              {screen === "Assistant" && renderAssistant()}
              {screen === "Presets" && renderPresetsManager()}
              {screen === "Auth" && renderAuth()}
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
  hero: {
    backgroundColor: THEME.panel,
    borderRadius: 18,
    borderWidth: 1,
    borderColor: THEME.border,
    padding: 18,
    gap: 10,
    shadowColor: "#000",
    shadowOpacity: 0.05,
    shadowRadius: 12,
    shadowOffset: { width: 0, height: 6 },
  },
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
  inlineRow: { gap: 8, flexDirection: "row", alignItems: "center", justifyContent: "space-between" },
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
  timelineLegendRow: { gap: 4 },
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
});
