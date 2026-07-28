import React, { useEffect, useRef, useState, useCallback, useMemo } from "react";
import { View, Text, Pressable, StyleSheet } from "react-native";
import { ComposedChart, Bar, Line, Area, XAxis, YAxis, Tooltip, ReferenceLine, ResponsiveContainer } from "recharts";
import THEME from "../../theme/colors";
import ErrorBoundary from "../shared/ErrorBoundary";
import { formatChartDate } from "../../utils/dateFormat";

const SPEEDS = { 0.5: 100, 1: 50, 2: 25, 4: 12 };
const WINDOW_SIZE = 120;

function Candle(props) {
  const { x, y, width, height, payload } = props;
  if (!payload || payload.close == null || height == null) return null;

  const close = Number(payload.close);
  const open = Number(payload.open);
  const dMin = Number(payload._dMin);
  if (isNaN(close) || isNaN(open) || isNaN(dMin)) return null;

  if (payload.tradeEntry) {
    const cx = x + width / 2;
    return (
      <g>
        <polygon points={`${cx},${y - 10} ${cx + 7},${y} ${cx - 7},${y}`} fill={THEME.profit} stroke="#fff" strokeWidth={0.5} />
        <text x={cx} y={y - 14} textAnchor="middle" fill={THEME.profit} fontSize={9} fontWeight="800">E</text>
      </g>
    );
  }
  if (payload.tradeExit) {
    const cx = x + width / 2;
    return (
      <g>
        <polygon points={`${cx},${y + 10} ${cx + 7},${y} ${cx - 7},${y}`} fill={THEME.loss} stroke="#fff" strokeWidth={0.5} />
        <text x={cx} y={y + 24} textAnchor="middle" fill={THEME.loss} fontSize={9} fontWeight="800">X</text>
      </g>
    );
  }

  const pixelPerUnit = height / (close - dMin || 1);
  const baseline = y + height;
  const pClose = y;
  const pOpen = baseline - (open - dMin) * pixelPerUnit;
  const pHigh = baseline - (Number(payload.high || close) - dMin) * pixelPerUnit;
  const pLow = baseline - (Number(payload.low || close) - dMin) * pixelPerUnit;

  const isUp = close >= open;
  const color = isUp ? THEME.profit : THEME.loss;
  const bw = Math.max(1, width * 0.7);
  const cx = x + width / 2;
  const bt = isUp ? pClose : pOpen;
  const bh = Math.max(1, Math.abs(pClose - pOpen));

  return (
    <g>
      <line x1={cx} y1={pHigh} x2={cx} y2={pLow} stroke={color} strokeWidth={1} />
      <rect x={cx - bw / 2} y={bt} width={bw} height={bh} fill={color} />
    </g>
  );
}

function formatDate(d) {
  return formatChartDate(d, "short");
}

export default function BacktestProgress({ equityCurve, trades, signals, rsiSeries, running, onComplete, ticker, token }) {
  const [currentBar, setCurrentBar] = useState(0);
  const [speed, setSpeed] = useState(1);
  const [paused, setPaused] = useState(false);
  const [priceData, setPriceData] = useState([]);
  const [mlSignal, setMlSignal] = useState(null);
  const [mlForecast, setMlForecast] = useState(null);
  const timerRef = useRef(null);
  const completedRef = useRef(false);

  // Fetch ML signal for this ticker
  const loadMlSignal = useCallback(async () => {
    if (!ticker) return;
    try {
      const base = typeof process !== "undefined" && process.env?.EXPO_PUBLIC_ML_ENGINE_URL || "http://127.0.0.1:8000";
      const res = await fetch(`${base}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker, include_trajectory: false }),
      });
      if (res.ok) {
        const data = await res.json();
        setMlSignal(data);
      }
    } catch {}
    try {
      const apiBase = typeof process !== "undefined" && process.env?.EXPO_PUBLIC_API_BASE_URL || "http://127.0.0.1:3000";
      const fcRes = await fetch(`${apiBase}/charts/forecast?ticker=${encodeURIComponent(ticker)}&interval=1D`, {
        headers: { "Content-Type": "application/json" },
        ...(token ? { "Authorization": `Bearer ${token}` } : {}),
      });
      if (fcRes.ok) {
        const fcData = await fcRes.json();
        setMlForecast(fcData.forecast || []);
      }
    } catch {}
  }, [ticker, token]);

  useEffect(() => { loadMlSignal(); }, [loadMlSignal]);

  const loadPriceData = useCallback(async () => {
    if (!ticker || priceData.length > 0) return;
    try {
      const base = typeof process !== "undefined" && process.env?.EXPO_PUBLIC_API_BASE_URL || "http://127.0.0.1:3000";
      const headers = { "Content-Type": "application/json" };
      if (token) headers["Authorization"] = `Bearer ${token}`;
      const since = new Date(Date.now() - 1000 * 60 * 60 * 24 * 30).toISOString();
      let res = await fetch(`${base}/market/alpaca/bars/${encodeURIComponent(ticker)}?timeframe=5Min&limit=500&start=${encodeURIComponent(since)}`, { headers });
      if (!res.ok) {
        res = await fetch(`${base}/market/series/cache?ticker=${encodeURIComponent(ticker)}&interval=1d`, { method: "POST", headers });
      }
      const data = await res.json();
      const items = data?.bars || data?.items || [];
      if (items.length) setPriceData(items);
    } catch {}
  }, [ticker, token, priceData.length]);

  useEffect(() => { loadPriceData(); }, [loadPriceData]);
  useEffect(() => {
    if (!running || equityCurve.length === 0) { setCurrentBar(0); return; }
    setCurrentBar(0);
    completedRef.current = false;
  }, [running, equityCurve]);

  useEffect(() => {
    if (!running || paused || equityCurve.length === 0) return;
    const delay = SPEEDS[speed] || 50;
    timerRef.current = setInterval(() => {
      setCurrentBar((prev) => {
        const next = prev + 2; // advance 2 bars at a time for speed
        if (next >= equityCurve.length) {
          clearInterval(timerRef.current);
          if (!completedRef.current) { completedRef.current = true; setTimeout(() => onComplete?.(), 500); }
          return equityCurve.length - 1;
        }
        return next;
      });
    }, delay);
    return () => clearInterval(timerRef.current);
  }, [running, paused, speed, equityCurve.length, onComplete]);

  const currentDate = equityCurve[currentBar]?.date || "";

  // Build full chart data (no scrolling window - show all with playhead)
  const fullBars = useMemo(() => {
    try {
      // Use equityCurve dates as the time axis
      // Generate synthetic OHLC from equity (since we don't have real price data in backtest)
      const eqData = equityCurve || [];
      const sigMap = {};
      if (signals) {
        signals.forEach(s => {
          const sd = String(s.date).slice(0, 10);
          sigMap[sd] = s;
        });
      }
      return eqData.map((d, i) => {
        const dk = String(d.date).slice(0, 10);
        const sig = sigMap[dk];
        const eq = Number(d.equity) || 1;
        // Synthesize OHLC from equity value (scale around 100 for readability)
        const price = eq * 100; // scale to look like real prices
        const spread = price * 0.01;
        return {
          idx: i,
          date: dk,
          open: price - spread * 0.5,
          close: price + spread * 0.5,
          high: price + spread,
          low: price - spread,
          equity: eq,
          tradeEntry: sig?.type === "buy" || null,
          tradeExit: sig?.type === "sell" || null,
          signalReason: sig?.reason || null,
          _dMin: 0,
        };
      });
    } catch (e) {
      console.error("fullBars error:", e);
      return [];
    }
  }, [equityCurve, signals]);

  const chartData = useMemo(() => {
    try {
      if (!fullBars || fullBars.length === 0) return [];
      const sVals = fullBars.flatMap(d => [d.high, d.low, d.open, d.close]);
      const vMin = Math.min(...sVals);
      const vMax = Math.max(...sVals);
      const vRng = (vMax - vMin) || 1;
      return fullBars.map(d => ({ ...d, _dMin: vMin - vRng * 0.02 }));
    } catch (e) {
      return [];
    }
  }, [fullBars]);

  const eqData = useMemo(() => {
    try {
      return (equityCurve || []).map((d, i) => ({
        idx: i, date: d.date, equity: Number(d.equity) || 0
      }));
    } catch (e) {
      return [];
    }
  }, [equityCurve]);

  // Find current rsi
  const currentRsi = useMemo(() => {
    if (!rsiSeries || currentBar >= rsiSeries.length) return null;
    return rsiSeries[currentBar]?.rsi;
  }, [rsiSeries, currentBar]);

  // Find current signal
  const currentSignal = useMemo(() => {
    if (!signals) return null;
    const curDate = String(equityCurve?.[currentBar]?.date || "").slice(0, 10);
    return signals.find(s => String(s.date).slice(0, 10) === curDate);
  }, [signals, currentBar, equityCurve]);

  const formatDate = (d) => d ? String(d).split("-").slice(1, 3).join("/") : "";
  const priceStep = Math.max(1, Math.floor((chartData?.length || 0) / 6));
  const eqIsUp = (eqData?.length || 0) >= 2 && eqData[eqData.length - 1].equity >= eqData[0].equity;

  const buySignalCount = (signals || []).filter(s => s.type === "buy").length;
  const sellSignalCount = (signals || []).filter(s => s.type === "sell").length;

  if (!chartData.length) return null;

  return (
    <ErrorBoundary>
      <View style={styles.wrap}>
        <View style={styles.header}>
          <View style={{ flex: 1 }}>
            <Text style={styles.title}>Live Trading Playback</Text>
            <Text style={styles.sub}>
              {ticker || ""} · Bar {Math.min(currentBar + 1, equityCurve.length)}/{equityCurve.length} · {buySignalCount} buy / {sellSignalCount} sell
            </Text>
          </View>
          <View style={styles.controls}>
            {Object.keys(SPEEDS).map(s => (
              <Pressable key={s} style={[styles.sb, speed === Number(s) && styles.sbOn]} onPress={() => setSpeed(Number(s))}>
                <Text style={[styles.st, speed === Number(s) && styles.stOn]}>{s}x</Text>
              </Pressable>
            ))}
            <Pressable style={[styles.pb, paused && styles.pbOn]} onPress={() => setPaused(v => !v)}>
              <Text style={styles.pt}>{paused ? "▶" : "⏸"}</Text>
            </Pressable>
          </View>
        </View>

        {currentRsi != null && (
          <View style={styles.indicatorRow}>
            <View style={styles.indicatorBox}>
              <Text style={styles.indicatorLabel}>RSI</Text>
              <Text style={[styles.indicatorValue, { color: currentRsi < 30 ? THEME.profit : currentRsi > 70 ? THEME.loss : THEME.text }]}>
                {currentRsi.toFixed(1)}
              </Text>
            </View>
            {currentSignal && (
              <View style={[styles.indicatorBox, { backgroundColor: currentSignal.type === "buy" ? THEME.profitDim : THEME.lossDim }]}>
                <Text style={styles.indicatorLabel}>Signal</Text>
                <Text style={[styles.indicatorValue, { color: currentSignal.type === "buy" ? THEME.profit : THEME.loss }]}>
                  {currentSignal.type.toUpperCase()}
                </Text>
                <Text style={styles.indicatorSub}>{currentSignal.reason}</Text>
              </View>
            )}
          </View>
        )}

        {mlSignal && (
          <View style={[styles.indicatorRow, { marginTop: 4 }]}>
            <View style={[styles.indicatorBox, {
              backgroundColor: mlSignal.direction === "BUY" ? THEME.profitDim : mlSignal.direction === "SELL" ? THEME.lossDim : THEME.surfaceLight,
              borderColor: mlSignal.direction === "BUY" ? THEME.profit : mlSignal.direction === "SELL" ? THEME.loss : THEME.border,
            }]}>
              <Text style={styles.indicatorLabel}>ML Signal</Text>
              <Text style={[styles.indicatorValue, {
                color: mlSignal.direction === "BUY" ? THEME.profit : mlSignal.direction === "SELL" ? THEME.loss : THEME.textMuted,
              }]}>
                {mlSignal.direction || "HOLD"} {(mlSignal.confidence * 100).toFixed(0)}%
              </Text>
              <Text style={styles.indicatorSub}>Predicted: {((mlSignal.predicted_return || 0) * 100).toFixed(2)}%</Text>
            </View>
            {mlForecast && mlForecast.length > 0 && (
              <View style={styles.indicatorBox}>
                <Text style={styles.indicatorLabel}>ML Forecast</Text>
                <Text style={[styles.indicatorValue, { color: mlForecast[mlForecast.length - 1]?.price > mlForecast[0]?.price ? THEME.profit : THEME.loss }]}>
                  ${mlForecast[mlForecast.length - 1]?.price?.toFixed(2) || "—"}
                </Text>
                <Text style={styles.indicatorSub}>21-step projection</Text>
              </View>
            )}
          </View>
        )}

        <Text style={styles.cl}>Price &amp; Signals (Full History)</Text>
        <View style={styles.chart}>
          <ResponsiveContainer width="100%" height={CHART_H}>
            <ComposedChart data={chartData} margin={{ top: 5, right: 5, bottom: 25, left: 0 }}>
              <XAxis
                dataKey="idx"
                tickFormatter={(i) => i % priceStep === 0 ? formatDate(chartData[i]?.date) : ""}
                tick={{ fill: THEME.textMuted, fontSize: 9 }}
                axisLine={{ stroke: THEME.border }}
                tickLine={false}
                interval={0}
              />
              <YAxis domain={['dataMin - 1', 'dataMax + 1']} tick={{ fill: THEME.textMuted, fontSize: 9 }} axisLine={{ stroke: THEME.border }} tickLine={false} tickFormatter={(v) => { try { return `$${Number(v).toFixed(0)}`; } catch { return ""; } }} width={55} />
              <Tooltip contentStyle={{ background: THEME.surface, border: `1px solid ${THEME.border}`, borderRadius: 4, fontSize: 11 }} labelFormatter={(i) => { try { return formatDate(chartData[i]?.date); } catch { return ""; } }} />
              <Bar dataKey="close" shape={<Candle />} isAnimationActive={false} maxBarSize={8} />
              <ReferenceLine x={currentBar} stroke={THEME.accent} strokeWidth={1.5} strokeDasharray="3 2" label={{ value: "NOW", fill: THEME.accent, fontSize: 8, position: "top" }} />
            </ComposedChart>
          </ResponsiveContainer>
        </View>

        <Text style={styles.cl}>Equity Curve</Text>
        <View style={styles.chart}>
          <ResponsiveContainer width="100%" height={120}>
            <ComposedChart data={eqData} margin={{ top: 2, right: 5, bottom: 25, left: 0 }}>
              <XAxis dataKey="idx" tickFormatter={(i) => i % priceStep === 0 ? formatDate(eqData[i]?.date) : ""} tick={{ fill: THEME.textMuted, fontSize: 9 }} axisLine={{ stroke: THEME.border }} tickLine={false} interval={0} />
              <YAxis hide domain={['dataMin - 0.001', 'dataMax + 0.001']} />
              <Tooltip contentStyle={{ background: THEME.surface, border: `1px solid ${THEME.border}`, borderRadius: 4, fontSize: 11 }} labelFormatter={(i) => formatDate(eqData[i]?.date)} />
              <defs>
                <linearGradient id="eqLG" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={eqIsUp ? THEME.profit : THEME.loss} stopOpacity={0.3} />
                  <stop offset="100%" stopColor={eqIsUp ? THEME.profit : THEME.loss} stopOpacity={0} />
                </linearGradient>
              </defs>
              <Area type="monotone" dataKey="equity" stroke={eqIsUp ? THEME.profit : THEME.loss} strokeWidth={2} fill="url(#eqLG)" isAnimationActive={false} />
              <ReferenceLine x={currentBar} stroke={THEME.accent} strokeWidth={1.5} strokeDasharray="3 2" />
            </ComposedChart>
          </ResponsiveContainer>
        </View>

        {signals && signals.length > 0 && (
          <View style={styles.signalsList}>
            <Text style={styles.cl}>All Signals ({signals.length})</Text>
            {signals.slice(0, 10).map((s, i) => (
              <View key={i} style={styles.signalRow}>
                <Text style={[styles.signalIcon, { color: s.type === "buy" ? THEME.profit : THEME.loss }]}>
                  {s.type === "buy" ? "▲" : "▼"}
                </Text>
                <Text style={styles.signalDate}>{String(s.date).slice(0, 10)}</Text>
                <Text style={styles.signalPrice}>${s.price}</Text>
                <Text style={styles.signalReason}>{s.reason}</Text>
              </View>
            ))}
            {signals.length > 10 && <Text style={styles.more}>+ {signals.length - 10} more</Text>}
          </View>
        )}
      </View>
    </ErrorBoundary>
  );
}

const CHART_H = 220;
const styles = StyleSheet.create({
  wrap: { marginBottom: 16, backgroundColor: THEME.surface, borderRadius: 10, padding: 12, borderLeftWidth: 3, borderLeftColor: THEME.accent },
  header: { flexDirection: "row", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 10 },
  title: { color: THEME.text, fontSize: 14, fontWeight: "800" },
  sub: { color: THEME.textMuted, fontSize: 10, marginTop: 2 },
  controls: { flexDirection: "row", gap: 3, alignItems: "center" },
  sb: { paddingHorizontal: 6, paddingVertical: 3, borderRadius: 3, borderWidth: 1, borderColor: THEME.border, backgroundColor: THEME.bg },
  sbOn: { backgroundColor: THEME.accentGlow, borderColor: THEME.accent },
  st: { color: THEME.textMuted, fontSize: 9, fontWeight: "600" },
  stOn: { color: THEME.accent },
  pb: { paddingHorizontal: 6, paddingVertical: 3, borderRadius: 3, backgroundColor: THEME.bg, borderWidth: 1, borderColor: THEME.border, marginLeft: 2 },
  pbOn: { backgroundColor: THEME.accentGlow, borderColor: THEME.accent },
  pt: { color: THEME.text, fontSize: 10 },
  indicatorRow: { flexDirection: "row", gap: 6, marginBottom: 8 },
  indicatorBox: { flex: 1, backgroundColor: THEME.surfaceLight, borderRadius: 4, paddingHorizontal: 8, paddingVertical: 4, borderWidth: 1, borderColor: THEME.border },
  indicatorLabel: { color: THEME.textMuted, fontSize: 9, fontWeight: "600", textTransform: "uppercase" },
  indicatorValue: { fontSize: 16, fontWeight: "800", fontVariant: ["tabular-nums"] },
  indicatorSub: { color: THEME.textMuted, fontSize: 8, marginTop: 1 },
  cl: { color: THEME.textMuted, fontSize: 10, fontWeight: "600", textTransform: "uppercase", marginTop: 8, marginBottom: 4 },
  chart: { backgroundColor: THEME.surfaceLight, borderRadius: 6, padding: 4 },
  signalsList: { marginTop: 8, padding: 8, backgroundColor: THEME.surfaceLight, borderRadius: 6 },
  signalRow: { flexDirection: "row", alignItems: "center", paddingVertical: 2, gap: 8 },
  signalIcon: { fontSize: 12, fontWeight: "800", minWidth: 12 },
  signalDate: { color: THEME.textMuted, fontSize: 10, minWidth: 70, fontVariant: ["tabular-nums"] },
  signalPrice: { color: THEME.text, fontSize: 10, minWidth: 50, fontVariant: ["tabular-nums"] },
  signalReason: { color: THEME.textMuted, fontSize: 9, flex: 1 },
  more: { color: THEME.textMuted, fontSize: 9, marginTop: 4, fontStyle: "italic" },
});
