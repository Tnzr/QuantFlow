import React, { useEffect, useRef, useState, useCallback, useMemo } from "react";
import { View, Text, Pressable, StyleSheet } from "react-native";
import {
  ComposedChart, Bar, Line, Area, XAxis, YAxis, Tooltip, ResponsiveContainer,
} from "recharts";
import THEME from "../../theme/colors";

const SPEEDS = { 1: 60, 2: 15, 4: 5, 16: 1 };
const CHART_H = 260;

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

export default function BacktestProgress({ equityCurve, trades, running, onComplete, ticker, token }) {
  const [currentBar, setCurrentBar] = useState(0);
  const [speed, setSpeed] = useState(1);
  const [paused, setPaused] = useState(false);
  const [priceData, setPriceData] = useState([]);
  const timerRef = useRef(null);
  const completedRef = useRef(false);

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
        const next = prev + 1;
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

  const WINDOW_SIZE = 80;
  const fullBars = useMemo(() => {
    try {
      if (!priceData || priceData.length === 0) return [];
      const eqMap = {};
      if (equityCurve) equityCurve.forEach(e => { eqMap[String(e.date).slice(0, 10)] = e.equity; });
      const tradeMap = {};
      if (trades) {
        trades.forEach(t => {
          const ed = String(t.entry_date || "").slice(0, 10);
          const xd = String(t.exit_date || "").slice(0, 10);
          if (ed) tradeMap[ed] = { ...t, type: "entry" };
          if (xd) tradeMap[xd] = { ...t, type: "exit" };
        });
      }
      return priceData.map((d, i) => {
        const o = Number(d.open || d.close || 0);
        const c = Number(d.close || 0);
        const h = Number(d.high || c);
        const l = Number(d.low || c);
        const dk = String(d.date).slice(0, 10);
        const t = tradeMap[dk];
        return { idx: i, date: dk, open: o, close: c, high: h, low: l, equity: eqMap[dk] || null, tradeEntry: t?.type === "entry" || null, tradeExit: t?.type === "exit" || null, _dMin: 0 };
      });
    } catch (e) {
      console.error("fullBars error:", e);
      return [];
    }
  }, [priceData, equityCurve, trades]);

  const visibleBars = useMemo(() => {
    try {
      if (!fullBars || fullBars.length <= WINDOW_SIZE) return fullBars || [];
      const start = Math.max(0, currentBar - WINDOW_SIZE + 10);
      return fullBars.slice(start, start + WINDOW_SIZE);
    } catch (e) {
      console.error("visibleBars error:", e);
      return [];
    }
  }, [fullBars, currentBar]);

  const chartData = useMemo(() => {
    try {
      if (!visibleBars || visibleBars.length === 0) return [];
      const sVals = visibleBars.flatMap(d => {
        const vals = [d.high, d.low, d.open, d.close].map(v => Number(v) || 0);
        return vals.filter(v => isFinite(v));
      });
      if (sVals.length === 0) return [];
      const vMin = Math.min(...sVals);
      const vMax = Math.max(...sVals);
      const vRng = (vMax - vMin) || 1;
      return visibleBars.map(d => ({ ...d, _dMin: vMin - vRng * 0.02 }));
    } catch (e) {
      console.error("chartData error:", e);
      return [];
    }
  }, [visibleBars]);

  const eqData = useMemo(() => {
    try {
      return (equityCurve || []).slice(0, currentBar + 1).map((d, i) => ({
        idx: i, date: d.date, equity: Number(d.equity) || 0
      }));
    } catch (e) {
      return [];
    }
  }, [equityCurve, currentBar]);

  const tradeCount = useMemo(() => {
    if (!trades || !equityCurve.length) return 0;
    const seenDates = new Set(equityCurve.slice(0, currentBar + 1).map(e => String(e.date).slice(0,10)));
    return trades.filter(t => seenDates.has(String(t.entry_date||'').slice(0,10)) || seenDates.has(String(t.exit_date||'').slice(0,10))).length;
  }, [trades, equityCurve, currentBar]);

  const visibleTrade = useMemo(() => {
    if (!trades || !equityCurve.length) return [];
    const seenDates = new Set(equityCurve.slice(0, currentBar + 1).map(e => String(e.date).slice(0,10)));
    return trades.filter(t => seenDates.has(String(t.entry_date||'').slice(0,10)) || seenDates.has(String(t.exit_date||'').slice(0,10))).slice(-3);
  }, [trades, equityCurve, currentBar]);

  const formatDate = (d) => d ? String(d).split("-").slice(1, 3).join("/") : "";
  const priceStep = Math.max(1, Math.floor((chartData?.length || 0) / 5));
  const eqIsUp = (eqData?.length || 0) >= 2 && eqData[eqData.length - 1].equity >= eqData[0].equity;

  if (!chartData.length) return null;

  return (
    <View style={styles.wrap}>
      <View style={styles.header}>
        <View>
          <Text style={styles.title}>Live Trading Playback</Text>
          <Text style={styles.sub}>
            {ticker || ""} · Daily · Bar {currentBar + 1}/{equityCurve.length} · {tradeCount} sig · {formatDate(currentDate)}
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

      <Text style={styles.cl}>Price & Signals</Text>
      <View style={styles.chart}>
        <ResponsiveContainer width="100%" height={CHART_H}>
          <ComposedChart data={chartData} margin={{ top: 5, right: 5, bottom: 20, left: 0 }}>
            <XAxis dataKey="idx" tickFormatter={(i) => {
              try { return i % priceStep === 0 ? formatDate(chartData[i]?.date) : ""; }
              catch { return ""; }
            }} tick={{ fill: THEME.textMuted, fontSize: 9 }} axisLine={{ stroke: THEME.border }} tickLine={false} />
            <YAxis domain={['dataMin - 1', 'dataMax + 1']} tick={{ fill: THEME.textMuted, fontSize: 9 }} axisLine={{ stroke: THEME.border }} tickLine={false} tickFormatter={(v) => { try { return `$${Number(v).toFixed(0)}`; } catch { return ""; } }} width={55} />
            <Tooltip contentStyle={{ background: THEME.surface, border: `1px solid ${THEME.border}`, borderRadius: 4, fontSize: 11 }} labelFormatter={(i) => { try { return formatDate(chartData[i]?.date); } catch { return ""; } }} />
            <Bar dataKey="close" shape={<Candle />} isAnimationActive={false} maxBarSize={10} />
          </ComposedChart>
        </ResponsiveContainer>
      </View>

      <Text style={styles.cl}>Equity Curve</Text>
      <View style={styles.chart}>
        <ResponsiveContainer width="100%" height={120}>
          <ComposedChart data={eqData} margin={{ top: 2, right: 5, bottom: 20, left: 0 }}>
            <XAxis dataKey="idx" tickFormatter={(i) => { try { return i % priceStep === 0 ? formatDate(eqData[i]?.date) : ""; } catch { return ""; } }} tick={{ fill: THEME.textMuted, fontSize: 9 }} axisLine={{ stroke: THEME.border }} tickLine={false} />
            <YAxis hide domain={['dataMin - 0.001', 'dataMax + 0.001']} />
            <Tooltip contentStyle={{ background: THEME.surface, border: `1px solid ${THEME.border}`, borderRadius: 4, fontSize: 11 }} />
            <defs>
              <linearGradient id="eqLG" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={eqIsUp ? THEME.profit : THEME.loss} stopOpacity={0.3} />
                <stop offset="100%" stopColor={eqIsUp ? THEME.profit : THEME.loss} stopOpacity={0} />
              </linearGradient>
            </defs>
            <Area type="monotone" dataKey="equity" stroke={eqIsUp ? THEME.profit : THEME.loss} strokeWidth={2} fill="url(#eqLG)" isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </View>

      {visibleTrade.length > 0 && (
        <View style={styles.trades}>
          <Text style={styles.cl}>Recent Trades</Text>
          {visibleTrade.map((t, i) => {
            const ret = (t.return_pct || 0);
            return (
              <View key={i} style={styles.tr}>
                <Text style={[styles.tt, { color: ret >= 0 ? THEME.profit : THEME.loss }]}>{ret >= 0 ? "+" : ""}{(ret * 100).toFixed(1)}%</Text>
                <Text style={styles.td}>{String(t.entry_date || "").slice(0, 10)}→{String(t.exit_date || "").slice(0, 10)}</Text>
              </View>
            );
          })}
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: { marginBottom: 16, backgroundColor: THEME.surface, borderRadius: 10, padding: 12, borderLeftWidth: 3, borderLeftColor: THEME.accent },
  header: { flexDirection: "row", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 10 },
  title: { color: THEME.text, fontSize: 14, fontWeight: "800" },
  sub: { color: THEME.textMuted, fontSize: 10, marginTop: 2 },
  controls: { flexDirection: "row", gap: 4, alignItems: "center" },
  sb: { paddingHorizontal: 7, paddingVertical: 4, borderRadius: 3, backgroundColor: THEME.surfaceLight, borderWidth: 1, borderColor: THEME.border },
  sbOn: { backgroundColor: THEME.accentGlow, borderColor: THEME.accent },
  st: { color: THEME.textMuted, fontSize: 10, fontWeight: "600" },
  stOn: { color: THEME.accent },
  pb: { paddingHorizontal: 8, paddingVertical: 4, borderRadius: 3, backgroundColor: THEME.surfaceLight, borderWidth: 1, borderColor: THEME.border },
  pbOn: { backgroundColor: THEME.accent, borderColor: THEME.accent },
  pt: { color: THEME.text, fontSize: 12 },
  cl: { color: THEME.textMuted, fontSize: 10, fontWeight: "600", textTransform: "uppercase", marginBottom: 4 },
  chart: { backgroundColor: THEME.bg, borderRadius: 6, padding: 4, marginBottom: 8 },
  trades: { marginTop: 4 },
  tr: { flexDirection: "row", alignItems: "center", paddingVertical: 2, gap: 8 },
  tt: { fontWeight: "800", fontSize: 12, minWidth: 50 },
  td: { color: THEME.textMuted, fontSize: 10 },
});
