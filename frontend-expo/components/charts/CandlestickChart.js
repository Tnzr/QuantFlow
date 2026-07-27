import React, { useMemo } from "react";
import { View, Text, StyleSheet } from "react-native";
import {
  ComposedChart,
  Bar,
  Line,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  Cell,
} from "recharts";
import THEME from "../../theme/colors";

const CHART_H = 280;
const MARGIN = { top: 5, right: 5, bottom: 20, left: 0 };

function Candlestick(props) {
  const { x, y, width, height, payload } = props;
  if (!payload || !payload.close || !height || height <= 0) return null;

  if (payload.barType === "forecast") {
    const cx = x + width / 2;
    const cy = y;
    return <circle cx={cx} cy={cy} r={3} fill={THEME.accent} />;
  }

  const close = Number(payload.close);
  const open = Number(payload.open);
  const high = Number(payload.high);
  const low = Number(payload.low);
  const dMin = Number(payload._dMin);
  const dMax = Number(payload._dMax);

  if (isNaN(close) || isNaN(open) || isNaN(dMin)) return null;

  const pixelPerUnit = height / (close - dMin || 1);
  const baseline = y + height;
  const pClose = y;
  const pOpen = baseline - (open - dMin) * pixelPerUnit;
  const pHigh = baseline - (high - dMin) * pixelPerUnit;
  const pLow = baseline - (low - dMin) * pixelPerUnit;

  const isUp = close >= open;
  const color = isUp ? THEME.profit : THEME.loss;
  const bodyWidth = Math.max(1, width * 0.65);
  const centerX = x + width / 2;
  const bodyTop = isUp ? pClose : pOpen;
  const bodyH = Math.max(1, Math.abs(pClose - pOpen));

  return (
    <g>
      <line x1={centerX} y1={pHigh} x2={centerX} y2={pLow} stroke={color} strokeWidth={1} />
      <rect x={centerX - bodyWidth / 2} y={bodyTop} width={bodyWidth} height={bodyH} fill={color} />
    </g>
  );
}

export default function CandlestickChart({ data, forecast, showForecast }) {
  const chartData = useMemo(() => {
    if (!data || data.length === 0) return { combined: [], dMin: 0, dMax: 1, actualBars: [] };

    const raw = data.map((d, i) => ({
      idx: i,
      date: d.date || d.time || String(i),
      open: Number(d.open) || 0,
      close: Number(d.close) || 0,
      high: Number(d.high) || 0,
      low: Number(d.low) || 0,
      volume: Number(d.volume) || 0,
      sma20: d.sma20 != null ? Number(d.sma20) : null,
      sma50: d.sma50 != null ? Number(d.sma50) : null,
      barType: "actual",
    }));

    let allPrices = raw.flatMap(d => [d.high, d.low, d.open, d.close]);

    let fData = [];
    if (showForecast && forecast && forecast.length > 0) {
      fData = forecast.map((f, i) => ({
        idx: raw.length + i,
        date: f.date || `F${i + 1}`,
        open: Number(f.price) || 0,
        close: Number(f.price) || 0,
        high: Number(f.upper || f.price) || 0,
        low: Number(f.lower || f.price) || 0,
        volume: null,
        sma20: null,
        sma50: null,
        barType: "forecast",
        pred: Number(f.price) || 0,
        upper: f.upper != null ? Number(f.upper) : null,
        lower: f.lower != null ? Number(f.lower) : null,
      }));
      allPrices = [...allPrices, ...fData.flatMap(d => [d.high, d.low, d.open, d.close, d.pred || 0, d.upper || 0, d.lower || 0])];
    }

    const combined = [...raw, ...fData];
    const validPrices = allPrices.filter(v => !isNaN(v) && v > 0);
    const dMin = Math.min(...validPrices) * 0.995;
    const dMax = Math.max(...validPrices) * 1.005;

    const result = combined.map(d => ({
      ...d,
      _dMin: dMin,
      _dMax: dMax,
    }));

    return { combined: result, dMin, dMax, actualBars: raw };
  }, [data, forecast, showForecast]);

  const { combined, dMin, dMax, actualBars } = chartData;

  if (!combined.length) {
    return (
      <View style={styles.empty}>
        <Text style={styles.emptyText}>No price data</Text>
      </View>
    );
  }

  const formatDate = (d) => {
    if (!d) return "";
    const parts = String(d).split("-");
    if (parts.length === 3) return `${parts[1]}/${parts[2]}`;
    return String(d).slice(0, 10);
  };

  const priceStep = Math.max(1, Math.floor(combined.length / 5));
  const volStep = Math.max(1, Math.floor(actualBars.length / 5));
  const hasForecast = combined.some(d => d.barType === "forecast");

  return (
    <View style={styles.wrap}>
      <Text style={styles.heading}>Candlestick</Text>
      <View style={styles.chart}>
        <ResponsiveContainer width="100%" height={CHART_H}>
          <ComposedChart data={combined} margin={MARGIN}>
            <XAxis
              dataKey="idx"
              tickFormatter={(i) => (i % priceStep === 0 ? formatDate(combined[i]?.date) : "")}
              tick={{ fill: THEME.textMuted, fontSize: 10 }}
              axisLine={{ stroke: THEME.border }}
              tickLine={false}
            />
            <YAxis
              domain={[dMin, dMax]}
              tick={{ fill: THEME.textMuted, fontSize: 10 }}
              axisLine={{ stroke: THEME.border }}
              tickLine={false}
              tickFormatter={(v) => `$${v.toFixed(0)}`}
              width={60}
            />
            <Tooltip
              contentStyle={{ backgroundColor: THEME.surface, border: `1px solid ${THEME.border}`, borderRadius: 4 }}
              labelStyle={{ color: THEME.textMuted, fontSize: 11 }}
              labelFormatter={(i) => formatDate(combined[i]?.date)}
              formatter={(val, name) => {
                if (name === "sma20") return [`$${Number(val).toFixed(2)}`, "SMA20"];
                if (name === "sma50") return [`$${Number(val).toFixed(2)}`, "SMA50"];
                if (name === "pred") return [`$${Number(val).toFixed(2)}`, "Forecast"];
                return [val, name];
              }}
            />

            <Bar dataKey="close" shape={<Candlestick />} isAnimationActive={false} maxBarSize={12} />

            <Line type="monotone" dataKey="sma20" stroke={THEME.info} dot={false} strokeWidth={1.5} connectNulls={false} />
            <Line type="monotone" dataKey="sma50" stroke={THEME.warn} dot={false} strokeWidth={1.5} connectNulls={false} />

            {hasForecast && (
              <>
                <ReferenceLine
                  x={combined.findIndex(d => d.barType === "forecast") - 1}
                  stroke={THEME.accent}
                  strokeWidth={1}
                  strokeDasharray="4 4"
                />
                <Line type="monotone" dataKey="pred" stroke={THEME.accent} strokeWidth={2.5} strokeDasharray="6 3" dot={true} dot={{ r: 2, fill: THEME.accent }} connectNulls />
                <Line type="monotone" dataKey="upper" stroke={THEME.accentLight} strokeWidth={1} dot={false} connectNulls strokeOpacity={0.5} />
                <Line type="monotone" dataKey="lower" stroke={THEME.accentLight} strokeWidth={1} dot={false} connectNulls strokeOpacity={0.5} />
              </>
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </View>

      <Text style={styles.heading}>Volume</Text>
      <View style={styles.volChart}>
        <ResponsiveContainer width="100%" height={80}>
          <ComposedChart data={actualBars} margin={{ top: 2, right: 5, bottom: 20, left: 0 }}>
            <XAxis
              dataKey="idx"
              tickFormatter={(i) => (i % volStep === 0 ? formatDate(actualBars[i]?.date) : "")}
              tick={{ fill: THEME.textMuted, fontSize: 10 }}
              axisLine={{ stroke: THEME.border }}
              tickLine={false}
            />
            <YAxis hide domain={[0, "auto"]} />
            <Tooltip
              contentStyle={{ backgroundColor: THEME.surface, border: `1px solid ${THEME.border}`, borderRadius: 4 }}
              labelFormatter={(i) => formatDate(actualBars[i]?.date)}
              formatter={(val) => [Number(val).toLocaleString()]}
            />
            <Bar dataKey="volume" maxBarSize={8} isAnimationActive={false}>
              {actualBars.map((entry, i) => (
                <Cell key={`vol-${i}`} fill={entry.close >= entry.open ? THEME.profit : THEME.loss} opacity={0.4} />
              ))}
            </Bar>
          </ComposedChart>
        </ResponsiveContainer>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: { marginBottom: 12 },
  heading: { color: THEME.text, fontSize: 13, fontWeight: "700", marginBottom: 4 },
  chart: { backgroundColor: THEME.surface, borderRadius: 8, padding: 4 },
  volChart: { backgroundColor: THEME.surface, borderRadius: 8, padding: 4, marginTop: 4 },
  empty: { backgroundColor: THEME.surface, borderRadius: 8, padding: 32, alignItems: "center" },
  emptyText: { color: THEME.textMuted, fontSize: 13 },
});
