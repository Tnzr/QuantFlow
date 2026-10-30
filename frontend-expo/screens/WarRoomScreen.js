import React, { useCallback, useState } from "react";
import { ScrollView, StyleSheet, View, Text, TextInput, Pressable } from "react-native";
import THEME from "../theme/colors";
import { apiGet, apiPost } from "../services/api";
import ScreenTitle from "../components/shared/ScreenTitle";
import MetricCard from "../components/shared/MetricCard";
import LoadingSpinner from "../components/shared/LoadingSpinner";
import ErrorBanner from "../components/shared/ErrorBanner";

const pct = (v) => (typeof v === "number" ? `${(v * 100).toFixed(1)}%` : "-");
const num = (v, d = 2) => (typeof v === "number" ? v.toFixed(d) : "-");

function AgentRow({ label, score, detail }) {
  const width = `${Math.round(Math.max(0, Math.min(1, score || 0)) * 100)}%`;
  return (
    <View style={styles.agentRow}>
      <Text style={styles.agentLabel}>{label}</Text>
      <View style={styles.barTrack}>
        <View style={[styles.barFill, { width }]} />
      </View>
      <Text style={styles.agentScore}>{num(score)}</Text>
      {detail ? <Text style={styles.agentDetail}>{detail}</Text> : null}
    </View>
  );
}

export default function WarRoomScreen({ token }) {
  const [ticker, setTicker] = useState("SPY");
  const [agents, setAgents] = useState(null);
  const [scenario, setScenario] = useState(null);
  const [resilience, setResilience] = useState(null);
  const [odds, setOdds] = useState(null);
  const [proposal, setProposal] = useState(null);
  const [busy, setBusy] = useState(null);
  const [error, setError] = useState(null);

  const guard = useCallback(async (name, fn) => {
    setBusy(name);
    setError(null);
    try {
      await fn();
    } catch (e) {
      setError(e?.message || String(e));
    } finally {
      setBusy(null);
    }
  }, []);

  const analyze = () => guard("agents", async () => setAgents(await apiGet(`/agents/analyze?ticker=${encodeURIComponent(ticker)}`, token)));
  const loadScenario = () => guard("scenario", async () => setScenario(await apiGet("/election/scenario?horizon=3m", token)));
  const runResilience = () =>
    guard("resilience", async () => setResilience(await apiGet(`/backtest/resilience?ticker=${encodeURIComponent(ticker)}`, token)));
  const ingest = () =>
    guard("ingest", async () => {
      const r = await apiPost("/election/ingest", {}, token);
      setOdds(r.odds || {});
    });
  const propose = (bias) =>
    guard("proposal", async () => setProposal(await apiPost("/proposals", { ticker, bias, equity: 100000 }, token)));

  const gate = agents?.gate;

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <ScreenTitle title="Agent War Room" subtitle="Election-aware agents, gating and resilience" icon="W" />
      <ErrorBanner message={error} />

      <View style={styles.row}>
        <TextInput
          style={styles.input}
          value={ticker}
          onChangeText={(t) => setTicker(t.toUpperCase())}
          autoCapitalize="characters"
          placeholder="Ticker"
          placeholderTextColor={THEME.textMuted}
        />
        <Pressable style={styles.btn} onPress={analyze}><Text style={styles.btnText}>Run agents</Text></Pressable>
        <Pressable style={styles.btn} onPress={runResilience}><Text style={styles.btnText}>Resilience</Text></Pressable>
      </View>
      <View style={styles.row}>
        <Pressable style={styles.btnAlt} onPress={ingest}><Text style={styles.btnText}>Ingest news + odds</Text></Pressable>
        <Pressable style={styles.btnAlt} onPress={loadScenario}><Text style={styles.btnText}>Senate scenario</Text></Pressable>
      </View>
      {busy ? (
        <View style={styles.row}>
          <LoadingSpinner size={24} />
          <Text style={styles.muted}>Running {busy}...</Text>
        </View>
      ) : null}

      {odds ? (
        <View style={styles.card}>
          <Text style={styles.cardTitle}>Polymarket: Democratic control</Text>
          <View style={styles.metrics}>
            <MetricCard label="Senate" value={odds.senate_control != null ? pct(odds.senate_control) : "-"} />
            <MetricCard label="House" value={odds.house_control != null ? pct(odds.house_control) : "-"} />
          </View>
        </View>
      ) : null}

      {agents ? (
        <View style={styles.card}>
          <Text style={styles.cardTitle}>{agents.ticker} as of {agents.as_of}: {String(agents.bias).toUpperCase()} ({num(agents.final_confidence)})</Text>
          <AgentRow label="Technical" score={agents.technical?.score} />
          <AgentRow label="Seasonal" score={agents.seasonal?.score} detail={`${agents.seasonal?.phase} year, n=${agents.seasonal?.n_years}`} />
          <AgentRow label="Sentiment" score={agents.sentiment?.score} detail={`source ${agents.sentiment?.source}`} />
          <AgentRow label="Rule baseline" score={gate?.base_confidence} />
          <View style={[styles.gate, gate?.override_applied ? styles.gateOn : styles.gateOff]}>
            <Text style={styles.gateText}>{gate?.reason}</Text>
            <Text style={styles.gateSub}>
              vol ratio {num(agents.volatility?.vol_ratio)} | election window {agents.volatility?.election_window ? "yes" : "no"} | gap {num(gate?.confidence_gap)}
            </Text>
          </View>
          <View style={styles.row}>
            <Pressable style={styles.btn} onPress={() => propose("long")}><Text style={styles.btnText}>Propose bull vertical</Text></Pressable>
            <Pressable style={styles.btn} onPress={() => propose("short")}><Text style={styles.btnText}>Propose bear vertical</Text></Pressable>
          </View>
        </View>
      ) : null}

      {proposal?.proposal ? (
        <View style={styles.card}>
          <Text style={styles.cardTitle}>Signed proposal {proposal.proposal.structure}</Text>
          {proposal.proposal.legs.map((l, i) => (
            <Text key={i} style={styles.mono}>{l.action} {l.qty}x {l.strike}{l.right} {l.expiry}</Text>
          ))}
          <Text style={styles.mono}>max loss ${num(proposal.proposal.max_loss, 0)} | risk cap {pct(proposal.proposal.max_risk_pct)} | slippage cap {pct(proposal.proposal.slippage_cap_pct)}</Text>
          <Text style={styles.muted} numberOfLines={1}>sig {proposal.proposal.signature}</Text>
        </View>
      ) : null}

      {resilience ? (
        <View style={styles.card}>
          <Text style={styles.cardTitle}>Resilience {resilience.ticker} {resilience.period}</Text>
          <View style={styles.metrics}>
            <MetricCard label="Sharpe (agents)" value={Number(num(resilience.agentic?.sharpe))} />
            <MetricCard label="Sharpe (rules)" value={Number(num(resilience.baseline_rules_only?.sharpe))} />
            <MetricCard label="Max DD" value={Number(((resilience.agentic?.max_drawdown || 0) * 100).toFixed(1))} suffix="%" />
            <MetricCard label="OOS Sharpe" value={Number(num(resilience.walk_forward?.oos_mean_sharpe))} />
          </View>
          <Text style={styles.muted}>Walk-forward folds: {resilience.walk_forward?.folds?.length} | OOS positive {pct(resilience.walk_forward?.oos_positive_share)} | bootstrap p95 DD {pct(resilience.bootstrap_drawdown?.p95)}</Text>
          <Text style={styles.muted}>Election windows: {resilience.election_windows?.trades} trades, win {pct(resilience.election_windows?.win_rate)}</Text>
        </View>
      ) : null}

      {scenario?.baskets ? (
        <View style={styles.card}>
          <Text style={styles.cardTitle}>If the president's party loses the Senate ({scenario.horizon} after election, vs SPY)</Text>
          {scenario.baskets.map((b) => (
            <View key={b.basket} style={styles.basket}>
              <Text style={[styles.basketName, { color: b.side === "long" ? THEME.profit : THEME.loss }]}>
                {b.side.toUpperCase()} {b.basket} {b.consistent ? "(history agrees)" : "(history disagrees)"}
              </Text>
              <Text style={styles.muted}>{b.tickers.join(", ")} | analog {pct(b.analog_excess)} (n={b.analog_n}) | divided gov {pct(b.divided_excess)} (n={b.divided_n})</Text>
            </View>
          ))}
          <Text style={styles.muted}>Educational research, not financial advice. Small historical samples.</Text>
        </View>
      ) : null}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: THEME.bg },
  content: { padding: 16, paddingBottom: 48 },
  row: { flexDirection: "row", flexWrap: "wrap", gap: 8, marginVertical: 6, alignItems: "center" },
  input: { backgroundColor: THEME.surface, color: THEME.textBright, borderColor: THEME.border, borderWidth: 1, borderRadius: 8, paddingHorizontal: 12, paddingVertical: 8, minWidth: 90 },
  btn: { backgroundColor: THEME.accent, paddingHorizontal: 12, paddingVertical: 9, borderRadius: 8 },
  btnAlt: { backgroundColor: THEME.surfaceLight, borderColor: THEME.border, borderWidth: 1, paddingHorizontal: 12, paddingVertical: 9, borderRadius: 8 },
  btnText: { color: THEME.textBright, fontWeight: "600" },
  card: { backgroundColor: THEME.surface, borderRadius: 12, borderColor: THEME.border, borderWidth: 1, padding: 14, marginTop: 12 },
  cardTitle: { color: THEME.textBright, fontWeight: "700", marginBottom: 10 },
  metrics: { flexDirection: "row", flexWrap: "wrap", gap: 8 },
  agentRow: { marginBottom: 8 },
  agentLabel: { color: THEME.text, marginBottom: 3 },
  barTrack: { height: 8, backgroundColor: THEME.surfaceLight, borderRadius: 4, overflow: "hidden" },
  barFill: { height: 8, backgroundColor: THEME.accentLight },
  agentScore: { color: THEME.textMuted, fontSize: 12, marginTop: 2 },
  agentDetail: { color: THEME.textMuted, fontSize: 11 },
  gate: { borderRadius: 8, padding: 10, marginVertical: 8 },
  gateOn: { backgroundColor: THEME.warnDim },
  gateOff: { backgroundColor: THEME.infoDim },
  gateText: { color: THEME.textBright, fontWeight: "600" },
  gateSub: { color: THEME.textMuted, fontSize: 12, marginTop: 4 },
  mono: { color: THEME.text, fontFamily: "monospace", fontSize: 12 },
  muted: { color: THEME.textMuted, fontSize: 12, marginTop: 4 },
  basket: { marginBottom: 8 },
  basketName: { fontWeight: "700" },
});
