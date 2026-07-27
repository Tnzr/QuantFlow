import React, { useState } from "react";
import { View, Text, TextInput, Pressable, StyleSheet } from "react-native";
import { apiPost } from "../../services/api";
import THEME from "../../theme/colors";

export default function OrderTicket({ token, portfolio, onOrderPlaced }) {
  const [ticker, setTicker] = useState("");
  const [action, setAction] = useState("buy");
  const [qty, setQty] = useState("1");
  const [orderType, setOrderType] = useState("market");
  const [limitPrice, setLimitPrice] = useState("");
  const [stopLoss, setStopLoss] = useState("");
  const [takeProfit, setTakeProfit] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [status, setStatus] = useState(null);
  const [riskPctInput, setRiskPctInput] = useState("1");
  const [stopDollar, setStopDollar] = useState("");

  const equity = portfolio?.account?.equity || portfolio?.account?.portfolio_value || 100000;
  const estPrice = limitPrice ? parseFloat(limitPrice) : 100;
  const calcSize = (() => {
    const riskP = (parseFloat(riskPctInput) || 0) / 100;
    const stopDist = parseFloat(stopDollar) || 0;
    const riskAmount = equity * riskP;
    const shares = stopDist > 0 ? Math.floor(riskAmount / stopDist) : 0;
    return { shares, cost: shares * estPrice, riskAmount };
  })();

  const price = limitPrice ? parseFloat(limitPrice) : 100;
  const notional = (parseFloat(qty) || 0) * price;
  const riskPct = equity > 0 ? (notional / equity) * 100 : 0;
  const riskColor = riskPct > 20 ? THEME.loss : riskPct > 10 ? THEME.warn : THEME.profit;

  const handleSubmit = async () => {
    if (!ticker.trim() || !parseFloat(qty)) {
      setStatus({ type: "error", text: "Fill ticker and quantity" });
      return;
    }
    setSubmitting(true);
    setStatus(null);
    try {
      const body = {
        ticker: ticker.trim().toUpperCase(),
        side: action,
        qty: parseFloat(qty),
        order_type: orderType,
      };
      if (orderType === "limit" && limitPrice) body.limit_price = parseFloat(limitPrice);
      if (stopLoss) body.stop_loss_pct = parseFloat(stopLoss) / 100;
      if (takeProfit) body.take_profit_pct = parseFloat(takeProfit) / 100;

      const res = await apiPost("/trade/paper", body, token);
      const order = res.order || res;
      setStatus({ type: "ok", text: `Order ${order.id || "placed"} — ${order.status || "submitted"}` });
      if (onOrderPlaced) onOrderPlaced();
    } catch (err) {
      setStatus({ type: "error", text: err.message });
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <View style={styles.wrap}>
      <Text style={styles.heading}>New Order</Text>

      <Text style={styles.label}>Ticker</Text>
      <TextInput
        style={styles.input}
        placeholder="e.g. AAPL"
        placeholderTextColor={THEME.neutral}
        value={ticker}
        onChangeText={(t) => setTicker(t.toUpperCase())}
        autoCapitalize="characters"
        accessibilityLabel="Ticker"
      />

      <Text style={styles.label}>Side</Text>
      <View style={styles.segRow}>
        <Pressable
          style={[styles.segBtn, action === "buy" && styles.segBuy]}
          onPress={() => setAction("buy")}
        >
          <Text style={[styles.segText, action === "buy" && styles.segTextActive]}>Buy</Text>
        </Pressable>
        <Pressable
          style={[styles.segBtn, action === "sell" && styles.segSell]}
          onPress={() => setAction("sell")}
        >
          <Text style={[styles.segText, action === "sell" && styles.segTextActive]}>Sell</Text>
        </Pressable>
      </View>

      <Text style={styles.label}>Quantity</Text>
      <View style={styles.row}>
        <TextInput
          style={[styles.input, styles.half]}
          placeholder="e.g. 10"
          placeholderTextColor={THEME.neutral}
          value={qty}
          onChangeText={setQty}
          keyboardType="decimal-pad"
          accessibilityLabel="Quantity"
        />
        <View style={styles.stepper}>
          <Pressable style={styles.stepBtn} onPress={() => setQty(String(Math.max(0, (parseFloat(qty) || 0) - 1)))}>
            <Text style={styles.stepText}>-</Text>
          </Pressable>
          <Pressable style={styles.stepBtn} onPress={() => setQty(String((parseFloat(qty) || 0) + 1))}>
            <Text style={styles.stepText}>+</Text>
          </Pressable>
        </View>
      </View>

      <Text style={styles.label}>Order Type</Text>
      <View style={styles.segRow}>
        <Pressable
          style={[styles.segBtn, orderType === "market" && styles.segActive]}
          onPress={() => setOrderType("market")}
        >
          <Text style={[styles.segText, orderType === "market" && styles.segTextActive]}>Market</Text>
        </Pressable>
        <Pressable
          style={[styles.segBtn, orderType === "limit" && styles.segActive]}
          onPress={() => setOrderType("limit")}
        >
          <Text style={[styles.segText, orderType === "limit" && styles.segTextActive]}>Limit</Text>
        </Pressable>
      </View>

      {orderType === "limit" && (
        <>
          <Text style={styles.label}>Limit Price</Text>
          <TextInput
            style={styles.input}
            placeholder="e.g. 195.50"
            placeholderTextColor={THEME.neutral}
            value={limitPrice}
            onChangeText={setLimitPrice}
            keyboardType="decimal-pad"
            accessibilityLabel="Limit price"
          />
        </>
      )}

      <View style={styles.row}>
        <View style={styles.half}>
          <Text style={styles.label}>Stop Loss %</Text>
          <TextInput
            style={styles.input}
            placeholder="e.g. 5"
            placeholderTextColor={THEME.neutral}
            value={stopLoss}
            onChangeText={setStopLoss}
            keyboardType="decimal-pad"
            accessibilityLabel="Stop loss percent"
          />
        </View>
        <View style={styles.half}>
          <Text style={styles.label}>Take Profit %</Text>
          <TextInput
            style={styles.input}
            placeholder="e.g. 10"
            placeholderTextColor={THEME.neutral}
            value={takeProfit}
            onChangeText={setTakeProfit}
            keyboardType="decimal-pad"
            accessibilityLabel="Take profit percent"
          />
        </View>
      </View>

      <View style={[styles.riskBar, { borderLeftColor: riskColor }]}>
        <Text style={styles.riskText}>
          Position: ${notional.toFixed(2)} ({riskPct.toFixed(1)}% of portfolio)
        </Text>
      </View>

      <View style={styles.sizerWrap}>
        <Text style={styles.sizerTitle}>Position Size Calculator</Text>
        <View style={styles.sizerRow}>
          <View style={styles.sizerCol}>
            <Text style={styles.label}>Risk %</Text>
            <TextInput
              style={[styles.input, styles.sizerInput]}
              placeholder="e.g. 1"
              placeholderTextColor={THEME.neutral}
              value={riskPctInput}
              onChangeText={setRiskPctInput}
              keyboardType="decimal-pad"
              accessibilityLabel="Risk percent"
            />
          </View>
          <View style={styles.sizerCol}>
            <Text style={styles.label}>Stop $</Text>
            <TextInput
              style={[styles.input, styles.sizerInput]}
              placeholder="e.g. 2.00"
              placeholderTextColor={THEME.neutral}
              value={stopDollar}
              onChangeText={setStopDollar}
              keyboardType="decimal-pad"
              accessibilityLabel="Stop dollar amount"
            />
          </View>
        </View>
        <Text style={styles.sizerResult}>
          Recommended: {calcSize.shares} shares (${calcSize.cost.toFixed(2)})
        </Text>
      </View>

      <Pressable
        style={[styles.submitBtn, submitting && styles.btnDisabled]}
        onPress={handleSubmit}
        disabled={submitting}
      >
        <Text style={styles.submitText}>
          {submitting ? "Placing..." : `Place ${action.toUpperCase()} Order`}
        </Text>
      </Pressable>

      {status && (
        <Text style={[styles.statusText, { color: status.type === "ok" ? THEME.profit : THEME.loss }]}>
          {status.text}
        </Text>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: { marginBottom: 16 },
  heading: { color: THEME.text, fontSize: 14, fontWeight: "700", marginBottom: 8 },
  input: {
    backgroundColor: THEME.surface,
    borderWidth: 1,
    borderColor: THEME.border,
    borderRadius: 6,
    paddingHorizontal: 12,
    paddingVertical: 10,
    color: THEME.text,
    fontSize: 14,
    marginBottom: 8,
  },
  half: { flex: 1 },
  row: { flexDirection: "row", gap: 8 },
  segRow: { flexDirection: "row", marginBottom: 8, gap: 4 },
  segBtn: {
    flex: 1,
    paddingVertical: 9,
    borderRadius: 6,
    backgroundColor: THEME.surface,
    alignItems: "center",
    borderWidth: 1,
    borderColor: THEME.border,
  },
  segBuy: { backgroundColor: THEME.profitDim, borderColor: THEME.profit },
  segSell: { backgroundColor: THEME.lossDim, borderColor: THEME.loss },
  segActive: { backgroundColor: THEME.accentGlow, borderColor: THEME.accent },
  segText: { color: THEME.textMuted, fontWeight: "600", fontSize: 13 },
  segTextActive: { color: THEME.textBright },
  stepper: { flexDirection: "row", gap: 4, alignItems: "center" },
  stepBtn: {
    width: 36,
    height: 36,
    borderRadius: 6,
    backgroundColor: THEME.surface,
    borderWidth: 1,
    borderColor: THEME.border,
    justifyContent: "center",
    alignItems: "center",
  },
  stepText: { color: THEME.text, fontWeight: "700", fontSize: 18 },
  riskBar: {
    backgroundColor: THEME.surface,
    borderLeftWidth: 3,
    paddingHorizontal: 10,
    paddingVertical: 8,
    borderRadius: 4,
    marginBottom: 10,
  },
  riskText: { color: THEME.textMuted, fontSize: 12 },
  submitBtn: {
    backgroundColor: THEME.accent,
    borderRadius: 8,
    paddingVertical: 13,
    alignItems: "center",
  },
  btnDisabled: { opacity: 0.6 },
  submitText: { color: THEME.textBright, fontWeight: "700", fontSize: 15 },
  statusText: { marginTop: 8, fontSize: 13, textAlign: "center" },
  sizerWrap: { marginBottom: 10, padding: 10, backgroundColor: THEME.surface, borderRadius: 6, borderWidth: 1, borderColor: THEME.border },
  sizerTitle: { color: THEME.textMuted, fontSize: 10, fontWeight: "600", textTransform: "uppercase", marginBottom: 6 },
  sizerRow: { flexDirection: "row", gap: 6, marginBottom: 6 },
  sizerCol: { flex: 1 },
  sizerInput: { marginBottom: 0 },
  sizerResult: { color: THEME.text, fontSize: 12, fontWeight: "600" },
  label: { color: THEME.textMuted, fontSize: 11, fontWeight: "600", marginBottom: 4 },
});
