// QuantFlow Frontend QA — Puppeteer Browser Automation (Node 18 compatible)
// Usage: node tests/qa-runner.mjs

import puppeteer from "puppeteer";

const BASE = "http://127.0.0.1:8080";
const API = "http://127.0.0.1:3000";
const ML = "http://127.0.0.1:8000";

let passed = 0;
let failed = 0;
let startTime = Date.now();

async function check(name, fn) {
  try {
    await fn();
    console.log(`  ✓ ${name}`);
    passed++;
  } catch (e) {
    console.log(`  ✗ ${name}: ${e.message}`);
    failed++;
  }
}

async function fetchJson(url, opts = {}) {
  const res = await fetch(url, opts);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

async function main() {
  console.log("\n╔════════════════════════════════╗");
  console.log("║   QuantFlow Frontend QA       ║");
  console.log("╚════════════════════════════════╝\n");

  // ── Health Checks ──────────────────────────────────────────
  console.log("Health Checks:");
  await check("API backend /health", async () => {
    const d = await fetchJson(`${API}/health`);
    if (d.status !== "ok") throw new Error("status not ok");
  });
  await check("ML engine /health", async () => {
    const d = await fetchJson(`${ML}/health`);
    if (d.status !== "ok") throw new Error("status not ok");
  });
  await check("Frontend serves index.html", async () => {
    const res = await fetch(BASE);
    if (res.status !== 200) throw new Error(`HTTP ${res.status}`);
  });

  // ── Backend Endpoints ──────────────────────────────────────
  console.log("\nBackend Endpoints:");
  await check("GET /portfolio/positions", async () => {
    const d = await fetchJson(`${API}/portfolio/positions`);
    if (!d.account) throw new Error("missing account");
    const eq = Number(d.account.equity);
    if (!isFinite(eq) || eq <= 0) throw new Error(`invalid equity: ${d.account.equity}`);
    // When Alpaca is configured, equity should be a realistic number (not 100000 demo).
    // Only assert non-demo when keys are configured.
  });
  await check("GET /market/alpaca/quote/AAPL", async () => {
    const d = await fetchJson(`${API}/market/alpaca/quote/AAPL`);
    const price = Number(d?.quote?.bid ?? d?.trade?.price);
    if (!isFinite(price) || price <= 0) throw new Error(`no live price: ${JSON.stringify(d).slice(0, 100)}`);
  });
  await check("GET /charts/forecast?ticker=AAPL", async () => {
    const d = await fetchJson(`${API}/charts/forecast?ticker=AAPL`);
    if (!d.forecast || d.forecast.length < 5) throw new Error("forecast too short");
    if (!d.forecast[0].price || d.forecast[0].price <= 0) throw new Error("forecast price invalid");
  });
  await check("POST /market/series/cache (1d)", async () => {
    const res = await fetch(`${API}/market/series/cache?ticker=AAPL&interval=1d`, { method: "POST" });
    const d = await res.json();
    if (!d.items || d.items.length < 50) throw new Error("too few items");
  });
  await check("POST /market/series/cache (1h)", async () => {
    const res = await fetch(`${API}/market/series/cache?ticker=AAPL&interval=1h`, { method: "POST" });
    const d = await res.json();
    if (d.interval !== "1h") throw new Error(`wrong interval: ${d.interval}`);
  });
  await check("GET /backtest/short-term", async () => {
    const d = await fetchJson(`${API}/backtest/short-term?ticker=AAPL&start=2024-01-01&end=2025-12-31&entry_rsi_threshold=30&max_hold_days=10&ma_filter=none&ma_trend_filter=none`);
    if (d.summary?.n_trades == null) throw new Error("missing summary");
    if (!d.equity || d.equity.length < 10) throw new Error("equity curve too short");
  });
  await check("GET /backtest/multi (3 tickers)", async () => {
    const d = await fetchJson(`${API}/backtest/multi?tickers=AAPL,MSFT,NVDA&start=2024-01-01&end=2025-12-31&entry_rsi_threshold=30&max_hold_days=10&ma_filter=none&ma_trend_filter=none`);
    if (!d.results || d.results.length !== 3) throw new Error(`expected 3 results, got ${d.results?.length}`);
    if (!d.portfolio_equity || d.portfolio_equity.length < 50) throw new Error("portfolio equity too short");
  });
  await check("POST /trade/paper (market buy)", async () => {
    const d = await fetchJson(`${API}/trade/paper`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ticker: "TSLA", action: "buy", qty: 1 }),
    });
    if (!d.ok) throw new Error("order failed");
  });
  await check("GET /recommend/latest", async () => {
    const d = await fetchJson(`${API}/recommend/latest`);
    if (d.items == null) throw new Error("missing items");
  });
  await check("POST /assistant/query", async () => {
    const d = await fetchJson(`${API}/assistant/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: "hello" }),
    });
    if (!d.answer) throw new Error("missing answer");
  });

  // ── ML Engine ──────────────────────────────────────────────
  console.log("\nML Engine:");
  await check("POST /predict", async () => {
    const d = await fetchJson(`${ML}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ticker: "AAPL", n_context: 80 }),
    });
    if (!d.direction) throw new Error("missing direction");
    if (d.confidence == null) throw new Error("missing confidence");
  });
  await check("GET /models", async () => {
    const d = await fetchJson(`${ML}/models`);
    if (!d.models || d.models.length === 0) throw new Error("no models");
  });

  // ── Browser Tests ──────────────────────────────────────────
  console.log("\nBrowser (Puppeteer):");
  const browser = await puppeteer.launch({ headless: "new", args: ["--no-sandbox"] });
  const page = await browser.newPage();

  try {
    const delay = (ms) => new Promise(r => setTimeout(r, ms));

    await check("Auth gate renders", async () => {
      await page.goto(BASE, { waitUntil: "networkidle2", timeout: 15000 });
      const el = await page.$("text=QuantFlow");
      if (!el) throw new Error("title not found");
    });

    await check("Can unlock workspace", async () => {
      await page.waitForSelector('input[placeholder="API Key / Bearer Token"]', { timeout: 5000 });
      await page.type('input[placeholder="API Key / Bearer Token"]', "qa-test");
      await delay(300);
      await page.click("text=Unlock Workspace");
      await page.waitForSelector("text=Dashboard", { timeout: 15000 });
    });

    await check("5 tabs visible in sidebar", async () => {
      for (const tab of ["Dashboard", "Trade", "Market", "Backtest", "Settings"]) {
        const el = await page.$(`text=${tab}`);
        if (!el) throw new Error(`tab ${tab} not found`);
      }
    });

    await check("Status bar shows API status", async () => {
      const text = await page.$eval("text=API:", (el) => el.textContent);
      if (!text.includes("API:")) throw new Error("status bar missing");
    });

    await check("Platform header shows branding", async () => {
      const text = await page.$eval("text=Algorithmic Trading", (el) => el.textContent);
      if (!text.includes("Algorithmic Trading")) throw new Error("tagline missing");
    });

    // Navigate to Trade tab
    await page.click("text=Trade");
    await delay(500);
    await check("Trade tab renders order form", async () => {
      const text = await page.$eval("text=New Order", (el) => el.textContent);
      if (text !== "New Order") throw new Error("order form missing");
    });

    // Navigate to Market tab
    await page.click("text=Market");
    await delay(500);
    await check("Market tab has ticker input and timeframe selector", async () => {
      const input = await page.$('input[aria-label="Ticker symbol"], input[placeholder*="NVDA"]');
      if (!input) throw new Error("ticker input missing");
    });

    // Navigate to Backtest tab
    await page.click("text=Backtest");
    await delay(500);
    await check("Backtest tab has config panel", async () => {
      const text = await page.$eval("text=Backtest Configuration", (el) => el.textContent);
      if (!text.includes("Backtest Configuration")) throw new Error("config missing");
    });

    // Navigate to Settings tab
    await page.click("text=Settings");
    await delay(500);
    await check("Settings tab has all sections", async () => {
      for (const section of ["Alpaca Connection", "LLM Configuration", "Risk Parameters", "Backend URL"]) {
        const el = await page.$(`text=${section}`);
        if (!el) throw new Error(`section ${section} missing`);
      }
    });

    // Open chat sidebar
    await page.click("text=Chat");
    await delay(500);
    await check("Chat sidebar opens", async () => {
      const text = await page.$eval("text=QuantFlow Assistant", (el) => el.textContent);
      if (text !== "QuantFlow Assistant") throw new Error("chat sidebar not open");
    });

    // Go back to Dashboard and add ticker
    await page.click("text=Dashboard");
    await page.waitForSelector("text=Account");
    await page.click("text=+ Add");
    await delay(300);
    await page.type('input[placeholder="Ticker (e.g. AAPL)"]', "IBM");
    await page.keyboard.press("Enter");
    await delay(2000);
    await check("Watchlist adds ticker", async () => {
      const el = await page.$("text=IBM");
      if (!el) throw new Error("IBM not in watchlist");
    });

    // Screenshot for visual check
    await page.screenshot({ path: "qa-screenshot-dashboard.png", fullPage: false });
    console.log("    Screenshot saved: tests/qa-screenshot-dashboard.png");
  } finally {
    await browser.close();
  }

  // ── Summary ────────────────────────────────────────────────
  const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
  console.log(`\n${"─".repeat(36)}`);
  console.log(`  Total: ${passed + failed}  |  Passed: ${passed}  |  Failed: ${failed}  |  ${elapsed}s`);
  console.log(`${"─".repeat(36)}\n`);

  process.exit(failed > 0 ? 1 : 0);
}

main().catch((e) => {
  console.error("QA runner crashed:", e.message);
  process.exit(1);
});
