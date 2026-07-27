// QuantFlow Frontend QA — Playwright Browser Automation
// Usage: npx playwright test --config=playwright.config.js
// Install: npm install -D @playwright/test && npx playwright install chromium

const { test, expect } = require("@playwright/test");

const BASE = "http://127.0.0.1:8080";
const API = "http://127.0.0.1:3000";
const ML = "http://127.0.0.1:8000";

test.describe("Health Checks", () => {
  test("API backend responds", async ({ request }) => {
    const r = await request.get(`${API}/health`);
    expect(r.status()).toBe(200);
    const body = await r.json();
    expect(body.status).toBe("ok");
  });

  test("ML engine responds", async ({ request }) => {
    const r = await request.get(`${ML}/health`);
    expect(r.status()).toBe(200);
    const body = await r.json();
    expect(body.status).toBe("ok");
  });

  test("Frontend serves index.html", async ({ page }) => {
    const r = await page.goto(BASE);
    expect(r.status()).toBe(200);
  });
});

test.describe("Auth Gate", () => {
  test("shows unlock screen on first visit", async ({ page }) => {
    await page.goto(BASE);
    await expect(page.locator("text=QuantFlow")).toBeVisible({ timeout: 10000 });
    await expect(page.locator("text=Unlock Workspace")).toBeVisible();
  });

  test("unlock with any key when auth disabled", async ({ page }) => {
    await page.goto(BASE);
    const input = page.locator('input[placeholder="API Key / Bearer Token"]');
    await input.fill("dev-test");
    await page.locator("text=Unlock Workspace").click();
    await expect(page.locator("text=Dashboard")).toBeVisible({ timeout: 10000 });
    await expect(page.locator("text=Trade")).toBeVisible();
    await expect(page.locator("text=Market")).toBeVisible();
    await expect(page.locator("text=Backtest")).toBeVisible();
    await expect(page.locator("text=Settings")).toBeVisible();
  });
});

test.describe("Navigation", () => {
  test("all tabs navigate and render", async ({ page }) => {
    await page.goto(BASE);
    await page.locator('input[placeholder="API Key / Bearer Token"]').fill("dev");
    await page.locator("text=Unlock Workspace").click();
    await page.waitForSelector("text=Dashboard");

    const tabs = ["Trade", "Market", "Backtest", "Settings", "Dashboard"];
    for (const tab of tabs) {
      await page.locator(`text=${tab}`).first().click();
      await page.waitForTimeout(500);
    }
  });
});

test.describe("Dashboard", () => {
  test("shows account summary and watchlist", async ({ page }) => {
    await page.goto(BASE);
    await page.locator('input[placeholder="API Key / Bearer Token"]').fill("dev");
    await page.locator("text=Unlock Workspace").click();
    await page.waitForSelector("text=Account");
    await page.waitForTimeout(2000);

    await expect(page.locator("text=Equity")).toBeVisible();
    await expect(page.locator("text=Watchlist")).toBeVisible();
    await expect(page.locator("text=+ Add")).toBeVisible();
  });

  test("can add ticker to watchlist", async ({ page }) => {
    await page.goto(BASE);
    await page.locator('input[placeholder="API Key / Bearer Token"]').fill("dev");
    await page.locator("text=Unlock Workspace").click();
    await page.waitForSelector("text=Account");

    await page.locator("text=+ Add").click();
    const addInput = page.locator('input[placeholder="Ticker (e.g. AAPL)"]');
    await addInput.fill("AAPL");
    await page.locator("text=Add").last().click();
    await page.waitForTimeout(2000);

    await expect(page.locator("text=AAPL")).toBeVisible();
  });
});

test.describe("Trade Screen", () => {
  test("shows order ticket and can place order", async ({ page }) => {
    await page.goto(BASE);
    await page.locator('input[placeholder="API Key / Bearer Token"]').fill("dev");
    await page.locator("text=Unlock Workspace").click();
    await page.waitForSelector("text=Dashboard");
    await page.locator("text=Trade").first().click();
    await page.waitForTimeout(1000);

    await expect(page.locator("text=New Order")).toBeVisible();

    const tickerInput = page.locator('input[placeholder="Ticker (e.g. AAPL)"]');
    await tickerInput.fill("MSFT");
    await page.locator("text=Place BUY Order").click();
    await page.waitForTimeout(2000);

    await expect(page.locator("text=Order")).toBeVisible();
  });
});

test.describe("Market Screen", () => {
  test("shows chart after loading ticker", async ({ page }) => {
    await page.goto(BASE);
    await page.locator('input[placeholder="API Key / Bearer Token"]').fill("dev");
    await page.locator("text=Unlock Workspace").click();
    await page.waitForSelector("text=Dashboard");
    await page.locator("text=Market").first().click();
    await page.waitForTimeout(500);

    const tickerInput = page.locator('input[placeholder="Ticker symbol"]');
    await tickerInput.fill("AAPL");
    await page.locator("text=Load").click();
    await page.waitForTimeout(3000);

    await expect(page.locator("text=Candlestick")).toBeVisible();
    await expect(page.locator("text=Volume")).toBeVisible();
    await expect(page.locator("text=RSI")).toBeVisible();
  });

  test("timeframe selector renders all options", async ({ page }) => {
    await page.goto(BASE);
    await page.locator('input[placeholder="API Key / Bearer Token"]').fill("dev");
    await page.locator("text=Unlock Workspace").click();
    await page.waitForSelector("text=Dashboard");
    await page.locator("text=Market").first().click();

    for (const tf of ["1m", "5m", "15m", "1H", "1D", "1W", "1M"]) {
      await expect(page.locator("text=" + tf)).toBeVisible();
    }
  });
});

test.describe("Backtest Screen", () => {
  test("config panel renders", async ({ page }) => {
    await page.goto(BASE);
    await page.locator('input[placeholder="API Key / Bearer Token"]').fill("dev");
    await page.locator("text=Unlock Workspace").click();
    await page.waitForSelector("text=Dashboard");
    await page.locator("text=Backtest").first().click();

    await expect(page.locator("text=Backtest Configuration")).toBeVisible();
  });

  test("can run backtest and see results", async ({ page }) => {
    await page.goto(BASE);
    await page.locator('input[placeholder="API Key / Bearer Token"]').fill("dev");
    await page.locator("text=Unlock Workspace").click();
    await page.waitForSelector("text=Dashboard");
    await page.locator("text=Backtest").first().click();
    await page.waitForTimeout(500);

    const tickerInput = page.locator('input[placeholder="Tickers (comma-sep, e.g. AAPL,MSFT)"]');
    await tickerInput.fill("AAPL");
    await page.locator("text=Run").first().click();
    await page.waitForTimeout(5000);

    await expect(page.locator("text=Results")).toBeVisible({ timeout: 10000 });
    await expect(page.locator("text=Trades")).toBeVisible();
  });
});

test.describe("Settings Screen", () => {
  test("renders all settings sections", async ({ page }) => {
    await page.goto(BASE);
    await page.locator('input[placeholder="API Key / Bearer Token"]').fill("dev");
    await page.locator("text=Unlock Workspace").click();
    await page.waitForSelector("text=Dashboard");
    await page.locator("text=Settings").first().click();

    await expect(page.locator("text=Alpaca Connection")).toBeVisible();
    await expect(page.locator("text=LLM Configuration")).toBeVisible();
    await expect(page.locator("text=Risk Parameters")).toBeVisible();
    await expect(page.locator("text=Backend URL")).toBeVisible();
    await expect(page.locator("text=Save Settings")).toBeVisible();
  });
});

test.describe("Chatbot Sidebar", () => {
  test("opens chat and sends a message", async ({ page }) => {
    await page.goto(BASE);
    await page.locator('input[placeholder="API Key / Bearer Token"]').fill("dev");
    await page.locator("text=Unlock Workspace").click();
    await page.waitForSelector("text=Dashboard");

    await page.locator("text=Chat").first().click();
    await page.waitForTimeout(500);

    await expect(page.locator("text=QuantFlow Assistant")).toBeVisible();
    await expect(page.locator("text=Suggested")).toBeVisible();

    const chatInput = page.locator('textarea[placeholder="Ask QuantFlow..."]');
    await chatInput.fill("Hello");
    await chatInput.press("Enter");
    await page.waitForTimeout(2000);

    await expect(page.locator("text=Ready when you are")).toBeVisible({ timeout: 10000 });
  });
});

test.describe("Responsive and Layout", () => {
  test("left sidebar navigation is present", async ({ page }) => {
    await page.goto(BASE);
    await page.locator('input[placeholder="API Key / Bearer Token"]').fill("dev");
    await page.locator("text=Unlock Workspace").click();
    await page.waitForSelector("text=Dashboard");

    const sidebar = page.locator("text=Dashboard").first();
    await expect(sidebar).toBeVisible();
  });

  test("platform header shows branding", async ({ page }) => {
    await page.goto(BASE);
    await page.locator('input[placeholder="API Key / Bearer Token"]').fill("dev");
    await page.locator("text=Unlock Workspace").click();
    await page.waitForSelector("text=Dashboard");

    await expect(page.locator("text=QuantFlow").first()).toBeVisible();
    await expect(page.locator("text=Algorithmic Trading")).toBeVisible();
  });

  test("status bar shows API status", async ({ page }) => {
    await page.goto(BASE);
    await page.locator('input[placeholder="API Key / Bearer Token"]').fill("dev");
    await page.locator("text=Unlock Workspace").click();
    await page.waitForSelector("text=Dashboard");

    await expect(page.locator("text=API:")).toBeVisible();
  });
});
