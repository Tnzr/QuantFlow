import puppeteer from "puppeteer";
const browser = await puppeteer.launch({ headless: "new", args: ["--no-sandbox"] });
const page = await browser.newPage();
await page.setViewport({ width: 1400, height: 900 });
page.on("pageerror", (e) => console.log("PAGEERR:", e.message.slice(0, 500)));
await page.goto("http://127.0.0.1:8080", { waitUntil: "networkidle0", timeout: 10000 });
await new Promise(r => setTimeout(r, 1500));
await page.evaluate(() => {
  const input = document.querySelector('input');
  const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
  setter.call(input, 'demo-token');
  input.dispatchEvent(new Event('input', { bubbles: true }));
});
await page.evaluate(() => {
  const btn = Array.from(document.querySelectorAll('*')).find(el => el.textContent && el.textContent.trim() === "Unlock Workspace");
  if (btn) btn.click();
});
await new Promise(r => setTimeout(r, 2500));
await page.evaluate(() => {
  for (const el of document.querySelectorAll('*')) {
    if (el.children.length === 0 && el.textContent === "Backtest") {
      let p = el; for (let i=0;i<6;i++){ const r=p.getBoundingClientRect(); if (r.width>30 && r.width<100){p.click();return;} p=p.parentElement; }
    }
  }
});
await new Promise(r => setTimeout(r, 2000));

// Type ticker
const tickerInput = await page.evaluate(() => {
  const inputs = Array.from(document.querySelectorAll('input'));
  const target = inputs.find(i => i.getAttribute('aria-label') === 'Tickers');
  if (target) {
    const r = target.getBoundingClientRect();
    return { x: r.x, y: r.y, w: r.width, h: r.height };
  }
  return null;
});
await page.mouse.click(tickerInput.x + 50, tickerInput.y + 18);
await page.keyboard.type("AAPL", { delay: 80 });
await new Promise(r => setTimeout(r, 300));

// Click Run
const runBtn = await page.evaluate(() => {
  const els = Array.from(document.querySelectorAll('*'));
  const runBtn = els.find(el => el.children.length === 0 && el.textContent && el.textContent.trim() === "Run");
  if (runBtn) {
    const r = runBtn.getBoundingClientRect();
    return { x: r.x, y: r.y, w: r.width, h: r.height };
  }
  return null;
});
await page.mouse.click(runBtn.x + runBtn.w / 2, runBtn.y + runBtn.h / 2);
await new Promise(r => setTimeout(r, 10000));
const body = await page.evaluate(() => document.body.innerText);
const hasResults = body.includes("Results") || body.includes("Sharpe") || body.includes("Trades");
const hasError = body.includes("Something went wrong");
console.log("Has results:", hasResults);
console.log("Has error boundary triggered:", hasError);
console.log("Body length:", body.length);
if (hasError) {
  const match = body.match(/Something went wrong\n([^\n]+)/);
  console.log("Error msg:", match ? match[1] : "unknown");
}
await page.screenshot({ path: "/tmp/bt-fixed.png", fullPage: true });
await browser.close();
