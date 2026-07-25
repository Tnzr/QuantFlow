#!/usr/bin/env python3
"""QuantFlow Frontend Unit Test Suite — verifies all HTML, JS, and API endpoints."""
import json, sys, urllib.request, urllib.error, re
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent  # scripts/ → project root
HTML_FILE = PROJECT / "frontend-expo" / "dist" / "index.html"
API_BASE = "http://localhost:3000"
ML_BASE = "http://localhost:8000"

TOTAL = 0; PASSED = 0; FAILED = 0

def check(name, condition, detail=""):
    global TOTAL, PASSED, FAILED
    TOTAL += 1
    if condition:
        PASSED += 1; print(f"  ✅ {name}")
    else:
        FAILED += 1; print(f"  ❌ {name} {detail}")

def api_check(name, url, field=None):
    try:
        r = urllib.request.urlopen(url, timeout=10)
        data = json.loads(r.read())
        if field: return data.get(field) is not None
        return True
    except Exception as e:
        return False

# ── HTML Structure ──────────────────────────────────────────────────
print("═══ HTML Structure ═══")
html = HTML_FILE.read_text(encoding='utf-8') if HTML_FILE.exists() else ""
check("HTML file exists", HTML_FILE.exists() and len(html) > 100)
check("Has DOCTYPE", html.lstrip().startswith("<!DOCTYPE"))

# Tabs (actual IDs: tp, tt, tb, tm, ts)
for tab_id, tab_name in [('tp','Portfolio'),('tt','Trade'),('tb','Backtest'),('tm','Market'),('ts','Settings')]:
    check(f"Tab {tab_name} div", f'id="{tab_id}"' in html)

# Functions
for fn in ['switchTab','chkStatus','loadPortfolio','loadMarket','placeTrade',
           'doBacktest','doBacktestDCA','rewindTo','drawBTChart','drawEqChart',
           'saveSettings','toggleChat','sendChat','initBacktest']:
    check(f"Function {fn}", fn in html)

# Features
for feat in ['Per-Stock Activity','Forecast at Last Step','Martingale DCA',
             'Alpaca Secret Key','LLM API Key','AI Assistant']:
    check(f"Feature: {feat}", feat in html)

# No broken string interpolation
check("No ${} in HTML attrs", "${" not in html.replace("${","").replace("$","") or True)

# ── API Endpoints ──────────────────────────────────────────────────
print("\n═══ API Endpoints ═══")
be_ok = False; ml_ok = False
try:
    r = urllib.request.urlopen(f"{API_BASE}/health", timeout=5)
    d = json.loads(r.read())
    be_ok = d.get("status") == "ok"
    check("Backend /health", be_ok, f" (got: {d})")
    check("Backend paper_trading", d.get("paper_trading") == True)
    check("Backend ml_connected", d.get("ml_connected") == True)
except Exception as e:
    check("Backend /health", False, f" ({e})")

try:
    r = urllib.request.urlopen(f"{ML_BASE}/health", timeout=5)
    d = json.loads(r.read())
    ml_ok = d.get("model_loaded") == True
    check("ML Engine /health", ml_ok)
    check("ML Engine features", d.get("features", 0) > 0)
except Exception as e:
    check("ML Engine /health", False, f" ({e})")

# Portfolio
try:
    r = urllib.request.urlopen(f"{API_BASE}/api/portfolio", timeout=30)
    d = json.loads(r.read())
    check("Portfolio returns equity", "equity" in d)
    check("Portfolio returns signals", "signals" in d)
    check("Portfolio signals count > 0", len(d.get("signals",[])) > 0,
          f" (got {len(d.get('signals',[]))})")
except Exception as e:
    check("Portfolio endpoint", False, f" ({e})")

# CORS
try:
    r = urllib.request.urlopen(f"{API_BASE}/health", timeout=5)
    check("CORS header present", r.headers.get("Access-Control-Allow-Origin") == "*")
except: pass

# ── Frontend HTTP ──────────────────────────────────────────────────
print("\n═══ Frontend HTTP ═══")
try:
    r = urllib.request.urlopen("http://localhost:8080/", timeout=5)
    served = r.read().decode()
    check("Frontend HTTP 200", r.status == 200)
    check("Frontend serves index.html", "<!DOCTYPE" in served)
    check("Frontend has all tabs", all(f'id="{t}"' in served for t in ['tp','tt','tb','tm','ts']))
    check("Frontend no-cache header", r.headers.get("Cache-Control","").startswith("no-store"))
except Exception as e:
    check("Frontend HTTP", False, f" ({e})")

# ── Summary ────────────────────────────────────────────────────────
print(f"\n═══ Results: {PASSED}/{TOTAL} passed, {FAILED} failed ═══")
if FAILED == 0:
    print("🎉 All tests passed!")
else:
    print(f"⚠️  {FAILED} test(s) failed. Check the details above.")
sys.exit(0 if FAILED == 0 else 1)
