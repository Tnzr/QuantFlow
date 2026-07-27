"""QuantFlow Backend API — Alpaca paper trading, live quotes, AI signals, portfolio, auth."""
import hashlib, hmac, json, os, time, httpx, asyncio, logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ML_SERVICE = os.environ.get("ML_ENGINE_URL", "http://ml-engine:8000")
JWT_SECRET = os.environ.get("JWT_SECRET", "dev-secret")
STRIPE_SECRET = os.environ.get("STRIPE_SECRET", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
PAPER_TRADING = os.environ.get("PAPER_TRADING", "false").lower() == "true"

# ── App ────────────────────────────────────────────────────────────────
app = FastAPI(title="QuantFlow Backend", version="2.0")

@app.middleware("http")
async def add_cors(request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

# ── Models ─────────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str = "ok"; service: str = "quantflow-backend"
    paper_trading: bool = False; ml_connected: bool = False

class PortfolioResponse(BaseModel):
    equity: float; cash: float; buying_power: float
    day_pnl: float; day_pnl_pct: float
    positions: List[dict]; signals: List[dict]

class OrderRequest(BaseModel):
    ticker: str; side: str; qty: int
    order_type: str = "market"; limit_price: Optional[float] = None
    stop_loss_pct: Optional[float] = None; take_profit_pct: Optional[float] = None

class AuthRequest(BaseModel):
    credential: str  # Google ID token
class AuthResponse(BaseModel):
    token: str; email: str; name: str; picture: Optional[str] = None
class SubscriptionResponse(BaseModel):
    active: bool; plan: str; expires: Optional[str] = None

# ── Paper Trading ──────────────────────────────────────────────────────
class PaperTradingService:
    def __init__(self):
        self.enabled = PAPER_TRADING
        self.key = os.environ.get("ALPACA_API_KEY", "")
        self.secret = os.environ.get("ALPACA_SECRET_KEY", "")
        self.base = "https://paper-api.alpaca.markets"
        self.data_base = "https://data.alpaca.markets"

    def _headers(self):
        return {"APCA-API-KEY-ID": self.key, "APCA-API-SECRET-KEY": self.secret}

    async def get_account(self):
        if os.environ.get("QUANTFLOW_DEMO_MODE", "").lower() in ("1", "true"):
            return {"equity": 100000, "cash": 50000, "buying_power": 100000,
                    "portfolio_value": 100000, "day_pnl": 0, "day_pnl_pct": 0}
        if not self.enabled or not self.key:
            raise HTTPException(status_code=503, detail="Alpaca credentials missing. Configure ALPACA_API_KEY and ALPACA_SECRET_KEY in .env, or set QUANTFLOW_DEMO_MODE=true for offline testing.")
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                r = await c.get(f"{self.base}/v2/account", headers=self._headers())
                r.raise_for_status()
                a = r.json()
                return {"equity": float(a.get("equity", 100000)), "cash": float(a.get("cash", 50000)),
                        "buying_power": float(a.get("buying_power", 100000)),
                        "portfolio_value": float(a.get("portfolio_value", 100000)),
                        "day_pnl": 0, "day_pnl_pct": 0}
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"Alpaca account fetch failed: {e}")

    async def _get(self, path: str):
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                r = await c.get(f"{self.base}/v2/{path}", headers=self._headers())
                r.raise_for_status()
                return r.json()
        except Exception as e:
            logger.warning(f"Alpaca GET {path}: {e}")
            return None

    async def _post(self, path: str, data: dict):
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                r = await c.post(f"{self.base}/v2/{path}", json=data, headers=self._headers())
                r.raise_for_status()
                return r.json()
        except Exception as e:
            logger.warning(f"Alpaca POST {path}: {e}")
            return None

    async def get_positions(self):
        if os.environ.get("QUANTFLOW_DEMO_MODE", "").lower() in ("1", "true"): return []
        if not self.enabled: raise HTTPException(status_code=503, detail="Alpaca credentials missing")
        data = await self._get("positions")
        if data is None: return []
        result = []
        for p in (data if isinstance(data, list) else []):
            result.append({"ticker": p.get("symbol","?"), "shares": int(p.get("qty",0)),
                "avgPrice": float(p.get("avg_entry_price",0)),
                "currentPrice": float(p.get("current_price",0)),
                "unrealizedPnl": float(p.get("unrealized_pl",0)),
                "unrealizedPnlPct": float(p.get("unrealized_plpc",0))*100,
                "marketValue": float(p.get("market_value",0))})
        return result

    async def place_order(self, req: OrderRequest):
        if os.environ.get("QUANTFLOW_DEMO_MODE", "").lower() in ("1", "true"):
            return {"id": f"demo-{int(time.time())}", "status": "filled",
                    "filled_price": 195.40, **req.model_dump()}
        if not self.enabled or not self.key:
            raise HTTPException(status_code=503, detail="Alpaca credentials missing")
        data = {"symbol": req.ticker, "qty": str(req.qty), "side": req.side,
                "type": req.order_type, "time_in_force": "day"}
        if req.limit_price:
            data["limit_price"] = str(req.limit_price)
        if req.stop_loss_pct:
            data["order_class"] = "bracket"
            data["stop_loss"] = {"stop_price": str(round(
                (req.limit_price or 100) * (1 - req.stop_loss_pct), 2))}
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(f"{self.base}/v2/orders", json=data, headers=self._headers())
            r.raise_for_status()
            o = r.json()
            return {"id": o["id"], "status": o["status"], "ticker": req.ticker,
                    "side": req.side, "qty": req.qty,
                    "filled_price": float(o.get("filled_avg_price", 0))}

    async def get_quote(self, ticker: str):
        if not self.enabled or not self.key:
            raise HTTPException(status_code=503, detail="Alpaca credentials missing")
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(f"{self.data_base}/v2/stocks/{ticker}/trades/latest",
                           headers=self._headers())
            r.raise_for_status()
            p = float(r.json()["trade"]["p"])
            return {"ticker": ticker, "price": p, "bid": p, "ask": p}

    async def get_order_history(self, limit=50):
        if not self.enabled: return []
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"{self.base}/v2/orders?status=all&limit={limit}",
                           headers=self._headers())
            r.raise_for_status()
            return [{"id": o["id"], "ticker": o["symbol"], "side": o["side"],
                     "qty": o["qty"], "status": o["status"],
                     "filled_price": float(o.get("filled_avg_price",0)),
                     "created": o["created_at"]} for o in r.json()]

trading = PaperTradingService()

# ── Simple User Store ──────────────────────────────────────────────────
users_db: Dict[str, dict] = {}  # email → {name, picture, plan, subscribed_at, expires}
def _load_users():
    p = Path("data/users.json")
    if p.exists():
        return json.loads(p.read_text())
    return {}
def _save_users():
    Path("data/users.json").write_text(json.dumps(users_db))
users_db = _load_users()

# ── Health ─────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            await c.get(f"{ML_SERVICE}/health")
            ml_ok = True
    except Exception:
        ml_ok = False
    return HealthResponse(paper_trading=trading.enabled, ml_connected=ml_ok)

# ── Portfolio ───────────────────────────────────────────────────────────
@app.get("/api/portfolio")
async def portfolio():
    acc = await trading.get_account()
    positions = await trading.get_positions()
    try:
        async with httpx.AsyncClient(timeout=20) as c:
            r = await c.post(f"{ML_SERVICE}/predict/batch",
                json={"tickers": ["AAPL","MSFT","GOOGL","AMZN"]})
            if r.status_code == 200:
                signals = r.json().get("predictions", [])
            else:
                signals = []
    except Exception:
        signals = []

    return PortfolioResponse(positions=positions, signals=signals, **acc)

# ── Orders ─────────────────────────────────────────────────────────────
@app.post("/api/orders")
async def place_order(req: OrderRequest):
    return await trading.place_order(req)

@app.get("/api/orders")
async def order_history(limit: int = 50):
    return await trading.get_order_history(limit)

# ── Market Data ────────────────────────────────────────────────────────
@app.get("/api/market-data/{ticker}")
async def market_data(ticker: str):
    return await trading.get_quote(ticker)

# ── Auth ───────────────────────────────────────────────────────────────
@app.post("/api/auth/google")
async def google_auth(req: AuthRequest):
    """Verify Google ID token and issue JWT."""
    if not GOOGLE_CLIENT_ID:
        # Demo mode: accept any
        import base64
        try:
            payload = json.loads(base64.b64decode(req.credential.split(".")[1] + "=="))
            email = payload.get("email", "demo@quantflow.io")
            name = payload.get("name", "Demo User")
            picture = payload.get("picture")
        except Exception:
            email, name, picture = "demo@quantflow.io", "Demo User", None
    else:
        async with httpx.AsyncClient() as c:
            r = await c.get(f"https://oauth2.googleapis.com/tokeninfo?id_token={req.credential}")
            if r.status_code != 200:
                raise HTTPException(401, "Invalid Google token")
            payload = r.json()
            email, name, picture = payload["email"], payload.get("name",""), payload.get("picture")

    # Register if new
    if email not in users_db:
        users_db[email] = {"name": name, "picture": picture, "plan": "free",
                           "subscribed_at": None, "expires": None}
        _save_users()

    # Issue JWT
    header = base64_encode(json.dumps({"alg": "HS256", "typ": "JWT"}))
    payload = base64_encode(json.dumps({"sub": email, "name": name,
        "iat": int(time.time()), "exp": int(time.time()) + 86400}))
    sig = hmac.new(JWT_SECRET.encode(), f"{header}.{payload}".encode(), hashlib.sha256).hexdigest()
    token = f"{header}.{payload}.{sig}"

    return AuthResponse(token=token, email=email, name=name, picture=picture)

def base64_encode(s: str) -> str:
    import base64
    return base64.urlsafe_b64encode(s.encode()).decode().rstrip("=")

# ── Stripe ─────────────────────────────────────────────────────────────
@app.get("/api/subscription")
async def subscription(request: Request):
    """Get current user's subscription status."""
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    email = _verify_token(token)
    if not email or email not in users_db:
        raise HTTPException(401, "Unauthorized")
    u = users_db[email]
    active = u.get("plan", "free") != "free"
    return SubscriptionResponse(active=active, plan=u.get("plan", "free"),
                                expires=u.get("expires"))

@app.post("/api/stripe/webhook")
async def stripe_webhook(request: Request):
    """Handle Stripe checkout.session.completed events."""
    body = await request.body()
    sig = request.headers.get("stripe-signature", "")
    if STRIPE_WEBHOOK_SECRET:
        import stripe
        try:
            event = stripe.Webhook.construct_event(body, sig, STRIPE_WEBHOOK_SECRET)
        except Exception:
            raise HTTPException(400, "Invalid signature")
    else:
        event = json.loads(body)

    if event.get("type") == "checkout.session.completed":
        session = event["data"]["object"]
        email = session.get("customer_email") or session.get("customer_details", {}).get("email")
        if email:
            if email not in users_db:
                users_db[email] = {"name": "", "picture": None, "plan": "pro",
                                   "subscribed_at": datetime.now(timezone.utc).isoformat(),
                                   "expires": None}
            else:
                users_db[email]["plan"] = "pro"
                users_db[email]["subscribed_at"] = datetime.now(timezone.utc).isoformat()
            _save_users()

    return {"status": "ok"}

def _verify_token(token: str) -> Optional[str]:
    try:
        parts = token.split(".")
        if len(parts) != 3: return None
        payload = json.loads(base64_decode(parts[1]))
        if payload.get("exp", 0) < time.time(): return None
        return payload.get("sub")
    except Exception:
        return None

def base64_decode(s: str) -> str:
    import base64
    s += "=" * (4 - len(s) % 4)
    return base64.urlsafe_b64decode(s).decode()

# ── WebSocket for live portfolio ────────────────────────────────────────
@app.websocket("/ws/portfolio")
async def ws_portfolio(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            acc = await trading.get_account()
            positions = await trading.get_positions()
            await websocket.send_json({"type": "portfolio_update", "data": {
                "equity": acc["equity"], "cash": acc["cash"],
                "positions": positions, "timestamp": datetime.now(timezone.utc).isoformat(),
            }})
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        pass
