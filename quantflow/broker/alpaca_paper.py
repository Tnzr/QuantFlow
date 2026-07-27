"""Alpaca Paper Trading Service — live quotes, account, positions, order placement.

Uses Alpaca's free tier Paper Trading API. Credentials from .env: ALPACA_API_KEY, ALPACA_SECRET_KEY.
Paper base URL: https://paper-api.alpaca.markets
"""

import os, time, logging, threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import requests

logger = logging.getLogger(__name__)

ALPACA_KEY = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET = os.environ.get("ALPACA_SECRET_KEY", "")
PAPER_BASE = "https://paper-api.alpaca.markets"
DATA_BASE = "https://data.alpaca.markets"

HEADERS = {
    "APCA-API-KEY-ID": ALPACA_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET,
}


@dataclass
class AccountInfo:
    equity: float; cash: float; buying_power: float
    portfolio_value: float; day_pnl: float; day_pnl_pct: float


@dataclass
class PositionInfo:
    ticker: str; shares: int; avg_entry_price: float
    current_price: float; unrealized_pnl: float; unrealized_pnl_pct: float
    market_value: float


@dataclass
class OrderResult:
    id: str; ticker: str; side: str; qty: int
    type: str; status: str; filled_price: Optional[float] = None


class AlpacaPaperTrading:
    """Alpaca paper trading API wrapper."""

    def __init__(self):
        self.enabled = bool(ALPACA_KEY and ALPACA_SECRET)
        self._account_cache = None
        self._cache_time = 0
        self._cache_ttl = 5  # seconds

    def _require_keys(self) -> None:
        if not self.enabled:
            raise RuntimeError(
                "Alpaca credentials missing. Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env "
                "or QUANTFLOW_DEMO_MODE=true for offline testing."
            )

    def _get(self, path: str) -> dict:
        resp = requests.get(f"{PAPER_BASE}/v2/{path}", headers=HEADERS, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, data: dict) -> dict:
        resp = requests.post(f"{PAPER_BASE}/v2/{path}", json=data, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        return resp.json()

    # ── Account ──────────────────────────────────────────────────────

    def get_account(self) -> AccountInfo:
        if os.environ.get("QUANTFLOW_DEMO_MODE", "").lower() in ("1", "true"):
            return AccountInfo(equity=100000, cash=50000, buying_power=100000,
                              portfolio_value=100000, day_pnl=0, day_pnl_pct=0)
        self._require_keys()
        now = time.time()
        if self._account_cache and (now - self._cache_time) < self._cache_ttl:
            return self._account_cache

        acc = self._get("account")
        self._account_cache = AccountInfo(
            equity=float(acc["equity"]),
            cash=float(acc["cash"]),
            buying_power=float(acc["buying_power"]),
            portfolio_value=float(acc["portfolio_value"]),
            day_pnl=float(acc.get("equity", 0)) - float(acc.get("last_equity", 0)),
            day_pnl_pct=(float(acc.get("equity", 0)) / max(float(acc.get("last_equity", 0)), 1)) - 1,
        )
        self._cache_time = now
        return self._account_cache

    # ── Positions ────────────────────────────────────────────────────

    def get_positions(self) -> List[PositionInfo]:
        if os.environ.get("QUANTFLOW_DEMO_MODE", "").lower() in ("1", "true"):
            return []
        self._require_keys()

        positions = self._get("positions")
        result = []
        for p in positions:
            result.append(PositionInfo(
                ticker=p["symbol"],
                shares=int(p["qty"]),
                avg_entry_price=float(p["avg_entry_price"]),
                current_price=float(p["current_price"]),
                unrealized_pnl=float(p.get("unrealized_pl", 0)),
                unrealized_pnl_pct=float(p.get("unrealized_plpc", 0)) * 100,
                market_value=float(p["market_value"]),
            ))
        return result

    # ── Orders ───────────────────────────────────────────────────────

    def place_order(self, ticker: str, side: str, qty: float,
                    order_type: str = "market",
                    limit_price: Optional[float] = None,
                    stop_loss_pct: Optional[float] = None,
                    take_profit_pct: Optional[float] = None) -> OrderResult:
        """Place a paper trade order."""
        if os.environ.get("QUANTFLOW_DEMO_MODE", "").lower() in ("1", "true"):
            return OrderResult(id="demo-001", ticker=ticker, side=side,
                             qty=qty, type=order_type, status="demo_filled",
                             filled_price=100.0)
        self._require_keys()

        data = {
            "symbol": ticker, "qty": str(qty), "side": side,
            "type": order_type, "time_in_force": "day",
        }
        if limit_price:
            data["limit_price"] = str(limit_price)

        if stop_loss_pct or take_profit_pct:
            entry_price = limit_price
            if not entry_price:
                try:
                    quote = self.get_last_quote(ticker)
                    entry_price = quote.get("last", quote.get("ask"))
                except Exception:
                    entry_price = None
            if not entry_price or entry_price <= 0:
                raise RuntimeError(f"Could not determine entry price for {ticker}")

            data["order_class"] = "bracket"
            data["type"] = "limit"
            data["limit_price"] = str(round(entry_price, 2))

            if stop_loss_pct:
                data["stop_loss"] = {
                    "stop_price": str(round(entry_price * (1 - stop_loss_pct), 2))
                }
            if take_profit_pct:
                data["take_profit"] = {
                    "limit_price": str(round(entry_price * (1 + take_profit_pct), 2))
                }

        order = self._post("orders", data)
        return OrderResult(
            id=order["id"], ticker=order["symbol"], side=order["side"],
            qty=float(order["qty"]), type=order["type"],
            status=order["status"],
            filled_price=float(order.get("filled_avg_price") or 0)
        )

    def cancel_order(self, order_id: str) -> bool:
        if not self.enabled:
            raise RuntimeError("Alpaca credentials missing")
        resp = requests.delete(f"{PAPER_BASE}/v2/orders/{order_id}", headers=HEADERS)
        return resp.status_code == 204

    def get_orders(self, status: str = "all", limit: int = 50) -> List[dict]:
        if not self.enabled:
            raise RuntimeError("Alpaca credentials missing")
        return self._get(f"orders?status={status}&limit={limit}")

    # ── Market Data (snapshot, not streaming) ────────────────────────

    def get_last_quote(self, ticker: str) -> dict:
        """Get latest quote from Alpaca."""
        if not self.enabled:
            raise RuntimeError("Alpaca credentials missing")
        try:
            resp = requests.get(
                f"{DATA_BASE}/v2/stocks/{ticker}/quotes/latest",
                headers=HEADERS, timeout=5,
            )
            resp.raise_for_status()
            q = resp.json()["quote"]
            return {"ticker": ticker, "bid": q["bp"], "ask": q["ap"],
                    "last": (q["bp"] + q["ap"]) / 2}
        except Exception as e:
            return {"ticker": ticker, "error": str(e)}

    def get_last_trade(self, ticker: str) -> dict:
        if not self.enabled:
            raise RuntimeError("Alpaca credentials missing")
        try:
            resp = requests.get(
                f"{DATA_BASE}/v2/stocks/{ticker}/trades/latest",
                headers=HEADERS, timeout=5,
            )
            resp.raise_for_status()
            t = resp.json()["trade"]
            return {"ticker": ticker, "price": float(t["p"])}
        except Exception as e:
            return {"ticker": ticker, "error": str(e)}
