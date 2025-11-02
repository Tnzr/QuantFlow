from __future__ import annotations

import os
from typing import List, Optional
from pathlib import Path

from .base import Broker
from .types import Position, Account

import robin_stocks.robinhood as r
from .credentials import get_creds


def _default_session_path() -> str:
    base = Path(os.getenv("XDG_CONFIG_HOME", Path.home() / ".config")) / "quantflow"
    base.mkdir(parents=True, exist_ok=True)
    return str(base / "robinhood.pickle")


class RobinhoodBroker(Broker):
    def __init__(self, username: str | None = None, password: str | None = None, mfa: str | None = None):
        self.username = username or os.getenv("ROBINHOOD_USERNAME")
        self.password = password or os.getenv("ROBINHOOD_PASSWORD")
        self.mfa = mfa or os.getenv("ROBINHOOD_MFA")
        self._logged_in = False

    def login(self, mfa_code: Optional[str] = None, store_session: bool = False, expires_in: int = 86400) -> None:
        if self._logged_in:
            return
        if not (self.username and self.password):
            creds = get_creds()
            self.username, self.password = creds.username, creds.password
        # prefer provided MFA code, else env-provided, else None (SMS challenge may be triggered)
        mfa_val = mfa_code or self.mfa
        pickle_path = os.getenv("QUANTFLOW_RH_SESSION_PATH", _default_session_path())
        r.login(
            username=self.username,
            password=self.password,
            mfa_code=mfa_val,
            by_sms=True,
            store_session=store_session,
            expiresIn=expires_in,
            pickle_path=pickle_path,
        )
        self._logged_in = True

    def account(self) -> Account:
        self.login()
        profile = r.profiles.load_portfolio_profile()
        acct = r.profiles.load_account_profile()
        equity = float(profile.get("equity", 0.0) or 0.0)
        cash = float(acct.get("cash", 0.0) or 0.0)
        buying_power = float(acct.get("buying_power", 0.0) or 0.0)
        return Account(equity=equity, cash=cash, buying_power=buying_power)

    def positions(self) -> List[Position]:
        self.login()
        pos = r.account.build_holdings()  # dict keyed by symbol
        out: List[Position] = []
        for sym, data in pos.items():
            qty = float(data.get("quantity", 0.0) or 0.0)
            avgp = float(data.get("average_buy_price", 0.0) or 0.0)
            out.append(Position(ticker=sym.upper(), qty=qty, avg_price=avgp, side="long" if qty >= 0 else "short"))
        return out
