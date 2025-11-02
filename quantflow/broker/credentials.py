from __future__ import annotations

import os
from dataclasses import dataclass
from getpass import getpass
from typing import Optional

try:
    import keyring  # type: ignore
except Exception:  # keyring may be unavailable (e.g., some containers)
    keyring = None  # type: ignore

SERVICE = "quantflow_robinhood"


@dataclass
class RobinhoodCreds:
    username: str
    password: str


def load_from_env() -> Optional[RobinhoodCreds]:
    u = os.getenv("ROBINHOOD_USERNAME")
    p = os.getenv("ROBINHOOD_PASSWORD")
    if u and p:
        return RobinhoodCreds(u, p)
    return None


def load_from_keyring() -> Optional[RobinhoodCreds]:
    if not keyring:
        return None
    try:
        u = keyring.get_password(SERVICE, "username")
        if not u:
            return None
        p = keyring.get_password(SERVICE, u)
        if not p:
            return None
        return RobinhoodCreds(u, p)
    except Exception:
        return None


def save_to_keyring(username: str, password: str) -> None:
    if not keyring:
        raise RuntimeError("Keyring backend not available. Install 'keyring' and a backend (SecretStorage on Linux).")
    keyring.set_password(SERVICE, "username", username)
    keyring.set_password(SERVICE, username, password)


def delete_from_keyring() -> None:
    if not keyring:
        return
    try:
        u = keyring.get_password(SERVICE, "username")
        if u:
            try:
                keyring.delete_password(SERVICE, u)
            except Exception:
                pass
            try:
                keyring.delete_password(SERVICE, "username")
            except Exception:
                pass
    except Exception:
        pass


def prompt_for_creds() -> RobinhoodCreds:
    u = input("Robinhood username (email): ").strip()
    p = getpass("Robinhood password: ")
    return RobinhoodCreds(u, p)


def get_creds(prefer: str = "env,keyring,prompt") -> RobinhoodCreds:
    order = [x.strip() for x in prefer.split(",") if x.strip()]
    for source in order:
        if source == "env":
            c = load_from_env()
            if c:
                return c
        elif source == "keyring":
            c = load_from_keyring()
            if c:
                return c
        elif source == "prompt":
            return prompt_for_creds()
    return prompt_for_creds()
