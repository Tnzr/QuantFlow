from __future__ import annotations

from typing import Optional

from ..data.persistence import (
    save_execution_intent,
    save_execution_event,
    save_policy_decision,
)


def audit_intent(
    ticker: str,
    action: str,
    qty: float,
    notional: float,
    broker_mode: str,
    source: str = "quantflow",
    status: str = "proposed",
    db_path: str = "sqlite:///quantflow.db",
) -> int:
    return save_execution_intent(
        ticker=ticker,
        action=action,
        qty=qty,
        notional=notional,
        broker_mode=broker_mode,
        source=source,
        status=status,
        db_path=db_path,
    )


def audit_event(
    intent_id: int,
    event_type: str,
    message: str,
    db_path: str = "sqlite:///quantflow.db",
) -> None:
    save_execution_event(
        intent_id=intent_id,
        event_type=event_type,
        message=message,
        db_path=db_path,
    )


def audit_policy(
    intent_id: Optional[int],
    allow: bool,
    reason: str,
    db_path: str = "sqlite:///quantflow.db",
) -> None:
    save_policy_decision(
        intent_id=intent_id,
        allow=allow,
        reason=reason,
        db_path=db_path,
    )
