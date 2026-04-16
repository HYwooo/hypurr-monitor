"""Structured alert event model for webhook notifications."""

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

ALERT_CATEGORY_BY_TYPE = {
    "ATR_Ch": "signal",
    "ClusterST": "signal",
    "BREAKOUT": "signal",
    "SYSTEM": "system",
    "ERROR": "error",
    "CONFIG": "config",
    "CONFIG ERROR": "config",
    "REPORT": "report",
}

ALERT_SEVERITY_BY_TYPE = {
    "ATR_Ch": "info",
    "ClusterST": "info",
    "BREAKOUT": "warn",
    "SYSTEM": "info",
    "ERROR": "error",
    "CONFIG": "info",
    "CONFIG ERROR": "error",
    "REPORT": "info",
}


@dataclass(frozen=True, slots=True)
class AlertEvent:
    """Structured alert event for downstream rendering and delivery."""

    alert_type: str
    message: str
    extra: dict[str, Any]
    category: str
    severity: str
    event: str
    symbol: str | None = None
    direction: str | None = None
    timeframe: str | None = None
    dedupe_key: str | None = None


def _infer_event_name(alert_type: str, extra: Mapping[str, Any]) -> str:
    """Infer a stable event name from legacy webhook fields."""
    reason = str(extra.get("reason", "")).strip().lower()
    if reason:
        return reason
    direction = str(extra.get("direction", "")).strip().lower()
    if direction:
        return direction
    return alert_type.lower().replace(" ", "_")


def _build_dedupe_key(alert_type: str, symbol: str | None, event: str, timeframe: str | None) -> str:
    """Build a stable dedupe key for alert delivery semantics."""
    parts = [alert_type]
    if symbol:
        parts.append(symbol)
    if timeframe:
        parts.append(timeframe)
    parts.append(event)
    return ":".join(parts)


def build_alert_event(alert_type: str, message: str, extra: Mapping[str, Any] | None = None) -> AlertEvent:
    """Convert legacy webhook arguments into a structured AlertEvent."""
    payload = dict(extra or {})
    symbol = str(payload.get("symbol", "")).strip() or None
    direction = str(payload.get("direction", "")).strip() or None
    timeframe = str(payload.get("timeframe", "")).strip() or None
    event = _infer_event_name(alert_type, payload)
    return AlertEvent(
        alert_type=alert_type,
        message=message,
        extra=payload,
        category=ALERT_CATEGORY_BY_TYPE.get(alert_type, "other"),
        severity=ALERT_SEVERITY_BY_TYPE.get(alert_type, "info"),
        event=event,
        symbol=symbol,
        direction=direction,
        timeframe=timeframe,
        dedupe_key=_build_dedupe_key(alert_type, symbol, event, timeframe),
    )


async def emit_alert(
    send_webhook_fn: Callable[[str, str, Mapping[str, Any] | None], Awaitable[None]],
    alert_type: str,
    message: str,
    extra: Mapping[str, Any] | None = None,
    send_event_fn: Callable[[AlertEvent], Awaitable[None]] | None = None,
) -> None:
    """Send a legacy alert or a structured AlertEvent when available."""
    if send_event_fn is not None:
        await send_event_fn(build_alert_event(alert_type, message, extra))
        return
    await send_webhook_fn(alert_type, message, extra)
