"""
Feishu Webhook notification module - builds message body and sends Webhook requests.

Main functions:
- log_warning / log_error: Log forwarding to standard logger
- _rotate_webhook_log_if_needed: Webhook log file auto-rotation
- build_feishu_card: Build Feishu Interactive Card format message
- send_webhook: Send Webhook request (supports card and text formats)
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .alert_event import AlertEvent, build_alert_event
from .formatters import render_value
from .webhook_sender import WebhookSender

WEBHOOK_LOG_FILE = "webhook.log"
WEBHOOK_SUCCESS_STATUS_CODE = 200


@dataclass(slots=True)
class WebhookDeliveryResult:
    """Structured webhook delivery outcome for observability."""

    attempted: bool
    sent: bool
    failed: bool
    deduped: bool = False


def _get_logger() -> logging.Logger:
    """Get logger instance."""
    return logging.getLogger(__name__)


def log_warning(msg: str) -> None:
    """Forward warning log to standard logger."""
    _get_logger().warning(msg)


def log_error(msg: str) -> None:
    """Forward error log to standard logger."""
    _get_logger().error(msg)


def _rotate_webhook_log_if_needed(log_file_path: str = WEBHOOK_LOG_FILE, max_log_lines: int = 1000) -> None:
    """
    If webhook.log exceeds max_log_lines, truncate to latest max_log_lines.

    Args:
        max_log_lines: Maximum lines to retain (default 1000)
    """
    try:
        log_path = Path(log_file_path)
        if not log_path.exists():
            return
        if max_log_lines <= 0:
            log_path.write_text("", encoding="utf-8")
            return
        with log_path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) <= max_log_lines:
            return
        with log_path.open("w", encoding="utf-8") as f:
            f.writelines(lines[-max_log_lines:])
    except Exception as e:
        _get_logger().warning(f"Log rotation failed: {e}")


def _render_extra_value(extra: dict[str, Any], key: str) -> str:
    """Render a payload field for display without mutating the payload type upstream."""
    return render_value(extra.get(key, ""))


def build_feishu_card(  # noqa: PLR0912, PLR0915
    alert_type: str, message: str, extra: dict[str, Any] | None, timestamp: str
) -> dict[str, Any]:
    """
    Build Feishu Interactive Card message based on alert type.

    Supported alert_type:
    - ATR_Ch: ATR Channel signal (long, short, trailing stop)
    - ClusterST: Clustering SuperTrend signal
    - SYSTEM: System message (connect/disconnect)
    - ERROR: Error message
    - CONFIG: Hot reload success
    - CONFIG ERROR: Hot reload failure
    - REPORT: Daily report
    - BREAKOUT: Breakout confirm/fail signal

    Args:
        alert_type: Alert type string
        message: Message content
        extra: Extended data dict containing symbol, direction, price, etc.
        timestamp: Trigger time string

    Returns:
        Feishu Card format dict, can be used directly as HTTP POST json body
    """
    extra = extra or {}
    direction = extra.get("direction", "").lower()
    symbol = extra.get("symbol", "")
    reason = extra.get("reason", "")

    # ATR_Ch signal
    if alert_type in ("ATR_Ch", "ClusterST"):
        is_trailing = reason == "trailing_stop"
        if is_trailing:
            color = "orange"
            emoji = "\U0001f6d1"
        elif direction == "long":
            color = "green"
            emoji = "\U0001f4c8"
        elif direction == "short":
            color = "red"
            emoji = "\U0001f4c9"
        else:
            color = "blue"
            emoji = "\U0001f4ca"

        price = _render_extra_value(extra, "price")
        atr_upper = _render_extra_value(extra, "atr_upper")
        atr_lower = _render_extra_value(extra, "atr_lower")
        stop_line = _render_extra_value(extra, "stop_line")
        entry_price = _render_extra_value(extra, "entry_price")
        timeframe = _render_extra_value(extra, "timeframe")

        if is_trailing:
            elements = [
                {
                    "tag": "markdown",
                    "content": f"**Direction:** {direction.upper()} TRAILING STOP",
                },
                {"tag": "markdown", "content": f"**Price:** {price}"},
                {"tag": "markdown", "content": f"**Stop Line:** {stop_line}"},
                {"tag": "markdown", "content": f"**Entry:** {entry_price}"},
            ]
        else:
            elements = [
                {"tag": "markdown", "content": f"**Direction:** {direction.upper()}"},
                {"tag": "markdown", "content": f"**Price:** {price}"},
            ]
        if timeframe and not is_trailing:
            elements.append({"tag": "markdown", "content": f"**Timeframe:** {timeframe}"})
        if stop_line:
            elements.append({"tag": "markdown", "content": f"**Stop Line:** {stop_line}"})
        if atr_upper and atr_lower and not is_trailing:
            elements.append(
                {
                    "tag": "markdown",
                    "content": f"**ATR Channel:** {atr_lower} ~ {atr_upper}",
                }
            )
        natr = extra.get("natr")
        if natr is not None and not is_trailing:
            natr_display = f"{float(natr):.2f}" if isinstance(natr, int | float) else render_value(natr)
            elements.append(
                {
                    "tag": "markdown",
                    "content": f"**NATR20:** {natr_display}%",
                }
            )
        if alert_type == "ClusterST":
            ts = _render_extra_value(extra, "ts")
            perf_ama = _render_extra_value(extra, "perf_ama")
            target_factor = _render_extra_value(extra, "target_factor")
            if ts:
                elements.append({"tag": "markdown", "content": f"**TS:** {ts}"})
            if perf_ama:
                elements.append({"tag": "markdown", "content": f"**perf_ama:** {perf_ama}"})
            if target_factor:
                elements.append(
                    {
                        "tag": "markdown",
                        "content": f"**target_factor:** {target_factor}",
                    }
                )
        elements.extend(
            [
                {"tag": "hr"},
                {"tag": "markdown", "content": f"**Trigger Time:** {timestamp}"},
            ]
        )
        title = f"{emoji} <{symbol}> {direction.upper()}"

    # SYSTEM
    elif alert_type == "SYSTEM":
        color = "blue"
        title = "\U0001f514 System"
        elements = [
            {"tag": "markdown", "content": f"**{message}**"},
            {"tag": "hr"},
            {"tag": "markdown", "content": f"**Trigger Time:** {timestamp}"},
        ]

    # ERROR
    elif alert_type == "ERROR":
        color = "red"
        title = "\u26a0\ufe0f Error"
        elements = [
            {"tag": "markdown", "content": f"**{message}**"},
            {"tag": "hr"},
            {"tag": "markdown", "content": f"**Trigger Time:** {timestamp}"},
        ]

    # CONFIG
    elif alert_type == "CONFIG":
        color = "purple"
        title = "\u2699\ufe0f Config"
        elements = [
            {"tag": "markdown", "content": f"**{message}**"},
            {"tag": "hr"},
            {"tag": "markdown", "content": f"**Trigger Time:** {timestamp}"},
        ]

    # CONFIG ERROR
    elif alert_type == "CONFIG ERROR":
        color = "red"
        title = "\u2699\ufe0f Config Error"
        elements = [
            {"tag": "markdown", "content": f"**{message}**"},
            {"tag": "hr"},
            {"tag": "markdown", "content": f"**Trigger Time:** {timestamp}"},
        ]

    # REPORT
    elif alert_type == "REPORT":
        color = "purple"
        title = "\U0001f4ca Daily Report"
        elements = [
            {"tag": "markdown", "content": f"**{message}**"},
            {"tag": "hr"},
            {"tag": "markdown", "content": f"**Trigger Time:** {timestamp}"},
        ]

    # BREAKOUT
    elif alert_type == "BREAKOUT":
        color = "orange"
        emoji = "\U0001f4a5"
        title = f"{emoji} {symbol}"
        confirmed = extra.get("confirmed", False)
        direction_disp = render_value(extra.get("direction", ""))
        confirmed_text = "CONFIRMED" if confirmed else "FALSE"
        elements = [
            {
                "tag": "markdown",
                "content": f"**Breakout:** {direction_disp} {confirmed_text}",
            },
            {"tag": "markdown", "content": f"**Price:** {extra.get('price', '')}"},
        ]
        if confirmed:
            elements.append(
                {
                    "tag": "markdown",
                    "content": f"**Trigger:** {render_value(extra.get('trigger', ''))}",
                }
            )
        else:
            elements.append({"tag": "markdown", "content": f"**Reason:** {render_value(extra.get('reason', ''))}"})
        elements.extend(
            [
                {"tag": "hr"},
                {"tag": "markdown", "content": f"**Trigger Time:** {timestamp}"},
            ]
        )

    # Default
    else:
        color = "blue"
        title = f"hypurr-monitor - {alert_type}"
        elements = [
            {"tag": "markdown", "content": f"**{message}**"},
            {"tag": "hr"},
            {"tag": "markdown", "content": f"**Trigger Time:** {timestamp}"},
        ]

    return {
        "header": {
            "title": {"tag": "plain_text", "content": title},
            "template": color,
        },
        "elements": elements,
    }


async def send_webhook(  # noqa: PLR0913
    webhook_url: str,
    webhook_format: str,
    alert_type: str,
    message: str,
    extra: dict[str, Any] | None = None,
    log_file_path: str = WEBHOOK_LOG_FILE,
    max_log_lines: int = 1000,
    get_timestamp_fn: Any = None,
    sender: WebhookSender | None = None,
) -> bool:
    """
    Send Feishu Webhook message.

    Flow:
    1. Append log to webhook.log (with auto-rotation)
    2. Build message body based on format (card or text)
    3. Send HTTP POST request to webhook_url
    4. Log success/error

    Args:
        webhook_url: Feishu Webhook URL
        webhook_format: "card" or "text"
        alert_type: Alert type
        message: Message content
        extra: Extended data dict
        max_log_lines: Log rotation threshold
        get_timestamp_fn: Timestamp getter function (optional, default returns empty)
    """
    event = build_alert_event(alert_type, message, extra)
    result = await send_alert_event(
        webhook_url,
        webhook_format,
        event,
        log_file_path=log_file_path,
        max_log_lines=max_log_lines,
        get_timestamp_fn=get_timestamp_fn,
        sender=sender,
    )
    return result.sent


def _build_log_message(event: AlertEvent) -> str:
    """Build operator-facing log line from structured alert event."""
    price = event.extra.get("price", "")
    atr_upper = event.extra.get("atr_upper", "")
    atr_lower = event.extra.get("atr_lower", "")
    stop_line = event.extra.get("stop_line", "")
    entry_price = event.extra.get("entry_price", "")
    reason = event.extra.get("reason", "")

    if event.alert_type == "SYSTEM":
        return f"[WEBHOOK] {event.message}"
    if reason == "trailing_stop":
        return f"[WEBHOOK] {event.message} | Price={price} | Stop={stop_line} | Entry={entry_price}"
    if event.alert_type == "ATR_Ch":
        return f"[WEBHOOK] {event.message} | Price={price} | Channel={atr_lower}~{atr_upper}"
    return f"[WEBHOOK] {event.message}"


async def send_alert_event(  # noqa: PLR0913
    webhook_url: str,
    webhook_format: str,
    event: AlertEvent,
    log_file_path: str = WEBHOOK_LOG_FILE,
    max_log_lines: int = 1000,
    get_timestamp_fn: Any = None,
    sender: WebhookSender | None = None,
) -> WebhookDeliveryResult:
    """Send a structured AlertEvent and return a structured delivery outcome."""
    timestamp = get_timestamp_fn() if get_timestamp_fn else ""
    full_content = f"[{timestamp}] [{event.alert_type}] {event.message}"
    logger = _get_logger()

    try:
        _rotate_webhook_log_if_needed(log_file_path, max_log_lines)
        with Path(log_file_path).open("a", encoding="utf-8") as f:
            f.write(f"{full_content}\n")
    except Exception as e:
        logger.warning(f"Write webhook log failed: {e}")

    logger.info(_build_log_message(event))

    if webhook_format == "card":
        card = build_feishu_card(event.alert_type, event.message, event.extra, timestamp)
        msg = {"msg_type": "interactive", "card": card}
    else:
        msg = {"msg_type": "text", "content": {"text": full_content}}

    if sender is not None:
        ok = await sender.send_json(webhook_url, msg)
        result = WebhookDeliveryResult(attempted=True, sent=ok, failed=not ok)
        logger.info(
            "Webhook delivery %s | alert_type=%s | symbol=%s | dedupe_key=%s",
            "success" if ok else "failed",
            event.alert_type,
            event.symbol or "",
            event.dedupe_key or "",
        )
        return result

    import aiohttp

    try:
        async with (
            aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10.0)) as session,
            session.post(webhook_url, json=msg, timeout=aiohttp.ClientTimeout(total=10.0)) as resp,
        ):
            if resp.status != WEBHOOK_SUCCESS_STATUS_CODE:
                log_error(f"Webhook failed: {resp.status}")
                logger.info(
                    "Webhook delivery failed | alert_type=%s | symbol=%s | dedupe_key=%s",
                    event.alert_type,
                    event.symbol or "",
                    event.dedupe_key or "",
                )
                return WebhookDeliveryResult(attempted=True, sent=False, failed=True)
    except Exception as e:
        log_error(f"Webhook error: {e}")
        logger.info(
            "Webhook delivery failed | alert_type=%s | symbol=%s | dedupe_key=%s",
            event.alert_type,
            event.symbol or "",
            event.dedupe_key or "",
        )
        return WebhookDeliveryResult(attempted=True, sent=False, failed=True)
    logger.info(
        "Webhook delivery success | alert_type=%s | symbol=%s | dedupe_key=%s",
        event.alert_type,
        event.symbol or "",
        event.dedupe_key or "",
    )
    return WebhookDeliveryResult(attempted=True, sent=True, failed=False)
