"""
Feishu Webhook notification module - builds message body and sends Webhook requests.

Main functions:
- log_warning / log_error: Log forwarding to standard logger
- _rotate_webhook_log_if_needed: Webhook log file auto-rotation
- build_feishu_card: Build Feishu Interactive Card format message
- send_webhook: Send Webhook request (supports card and text formats)
"""

import logging
from pathlib import Path
from typing import Any

WEBHOOK_LOG_FILE = "webhook.log"
WEBHOOK_SUCCESS_STATUS_CODE = 200


def _get_logger() -> logging.Logger:
    """Get logger instance."""
    return logging.getLogger(__name__)


def log_warning(msg: str) -> None:
    """Forward warning log to standard logger."""
    _get_logger().warning(msg)


def log_error(msg: str) -> None:
    """Forward error log to standard logger."""
    _get_logger().error(msg)


def _rotate_webhook_log_if_needed(max_log_lines: int = 1000) -> None:
    """
    If webhook.log exceeds max_log_lines, truncate to latest max_log_lines.

    Args:
        max_log_lines: Maximum lines to retain (default 1000)
    """
    try:
        log_path = Path(WEBHOOK_LOG_FILE)
        if not log_path.exists():
            return
        lines = log_path.read_text(encoding="utf-8").splitlines()
        if len(lines) > max_log_lines:
            log_path.write_text("\n".join(lines[-max_log_lines:]) + "\n", encoding="utf-8")
    except Exception as e:
        _get_logger().warning(f"Log rotation failed: {e}")


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

        price = extra.get("price", "")
        atr_upper = extra.get("atr_upper", "")
        atr_lower = extra.get("atr_lower", "")
        stop_line = extra.get("stop_line", "")
        entry_price = extra.get("entry_price", "")
        timeframe = extra.get("timeframe", "")

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
            elements.append(
                {
                    "tag": "markdown",
                    "content": f"**NATR20:** {natr:.2f}%",
                }
            )
        if alert_type == "ClusterST":
            ts = extra.get("ts", "")
            perf_ama = extra.get("perf_ama", "")
            target_factor = extra.get("target_factor", "")
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
        direction_disp = extra.get("direction", "")
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
                    "content": f"**Trigger:** {extra.get('trigger', '')}",
                }
            )
        else:
            elements.append({"tag": "markdown", "content": f"**Reason:** {extra.get('reason', '')}"})
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
    max_log_lines: int = 1000,
    get_timestamp_fn: Any = None,
) -> None:
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
    timestamp = get_timestamp_fn() if get_timestamp_fn else ""
    full_content = f"[{timestamp}] [{alert_type}] {message}"

    extra = extra or {}
    price = extra.get("price", "")
    atr_upper = extra.get("atr_upper", "")
    atr_lower = extra.get("atr_lower", "")
    stop_line = extra.get("stop_line", "")
    entry_price = extra.get("entry_price", "")
    reason = extra.get("reason", "")

    if alert_type == "SYSTEM":
        log_msg = f"[WEBHOOK] {message}"
    elif reason == "trailing_stop":
        log_msg = f"[WEBHOOK] {message} | Price={price} | Stop={stop_line} | Entry={entry_price}"
    elif alert_type == "ATR_Ch":
        log_msg = f"[WEBHOOK] {message} | Price={price} | Channel={atr_lower}~{atr_upper}"
    else:
        log_msg = f"[WEBHOOK] {message}"

    # Step 1: Write to log file
    try:
        _rotate_webhook_log_if_needed(max_log_lines)
        with Path(WEBHOOK_LOG_FILE).open("a", encoding="utf-8") as f:
            f.write(f"{full_content}\n")
    except Exception as e:
        _get_logger().warning(f"Write webhook log failed: {e}")

    # Step 2: Always log to console BEFORE sending (even if request fails)
    _get_logger().info(log_msg)

    # Step 3: Build message
    if webhook_format == "card":
        card = build_feishu_card(alert_type, message, extra or {}, timestamp)
        msg = {"msg_type": "interactive", "card": card}
    else:
        msg = {"msg_type": "text", "content": {"text": full_content}}

    # Step 4: Send HTTP request
    import aiohttp

    try:
        async with (
            aiohttp.ClientSession() as session,
            session.post(webhook_url, json=msg, timeout=aiohttp.ClientTimeout(total=10)) as resp,
        ):
            if resp.status != WEBHOOK_SUCCESS_STATUS_CODE:
                log_error(f"Webhook failed: {resp.status}")
    except Exception as e:
        log_error(f"Webhook error: {e}")
