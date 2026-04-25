"""
Breakout monitoring module - monitors price breakout confirmation signals.

Responsibilities:
- start_breakout_monitor: Start monitoring breakout 15m K-lines
- _on_15m_kline: Process 15m K-line data (internal)
- check_breakout: Detect if breakout is confirmed or failed
"""

import logging
from typing import Any

from notifications import (
    ALERT_BREAKOUT,
    BREAKOUT_CONFIRMED,
    BREAKOUT_FALSE_NO_CONTINUATION,
    BREAKOUT_FALSE_REVERSE,
    DIRECTION_LONG,
    DIRECTION_SHORT,
    REASON_NO_CONTINUATION,
    REASON_REVERSE,
    emit_alert,
    format_breakout_message,
)
from signals.state import BreakoutMonitorState, get_breakout_monitor_state, set_breakout_monitor_state

MIN_KLINES_FOR_BREAKOUT = 2
MAX_KLINE_MONITOR_COUNT = 20
MIN_PAIR_PARTS = 2

logger = logging.getLogger(__name__)


def _split_pair_symbol(symbol: str) -> tuple[str, str] | None:
    """Split pair symbols supporting both legacy ':' and runtime '-' separators."""
    if "-" in symbol:
        left, right = symbol.split("-", 1)
        return left, right
    parts = symbol.split(":")
    if len(parts) >= MIN_PAIR_PARTS:
        return parts[0], parts[1]
    return None


async def start_breakout_monitor(  # noqa: PLR0913
    symbol: str,
    direction: str,
    price: float,
    trigger_time: float,
    breakout_monitor: dict[str, Any],
    is_pair_trading: bool,
    breakout_comp_prices: dict[str, Any],
    _ws_client: Any,
    _update_15m_atr_fn: Any,
    fetch_pair_klines_fn: Any = None,
    proxy: str | None = None,
) -> None:
    """Start monitoring breakout for specified trading pair."""
    if symbol in breakout_monitor:
        return

    _ = breakout_comp_prices
    from hyperliquid.rest_client import HyperliquidREST

    client = HyperliquidREST(proxy=proxy)
    try:
        if is_pair_trading:
            history = await (fetch_pair_klines_fn or client.fetch_klines)(symbol, interval="15m", limit=20)
        else:
            history = await client.fetch_klines(symbol, interval="15m", limit=20)
    except Exception:
        logger.exception("[start_breakout_monitor] symbol=%s stage=history_fetch", symbol)
        raise
    finally:
        await client.close()

    if not history:
        return
    set_breakout_monitor_state(
        breakout_monitor,
        symbol,
        BreakoutMonitorState(
        direction=direction,
        trigger_price=price,
        trigger_time=trigger_time,
        klines_15m=history,
        ),
    )
    if is_pair_trading:
        _ = _split_pair_symbol(symbol)
        # Design note: keep breakout state isolated from runtime price caches.


async def check_breakout(  # noqa: PLR0912, PLR0913, PLR0915
    symbol: str,
    breakout_monitor: dict[str, Any],
    send_webhook_fn: Any,
    increment_alert_count_fn: Any,
    stop_breakout_monitor_fn: Any = None,
    send_event_fn: Any = None,
) -> None:
    """Detect if breakout is confirmed or failed."""
    try:
        monitor = get_breakout_monitor_state(breakout_monitor, symbol)
        if not monitor:
            return
        direction = monitor.direction
        trigger_price = monitor.trigger_price
        klines = monitor.klines_15m
        count = monitor.kline_15m_count

        def deactivate() -> None:
            """Retain compatibility while leaving cleanup to the caller."""
            return
        if len(klines) < MIN_KLINES_FOR_BREAKOUT:
            return

        # Design note: breakout confirmation must use the latest completed 15m close, not intrabar high/low.
        latest_close = float(klines[-1].close)
        prev_closes = [float(k.close) for k in klines[:-1]]
        max_prev = max(prev_closes) if prev_closes else 0
        min_prev = min(prev_closes) if prev_closes else float("inf")

        if direction == "11":
            if latest_close > max_prev:
                await emit_alert(send_webhook_fn, ALERT_BREAKOUT, format_breakout_message(symbol, DIRECTION_LONG, BREAKOUT_CONFIRMED), {
                    "symbol": symbol, "direction": DIRECTION_LONG, "confirmed": True,
                    "price": latest_close, "trigger": trigger_price,
                }, send_event_fn)
                increment_alert_count_fn()
                if stop_breakout_monitor_fn:
                    await stop_breakout_monitor_fn(symbol)
                deactivate()
            elif latest_close < min_prev:
                await emit_alert(send_webhook_fn, ALERT_BREAKOUT, format_breakout_message(symbol, DIRECTION_LONG, BREAKOUT_FALSE_REVERSE), {
                    "symbol": symbol, "direction": DIRECTION_LONG, "confirmed": False,
                    "reason": REASON_REVERSE, "price": latest_close,
                }, send_event_fn)
                increment_alert_count_fn()
                if stop_breakout_monitor_fn:
                    await stop_breakout_monitor_fn(symbol)
                deactivate()
            elif count >= MAX_KLINE_MONITOR_COUNT:
                await emit_alert(send_webhook_fn, ALERT_BREAKOUT, format_breakout_message(symbol, DIRECTION_LONG, BREAKOUT_FALSE_NO_CONTINUATION), {
                    "symbol": symbol, "direction": DIRECTION_LONG, "confirmed": False,
                    "reason": REASON_NO_CONTINUATION, "price": latest_close,
                }, send_event_fn)
                increment_alert_count_fn()
                if stop_breakout_monitor_fn:
                    await stop_breakout_monitor_fn(symbol)
                deactivate()
        elif direction == "00":
            if latest_close < min_prev:
                await emit_alert(send_webhook_fn, ALERT_BREAKOUT, format_breakout_message(symbol, DIRECTION_SHORT, BREAKOUT_CONFIRMED), {
                    "symbol": symbol, "direction": DIRECTION_SHORT, "confirmed": True,
                    "price": latest_close, "trigger": trigger_price,
                }, send_event_fn)
                increment_alert_count_fn()
                if stop_breakout_monitor_fn:
                    await stop_breakout_monitor_fn(symbol)
                deactivate()
            elif latest_close > max_prev:
                await emit_alert(send_webhook_fn, ALERT_BREAKOUT, format_breakout_message(symbol, DIRECTION_SHORT, BREAKOUT_FALSE_REVERSE), {
                    "symbol": symbol, "direction": DIRECTION_SHORT, "confirmed": False,
                    "reason": REASON_REVERSE, "price": latest_close,
                }, send_event_fn)
                increment_alert_count_fn()
                if stop_breakout_monitor_fn:
                    await stop_breakout_monitor_fn(symbol)
                deactivate()
            elif count >= MAX_KLINE_MONITOR_COUNT:
                await emit_alert(send_webhook_fn, ALERT_BREAKOUT, format_breakout_message(symbol, DIRECTION_SHORT, BREAKOUT_FALSE_NO_CONTINUATION), {
                    "symbol": symbol, "direction": DIRECTION_SHORT, "confirmed": False,
                    "reason": REASON_NO_CONTINUATION, "price": latest_close,
                }, send_event_fn)
                increment_alert_count_fn()
                if stop_breakout_monitor_fn:
                    await stop_breakout_monitor_fn(symbol)
                deactivate()
    except Exception:
        logger.error("[check_breakout] symbol=%s stage=breakout_check", symbol, exc_info=True)
