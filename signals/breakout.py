"""
Breakout monitoring module - monitors price breakout confirmation signals.

Responsibilities:
- start_breakout_monitor: Start monitoring breakout 15m K-lines
- _on_15m_kline: Process 15m K-line data (internal)
- check_breakout: Detect if breakout is confirmed or failed
"""

from typing import Any

from notifications import format_number

MIN_KLINES_FOR_BREAKOUT = 2
MAX_KLINE_MONITOR_COUNT = 20


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
    """
    Start monitoring breakout for specified trading pair.

    Monitoring logic (LONG direction="11" as example):
    - Fetch last 20 15m K-lines as baseline
    - Wait for next 15m K-line to complete:
        - If new close > highest of previous 20 bars -> confirm breakout, push LONG CONFIRMED
        - If new close < lowest of previous 20 bars -> false breakout, push LONG FALSE (REVERSE)
        - If no breakout within 20 bars -> expire, push LONG FALSE (NO_CONTINUATION)
    - SHORT (direction="00") logic is symmetric

    Args:
        symbol: Trading pair name
        direction: Breakout direction, "11"=LONG, "00"=SHORT
        price: Trigger price
        trigger_time: Trigger timestamp
        breakout_monitor: Breakout monitoring state dict (will be written)
        is_pair_trading: Whether this is pair trading
        breakout_comp_prices: Pair trading component price cache
        ws_client: WebSocket client instance (for registering callbacks)
        update_15m_atr_fn: Async callback to trigger 15m ATR update
        fetch_pair_klines_fn: Async callback to fetch pair K-lines (optional)
        proxy: HTTP proxy
    """
    if symbol in breakout_monitor:
        return

    from hyperliquid.rest_client import HyperliquidREST

    client = HyperliquidREST(proxy=proxy)
    try:
        if is_pair_trading:
            history = await (fetch_pair_klines_fn or client.fetch_klines)(symbol, interval="15m", limit=20)
        else:
            history = await client.fetch_klines(symbol, interval="15m", limit=20)
    finally:
        await client.close()

    if not history:
        return

    breakout_monitor[symbol] = {
        "direction": direction,
        "trigger_price": price,
        "trigger_time": trigger_time,
        "kline_15m_count": 0,
        "klines_15m": history,
    }

    if is_pair_trading:
        parts = symbol.split(":")
        breakout_comp_prices[parts[0]] = 0
        breakout_comp_prices[parts[1]] = 0


async def check_breakout(  # noqa: PLR0912
    symbol: str,
    breakout_monitor: dict[str, Any],
    send_webhook_fn: Any,
    increment_alert_count_fn: Any,
    stop_breakout_monitor_fn: Any = None,
) -> None:
    """
    Detect if breakout is confirmed or failed.

    LONG (direction="11"):
    - Confirm: new close > highest of previous 20 bars
    - False breakout (reverse): new close < lowest of previous 20 bars
    - Expire (no continuation): no confirmation within 20 bars

    SHORT (direction="00"): symmetric logic

    Args:
        symbol: Trading pair name
        breakout_monitor: Breakout monitoring state dict
        send_webhook_fn: Async callback to send Webhook
        increment_alert_count_fn: Increment alert count
        stop_breakout_monitor_fn: Async callback to stop monitoring (optional)
    """
    monitor = breakout_monitor.get(symbol)
    if not monitor:
        return
    direction = monitor["direction"]
    trigger_price = monitor["trigger_price"]
    klines = monitor["klines_15m"]
    count = monitor["kline_15m_count"]
    if len(klines) < MIN_KLINES_FOR_BREAKOUT:
        return

    current_close = klines[-1].high
    prev_closes = [k.high for k in klines[:-1]]
    max_prev = max(prev_closes) if prev_closes else 0
    min_prev = min(prev_closes) if prev_closes else float("inf")

    if direction == "11":
        if current_close > max_prev:
            await send_webhook_fn(
                "BREAKOUT",
                f"{symbol} LONG CONFIRMED",
                {
                    "symbol": symbol,
                    "direction": "LONG",
                    "confirmed": True,
                    "price": format_number(current_close),
                    "trigger": format_number(trigger_price),
                },
            )
            increment_alert_count_fn()
            if stop_breakout_monitor_fn:
                await stop_breakout_monitor_fn(symbol)

        elif current_close < min_prev:
            await send_webhook_fn(
                "BREAKOUT",
                f"{symbol} LONG FALSE (REVERSE)",
                {
                    "symbol": symbol,
                    "direction": "LONG",
                    "confirmed": False,
                    "reason": "reverse",
                    "price": format_number(current_close),
                },
            )
            increment_alert_count_fn()
            if stop_breakout_monitor_fn:
                await stop_breakout_monitor_fn(symbol)

        elif count >= MAX_KLINE_MONITOR_COUNT:
            await send_webhook_fn(
                "BREAKOUT",
                f"{symbol} LONG FALSE (NO_CONTINUATION)",
                {
                    "symbol": symbol,
                    "direction": "LONG",
                    "confirmed": False,
                    "reason": "no_continuation",
                    "price": format_number(current_close),
                },
            )
            increment_alert_count_fn()
            if stop_breakout_monitor_fn:
                await stop_breakout_monitor_fn(symbol)

    elif direction == "00":
        if current_close < min_prev:
            await send_webhook_fn(
                "BREAKOUT",
                f"{symbol} SHORT CONFIRMED",
                {
                    "symbol": symbol,
                    "direction": "SHORT",
                    "confirmed": True,
                    "price": format_number(current_close),
                    "trigger": format_number(trigger_price),
                },
            )
            increment_alert_count_fn()
            if stop_breakout_monitor_fn:
                await stop_breakout_monitor_fn(symbol)

        elif current_close > max_prev:
            await send_webhook_fn(
                "BREAKOUT",
                f"{symbol} SHORT FALSE (REVERSE)",
                {
                    "symbol": symbol,
                    "direction": "SHORT",
                    "confirmed": False,
                    "reason": "reverse",
                    "price": format_number(current_close),
                },
            )
            increment_alert_count_fn()
            if stop_breakout_monitor_fn:
                await stop_breakout_monitor_fn(symbol)

        elif count >= MAX_KLINE_MONITOR_COUNT:
            await send_webhook_fn(
                "BREAKOUT",
                f"{symbol} SHORT FALSE (NO_CONTINUATION)",
                {
                    "symbol": symbol,
                    "direction": "SHORT",
                    "confirmed": False,
                    "reason": "no_continuation",
                    "price": format_number(current_close),
                },
            )
            increment_alert_count_fn()
            if stop_breakout_monitor_fn:
                await stop_breakout_monitor_fn(symbol)
