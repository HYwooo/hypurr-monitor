"""Alert type constants and legacy message helpers."""

ALERT_ATR_CHANNEL = "ATR_Ch"
ALERT_CLUSTER_ST = "ClusterST"
ALERT_BREAKOUT = "BREAKOUT"
ALERT_SYSTEM = "SYSTEM"
ALERT_ERROR = "ERROR"
ALERT_CONFIG = "CONFIG"
ALERT_CONFIG_ERROR = "CONFIG ERROR"
ALERT_REPORT = "REPORT"

DIRECTION_LONG = "LONG"
DIRECTION_SHORT = "SHORT"

REASON_TRAILING_STOP = "trailing_stop"
REASON_REVERSE = "reverse"
REASON_NO_CONTINUATION = "no_continuation"

BREAKOUT_CONFIRMED = "CONFIRMED"
BREAKOUT_FALSE_REVERSE = "FALSE (REVERSE)"
BREAKOUT_FALSE_NO_CONTINUATION = "FALSE (NO_CONTINUATION)"


def format_directional_signal_message(symbol: str, direction: str, timeframe: str | None = None) -> str:
    """Format legacy directional signal messages with optional timeframe prefix."""
    prefix = f"{timeframe} " if timeframe else ""
    return f"[{symbol}] {prefix}{direction}"


def format_trailing_stop_message(symbol: str) -> str:
    """Format legacy trailing-stop signal message."""
    return f"[{symbol}] TRAILING STOP"


def format_breakout_message(symbol: str, direction: str, outcome: str) -> str:
    """Format legacy breakout notification message."""
    return f"{symbol} {direction} {outcome}"


def format_connection_success_message(exchange_id: str) -> str:
    """Format websocket connection success message."""
    return f"hypurr-monitor connected to {exchange_id}"


def format_connection_failed_message(error: object) -> str:
    """Format websocket connection failure message."""
    return f"Connection failed: {error}"


def format_ws_reconnect_failure_message(reason: str) -> str:
    """Format websocket reconnect failure message."""
    return f"Hyperliquid WS disconnected: {reason}. Reconnecting..."


def format_ws_reconnect_success_message(reason: str, attempt: int) -> str:
    """Format websocket reconnect success message."""
    return f"Hyperliquid WS reconnected after {reason} (attempt {attempt})"


def format_ws_data_silence_message(silence_seconds: int) -> str:
    """Format market-data silence message."""
    return f"Hyperliquid market data silent for {silence_seconds}s. Reconnecting..."


def format_ws_data_resumed_message(silence_duration: int) -> str:
    """Format market-data resumed message."""
    return f"Hyperliquid market data resumed after {silence_duration}s silence"
