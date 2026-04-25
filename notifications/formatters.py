"""Notification formatting module - pure formatting utilities, no network or file I/O."""

import math
from typing import Any


def format_number(value: float) -> str:
    """
    Format floating point number to readable string.

    Rules:
    - Non-finite values (inf/nan) return original string
    - Zero returns "0.00000000"
    - Absolute value < 1: keep 8 decimal places
    - Absolute value >= 1: integer digits + decimal digits = 8 (trailing zeros omitted)

    Examples:
        0.00001234  -> "0.00001234"
        1.5         -> "1.5"
        123456.789  -> "123456.79"
        0           -> "0.00000000"
    """
    if not math.isfinite(value):
        return str(value)
    if value == 0:
        return "0.00000000"
    abs_val = abs(value)
    if abs_val < 1:
        return f"{value:.8f}"
    int_digits = len(str(int(abs_val)))
    decimal_places = max(0, 8 - int_digits)
    formatted = f"{value:.{decimal_places}f}"
    return formatted.rstrip("0").rstrip(".")


def render_value(value: Any) -> str:
    """Render a payload value for webhook output without forcing upstream stringification."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "YES" if value else "NO"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return format_number(value)
    return str(value)
