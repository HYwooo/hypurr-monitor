"""
Technical indicator calculation module - pure functions, no I/O or network operations.

Contains:
- calculate_supertrend: Supertrend indicator (dual period)
- calculate_vegas_tunnel: Vegas Tunnel indicator (3 EMA)
- calculate_dema: Double Exponential Moving Average
- calculate_hma: Hull Moving Average
- calculate_tr: True Range
- calculate_atr: ATR (Average True Range), supports multiple MA types
- run_atr_channel: ATR Channel state machine (core trailing stop logic)
"""

import math
from typing import Any

import numpy as np
import talib


def calculate_supertrend(
    high: np.ndarray[Any, Any],
    low: np.ndarray[Any, Any],
    close: np.ndarray[Any, Any],
    period: int,
    multiplier: float,
) -> np.ndarray[Any, Any]:
    """
    Calculate Supertrend indicator.

    Args:
        high: High price array (np.array)
        low: Low price array (np.array)
        close: Close price array (np.array)
        period: ATR calculation period
        multiplier: ATR multiplier

    Returns:
        supertrend array, nan means invalid value
    """
    close_shifted = np.roll(close, 1)
    close_shifted[0] = close[0]
    tr1 = high - low
    tr2 = np.abs(high - close_shifted)
    tr3 = np.abs(low - close_shifted)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))

    atr = np.zeros_like(close)
    if len(tr) >= period:
        sma = np.mean(tr[1 : period + 1])
        for i in range(period, len(tr)):
            if np.isnan(atr[i - 1]):
                atr[i] = sma
            else:
                atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    hl_avg = (high + low) / 2
    upper_band = hl_avg + (multiplier * atr)
    lower_band = hl_avg - (multiplier * atr)

    supertrend = np.full_like(close, np.nan)
    direction = np.ones_like(close)
    first_valid = period
    supertrend[first_valid] = lower_band[first_valid]
    direction[first_valid] = 1

    for i in range(first_valid + 1, len(close)):
        if close[i] > upper_band[i - 1]:
            direction[i] = 1
        elif close[i] < lower_band[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]
        supertrend[i] = lower_band[i] if direction[i] == 1 else upper_band[i]
    return supertrend


def calculate_vegas_tunnel(
    close: np.ndarray[Any, Any],
    vt_ema_signal: int = 9,
    vt_ema_upper: int = 144,
    vt_ema_lower: int = 169,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """
    Calculate Vegas Tunnel indicator (3 EMA lines).

    Args:
        close: Close price array
        vt_ema_signal: Signal line period (default 9)
        vt_ema_upper: Upper EMA period (default 144)
        vt_ema_lower: Lower EMA period (default 169)

    Returns:
        (ema_signal, ema_upper, ema_lower) three EMA arrays
    """
    ema_signal = talib.EMA(close, timeperiod=vt_ema_signal)
    ema_upper = talib.EMA(close, timeperiod=vt_ema_upper)
    ema_lower = talib.EMA(close, timeperiod=vt_ema_lower)
    return ema_signal, ema_upper, ema_lower


def calculate_dema(data: np.ndarray[Any, Any], period: int) -> np.ndarray[Any, Any]:
    """
    Double Exponential Moving Average (DEMA).
    DEMA = 2*EMA(data,period) - EMA(EMA(data,period),period)
    """
    ema1 = talib.EMA(data, timeperiod=period)
    ema2 = talib.EMA(ema1, timeperiod=period)
    return 2 * ema1 - ema2


def calculate_hma(data: np.ndarray[Any, Any], period: int) -> np.ndarray[Any, Any]:
    """
    Hull Moving Average (HMA).
    HMA = WMA(2*WMA(data,period/2) - WMA(data,period), sqrt(period))
    """
    return talib.WMA(
        2 * talib.WMA(data, period // 2) - talib.WMA(data, period),
        int(math.sqrt(period)),
    )


def calculate_tr(
    high: np.ndarray[Any, Any], low: np.ndarray[Any, Any], close: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    """
    True Range (TR).
    TR = max(H-L, |H-PC|, |L-PC|), PC = previous close
    First bar has no previous close, TR = H-L.
    """
    prev_close = np.empty_like(close)
    prev_close[0] = close[0]
    prev_close[1:] = close[:-1]
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    return np.maximum(tr1, np.maximum(tr2, tr3))  # type: ignore[no-any-return]


def _wilder_rma(tr: np.ndarray[Any, Any], period: int) -> np.ndarray[Any, Any]:
    """
    Wilder's RMA (Relative Moving Average).
    Equivalent to EMA(alpha=1/period), i.e. Wilder's smoothing.
    Initial values use SMA of first period TR.
    """
    n = len(tr)
    rma = np.zeros(n, dtype=float)
    if n == 0:
        return rma
    alpha = 1.0 / period
    cum = 0.0
    for i in range(n):
        cum += float(tr[i])
        if i < period - 1:
            rma[i] = cum / (i + 1)
        else:
            rma[i] = alpha * float(tr[i]) + (1 - alpha) * rma[i - 1]
    return rma


def calculate_atr(
    high: np.ndarray[Any, Any],
    low: np.ndarray[Any, Any],
    close: np.ndarray[Any, Any],
    period: int,
    ma_type: str = "DEMA",
) -> np.ndarray[Any, Any]:
    """
    Average True Range (ATR), supports multiple MA types.

    Args:
        high, low, close: Price arrays
        period: ATR period
        ma_type: MA type, DEMA | HMA | EMA | SMA | WMA | RMA (default DEMA)
                 RMA = Wilder's RMA (equivalent to TradingView ta.rma)

    Returns:
        ATR array
    """
    tr = calculate_tr(high, low, close)
    if ma_type in ("DEMA", "EMA", "WMA", "SMA", "HMA"):
        ma_lower = ma_type.lower()
        if ma_lower == "dema":
            atr = calculate_dema(tr, period)
        elif ma_lower == "hma":
            atr = calculate_hma(tr, period)
        elif ma_lower == "ema":
            atr = talib.EMA(tr, period)
        elif ma_lower == "sma":
            atr = talib.SMA(tr, period)
        elif ma_lower == "wma":
            atr = talib.WMA(tr, period)
        else:
            atr = _wilder_rma(tr, period)
    elif ma_type in ("RMA", "RMA (Standard ATR)"):
        atr = _wilder_rma(tr, period)
    else:
        atr = _wilder_rma(tr, period)
    return atr


def run_atr_channel(
    close: float, atr: float, mult: float, prev_state: tuple[float, float, int]
) -> tuple[float, float, int]:
    """
    ATR Channel state machine - updates channel upper/lower bands based on current price and ATR.

    Logic:
    - Initial: channel centered at current price, width = atr*mult
    - Price breaks above upper band: upper updates to current price, lower = max(old lower, new lower) (only rises, never falls)
    - Price breaks below lower band: lower updates to current price, upper = min(old upper, new upper) (only falls, never rises)

    Args:
        close: Current close price (scalar)
        atr: Current ATR value (scalar)
        mult: ATR multiplier (for channel width)
        prev_state: Previous state (prev_upper, prev_lower, prev_ch)
                    prev_ch: 0=neutral, 1=long channel, -1=short channel

    Returns:
        (upper_band, lower_band, ch_state) new state
    """
    upper_band, lower_band, ch_state = prev_state
    if math.isnan(atr) or atr <= 0:
        return upper_band, lower_band, ch_state

    # Initial state: channel centered at current price
    if math.isnan(upper_band) or math.isnan(lower_band):
        width = atr * mult
        upper_band = close + width / 2
        lower_band = close - width / 2
        ch_state = 0
    # Price breaks above upper band -> long channel (trailing stop rises)
    elif close > upper_band:
        width = atr * mult
        new_upper = close
        new_lower = close - width
        lower_band = max(lower_band, new_lower)
        upper_band = new_upper
        ch_state = 1
    # Price breaks below lower band -> short channel (trailing stop falls)
    elif close < lower_band:
        width = atr * mult
        new_lower = close
        new_upper = close + width
        upper_band = min(upper_band, new_upper)
        lower_band = new_lower
        ch_state = -1
    return upper_band, lower_band, ch_state
