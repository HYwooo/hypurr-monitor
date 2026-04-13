"""
Signal detection module - core trading signal logic.

Responsibilities:
- fetch_pair_klines: Fetch and merge pair trading K-lines
- update_klines: Fetch latest K-lines and trigger state recalculation
- recalculate_states: Recalculate all technical indicators (Supertrend, Vegas, ATR Channel)
- check_signals: ATR Channel breakout signal detection entry
- check_signals_impl: ATR Channel breakout signal detection implementation
- check_trailing_stop: Trailing stop trigger check
"""

import logging
import math
import time
from contextlib import suppress
from typing import Any

import numpy as np

from indicators import (
    calculate_atr,
    calculate_supertrend,
    calculate_vegas_tunnel,
    clustering_supertrend,
    run_atr_channel,
)
from notifications import format_number

logger = logging.getLogger(__name__)

MIN_KLINES = 200
STALE_PRICE_THRESHOLD = 300
SIGNAL_COOLDOWN = 3600

PRECISION_EPSILON = 1e-9


def price_ge(a: float, b: float) -> bool:
    """Safe greater-than-or-equal comparison for prices (IEEE 754 tolerance)."""
    return a > b + PRECISION_EPSILON or math.isclose(a, b, rel_tol=PRECISION_EPSILON)


def price_le(a: float, b: float) -> bool:
    """Safe less-than-or-equal comparison for prices (IEEE 754 tolerance)."""
    return a < b - PRECISION_EPSILON or math.isclose(a, b, rel_tol=PRECISION_EPSILON)


def price_gt(a: float, b: float) -> bool:
    """Safe greater-than comparison for prices (IEEE 754 tolerance)."""
    return a > b + PRECISION_EPSILON


def price_lt(a: float, b: float) -> bool:
    """Safe less-than comparison for prices (IEEE 754 tolerance)."""
    return a < b - PRECISION_EPSILON


async def fetch_pair_klines(  # noqa: PLR0913, PLR0912
    symbol: str,
    limit: int = 500,
    interval: str = "1h",
    proxy: str | None = None,
    kline_cache: dict[str, Any] | None = None,
    fetch_klines_fn: Any | None = None,
    exchange_id: str = "binance",  # noqa: ARG001
) -> list[Any]:
    """
    Fetch and merge pair trading K-lines.

    Pair trading (e.g. BTCUSDT:ETHUSDT) price = BTC price / ETH price.
    Align two trading pairs K-lines by timestamp and merge into one pair K-line:
    - open = ratio_open = open1 / open2
    - high = max(ratio_open, ratio_close)
    - low = min(ratio_open, ratio_close)
    - close = ratio_close = close1 / close2

    Args:
        symbol: Pair trading symbol, e.g. "BTCUSDT:ETHUSDT"
        limit: K-line count
        interval: K-line interval
        proxy: HTTP proxy
        kline_cache: Optional K-line cache dict {symbol: klines}. If provided,优先从 cache 读取
        fetch_klines_fn: Optional REST fetch function
        exchange_id: Exchange ID (deprecated, unused)

    Returns:
        Merged K-line list, format: [timestamp, ratio_open, ratio_high, ratio_low, ratio_close, volume]
    """
    parts = symbol.split(":")
    if fetch_klines_fn is None:
        from hyperliquid.rest_client import HyperliquidREST

        client = HyperliquidREST(proxy=proxy)
        try:
            klines1 = await client.fetch_klines(parts[0], interval=interval, limit=limit)
            klines2 = await client.fetch_klines(parts[1], interval=interval, limit=limit)
        finally:
            await client.close()
    else:
        klines1 = await fetch_klines_fn(parts[0], limit=limit, interval=interval, proxy=proxy)
        klines2 = await fetch_klines_fn(parts[1], limit=limit, interval=interval, proxy=proxy)

    if kline_cache is not None:
        if klines1:
            kline_cache[parts[0]] = klines1
        if klines2:
            kline_cache[parts[1]] = klines2
    elif fetch_klines_fn is None:
        if klines1 and parts[0] not in kline_cache:  # type: ignore[operator]
            kline_cache[parts[0]] = klines1  # type: ignore[index]
        if klines2 and parts[1] not in kline_cache:  # type: ignore[operator]
            kline_cache[parts[1]] = klines2  # type: ignore[index]

    if not klines1 or not klines2:
        return []

    k2_by_time = {int(k.open_time): k for k in klines2}
    merged = []
    for k1 in klines1:
        t = int(k1.open_time)
        if t not in k2_by_time:
            continue
        k2 = k2_by_time[t]
        o1, c1 = float(k1.open), float(k1.close)
        h1, l1 = float(k1.high), float(k1.low)
        o2, c2 = float(k2.open), float(k2.close)
        h2, l2 = float(k2.high), float(k2.low)
        if o2 == 0 or c2 == 0 or h2 == 0 or l2 == 0:
            continue
        ratio_o = o1 / o2
        ratio_c = c1 / c2
        ratio_h = h1 / ((h2 + l2) / 2)
        ratio_l = l1 / ((h2 + l2) / 2)
        merged.append(
            [
                t,
                ratio_o,
                max(ratio_o, ratio_c, ratio_h),
                min(ratio_o, ratio_c, ratio_l),
                ratio_c,
                float(k1.volume),
            ]
        )
    return sorted(merged, key=lambda x: x[0])


async def update_klines(  # noqa: PLR0913, PLR0912
    symbol: str,
    kline_cache: dict[str, Any],
    last_kline_time: dict[str, Any],
    is_pair_trading_fn: Any,
    proxy: str | None = None,
    recalculate_states_fn: Any | None = None,
    fetch_pair_klines_fn: Any | None = None,
    fetch_klines_fn: Any | None = None,
    recalculate_states_clustering_fn: Any | None = None,
    exchange_id: str = "binance",
) -> None:
    """
    Fetch latest K-lines for specified trading pair and trigger indicator recalculation.

    Args:
        symbol: Trading pair name
        kline_cache: K-line cache dict {symbol: klines}
        last_kline_time: Last K-line timestamp dict {symbol: unix_ms}
        is_pair_trading_fn: Function to check if pair trading fn(symbol) -> bool
        proxy: HTTP proxy
        recalculate_states_fn: Async callback for SINGLE indicator recalculation
        fetch_pair_klines_fn: Async callback for pair K-lines (optional)
        fetch_klines_fn: Async callback for single K-line fetch (optional)
        recalculate_states_clustering_fn: Async callback for PAIR Clustering recalculation (optional)
        exchange_id: Exchange ID (deprecated, unused)
    """
    try:
        _ = exchange_id
        is_pair = is_pair_trading_fn(symbol)
        if is_pair:
            klines = await (fetch_pair_klines_fn or fetch_pair_klines)(
                symbol,
                proxy=proxy,
                kline_cache=kline_cache,
            )
            if klines:
                kline_cache[symbol] = klines
                last_time = last_kline_time.get(symbol, 0)
                new_time = int(klines[-1].open_time)
                if new_time > last_time:
                    last_kline_time[symbol] = new_time
                    if recalculate_states_clustering_fn:
                        await recalculate_states_clustering_fn(symbol)
        else:
            from hyperliquid.rest_client import (
                get_cached_klines,
                update_cache,
            )

            interval = "1h"
            cached = get_cached_klines(symbol, interval)
            if cached and len(cached) >= 200:  # noqa: PLR2004
                klines = cached
                logger.debug(f"[update_klines] {symbol} cache hit, using {len(klines)} cached klines")
            elif fetch_klines_fn:
                klines = await fetch_klines_fn(symbol, proxy=proxy)
            else:
                from hyperliquid.rest_client import HyperliquidREST

                client = HyperliquidREST(proxy=proxy)
                try:
                    klines = await client.fetch_klines(symbol, interval=interval, limit=500)
                finally:
                    await client.close()
                if klines:
                    update_cache(symbol, interval, klines)
            if klines:
                kline_cache[symbol] = klines
                last_time = last_kline_time.get(symbol, 0)
                new_time = int(klines[-1].open_time)
                if new_time > last_time:
                    last_kline_time[symbol] = new_time
                    if recalculate_states_fn:
                        await recalculate_states_fn(symbol)
    except Exception:
        logger.warning(f"[update_klines] {symbol} fetch failed or timeout, skipping")


async def recalculate_states(  # noqa: PLR0913
    symbol: str,
    kline_cache: dict[str, Any],
    benchmark: dict[str, Any],
    is_pair_trading: bool,
    st_period1: int,
    st_multiplier1: float,
    st_period2: int,
    st_multiplier2: float,
    vt_ema_signal: int,
    vt_ema_upper: int,
    vt_ema_lower: int,
    atr1h_period: int,
    atr1h_ma_type: str,
    atr1h_mult: float,
    _atr15m_period: int,
    _atr15m_ma_type: str,
    _atr15m_mult: float,
    debug: bool = False,
) -> None:
    """
    Recalculate all technical indicators for specified trading pair and store in benchmark dict.

    Calculates:
    1. Supertrend (dual period: period1+multiplier1, period2+multiplier2)
    2. Vegas Tunnel (3 EMA lines)
    3. ATR Channel (1h ATR, upper/lower/state)

    Args:
        symbol: Trading pair name
        kline_cache: K-line cache
        benchmark: Indicator cache dict, recalculation results written here
        is_pair_trading: Whether this is a pair trading symbol
        st_period1/multiplier1: Supertrend period 1 parameters
        st_period2/multiplier2: Supertrend period 2 parameters
        vt_ema_signal/upper/lower: Vegas Tunnel EMA parameters
        atr1h_period/ma_type/mult: 1h ATR Channel parameters
        atr15m_period/ma_type/mult: 15m ATR Channel parameters (for trailing stop)
    """
    _ = is_pair_trading
    klines = kline_cache.get(symbol, [])
    if len(klines) < MIN_KLINES:
        return
    with suppress(Exception):
        close = np.array([float(k.close) for k in klines], dtype=float)
        high = np.array([float(k.high) for k in klines], dtype=float)
        low = np.array([float(k.low) for k in klines], dtype=float)

        if debug:
            invalid_mask = (high == 0) | (low == 0) | (close == 0) | np.isnan(high) | np.isnan(low) | np.isnan(close)
            invalid_count = np.sum(invalid_mask)
            if invalid_count > 0:
                logger.debug(f"{symbol} | invalid_bars={invalid_count}")
            else:
                logger.debug(
                    f"{symbol} | clean count={len(klines)} | "
                    f"H_range=[{high.min():.1f},{high.max():.1f}] | "
                    f"L_range=[{low.min():.1f},{low.max():.1f}] | "
                    f"C_range=[{close.min():.1f},{close.max():.1f}]"
                )

        st1 = calculate_supertrend(high, low, close, st_period1, st_multiplier1)
        st2 = calculate_supertrend(high, low, close, st_period2, st_multiplier2)

        ema_s, ema_u, ema_l = calculate_vegas_tunnel(close, vt_ema_signal, vt_ema_upper, vt_ema_lower)

        atr1h = calculate_atr(high, low, close, atr1h_period, atr1h_ma_type)
        atr1h_natrr = calculate_atr(high, low, close, 20, "RMA (Standard ATR)")
        prev_atr_state = benchmark.get(symbol, {}).get("atr1h_state", (float("nan"), float("nan"), 0))
        for i in range(len(close)):
            upper, lower, ch = run_atr_channel(close[i], atr1h[i], atr1h_mult, prev_atr_state)
            prev_atr_state = (upper, lower, ch)
        atr1h_upper, atr1h_lower, atr1h_ch = prev_atr_state

        if debug:
            last_close = float(close[-1])
            last_atr = float(atr1h[-1]) if math.isfinite(atr1h[-1]) else 0
            logger.debug(
                f"{symbol} | close={last_close:.4f} | ATR({atr1h_period},{atr1h_ma_type})={last_atr:.4f} | "
                f"mult={atr1h_mult:.3f} | upper={atr1h_upper:.4f} | lower={atr1h_lower:.4f} | "
                f"ch={atr1h_ch} | width={atr1h_upper - atr1h_lower:.4f}"
            )

        st1_val, st2_val = float(st1[-1]), float(st2[-1])
        ema_s_val, ema_u_val, ema_l_val = (
            float(ema_s[-1]),
            float(ema_u[-1]),
            float(ema_l[-1]),
        )
        if not all(math.isfinite(v) for v in [st1_val, st2_val, ema_s_val, ema_u_val, ema_l_val]):
            return

        benchmark[symbol] = {
            "st1": st1_val,
            "st2": st2_val,
            "ema_s": ema_s_val,
            "ema_u": ema_u_val,
            "ema_l": ema_l_val,
            "kline_time": int(klines[-1].open_time),
            "atr1h_upper": float(atr1h_upper) if not math.isnan(atr1h_upper) else 0,
            "atr1h_lower": float(atr1h_lower) if not math.isnan(atr1h_lower) else 0,
            "atr1h_ch": atr1h_ch,
            "atr1h_state": prev_atr_state,
            "atr1h_raw": float(atr1h[-1]) if math.isfinite(atr1h[-1]) else 0,
            "atr1h_natrr": float(atr1h_natrr[-1]) if math.isfinite(atr1h_natrr[-1]) else 0,
        }


async def check_signals(  # noqa: PLR0913
    symbol: str,
    mark_prices: dict[str, Any],
    mark_price_times: dict[str, Any],
    benchmark: dict[str, Any],
    trailing_stop: dict[str, Any],
    last_atr_state: dict[str, Any],
    last_alert_time: dict[str, Any],
    _initialized: bool,
    last_st_state: dict[str, Any],
    _atr1h_ma_type: str,
    _atr1h_period: int,
    _atr1h_mult: float,
    _atr15m_ma_type: str,
    _atr15m_period: int,
    _atr15m_mult: float,
    send_webhook_fn: Any,
    increment_alert_count_fn: Any,
) -> None:
    """
    ATR Channel signal detection entry function (wrapper).
    Catches exceptions to prevent signal check crash from affecting other logic.
    """
    if symbol not in benchmark:
        return
    with suppress(Exception):
        await check_signals_impl(
            symbol,
            mark_prices,
            mark_price_times,
            benchmark,
            trailing_stop,
            last_atr_state,
            last_alert_time,
            _initialized,
            last_st_state,
            _atr1h_ma_type,
            _atr1h_period,
            _atr1h_mult,
            _atr15m_ma_type,
            _atr15m_period,
            _atr15m_mult,
            send_webhook_fn,
            increment_alert_count_fn,
        )


async def check_signals_impl(  # noqa: PLR0913
    symbol: str,
    mark_prices: dict[str, Any],
    mark_price_times: dict[str, Any],
    benchmark: dict[str, Any],
    trailing_stop: dict[str, Any],
    last_atr_state: dict[str, Any],
    last_alert_time: dict[str, Any],
    _initialized: bool,
    last_st_state: dict[str, Any],
    _atr1h_ma_type: str,
    _atr1h_period: int,
    _atr1h_mult: float,
    _atr15m_ma_type: str,
    _atr15m_period: int,
    atr15m_mult: float,
    send_webhook_fn: Any,
    increment_alert_count_fn: Any,
) -> None:
    """
    ATR Channel signal detection implementation.

    Logic:
    - During initialization (_initialized=False), only record last_st_state, do not push signals
    - During non-initialization:
        - price >= atr1h_upper and previous state is not long -> push LONG, establish trailing stop
        - price <= atr1h_lower and previous state is not short -> push SHORT, establish trailing stop
    - Same-direction signals must be at least 1 hour (3600 seconds) apart
    """
    current_price = mark_prices.get(symbol)
    if not current_price:
        return
    last_update = mark_price_times.get(symbol, 0)
    if time.time() - last_update > STALE_PRICE_THRESHOLD:
        return
    bm = benchmark.get(symbol)
    if not bm:
        return

    st1_val = bm["st1"]
    st2_val = bm["st2"]
    st_state = ("1" if current_price > st1_val else "0") + ("1" if current_price > st2_val else "0")

    if not _initialized:
        last_st_state[symbol] = st_state
        return

    now = time.time()
    atr1h_upper = bm.get("atr1h_upper", 0)
    atr1h_lower = bm.get("atr1h_lower", 0)
    atr1h_natrr = bm.get("atr1h_natrr", 0)
    prev_atr_state = last_atr_state.get(symbol, {"ch": 0, "sent": None})

    if price_ge(current_price, atr1h_upper) and prev_atr_state["ch"] != 1:
        last_alert = last_alert_time.get(f"ATR_Ch_{symbol}", 0)
        if now - last_alert > SIGNAL_COOLDOWN:
            last_alert_time[f"ATR_Ch_{symbol}"] = now
            last_atr_state[symbol] = {"ch": 1, "sent": "LONG"}
            natr = (atr1h_natrr / current_price * 100) if current_price > 0 and atr1h_natrr > 0 else None
            await send_webhook_fn(
                "ATR_Ch",
                f"[{symbol}] LONG",
                {
                    "symbol": symbol,
                    "direction": "LONG",
                    "price": format_number(current_price),
                    "atr_upper": format_number(atr1h_upper),
                    "atr_lower": format_number(atr1h_lower),
                    "natr": natr,
                },
            )
            increment_alert_count_fn()
            trailing_stop[symbol] = {
                "direction": "LONG",
                "entry_price": current_price,
                "entry_time": now,
                "atr_mult": atr15m_mult,
                "atr15m_upper": 0,
                "atr15m_lower": 0,
                "atr15m_state": (float("nan"), float("nan"), 0),
                "active": True,
            }

    elif price_le(current_price, atr1h_lower) and prev_atr_state["ch"] != -1:
        last_alert = last_alert_time.get(f"ATR_Ch_{symbol}", 0)
        if now - last_alert > SIGNAL_COOLDOWN:
            last_alert_time[f"ATR_Ch_{symbol}"] = now
            last_atr_state[symbol] = {"ch": -1, "sent": "SHORT"}
            natr = (atr1h_natrr / current_price * 100) if current_price > 0 and atr1h_natrr > 0 else None
            await send_webhook_fn(
                "ATR_Ch",
                f"[{symbol}] SHORT",
                {
                    "symbol": symbol,
                    "direction": "SHORT",
                    "price": format_number(current_price),
                    "atr_upper": format_number(atr1h_upper),
                    "atr_lower": format_number(atr1h_lower),
                    "natr": natr,
                },
            )
            increment_alert_count_fn()
            trailing_stop[symbol] = {
                "direction": "SHORT",
                "entry_price": current_price,
                "entry_time": now,
                "atr_mult": atr15m_mult,
                "atr15m_upper": 0,
                "atr15m_lower": 0,
                "atr15m_state": (float("nan"), float("nan"), 0),
                "active": True,
            }


async def check_trailing_stop(  # noqa: PLR0913, PLR0912
    symbol: str,
    current_price: float,
    trailing_stop: dict[str, Any],
    send_webhook_fn: Any,
    increment_alert_count_fn: Any,
    last_alert_time: dict[str, Any] | None = None,
) -> None:
    """
    Check if trailing stop is triggered.

    Logic:
    - ATR Channel mode (SINGLE):
        - LONG direction: current price < atr15m_lower (15m lower band) -> trigger stop
        - SHORT direction: current price > atr15m_upper (15m upper band) -> trigger stop
    - Clustering SuperTrend mode (PAIR):
        - LONG direction: current price < clustering_ts (trailing stop line) -> trigger stop
        - SHORT direction: current price > clustering_ts -> trigger stop

    Args:
        symbol: Trading pair name
        current_price: Current price
        trailing_stop: Trailing stop state dict
        send_webhook_fn: Async callback to send Webhook
        increment_alert_count_fn: Increment alert count
    """
    if current_price <= 0:
        return
    if symbol not in trailing_stop:
        return
    ts_entry = trailing_stop.get(symbol)
    if not ts_entry.get("active"):  # type: ignore[union-attr]
        return
    with suppress(Exception):
        direction = ts_entry.get("direction", "")  # type: ignore[union-attr]

        if ts_entry.get("use_clustering_ts"):  # type: ignore[union-attr]
            clustering_ts_val = ts_entry.get("clustering_ts", 0)  # type: ignore[union-attr]
            if clustering_ts_val > 0:
                if direction == "LONG" and price_lt(current_price, clustering_ts_val):
                    await send_webhook_fn(
                        "ATR_Ch",
                        f"[{symbol}] TRAILING STOP",
                        {
                            "symbol": symbol,
                            "direction": "LONG",
                            "price": format_number(current_price),
                            "stop_line": format_number(clustering_ts_val),
                            "entry_price": format_number(
                                ts_entry.get("entry_price", 0)  # type: ignore[union-attr]
                            ),
                            "reason": "trailing_stop",
                        },
                    )
                    increment_alert_count_fn()
                    ts_entry["active"] = False  # type: ignore[index]
                    if last_alert_time is not None:
                        last_alert_time[symbol] = 0
                elif direction == "SHORT" and price_gt(current_price, clustering_ts_val):
                    await send_webhook_fn(
                        "ATR_Ch",
                        f"[{symbol}] TRAILING STOP",
                        {
                            "symbol": symbol,
                            "direction": "SHORT",
                            "price": format_number(current_price),
                            "stop_line": format_number(clustering_ts_val),
                            "entry_price": format_number(
                                ts_entry.get("entry_price", 0)  # type: ignore[union-attr]
                            ),
                            "reason": "trailing_stop",
                        },
                    )
                    increment_alert_count_fn()
                    ts_entry["active"] = False  # type: ignore[index]
                    if last_alert_time is not None:
                        last_alert_time[symbol] = 0
            return

        upper = ts_entry.get("atr15m_upper", 0)  # type: ignore[union-attr]
        lower = ts_entry.get("atr15m_lower", 0)  # type: ignore[union-attr]

        if direction == "LONG" and lower > 0 and price_lt(current_price, lower):
            await send_webhook_fn(
                "ATR_Ch",
                f"[{symbol}] TRAILING STOP",
                {
                    "symbol": symbol,
                    "direction": "LONG",
                    "price": format_number(current_price),
                    "stop_line": format_number(lower),
                    "entry_price": format_number(ts_entry.get("entry_price", 0)),  # type: ignore[union-attr]
                    "reason": "trailing_stop",
                },
            )
            increment_alert_count_fn()
            ts_entry["active"] = False  # type: ignore[index]
            if last_alert_time is not None:
                last_alert_time[symbol] = 0

        elif direction == "SHORT" and upper > 0 and price_gt(current_price, upper):
            await send_webhook_fn(
                "ATR_Ch",
                f"[{symbol}] TRAILING STOP",
                {
                    "symbol": symbol,
                    "direction": "SHORT",
                    "price": format_number(current_price),
                    "stop_line": format_number(upper),
                    "entry_price": format_number(ts_entry.get("entry_price", 0)),  # type: ignore[union-attr]
                    "reason": "trailing_stop",
                },
            )
            increment_alert_count_fn()
            ts_entry["active"] = False  # type: ignore[index]
            if last_alert_time is not None:
                last_alert_time[symbol] = 0


async def recalculate_states_clustering(  # noqa: PLR0913
    symbol: str,
    kline_cache: dict[str, Any],
    benchmark: dict[str, Any],
    clustering_states: dict[str, Any],
    is_pair_trading: bool,
    st_period1: int,
    st_multiplier1: float,
    st_period2: int,
    st_multiplier2: float,
    vt_ema_signal: int,
    vt_ema_upper: int,
    vt_ema_lower: int,
    atr1h_period: int,
    atr1h_ma_type: str,
    atr1h_mult: float,
    _atr15m_period: int,
    _atr15m_ma_type: str,
    _atr15m_mult: float,
    clustering_min_mult: float,
    clustering_max_mult: float,
    clustering_step: float,
    clustering_perf_alpha: float,
    clustering_from_cluster: str,
    clustering_max_iter: int,
    clustering_max_data: int,
    debug: bool = False,
) -> None:
    """
    Recalculate Clustering SuperTrend indicator for PairTrading symbols (batch mode).

    Args:
        symbol: Trading pair name
        kline_cache: K-line cache
        benchmark: Indicator cache dict
        clustering_states: ClusteringState cache dict {symbol: ClusteringState}
        is_pair_trading: Whether this is pair trading
        st_period1/multiplier1: Supertrend period 1 parameters
        st_period2/multiplier2: Supertrend period 2 parameters
        vt_ema_signal/upper/lower: Vegas Tunnel EMA parameters
        atr1h_period/ma_type/mult: 1h ATR Channel parameters
        atr15m_period/ma_type/mult: 15m ATR Channel parameters
        clustering_*: Clustering SuperTrend parameters
    """
    klines = kline_cache.get(symbol, [])
    if len(klines) < MIN_KLINES:
        return
    with suppress(Exception):
        close = np.array([float(k.close) for k in klines], dtype=float)
        if is_pair_trading:
            open_arr = np.array([float(k.open) for k in klines], dtype=float)
            high = np.maximum(open_arr, close)
            low = np.minimum(open_arr, close)
        else:
            high = np.array([float(k.high) for k in klines], dtype=float)
            low = np.array([float(k.low) for k in klines], dtype=float)

        if debug:
            invalid_mask = (high == 0) | (low == 0) | (close == 0) | np.isnan(high) | np.isnan(low) | np.isnan(close)
            invalid_count = np.sum(invalid_mask)
            if invalid_count > 0:
                logger.debug(f"[DEBUG KLINES] {symbol} | invalid_bars={invalid_count}")
            else:
                logger.debug(
                    f"[DEBUG KLINES] {symbol} | clean count={len(klines)} | H_range=[{high.min():.1f},{high.max():.1f}]"
                )

        st1 = calculate_supertrend(high, low, close, st_period1, st_multiplier1)
        st2 = calculate_supertrend(high, low, close, st_period2, st_multiplier2)

        ema_s, ema_u, ema_l = calculate_vegas_tunnel(close, vt_ema_signal, vt_ema_upper, vt_ema_lower)

        atr1h = calculate_atr(high, low, close, atr1h_period, atr1h_ma_type)
        atr1h_natrr = calculate_atr(high, low, close, 20, "RMA (Standard ATR)")
        prev_atr_state = benchmark.get(symbol, {}).get("atr1h_state", (float("nan"), float("nan"), 0))
        for i in range(len(close)):
            upper, lower, ch = run_atr_channel(close[i], atr1h[i], atr1h_mult, prev_atr_state)
            prev_atr_state = (upper, lower, ch)
        atr1h_upper, atr1h_lower, atr1h_ch = prev_atr_state

        if debug:
            last_close = float(close[-1])
            last_atr = float(atr1h[-1]) if math.isfinite(atr1h[-1]) else 0
            logger.debug(
                f"{symbol} | close={last_close:.4f} | ATR({atr1h_period},{atr1h_ma_type})={last_atr:.4f} | "
                f"mult={atr1h_mult:.3f} | upper={atr1h_upper:.4f} | lower={atr1h_lower:.4f} | ch={atr1h_ch}"
            )

        st1_val, st2_val = float(st1[-1]), float(st2[-1])
        ema_s_val = float(ema_s[-1])
        if not all(math.isfinite(v) for v in [st1_val, st2_val, ema_s_val]):
            return

        prev_state = clustering_states.get(symbol)
        ts, perf_ama, new_state = clustering_supertrend(
            close,
            high,
            low,
            atr1h,
            prev_state,
            min_mult=clustering_min_mult,
            max_mult=clustering_max_mult,
            step=clustering_step,
            perf_alpha=clustering_perf_alpha,
            from_cluster=clustering_from_cluster,
            max_iter=clustering_max_iter,
            max_data=clustering_max_data,
        )
        clustering_states[symbol] = new_state

        benchmark[symbol] = {
            "st1": st1_val,
            "st2": st2_val,
            "ema_s": ema_s_val,
            "ema_u": float(ema_u[-1]),
            "ema_l": float(ema_l[-1]),
            "kline_time": int(klines[-1].open_time),
            "atr1h_upper": float(atr1h_upper) if not math.isnan(atr1h_upper) else 0,
            "atr1h_lower": float(atr1h_lower) if not math.isnan(atr1h_lower) else 0,
            "atr1h_ch": atr1h_ch,
            "atr1h_state": prev_atr_state,
            "atr1h_raw": float(atr1h[-1]) if math.isfinite(atr1h[-1]) else 0,
            "atr1h_natrr": float(atr1h_natrr[-1]) if math.isfinite(atr1h_natrr[-1]) else 0,
            "ts": float(ts) if math.isfinite(ts) else 0,
            "perf_ama": float(perf_ama) if math.isfinite(perf_ama) else 0,
            "target_factor": new_state.target_factor,
        }


async def check_signals_clustering(  # noqa: PLR0913
    symbol: str,
    mark_prices: dict[str, Any],
    mark_price_times: dict[str, Any],
    benchmark: dict[str, Any],
    trailing_stop: dict[str, Any],
    last_clustering_state: dict[str, Any],
    last_alert_time: dict[str, Any],
    _initialized: bool,
    last_st_state: dict[str, Any],
    clustering_states: dict[str, Any],
    _atr1h_ma_type: str,
    _atr1h_period: int,
    _atr1h_mult: float,
    _atr15m_ma_type: str,
    _atr15m_period: int,
    atr15m_mult: float,
    clustering_min_mult: float,
    clustering_max_mult: float,
    _clustering_step: float,
    _clustering_perf_alpha: float,
    _clustering_from_cluster: str,
    _clustering_max_iter: int,
    send_webhook_fn: Any,
    increment_alert_count_fn: Any,
) -> None:
    """
    Clustering SuperTrend signal detection entry function (wrapper).
    Catches exceptions to prevent signal check crash from affecting other logic.
    """
    if symbol not in benchmark:
        return
    with suppress(Exception):
        await check_signals_clustering_impl(
            symbol,
            mark_prices,
            mark_price_times,
            benchmark,
            trailing_stop,
            last_clustering_state,
            last_alert_time,
            _initialized,
            last_st_state,
            clustering_states,
            _atr1h_ma_type,
            _atr1h_period,
            _atr1h_mult,
            _atr15m_ma_type,
            _atr15m_period,
            atr15m_mult,
            clustering_min_mult,
            clustering_max_mult,
            _clustering_step,
            _clustering_perf_alpha,
            _clustering_from_cluster,
            _clustering_max_iter,
            send_webhook_fn,
            increment_alert_count_fn,
        )


async def check_signals_clustering_impl(  # noqa: PLR0913
    symbol: str,
    mark_prices: dict[str, Any],
    mark_price_times: dict[str, Any],
    benchmark: dict[str, Any],
    trailing_stop: dict[str, Any],
    last_clustering_state: dict[str, Any],
    last_alert_time: dict[str, Any],
    _initialized: bool,
    last_st_state: dict[str, Any],
    clustering_states: dict[str, Any],
    _atr1h_ma_type: str,
    _atr1h_period: int,
    _atr1h_mult: float,
    _atr15m_ma_type: str,
    _atr15m_period: int,
    atr15m_mult: float,
    clustering_min_mult: float,
    clustering_max_mult: float,
    _clustering_step: float,
    _clustering_perf_alpha: float,
    _clustering_from_cluster: str,
    _clustering_max_iter: int,
    send_webhook_fn: Any,
    increment_alert_count_fn: Any,
) -> None:
    """
    Clustering SuperTrend signal detection implementation (for PairTrading).

    Logic:
    - During initialization (_initialized=False), only record last_st_state, do not push signals
    - During non-initialization:
        - trend changes from -1 to 1 (short to long) -> push LONG, establish trailing stop (use ts as stop line)
        - trend changes from 1 to -1 (long to short) -> push SHORT, establish trailing stop
    - Same-direction signals must be at least 1 hour (3600 seconds) apart
    - Trailing stop uses ts (Clustering SuperTrend trailing stop line)
    """
    current_price = mark_prices.get(symbol)
    if not current_price:
        return
    last_update = mark_price_times.get(symbol, 0)
    if time.time() - last_update > STALE_PRICE_THRESHOLD:
        return
    bm = benchmark.get(symbol)
    if not bm:
        return

    prev_trend = last_clustering_state.get(symbol, {}).get("trend", 0)
    target_factor = bm.get("target_factor", (clustering_min_mult + clustering_max_mult) / 2)
    ts = bm.get("ts", 0)
    perf_ama = bm.get("perf_ama", 0)

    if not _initialized:
        last_st_state[symbol] = f"clust_{prev_trend}"
        return

    now = time.time()
    current_trend = prev_trend

    cluster_state = clustering_states.get(symbol)
    if cluster_state and math.isfinite(cluster_state.trend):
        current_trend = int(cluster_state.trend)

    if prev_trend not in (current_trend, 0):
        last_alert = last_alert_time.get(f"ClusterST_{symbol}", 0)
        if now - last_alert > SIGNAL_COOLDOWN:
            last_alert_time[f"ClusterST_{symbol}"] = now
            direction = "LONG" if current_trend == 1 else "SHORT"
            last_clustering_state[symbol] = {"trend": current_trend, "sent": direction}
            await send_webhook_fn(
                "ClusterST",
                f"[{symbol}] {direction}",
                {
                    "symbol": symbol,
                    "direction": direction,
                    "price": format_number(current_price),
                    "ts": format_number(ts),
                    "perf_ama": format_number(perf_ama),
                    "target_factor": format_number(target_factor),
                },
            )
            increment_alert_count_fn()
            trailing_stop[symbol] = {
                "direction": direction,
                "entry_price": current_price,
                "entry_time": now,
                "atr_mult": atr15m_mult,
                "atr15m_upper": 0,
                "atr15m_lower": 0,
                "atr15m_state": (float("nan"), float("nan"), 0),
                "active": True,
                "use_clustering_ts": True,
                "clustering_ts": ts,
            }
    else:
        prev_sent = last_clustering_state.get(symbol, {}).get("sent")
        if prev_sent is None:
            prev_sent = "N/A"
        last_clustering_state[symbol] = {"trend": current_trend, "sent": prev_sent}

    if symbol in trailing_stop and trailing_stop[symbol].get("use_clustering_ts"):
        trailing_stop[symbol]["clustering_ts"] = ts
