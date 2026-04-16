"""Signal coordination facade extracted from NotificationService."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any

from models import Kline
from notifications import DIRECTION_LONG, DIRECTION_SHORT
from signals import (
    check_breakout,
    check_signals,
    check_signals_clustering,
    check_trailing_stop,
    start_breakout_monitor,
)


class SignalCoordinator:
    """Coordinate signal, trailing stop, and breakout connector flows."""

    def __init__(  # noqa: PLR0913
        self,
        mark_prices: dict[str, float],
        mark_price_times: dict[str, float],
        benchmark: dict[str, dict[str, Any]],
        trailing_stop: dict[str, dict[str, Any]],
        last_atr_state: dict[str, dict[str, Any]],
        last_clustering_state: dict[str, dict[str, Any]],
        last_alert_time: dict[str, float],
        last_st_state: dict[str, Any],
        clustering_states: dict[str, Any],
        breakout_monitor: dict[str, dict[str, Any]],
        kline_cache_15m: dict[str, list[Kline]],
        send_webhook_fn: Callable[[str, str, dict[str, Any] | None], Awaitable[None]],
        increment_alert_count_fn: Callable[[], None],
        send_event_fn: Callable[[Any], Awaitable[None]],
        refresh_trailing_stop_channel_fn: Callable[[str, bool], Awaitable[None]],
        start_breakout_monitor_fn: Callable[[str, str, float, float], Awaitable[None]],
        stop_breakout_monitor_fn: Callable[[str], Awaitable[None]],
        is_pair_symbol_fn: Callable[[str], bool],
        get_ws_fn: Callable[[], Any],
        update_15m_atr_fn: Callable[[str, dict[str, Any]], Awaitable[None]],
        fetch_pair_klines_fn: Callable[..., Awaitable[list[Kline]]],
        atr1h_ma_type: str,
        atr1h_period: int,
        atr1h_mult: float,
        atr15m_ma_type: str,
        atr15m_period: int,
        atr15m_mult: float,
        clustering_min_mult: float,
        clustering_max_mult: float,
        clustering_step: float,
        clustering_perf_alpha: float,
        clustering_from_cluster: str,
        clustering_max_iter: int,
        disable_single_trailing: bool,
        disable_pair_trailing: bool,
        proxy_enable: bool,
        proxy_url: str,
        breakout_direction_long: str,
        breakout_direction_short: str,
        min_trailing_klines: int,
    ) -> None:
        self.mark_prices = mark_prices
        self.mark_price_times = mark_price_times
        self.benchmark = benchmark
        self.trailing_stop = trailing_stop
        self.last_atr_state = last_atr_state
        self.last_clustering_state = last_clustering_state
        self.last_alert_time = last_alert_time
        self.last_st_state = last_st_state
        self.clustering_states = clustering_states
        self.breakout_monitor = breakout_monitor
        self.kline_cache_15m = kline_cache_15m
        self.send_webhook_fn = send_webhook_fn
        self.increment_alert_count_fn = increment_alert_count_fn
        self.send_event_fn = send_event_fn
        self.refresh_trailing_stop_channel_fn = refresh_trailing_stop_channel_fn
        self.start_breakout_monitor_fn = start_breakout_monitor_fn
        self.stop_breakout_monitor_fn = stop_breakout_monitor_fn
        self.is_pair_symbol_fn = is_pair_symbol_fn
        self.get_ws_fn = get_ws_fn
        self.update_15m_atr_fn = update_15m_atr_fn
        self.fetch_pair_klines_fn = fetch_pair_klines_fn
        self.atr1h_ma_type = atr1h_ma_type
        self.atr1h_period = atr1h_period
        self.atr1h_mult = atr1h_mult
        self.atr15m_ma_type = atr15m_ma_type
        self.atr15m_period = atr15m_period
        self.atr15m_mult = atr15m_mult
        self.clustering_min_mult = clustering_min_mult
        self.clustering_max_mult = clustering_max_mult
        self.clustering_step = clustering_step
        self.clustering_perf_alpha = clustering_perf_alpha
        self.clustering_from_cluster = clustering_from_cluster
        self.clustering_max_iter = clustering_max_iter
        self.disable_single_trailing = disable_single_trailing
        self.disable_pair_trailing = disable_pair_trailing
        self.proxy_enable = proxy_enable
        self.proxy_url = proxy_url
        self.breakout_direction_long = breakout_direction_long
        self.breakout_direction_short = breakout_direction_short
        self.min_trailing_klines = min_trailing_klines

    async def check_trailing_stop(self, symbol: str, price: float, is_pair: bool) -> None:
        """Run trailing stop connector logic with config gating."""
        if is_pair and self.disable_pair_trailing:
            return
        if not is_pair and self.disable_single_trailing:
            return
        await check_trailing_stop(
            symbol,
            price,
            self.trailing_stop,
            self.send_webhook_fn,
            self.increment_alert_count_fn,
            self.last_alert_time,
            self.send_event_fn,
        )

    async def check_signals(self, symbol: str, initialized: bool) -> None:
        """Run ATR signal checks and breakout monitor bootstrap."""
        had_active_trailing = bool(self.trailing_stop.get(symbol, {}).get("active"))
        await check_signals(
            symbol,
            self.mark_prices,
            self.mark_price_times,
            self.benchmark,
            self.trailing_stop,
            self.last_atr_state,
            self.last_alert_time,
            initialized,
            self.last_st_state,
            self.atr1h_ma_type,
            self.atr1h_period,
            self.atr1h_mult,
            self.atr15m_ma_type,
            self.atr15m_period,
            self.atr15m_mult,
            self.send_webhook_fn,
            self.increment_alert_count_fn,
            self.send_event_fn,
        )
        has_new_atr_trailing = bool(self.trailing_stop.get(symbol, {}).get("active")) and not bool(
            self.trailing_stop.get(symbol, {}).get("use_clustering_ts")
        )
        if not had_active_trailing and has_new_atr_trailing:
            await self.refresh_trailing_stop_channel_fn(symbol, True)
            direction = self.trailing_stop.get(symbol, {}).get("direction", "")
            if direction == DIRECTION_LONG:
                await self.start_breakout_monitor_fn(
                    symbol, self.breakout_direction_long, self.mark_prices.get(symbol, 0), time.time()
                )
            elif direction == DIRECTION_SHORT:
                await self.start_breakout_monitor_fn(
                    symbol, self.breakout_direction_short, self.mark_prices.get(symbol, 0), time.time()
                )

    async def check_signals_clustering(self, symbol: str, initialized: bool) -> None:
        """Run clustering signal checks through the shared alert path."""
        await check_signals_clustering(
            symbol,
            self.mark_prices,
            self.mark_price_times,
            self.benchmark,
            self.trailing_stop,
            self.last_clustering_state,
            self.last_alert_time,
            initialized,
            self.last_st_state,
            self.clustering_states,
            self.atr1h_ma_type,
            self.atr1h_period,
            self.atr1h_mult,
            self.atr15m_ma_type,
            self.atr15m_period,
            self.atr15m_mult,
            self.clustering_min_mult,
            self.clustering_max_mult,
            self.clustering_step,
            self.clustering_perf_alpha,
            self.clustering_from_cluster,
            self.clustering_max_iter,
            self.send_webhook_fn,
            self.increment_alert_count_fn,
            self.send_event_fn,
        )

    async def start_breakout_monitor(self, symbol: str, direction: str, price: float, trigger_time: float) -> None:
        """Start breakout monitor using service-owned runtime dependencies."""
        proxy = self.proxy_url if self.proxy_enable else None
        await start_breakout_monitor(
            symbol,
            direction,
            price,
            trigger_time,
            self.breakout_monitor,
            self.is_pair_symbol_fn(symbol),
            self.mark_prices,
            self.get_ws_fn(),
            self.update_15m_atr_fn,
            fetch_pair_klines_fn=self.fetch_pair_klines_fn,
            proxy=proxy,
        )
        monitor = self.breakout_monitor.get(symbol)
        if monitor:
            history = monitor.get("klines_15m", [])
            if isinstance(history, list) and history:
                self.kline_cache_15m[symbol] = history

    def sync_breakout_monitor_from_cache(self, symbol: str) -> None:
        """Advance breakout monitor using cached 15m bars refreshed elsewhere."""
        monitor = self.breakout_monitor.get(symbol)
        if not monitor:
            return

        cached_klines = self.kline_cache_15m.get(symbol, [])
        if len(cached_klines) < self.min_trailing_klines:
            return

        monitor_klines = monitor.get("klines_15m", [])
        if not isinstance(monitor_klines, list) or not monitor_klines:
            monitor["klines_15m"] = cached_klines[-20:]
            return

        latest_known_open = int(monitor_klines[-1].open_time)
        newer_bars = [kline for kline in cached_klines if int(kline.open_time) > latest_known_open]
        if not newer_bars:
            return

        updated_klines = [*monitor_klines, *newer_bars]
        monitor["klines_15m"] = updated_klines[-20:]
        monitor["kline_15m_count"] = int(monitor.get("kline_15m_count", 0)) + len(newer_bars)

    async def check_breakout(self, symbol: str) -> None:
        """Run breakout evaluation after syncing cached runtime 15m bars."""
        self.sync_breakout_monitor_from_cache(symbol)
        monitor = self.breakout_monitor.get(symbol)
        if monitor and int(monitor.get("kline_15m_count", 0)) <= 0:
            return
        await check_breakout(
            symbol,
            self.breakout_monitor,
            self.send_webhook_fn,
            self.increment_alert_count_fn,
            self.stop_breakout_monitor_fn,
            self.send_event_fn,
        )
