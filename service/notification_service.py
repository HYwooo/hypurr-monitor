"""
Main notification service - coordinates all modules, manages WebSocket and signal detection.
"""

import asyncio
import threading
import time
import tomllib
from collections.abc import MutableMapping
from contextlib import suppress
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np

from config import load_network_config, resolve_path_from_config
from hyperliquid import MarketGateway
from hyperliquid.rest_client import (
    HyperliquidREST,
    get_cached_klines,
    get_price_decimals,
    update_cache,
)
from indicators import (
    calculate_atr,
    run_atr_channel,
)
from logging_config import get_logger
from models import Kline
from notifications import (
    ALERT_ATR_CHANNEL,
    ALERT_ERROR,
    ALERT_SYSTEM,
    DIRECTION_LONG,
    DIRECTION_SHORT,
    AlertEvent,
    WebhookSender,
    format_connection_failed_message,
    format_connection_success_message,
    format_directional_signal_message,
    format_ws_data_resumed_message,
    format_ws_data_silence_message,
    format_ws_reconnect_failure_message,
    format_ws_reconnect_success_message,
    log_warning,
)
from service.alert_dispatcher import AlertDispatcher
from service.market_data_processor import MarketDataProcessor
from service.signal_coordinator import SignalCoordinator
from service.ws_runtime_supervisor import WSRuntimeSupervisor
from signals import (
    recalculate_states,
    recalculate_states_clustering,
    start_breakout_monitor,
    update_klines,
)

logger = get_logger(__name__)

WEBHOOK_LOG_FILE = "webhook.log"
PAIR_15M_PER_1H = 4
PAIR_15M_PER_4H = 16
ATR_REFRESH_THROTTLE_SECONDS = 60
ATR_4H_REFRESH_THROTTLE_SECONDS = 300
TRAILING_REFRESH_THROTTLE_SECONDS = 30
MIN_TRAILING_KLINES = 2
PRICE_STALE_THRESHOLD_SECONDS = 300
ATR_BREAKOUT_COOLDOWN_SECONDS = 3600
HEARTBEAT_WRITE_THROTTLE_SECONDS = 5
BREAKOUT_DIRECTION_LONG = "11"
BREAKOUT_DIRECTION_SHORT = "00"


def build_pair_15m_klines(symbol: str, klines1: list[Kline], klines2: list[Kline]) -> list[Kline]:
    """Build synthetic 15m pair klines from two aligned component legs."""
    k2_by_time = {int(k.open_time): k for k in klines2}
    merged: list[Kline] = []
    for k1 in klines1:
        t = int(k1.open_time)
        if t not in k2_by_time:
            continue
        k2 = k2_by_time[t]
        o1 = float(k1.open)
        c1 = float(k1.close)
        o2 = float(k2.open)
        c2 = float(k2.close)
        if o2 == 0 or c2 == 0:
            continue
        ratio_open = o1 / o2
        ratio_close = c1 / c2
        merged.append(
            Kline(
                symbol=symbol,
                interval="15m",
                open_time=t,
                open=ratio_open,
                high=max(ratio_open, ratio_close),
                low=min(ratio_open, ratio_close),
                close=ratio_close,
                volume=float(k1.volume),
                close_time=int(k1.close_time) if k1.close_time else 0,
                is_closed=bool(k1.is_closed and k2.is_closed),
            )
        )
    return sorted(merged, key=lambda x: x.open_time)


def aggregate_pair_15m_klines(
    symbol: str,
    klines_15m: list[Kline],
    bars_per_bucket: int,
    bucket_ms: int,
    interval: str,
) -> list[Kline]:
    """Aggregate synthetic pair 15m klines into a higher timeframe."""
    if not klines_15m:
        return []
    grouped: dict[int, list[Kline]] = {}
    for kline in klines_15m:
        bucket = int(kline.open_time) // bucket_ms
        grouped.setdefault(bucket, []).append(kline)

    aggregated: list[Kline] = []
    for _, group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda x: x.open_time)
        if len(ordered) < bars_per_bucket:
            continue
        aggregated.append(
            Kline(
                symbol=symbol,
                interval=interval,
                open_time=int(ordered[0].open_time),
                open=float(ordered[0].open),
                high=max(float(k.high) for k in ordered),
                low=min(float(k.low) for k in ordered),
                close=float(ordered[-1].close),
                volume=sum(float(k.volume) for k in ordered),
                close_time=int(ordered[-1].close_time) if ordered[-1].close_time else 0,
                is_closed=all(bool(k.is_closed) for k in ordered),
            )
        )
    return aggregated


def aggregate_pair_15m_to_1h(symbol: str, klines_15m: list[Kline]) -> list[Kline]:
    """Aggregate synthetic pair 15m klines into 1h klines."""
    return aggregate_pair_15m_klines(symbol, klines_15m, PAIR_15M_PER_1H, 3_600_000, "1h")


def aggregate_pair_15m_to_4h(symbol: str, klines_15m: list[Kline]) -> list[Kline]:
    """Aggregate synthetic pair 15m klines into 4h klines."""
    return aggregate_pair_15m_klines(symbol, klines_15m, PAIR_15M_PER_4H, 14_400_000, "4h")


class NotificationService:
    """
    Main service - coordinates all modules, manages WebSocket connections and trading signal detection.

    State dicts:
    - mark_prices: {symbol: price} current prices
    - mark_price_times: {symbol: unix_time} price update times
    - kline_cache: {symbol: klines} K-line cache
    - benchmark: {symbol: {st1, st2, ema_*, atr1h_*, ...}} indicator results
    - trailing_stop: {symbol: {direction, entry_price, atr15m_*, active}} trailing stop state
    """

    def __init__(self, config_path: str = "config.toml", debug: bool = False):  # noqa: PLR0915
        self.config_path = config_path
        self.debug = debug
        self.config = self._load_config(config_path)
        self.network = load_network_config(self.config)
        self.market_gateway = MarketGateway(self.network.rest, self.network.ws)
        sym_config = self.config.get("symbols", {})
        self.single_list: list[str] = sym_config.get("single_list", [])
        self.pair_list: list[str] = sym_config.get("pair_list", [])
        self._pair_components: dict[str, tuple[str, str]] = {}
        for p in self.pair_list:
            if "-" in p:
                parts = p.split("-", 1)
                self._pair_components[p] = (parts[0], parts[1])
        self.symbols: list[str] = self.single_list + self.pair_list
        self.webhook_url = self.config["webhook"]["url"]
        self.webhook_format = self.config["webhook"].get("format", "card")

        # Indicator parameters
        self.st_period1 = self.config.get("supertrend", {}).get("period1", 9)
        self.st_multiplier1 = self.config.get("supertrend", {}).get("multiplier1", 2.5)
        self.st_period2 = self.config.get("supertrend", {}).get("period2", 14)
        self.st_multiplier2 = self.config.get("supertrend", {}).get("multiplier2", 1.7)
        self.vt_ema_signal = self.config.get("vegas", {}).get("ema_signal", 9)
        self.vt_ema_upper = self.config.get("vegas", {}).get("ema_upper", 144)
        self.vt_ema_lower = self.config.get("vegas", {}).get("ema_lower", 169)
        self.atr1h_ma_type = self.config.get("atr_1h", {}).get("ma_type", "DEMA")
        self.atr1h_period = self.config.get("atr_1h", {}).get("period", 14)
        self.atr1h_mult = self.config.get("atr_1h", {}).get("mult", 1.618)
        self.atr4h_ma_type = self.config.get("atr_4h", {}).get("ma_type", self.atr1h_ma_type)
        self.atr4h_period = self.config.get("atr_4h", {}).get("period", self.atr1h_period)
        self.atr4h_mult = self.config.get("atr_4h", {}).get("mult", self.atr1h_mult)
        self.atr15m_ma_type = self.config.get("atr_15m", {}).get("ma_type", "HMA")
        self.atr15m_period = self.config.get("atr_15m", {}).get("period", 14)
        self.atr15m_mult = self.config.get("atr_15m", {}).get("mult", 1.3)

        cs_config = self.config.get("clustering_st", {})
        self.clustering_enabled = cs_config.get("enabled", False)
        self.clustering_min_mult = cs_config.get("min_mult", 1.0)
        self.clustering_max_mult = cs_config.get("max_mult", 5.0)
        self.clustering_step = cs_config.get("step", 0.5)
        self.clustering_perf_alpha = cs_config.get("perf_alpha", 10.0)
        self.clustering_from_cluster = cs_config.get("from_cluster", "Best")
        self.clustering_max_iter = cs_config.get("max_iter", 1000)
        self.clustering_history_klines = cs_config.get("history_klines", 500)

        self.heartbeat_file = resolve_path_from_config(config_path, str(self.config["service"]["heartbeat_file"]))
        self.webhook_log_file = resolve_path_from_config(config_path, WEBHOOK_LOG_FILE)
        self.heartbeat_timeout = int(self.config.get("service", {}).get("heartbeat_timeout", 120))
        self.proxy_enable = self.network.rest.proxy_url is not None
        self.proxy_url = self.network.rest.proxy_url or ""
        self.report_enable = self.config.get("report", {}).get("enable", False)
        self.report_times = self.config.get("report", {}).get("times", ["08:00", "20:00"])
        self.timezone = self.config.get("settings", {}).get("timezone", "Z")
        self.max_log_lines = self.config.get("settings", {}).get("max_log_lines", 1000)
        self.disable_single_trailing = self.config.get("settings", {}).get("disable_single_trailing", False)
        self.disable_pair_trailing = self.config.get("settings", {}).get("disable_pair_trailing", False)

        exchange_id = self.config.get("settings", {}).get("exchange", "binance")
        self._exchange_id = exchange_id

        self.mark_prices: dict[str, float] = {}
        self.mark_price_times: dict[str, float] = {}
        self._breakout_comp_prices: dict[str, float] = {}
        self.kline_cache: dict[str, list[Any]] = {}
        self.kline_cache_4h: dict[str, list[Kline]] = {}
        self.kline_cache_15m: dict[str, list[Kline]] = {}
        self.benchmark: dict[str, dict[str, Any]] = {}
        self.last_st_state: dict[str, str] = {}
        self.last_atr_state: dict[str, dict[str, Any]] = {}
        self.last_atr4h_state: dict[str, dict[str, Any]] = {}
        self.last_alert_time: dict[str, float] = {}
        self.last_kline_time: dict[str, int] = {}
        self.breakout_monitor: dict[str, dict[str, Any]] = {}
        self.trailing_stop: dict[str, dict[str, Any]] = {}
        self.last_atr1h_ch: dict[str, int] = {}
        self.clustering_states: dict[str, Any] = {}
        self.last_clustering_state: dict[str, dict[str, Any]] = {}
        self.connected = False
        self.running = False
        self.observer: Any | None = None
        self._initialized = False
        self._status_print_enabled = True
        self._alert_count = 0
        self._last_report_time = 0
        self._lock = threading.Lock()
        self._pending_status: set[str] = set()
        self._ws_tasks: list[asyncio.Task[None]] = []
        self._logged_initial_price: set[str] = set()
        self._last_atr_refresh_attempt: dict[str, float] = {}
        self._last_atr4h_refresh_attempt: dict[str, float] = {}
        self._last_trailing_refresh_attempt: dict[str, float] = {}
        self._hl_ws: aiohttp.ClientWebSocketResponse | None = None
        self._hl_session: aiohttp.ClientSession | None = None
        self._webhook_sender = WebhookSender(self.network.webhook)
        self._alert_dispatcher = AlertDispatcher(
            self.webhook_url,
            self.webhook_format,
            self.webhook_log_file,
            self.max_log_lines,
            self._get_timestamp,
            self._webhook_sender,
        )
        self._hl_ws_running = False
        self._ws_reconnect_alert_active = False
        self._ws_silence_alert_active = False
        self._ws_silence_started_at = 0.0
        self._last_ws_message_time = 0.0
        self._last_ws_data_time = 0.0
        self._last_heartbeat_write_time = 0.0
        self._market_data_processor = MarketDataProcessor(
            symbols_fn=lambda: self.symbols,
            pair_components_fn=lambda: self._pair_components,
            mark_prices=self.mark_prices,
            mark_price_times=self.mark_price_times,
            logged_initial_price=self._logged_initial_price,
            record_ws_data_activity_fn=self._record_ws_data_activity,
            log_symbol_state_fn=lambda symbol: self._log_symbol_state(symbol),
            maybe_refresh_runtime_atr_fn=lambda symbol: self._maybe_refresh_runtime_atr(symbol),
            maybe_refresh_runtime_atr_4h_fn=lambda symbol: self._maybe_refresh_runtime_atr_4h(symbol),
            refresh_trailing_stop_channel_fn=lambda symbol: self._refresh_trailing_stop_channel(symbol),
            check_trailing_stop_fn=lambda symbol, price: self._ct_check_trailing_stop(symbol, price),
            use_clustering_for_symbol_fn=lambda symbol: self._use_clustering_for_symbol(symbol),
            check_signals_clustering_fn=lambda symbol: self._ct_check_signals_clustering(symbol),
            is_pair_trading_fn=lambda symbol: self._is_pair_trading(symbol),
            is_pair_symbol_fn=lambda symbol: self._is_pair_symbol(symbol),
            check_signals_fn=lambda symbol: self._ct_check_signals(symbol),
            check_signals_4h_fn=lambda symbol: self._ct_check_signals_4h(symbol),
            check_breakout_fn=lambda symbol: self._ct_check_breakout(symbol),
        )
        self._signal_coordinator = SignalCoordinator(
            mark_prices=self.mark_prices,
            mark_price_times=self.mark_price_times,
            benchmark=self.benchmark,
            trailing_stop=self.trailing_stop,
            last_atr_state=self.last_atr_state,
            last_clustering_state=self.last_clustering_state,
            last_alert_time=self.last_alert_time,
            last_st_state=self.last_st_state,
            clustering_states=self.clustering_states,
            breakout_monitor=self.breakout_monitor,
            kline_cache_15m=self.kline_cache_15m,
            send_webhook_fn=self._send_webhook_current,
            increment_alert_count_fn=self._increment_alert_count,
            send_event_fn=self._send_event_current,
            refresh_trailing_stop_channel_fn=self._refresh_trailing_stop_channel_current,
            start_breakout_monitor_fn=self._start_breakout_monitor_impl,
            stop_breakout_monitor_fn=self._stop_breakout_monitor_impl,
            is_pair_symbol_fn=self._is_pair_symbol,
            get_ws_fn=lambda: self._hl_ws,
            update_15m_atr_fn=self._ct_update_15m_atr,
            fetch_pair_klines_fn=self._fetch_pair_klines_current,
            atr1h_ma_type=self.atr1h_ma_type,
            atr1h_period=self.atr1h_period,
            atr1h_mult=self.atr1h_mult,
            atr15m_ma_type=self.atr15m_ma_type,
            atr15m_period=self.atr15m_period,
            atr15m_mult=self.atr15m_mult,
            clustering_min_mult=self.clustering_min_mult,
            clustering_max_mult=self.clustering_max_mult,
            clustering_step=self.clustering_step,
            clustering_perf_alpha=self.clustering_perf_alpha,
            clustering_from_cluster=self.clustering_from_cluster,
            clustering_max_iter=self.clustering_max_iter,
            disable_single_trailing=self.disable_single_trailing,
            disable_pair_trailing=self.disable_pair_trailing,
            proxy_enable=self.proxy_enable,
            proxy_url=self.proxy_url,
            breakout_direction_long=BREAKOUT_DIRECTION_LONG,
            breakout_direction_short=BREAKOUT_DIRECTION_SHORT,
            min_trailing_klines=MIN_TRAILING_KLINES,
        )
        self._ws_runtime_supervisor = WSRuntimeSupervisor(
            should_run_fn=lambda: self._hl_ws_running and self._hl_ws is not None,
            check_data_silence_fn=lambda: self._check_ws_data_silence(),
            receive_message_fn=lambda: self._receive_hyperliquid_ws_message(),
            send_ping_fn=lambda: self._send_hyperliquid_ping(),
            reconnect_fn=lambda reason: self._reconnect_hyperliquid_ws(reason),
            mark_message_received_fn=lambda now: self._record_ws_message(now),
            process_payload_fn=lambda data: self._market_data_processor.process_payload(data),
        )

    def _touch_heartbeat_file(self) -> None:
        """Persist a lightweight heartbeat timestamp for external monitors."""
        now = time.time()
        if now - self._last_heartbeat_write_time < HEARTBEAT_WRITE_THROTTLE_SECONDS:
            return
        try:
            Path(self.heartbeat_file).write_text(f"{int(now)}\n", encoding="utf-8")
            self._last_heartbeat_write_time = now
        except Exception as e:
            logger.warning("Failed to write heartbeat file %s: %s", self.heartbeat_file, e)

    async def _record_ws_data_activity(self, now: float) -> None:
        """Record fresh market data activity and update external health state."""
        self._last_ws_data_time = now
        await self._notify_ws_data_recovered(now)
        self._touch_heartbeat_file()

    def _record_ws_message(self, now: float) -> None:
        """Record receipt of any websocket message for observability."""
        self._last_ws_message_time = now

    async def _receive_hyperliquid_ws_message(self) -> aiohttp.WSMessage:
        """Receive next websocket message from the active Hyperliquid stream."""
        if self._hl_ws is None:
            raise ConnectionError("ws unavailable")  # noqa: TRY003
        return await self.market_gateway.receive_ws_message(self._hl_ws)

    async def _send_webhook_current(self, alert_type: str, message: str, extra: dict[str, Any] | None = None) -> None:
        """Invoke the current webhook connector to support monkeypatch-friendly delegation."""
        await self._send_webhook(alert_type, message, extra)

    async def _send_event_current(self, event: AlertEvent) -> None:
        """Invoke the current structured event facade."""
        await self.send_event(event)

    async def _refresh_trailing_stop_channel_current(self, symbol: str, force: bool = False) -> None:
        """Invoke the current trailing-stop refresh implementation."""
        await self._refresh_trailing_stop_channel(symbol, force)

    async def _fetch_pair_klines_current(
        self,
        symbol: str,
        limit: int = 500,
        interval: str = "1h",
        proxy: str | None = None,
        kline_cache: dict[str, Any] | None = None,
        _fetch_klines_fn: Any = None,
    ) -> list[Kline]:
        """Invoke the current pair-kline fetcher while preserving test monkeypatching."""
        return await self._hl_fetch_pair_klines(symbol, limit, interval, proxy, kline_cache, _fetch_klines_fn)

    async def _start_breakout_monitor_impl(
        self, symbol: str, direction: str, price: float, trigger_time: float
    ) -> None:
        """Actual breakout monitor startup implementation used by the coordinator."""
        proxy = self.proxy_url if self.proxy_enable else None
        await start_breakout_monitor(
            symbol,
            direction,
            price,
            trigger_time,
            self.breakout_monitor,
            self._is_pair_symbol(symbol),
            self.mark_prices,
            self._hl_ws,
            self._ct_update_15m_atr,
            fetch_pair_klines_fn=self._fetch_pair_klines_current,
            proxy=proxy,
        )
        monitor = self.breakout_monitor.get(symbol)
        if monitor:
            history = monitor.get("klines_15m", [])
            if isinstance(history, list) and history:
                self.kline_cache_15m[symbol] = history

    async def _stop_breakout_monitor_impl(self, symbol: str) -> None:
        """Actual breakout monitor stop implementation used by the coordinator."""
        self.breakout_monitor.pop(symbol, None)

    def _create_rest_client(self) -> HyperliquidREST:
        """Create REST client using unified network settings."""
        return self.market_gateway.create_rest_client()

    async def send_event(self, event: AlertEvent) -> None:
        """Public structured alert facade for callers that already build AlertEvent."""
        await self._alert_dispatcher.send_event(event)

    async def send_alert(self, alert_type: str, message: str, extra: dict[str, Any] | None = None) -> None:
        """Public alert sending facade for callers outside the service internals."""
        await self._alert_dispatcher.send_alert(alert_type, message, extra)

    def _cleanup_symbol_state(self, symbol: str) -> None:
        """Remove symbol-scoped runtime state to avoid stale data retention."""
        for mapping in (
            self.mark_prices,
            self.mark_price_times,
            self.kline_cache,
            self.kline_cache_4h,
            self.kline_cache_15m,
            self.benchmark,
            self.last_st_state,
            self.last_atr_state,
            self.last_atr4h_state,
            self.last_kline_time,
            self.breakout_monitor,
            self.trailing_stop,
            self.last_atr1h_ch,
            self.clustering_states,
            self.last_clustering_state,
            self._last_atr_refresh_attempt,
            self._last_atr4h_refresh_attempt,
            self._last_trailing_refresh_attempt,
        ):
            mapping.pop(symbol, None)

        for alert_key in (
            f"ATR_Ch_{symbol}",
            f"ATR_4H_{symbol}",
            f"ClusterST_{symbol}",
            symbol,
        ):
            self.last_alert_time.pop(alert_key, None)

    def _prune_runtime_state(self) -> None:
        """Prune stale symbol state that no longer belongs to configured symbols."""
        configured_symbols = set(self.symbols)
        pair_components = {component for pair in self._pair_components.values() for component in pair}
        market_symbols = configured_symbols | pair_components

        market_mappings: tuple[MutableMapping[str, Any], ...] = (
            self.mark_prices,
            self.mark_price_times,
            self.kline_cache,
            self.kline_cache_15m,
        )
        for mapping in market_mappings:
            stale_keys = [key for key in mapping if key not in market_symbols]
            for key in stale_keys:
                mapping.pop(key, None)

        benchmark_mappings: tuple[MutableMapping[str, Any], ...] = (self.kline_cache_4h, self.benchmark)
        for mapping in benchmark_mappings:
            stale_keys = [key for key in mapping if key not in configured_symbols]
            for key in stale_keys:
                mapping.pop(key, None)

        tracked_symbol_mappings: tuple[MutableMapping[str, Any], ...] = (
            self.last_st_state,
            self.last_atr_state,
            self.last_atr4h_state,
            self.last_kline_time,
            self.breakout_monitor,
            self.trailing_stop,
            self.last_atr1h_ch,
            self.clustering_states,
            self.last_clustering_state,
            self._last_atr_refresh_attempt,
            self._last_atr4h_refresh_attempt,
            self._last_trailing_refresh_attempt,
        )
        for mapping in tracked_symbol_mappings:
            stale_keys = [key for key in mapping if key not in configured_symbols]
            for key in stale_keys:
                mapping.pop(key, None)

        valid_alert_keys = set(configured_symbols)
        for symbol in configured_symbols:
            valid_alert_keys.update({f"ATR_Ch_{symbol}", f"ATR_4H_{symbol}", f"ClusterST_{symbol}"})
        stale_alert_keys = [key for key in self.last_alert_time if key not in valid_alert_keys]
        for key in stale_alert_keys:
            self.last_alert_time.pop(key, None)

    def _is_ws_data_silent(self, now: float | None = None) -> bool:
        """Return whether market data stream has been silent beyond threshold."""
        if self.heartbeat_timeout <= 0:
            return False
        current = now if now is not None else time.time()
        last_data_time = self._last_ws_data_time
        if last_data_time <= 0:
            return False
        return current - last_data_time > self.heartbeat_timeout

    async def _check_ws_data_silence(self, now: float | None = None) -> bool:
        """Reconnect when market data has been silent beyond heartbeat_timeout."""
        current = now if now is not None else time.time()
        if not self._is_ws_data_silent(current):
            return False
        silence_seconds = int(current - self._last_ws_data_time)
        reason = f"market data silence > {self.heartbeat_timeout}s (last {silence_seconds}s ago)"
        await self._notify_ws_data_silence(silence_seconds)
        logger.error("Hyperliquid WS %s", reason)
        return await self._reconnect_hyperliquid_ws(reason)

    async def _notify_ws_data_silence(self, silence_seconds: int) -> None:
        """Send a single webhook when market data becomes silent."""
        if self._ws_silence_alert_active:
            return
        self._ws_silence_alert_active = True
        self._ws_silence_started_at = time.time()
        with suppress(Exception):
            await self._send_webhook(
                ALERT_ERROR,
                format_ws_data_silence_message(silence_seconds),
            )

    async def _notify_ws_data_recovered(self, recovered_at: float | None = None) -> None:
        """Send webhook when market data resumes after silence."""
        if not self._ws_silence_alert_active:
            return
        current = recovered_at if recovered_at is not None else time.time()
        silence_duration = max(0, int(current - self._ws_silence_started_at))
        self._ws_silence_alert_active = False
        self._ws_silence_started_at = 0.0
        with suppress(Exception):
            await self._send_webhook(
                ALERT_SYSTEM,
                format_ws_data_resumed_message(silence_duration),
            )

    def _use_clustering_for_symbol(self, symbol: str) -> bool:
        """Return whether a symbol should use clustering signal path."""
        return self.clustering_enabled and self._is_pair_symbol(symbol)

    def _expected_closed_open_time_ms(self, interval_seconds: int) -> int:
        """Return latest expected closed candle open_time in milliseconds."""
        now = int(time.time())
        current_bucket = now // interval_seconds
        last_closed_bucket = current_bucket - 1
        if last_closed_bucket < 0:
            return 0
        return last_closed_bucket * interval_seconds * 1000

    async def _maybe_refresh_runtime_atr(self, symbol: str, interval_seconds: int = 3600) -> None:
        """Refresh ATR benchmark only when a newer closed kline should exist."""
        expected_open_time = self._expected_closed_open_time_ms(interval_seconds)
        if expected_open_time <= 0:
            return
        klines = self.kline_cache.get(symbol, [])
        latest_open_time = int(klines[-1].open_time) if klines else 0
        if latest_open_time >= expected_open_time:
            return
        now = time.time()
        last_attempt = self._last_atr_refresh_attempt.get(symbol, 0.0)
        if now - last_attempt < ATR_REFRESH_THROTTLE_SECONDS:
            return
        self._last_atr_refresh_attempt[symbol] = now
        await self._ct_update_klines(symbol)

    async def _maybe_refresh_runtime_atr_4h(self, symbol: str) -> None:
        """Refresh 4h ATR breakout benchmark when a newer closed 4h kline should exist."""
        expected_open_time = self._expected_closed_open_time_ms(14_400)
        if expected_open_time <= 0:
            return
        klines = self.kline_cache_4h.get(symbol, [])
        latest_open_time = int(klines[-1].open_time) if klines else 0
        if latest_open_time >= expected_open_time:
            return
        now = time.time()
        last_attempt = self._last_atr4h_refresh_attempt.get(symbol, 0.0)
        if now - last_attempt < ATR_4H_REFRESH_THROTTLE_SECONDS:
            return
        self._last_atr4h_refresh_attempt[symbol] = now
        await self._recalculate_4h_breakout_state(symbol)

    async def _fetch_4h_klines(self, symbol: str, limit: int = 500) -> list[Kline]:
        """Fetch 4h klines for single or pair symbol."""
        proxy = self.proxy_url if self.proxy_enable else None
        if self._is_pair_symbol(symbol):
            klines = await self._hl_fetch_pair_klines(
                symbol,
                limit=limit,
                interval="4h",
                proxy=proxy,
                kline_cache=self.kline_cache_4h,
            )
        else:
            klines = await self._get_component_klines(symbol, "4h", limit, proxy, self.kline_cache_4h)
        if klines:
            self.kline_cache_4h[symbol] = klines
        return klines

    async def _recalculate_4h_breakout_state(self, symbol: str) -> None:
        """Recalculate 4h ATR breakout state for ATR-path symbols."""
        klines = await self._fetch_4h_klines(symbol)
        if len(klines) < MIN_TRAILING_KLINES:
            return

        high = np.array([float(k.high) for k in klines], dtype=float)
        low = np.array([float(k.low) for k in klines], dtype=float)
        close = np.array([float(k.close) for k in klines], dtype=float)
        atr4h = calculate_atr(high, low, close, self.atr4h_period, self.atr4h_ma_type)
        atr4h_natrr = calculate_atr(high, low, close, 20, "RMA (Standard ATR)")
        prev_state = self.benchmark.get(symbol, {}).get("atr4h_state", (float("nan"), float("nan"), 0))

        for i in range(len(close)):
            upper, lower, ch = run_atr_channel(close[i], atr4h[i], self.atr4h_mult, prev_state)
            prev_state = (upper, lower, ch)

        atr4h_upper, atr4h_lower, atr4h_ch = prev_state
        bm = self.benchmark.setdefault(symbol, {})
        bm.update(
            {
                "atr4h_upper": float(atr4h_upper) if not np.isnan(atr4h_upper) else 0,
                "atr4h_lower": float(atr4h_lower) if not np.isnan(atr4h_lower) else 0,
                "atr4h_ch": atr4h_ch,
                "atr4h_state": prev_state,
                "atr4h_raw": float(atr4h[-1]) if np.isfinite(atr4h[-1]) else 0,
                "atr4h_natrr": float(atr4h_natrr[-1]) if np.isfinite(atr4h_natrr[-1]) else 0,
                "atr4h_kline_time": int(klines[-1].open_time),
            }
        )

    async def _ct_check_signals_4h(self, symbol: str) -> None:
        """Check 4h ATR Channel breakout without affecting trailing stop state."""
        if self._use_clustering_for_symbol(symbol):
            return
        current_price = self.mark_prices.get(symbol)
        if not current_price:
            return
        last_update = self.mark_price_times.get(symbol, 0)
        if time.time() - last_update > PRICE_STALE_THRESHOLD_SECONDS:
            return
        bm = self.benchmark.get(symbol, {})
        atr_upper = bm.get("atr4h_upper", 0)
        atr_lower = bm.get("atr4h_lower", 0)
        if atr_upper <= 0 and atr_lower <= 0:
            return
        if not self._initialized:
            return

        now = time.time()
        prev_state = self.last_atr4h_state.get(symbol, {"ch": 0, "sent": None})
        natr_raw = bm.get("atr4h_natrr", 0)
        natr = (natr_raw / current_price * 100) if current_price > 0 and natr_raw > 0 else None
        alert_key = f"ATR_4H_{symbol}"

        if current_price >= atr_upper and prev_state["ch"] != 1:
            last_alert = self.last_alert_time.get(alert_key, 0)
            if now - last_alert > ATR_BREAKOUT_COOLDOWN_SECONDS:
                self.last_alert_time[alert_key] = now
                self.last_atr4h_state[symbol] = {"ch": 1, "sent": "LONG"}
                await self._send_webhook(
                    ALERT_ATR_CHANNEL,
                    format_directional_signal_message(symbol, DIRECTION_LONG, "4H"),
                    {
                        "symbol": symbol,
                        "direction": DIRECTION_LONG,
                        "timeframe": "4H",
                        "price": current_price,
                        "atr_upper": atr_upper,
                        "atr_lower": atr_lower,
                        "natr": natr,
                    },
                )
                self._increment_alert_count()
        elif current_price <= atr_lower and prev_state["ch"] != -1:
            last_alert = self.last_alert_time.get(alert_key, 0)
            if now - last_alert > ATR_BREAKOUT_COOLDOWN_SECONDS:
                self.last_alert_time[alert_key] = now
                self.last_atr4h_state[symbol] = {"ch": -1, "sent": "SHORT"}
                await self._send_webhook(
                    ALERT_ATR_CHANNEL,
                    format_directional_signal_message(symbol, DIRECTION_SHORT, "4H"),
                    {
                        "symbol": symbol,
                        "direction": DIRECTION_SHORT,
                        "timeframe": "4H",
                        "price": current_price,
                        "atr_upper": atr_upper,
                        "atr_lower": atr_lower,
                        "natr": natr,
                    },
                )
                self._increment_alert_count()

    async def _fetch_15m_klines(self, symbol: str, limit: int = 500) -> list[Kline]:
        """Fetch 15m klines for single or pair symbol."""
        proxy = self.proxy_url if self.proxy_enable else None
        if self._is_pair_symbol(symbol):
            klines = await self._hl_fetch_pair_klines(
                symbol,
                limit=limit,
                interval="15m",
                proxy=proxy,
                kline_cache=self.kline_cache_15m,
            )
        else:
            klines = await self._get_component_klines(symbol, "15m", limit, proxy, self.kline_cache_15m)
        if klines:
            self.kline_cache_15m[symbol] = klines
        return klines

    async def _refresh_trailing_stop_channel(self, symbol: str, force: bool = False) -> None:
        """Refresh ATR trailing stop lines from full 15m history."""
        ts = self.trailing_stop.get(symbol)
        if not ts or not ts.get("active") or ts.get("use_clustering_ts"):
            return
        expected_open_time = self._expected_closed_open_time_ms(900)
        cached = self.kline_cache_15m.get(symbol, [])
        latest_open_time = int(cached[-1].open_time) if cached else 0
        now = time.time()
        last_attempt = self._last_trailing_refresh_attempt.get(symbol, 0.0)
        if not force and latest_open_time >= expected_open_time:
            return
        if not force and now - last_attempt < TRAILING_REFRESH_THROTTLE_SECONDS:
            return
        self._last_trailing_refresh_attempt[symbol] = now
        klines = await self._fetch_15m_klines(symbol)
        if len(klines) < MIN_TRAILING_KLINES:
            return
        close = np.array([float(k.close) for k in klines], dtype=float)
        high = np.array([float(k.high) for k in klines], dtype=float)
        low = np.array([float(k.low) for k in klines], dtype=float)
        atr = calculate_atr(high, low, close, self.atr15m_period, self.atr15m_ma_type)
        prev_state = (float("nan"), float("nan"), 0)
        for i in range(len(close)):
            upper, lower, ch = run_atr_channel(close[i], atr[i], float(ts["atr_mult"]), prev_state)
            prev_state = (upper, lower, ch)
        ts["atr15m_upper"] = prev_state[0]
        ts["atr15m_lower"] = prev_state[1]
        ts["atr15m_state"] = prev_state

    def _is_pair_trading(self, symbol: str) -> bool:
        """Check if symbol is part of any pair (e.g. BTC in BTC-ETH -> True)."""
        return any(symbol in comp for comp in self._pair_components.values())

    def _get_pair_for_symbol(self, symbol: str) -> tuple[str, str] | None:
        """Get components for a pair symbol (e.g. BTC-ETH -> (BTC, ETH))."""
        return self._pair_components.get(symbol)

    def _is_pair_symbol(self, symbol: str) -> bool:
        """Check if symbol is a pair symbol itself (e.g. BTC-ETH -> True, BTC -> False)."""
        return symbol in self.pair_list

    def _get_timestamp(self) -> str:
        """Get formatted timestamp based on configured timezone."""
        tz = self.timezone
        if tz == "Z":
            return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S+00:00")
        if tz.startswith(("+", "-")):
            sign = 1 if tz[0] == "+" else -1
            parts = tz[1:].split(":")
            td = timedelta(hours=sign * int(parts[0]), minutes=sign * int(parts[1]))
            tz_obj = timezone(td)
            local_time = datetime.now(UTC).astimezone(tz_obj)
            return local_time.strftime("%Y-%m-%dT%H:%M:%S%z")
        return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S+00:00")

    def _load_config(self, config_path: str) -> dict[str, Any]:
        """Load config from TOML file."""
        config_file_path = Path(config_path)
        if not config_file_path.exists():
            raise FileNotFoundError("Config file not found")  # noqa: TRY003
        with config_file_path.open("rb") as config_file:
            return tomllib.load(config_file)

    def _increment_alert_count(self) -> None:
        """Increment alert count thread-safely."""
        with self._lock:
            self._alert_count += 1

    async def _send_webhook(self, alert_type: str, message: str, extra: dict[str, Any] | None = None) -> None:
        """Send webhook notification."""
        await self.send_alert(alert_type, message, extra)

    async def connect(self) -> None:
        """Connect WebSocket and subscribe to all trading pairs."""
        try:
            await self._check_hyperliquid_connection()
            await self._connect_hyperliquid_ws()
            self.connected = True
            await self._send_webhook(ALERT_SYSTEM, format_connection_success_message(self._exchange_id))
        except Exception as e:
            await self._send_webhook(ALERT_ERROR, format_connection_failed_message(e))
            raise

    async def _check_hyperliquid_connection(self) -> None:
        """Check Hyperliquid REST API connectivity with retry."""
        max_retries = self.network.rest.retry.max_retries + 1

        for attempt in range(1, max_retries + 1):
            try:
                data = await self.market_gateway.check_connectivity()
                if isinstance(data, dict):
                    logger.info("Hyperliquid API connectivity check passed")
                    return
                logger.warning("Hyperliquid API returned unexpected payload, attempt %s/%s", attempt, max_retries)
            except Exception as e:
                logger.warning("Hyperliquid connectivity check failed: %s, attempt %s/%s", e, attempt, max_retries)

            if attempt < max_retries:
                delay = self.network.rest.retry.base_delay_seconds * (2 ** (attempt - 1))
                logger.info("Retrying in %ss...", delay)
                await asyncio.sleep(delay)

        msg = f"Hyperliquid API unreachable after {max_retries} attempts"
        logger.error(msg)
        raise ConnectionError(msg)

    async def _close_hyperliquid_ws(self) -> None:
        """Close current Hyperliquid websocket resources."""
        ws = getattr(self, "_hl_ws", None)
        session = getattr(self, "_hl_session", None)
        self._hl_ws = None
        self._hl_session = None
        await self.market_gateway.close_ws_resources(session, ws)

    async def _send_hyperliquid_ping(self) -> None:
        """Send websocket ping heartbeat to keep connection alive."""
        if not self._hl_ws:
            raise ConnectionError("ws unavailable")  # noqa: TRY003
        await self.market_gateway.send_ws_ping(self._hl_ws)

    async def _notify_ws_reconnect_failure(self, reason: str) -> None:
        """Send a single webhook when websocket enters reconnect mode."""
        if self._ws_reconnect_alert_active:
            return
        self._ws_reconnect_alert_active = True
        with suppress(Exception):
            await self._send_webhook(ALERT_ERROR, format_ws_reconnect_failure_message(reason))

    async def _notify_ws_reconnect_success(self, reason: str, attempt: int) -> None:
        """Send webhook when websocket recovers from reconnect mode."""
        if not self._ws_reconnect_alert_active:
            return
        self._ws_reconnect_alert_active = False
        with suppress(Exception):
            await self._send_webhook(ALERT_SYSTEM, format_ws_reconnect_success_message(reason, attempt))

    async def _reconnect_hyperliquid_ws(self, reason: str) -> bool:
        """Reconnect Hyperliquid websocket with exponential backoff."""
        self.connected = False
        delay = self.network.ws.reconnect_base_delay_seconds
        attempt = 0

        while self._hl_ws_running:
            attempt += 1
            await self._notify_ws_reconnect_failure(reason)
            logger.warning("Hyperliquid WS reconnecting (%s), attempt %s", reason, attempt)
            try:
                await self._close_hyperliquid_ws()
                await self._connect_hyperliquid_ws(start_watch_task=False)
                self.connected = True
                await self._notify_ws_reconnect_success(reason, attempt)
                logger.info("Hyperliquid WS reconnected after %s, attempt %s", reason, attempt)
                return True
            except Exception as e:
                logger.warning("Hyperliquid WS reconnect failed on attempt %s: %s", attempt, e)
                if not self._hl_ws_running:
                    return False
                await asyncio.sleep(delay)
                delay = min(delay * 2, self.network.ws.reconnect_max_delay_seconds)

        return False

    async def _connect_hyperliquid_ws(self, start_watch_task: bool = True) -> None:
        """Connect to Hyperliquid native WebSocket for mark prices."""
        self._hl_ws_running = True
        now = time.time()
        self._last_ws_message_time = now
        self._last_ws_data_time = now

        self._hl_session, self._hl_ws = await self.market_gateway.open_mark_price_stream("wss://api.hyperliquid.xyz/ws")

        if start_watch_task:
            self._ws_tasks = [asyncio.create_task(self._watch_hyperliquid_marks())]

    async def _watch_hyperliquid_marks(self) -> None:
        """Receive mark price updates from Hyperliquid WebSocket."""
        await self._ws_runtime_supervisor.run()

    async def _on_ticker(self, symbol: str, ticker: dict[str, Any]) -> None:
        """Handle ticker update."""
        try:
            price_str = ticker.get("last") or ticker.get("c") or "0"
            price = float(price_str)
            self.mark_prices[symbol] = price
            self.mark_price_times[symbol] = time.time()
            await self._ct_check_trailing_stop(symbol, price)
        except Exception as e:
            logger.warning(f"ticker error: {e}")

    async def _on_kline_1h(self, symbol: str, _kline: dict[str, Any]) -> None:
        """Handle 1h kline update."""
        try:
            await self._ct_update_klines(symbol)
            await self._ct_check_signals(symbol)
        except Exception as e:
            logger.warning(f"kline_1h error: {e}")

    async def _on_kline_15m(self, symbol: str, kline: dict[str, Any]) -> None:
        """Handle 15m kline update."""
        try:
            await self._ct_update_15m_atr(symbol, kline)
        except Exception as e:
            logger.warning(f"kline_15m error: {e}")

    async def _ct_check_trailing_stop(self, symbol: str, price: float) -> None:
        """Connector: bridge check_trailing_stop to instance method."""
        is_pair = self._is_pair_symbol(symbol)
        await self._signal_coordinator.check_trailing_stop(symbol, price, is_pair)

    async def _ct_check_signals(self, symbol: str) -> None:
        """Connector: bridge check_signals to instance method."""
        await self._signal_coordinator.check_signals(symbol, self._initialized)

    async def _ct_check_signals_clustering(self, symbol: str) -> None:
        """Connector: bridge check_signals_clustering to instance method."""
        await self._signal_coordinator.check_signals_clustering(symbol, self._initialized)

    async def _ct_start_breakout_monitor(self, symbol: str, direction: str, price: float, trigger_time: float) -> None:
        """Connector: start breakout monitoring with service-owned dependencies."""
        await self._signal_coordinator.start_breakout_monitor(symbol, direction, price, trigger_time)

    def _sync_breakout_monitor_from_cache(self, symbol: str) -> None:
        """Advance breakout monitor using cached 15m klines refreshed elsewhere in runtime."""
        self._signal_coordinator.sync_breakout_monitor_from_cache(symbol)

    async def _ct_check_breakout(self, symbol: str) -> None:
        """Connector: route breakout notifications through structured service alerts."""
        await self._signal_coordinator.check_breakout(symbol)

    async def _stop_breakout_monitor(self, symbol: str) -> None:
        """Stop breakout monitoring for a symbol."""
        self.breakout_monitor.pop(symbol, None)

    async def _get_component_klines(
        self,
        sym: str,
        interval: str,
        limit: int,
        proxy: str | None,
        kline_cache: dict[str, Any] | None,
    ) -> list[Kline]:
        """Get klines for a component symbol, checking global cache first."""
        _ = proxy
        _min_klines = 200
        cached = get_cached_klines(sym, interval)
        if cached and len(cached) >= _min_klines:
            return cached
        local: list[Kline] | None = kline_cache.get(sym) if kline_cache else None
        if local and len(local) >= _min_klines:
            return local
        client = self._create_rest_client()
        try:
            klines = await client.fetch_klines(sym, interval=interval, limit=limit)
        finally:
            await client.close()
        if klines:
            update_cache(sym, interval, klines)
        return klines

    async def _hl_fetch_klines(self, symbol: str, proxy: str | None = None) -> list[Kline]:
        """Fetch K-lines using native Hyperliquid REST API."""
        _ = proxy
        client = self._create_rest_client()
        try:
            return await client.fetch_klines(symbol, interval="1h", limit=500)
        finally:
            await client.close()

    async def _hl_fetch_pair_klines(
        self,
        symbol: str,
        limit: int = 500,
        interval: str = "1h",
        proxy: str | None = None,
        kline_cache: dict[str, Any] | None = None,
        _fetch_klines_fn: Any = None,
    ) -> list[Kline]:
        """Fetch pair K-lines using native Hyperliquid REST API."""
        pair = self._get_pair_for_symbol(symbol)
        if pair is None:
            return []
        sym1, sym2 = pair[0], pair[1]

        if interval in {"1h", "4h"}:
            bars_per_bucket = PAIR_15M_PER_1H if interval == "1h" else PAIR_15M_PER_4H
            klines1 = await self._get_component_klines(
                sym1, "15m", limit * bars_per_bucket, proxy, self.kline_cache_15m
            )
            klines2 = await self._get_component_klines(
                sym2, "15m", limit * bars_per_bucket, proxy, self.kline_cache_15m
            )
            if not klines1 or not klines2:
                return []
            pair_15m = build_pair_15m_klines(symbol, klines1, klines2)
            self.kline_cache_15m[symbol] = pair_15m
            aggregated = (
                aggregate_pair_15m_to_1h(symbol, pair_15m)
                if interval == "1h"
                else aggregate_pair_15m_to_4h(symbol, pair_15m)
            )
            return aggregated[-limit:] if limit > 0 else aggregated

        klines1 = await self._get_component_klines(sym1, interval, limit, proxy, kline_cache)
        klines2 = await self._get_component_klines(sym2, interval, limit, proxy, kline_cache)

        if not klines1 or not klines2:
            return []

        if kline_cache is not None:
            kline_cache[sym1] = klines1
            kline_cache[sym2] = klines2

        merged = build_pair_15m_klines(symbol, klines1, klines2)
        return merged[-limit:] if limit > 0 else merged

    async def _ct_update_klines(self, symbol: str) -> None:
        """Connector: bridge update_klines to instance method."""
        proxy = self.proxy_url if self.proxy_enable else None
        await update_klines(
            symbol,
            self.kline_cache,
            self.last_kline_time,
            lambda s: self._is_pair_symbol(s),
            proxy=proxy,
            recalculate_states_fn=lambda s: self._recalculate_states(s),
            fetch_pair_klines_fn=self._hl_fetch_pair_klines,
            recalculate_states_clustering_fn=self._ct_recalculate_states_clustering,
            exchange_id="hyperliquid",
        )

    async def _ct_recalculate_states_clustering(self, symbol: str) -> None:
        """Connector: bridge recalculate_states_clustering to instance method."""
        await recalculate_states_clustering(
            symbol,
            self.kline_cache,
            self.benchmark,
            self.clustering_states,
            self._is_pair_trading(symbol),
            self.st_period1,
            self.st_multiplier1,
            self.st_period2,
            self.st_multiplier2,
            self.vt_ema_signal,
            self.vt_ema_upper,
            self.vt_ema_lower,
            self.atr1h_period,
            self.atr1h_ma_type,
            self.atr1h_mult,
            self.atr15m_period,
            self.atr15m_ma_type,
            self.atr15m_mult,
            self.clustering_min_mult,
            self.clustering_max_mult,
            self.clustering_step,
            self.clustering_perf_alpha,
            self.clustering_from_cluster,
            self.clustering_max_iter,
            self.clustering_history_klines,
            self.debug,
        )

    async def _ct_update_15m_atr(self, symbol: str, kline: dict[str, Any]) -> None:
        """Connector: bridge update_15m_atr instance method."""
        await self.update_15m_atr(symbol, kline)

    async def _update_pair_price(self, symbol: str, symbol1: str, symbol2: str) -> None:
        """Update pair trading price."""
        p1 = self.mark_prices.get(symbol1, 0)
        p2 = self.mark_prices.get(symbol2, 0)
        if p1 <= 0 or p2 <= 0:
            return
        pair_price = p1 / p2
        self.mark_prices[symbol] = pair_price
        self.mark_price_times[symbol] = time.time()
        await self._ct_check_trailing_stop(symbol, pair_price)
        await self._ct_check_signals_clustering(symbol)

    async def update_15m_atr(self, symbol: str, kline: dict[str, Any]) -> None:
        """Update trailing stop ATR channel via WebSocket 15m K-line."""
        _ = kline
        await self._refresh_trailing_stop_channel(symbol, force=True)

    async def _recalculate_states(self, symbol: str) -> None:
        """Recalculate indicators for symbol."""
        await recalculate_states(
            symbol,
            self.kline_cache,
            self.benchmark,
            self._is_pair_trading(symbol),
            self.st_period1,
            self.st_multiplier1,
            self.st_period2,
            self.st_multiplier2,
            self.vt_ema_signal,
            self.vt_ema_upper,
            self.vt_ema_lower,
            self.atr1h_period,
            self.atr1h_ma_type,
            self.atr1h_mult,
            self.atr15m_period,
            self.atr15m_ma_type,
            self.atr15m_mult,
            self.debug,
        )

    async def initialize(self) -> None:  # noqa: PLR0912
        """Initialize service: fetch all K-lines and calculate indicators."""
        logger.info(f"Initializing klines for {len(self.symbols)} symbols...")
        self._initialized = False
        self._prune_runtime_state()

        await self.market_gateway.fetch_meta()

        for symbol in self.single_list:
            await self._ct_update_klines(symbol)
            klines = self.kline_cache.get(symbol, [])
            if klines:
                self.mark_prices[symbol] = float(klines[-1].close)

        for symbol in self.pair_list:
            await self._ct_update_klines(symbol)
            pair = self._get_pair_for_symbol(symbol)
            if pair:
                c1, c2 = pair
                klines1 = self.kline_cache.get(c1, [])
                klines2 = self.kline_cache.get(c2, [])
                if klines1 and c1 not in self.mark_prices:
                    self.mark_prices[c1] = float(klines1[-1].close)
                if klines2 and c2 not in self.mark_prices:
                    self.mark_prices[c2] = float(klines2[-1].close)
                p1 = self.mark_prices.get(c1, 0)
                p2 = self.mark_prices.get(c2, 0)
                if p1 > 0 and p2 > 0:
                    self.mark_prices[symbol] = p1 / p2

        for symbol in self.single_list:
            await self._recalculate_states(symbol)
            await self._recalculate_4h_breakout_state(symbol)
        for symbol in self.pair_list:
            if self._use_clustering_for_symbol(symbol):
                await self._ct_recalculate_states_clustering(symbol)
            else:
                await self._recalculate_states(symbol)
                await self._recalculate_4h_breakout_state(symbol)

        for symbol in self.single_list:
            self._log_symbol_state(symbol)
        for symbol in self.pair_list:
            self._log_symbol_state(symbol)

        self._initialized = True
        logger.info("Initialization complete")

    def _log_symbol_state(self, symbol: str) -> None:
        """Log current state of a symbol."""
        price = self.mark_prices.get(symbol, 0)
        if price <= 0:
            klines = self.kline_cache.get(symbol, [])
            if klines:
                price = float(klines[-1].close)
        bm = self.benchmark.get(symbol, {})
        atr_ch = bm.get("atr1h_ch", 0)
        atr_upper = bm.get("atr1h_upper", 0)
        atr_lower = bm.get("atr1h_lower", 0)
        atr_natrr = bm.get("atr1h_natrr", 0)

        if atr_ch == 1:
            atr_dir = "LONG"
        elif atr_ch == -1:
            atr_dir = "SHORT"
        else:
            atr_dir = "NEUTRAL"

        if price > 0 and atr_natrr > 0:
            natr = (atr_natrr / price) * 100
            logger.info(
                f"[{symbol}] {atr_dir}@{price:.4f} | ATR_Ch[{atr_upper:.4f}, {atr_lower:.4f}] | NATR {natr:.2f}%"
            )
        else:
            logger.info(f"[{symbol}] {atr_dir}@{price:.4f} | ATR_Ch[{atr_upper:.4f}, {atr_lower:.4f}] | NATR N/A")

    async def _send_initial_state_summary(self) -> None:
        """Send initial state summary for all symbols after initialization."""
        lines: list[str] = []
        for sym in self.symbols:
            is_pair = self._is_pair_symbol(sym)
            price = self.mark_prices.get(sym, 0)
            if price <= 0:
                klines = self.kline_cache.get(sym, [])
                if klines:
                    price = float(klines[-1].close)
            bm = self.benchmark.get(sym, {})
            atr_ch = bm.get("atr1h_ch", 0)
            atr_upper = bm.get("atr1h_upper", 0)
            atr_lower = bm.get("atr1h_lower", 0)
            atr_natrr = bm.get("atr1h_natrr", 0)

            if atr_ch == 1:
                atr_dir = "LONG"
            elif atr_ch == -1:
                atr_dir = "SHORT"
            else:
                atr_dir = "NEUTRAL"

            if price <= 0:
                logger.info(
                    f"[{sym}] ATR_Ch={atr_dir} | Range=[{atr_lower:.4f}, {atr_upper:.4f}] | Waiting for WS price..."
                )
                continue

            pd_val = get_price_decimals(sym)

            if price > 0 and atr_natrr > 0:
                natr = (atr_natrr / price) * 100
                natr_str = f"NATR {natr:.2f}%"
            else:
                natr_str = "NATR N/A"

            if is_pair:
                st_state = self.last_st_state.get(sym, "neutral")
                lines.append(
                    f"{sym} | {atr_dir}@{price:.{pd_val}f} | ATR_Ch[{atr_upper:.{pd_val}f}, {atr_lower:.{pd_val}f}] | {natr_str} | ST:{st_state}"
                )
            else:
                lines.append(
                    f"{sym} | {atr_dir}@{price:.{pd_val}f} | ATR_Ch[{atr_upper:.{pd_val}f}, {atr_lower:.{pd_val}f}] | {natr_str}"
                )

        msg = "READY\n" + "\n".join(lines)
        await self._send_webhook(ALERT_SYSTEM, msg)

    async def run(self) -> None:
        """Main service run loop."""
        self.running = True
        await self.initialize()
        await self.connect()
        await asyncio.sleep(2)
        await self._send_initial_state_summary()
        try:
            while self.running:
                await asyncio.sleep(1)
        except (KeyboardInterrupt, asyncio.CancelledError):
            logger.info("Received shutdown signal, stopping service...")
            await self.stop()

    async def stop(self) -> None:
        """Stop service."""
        self.running = False
        self._hl_ws_running = False
        for task in self._ws_tasks:
            if not task.done():
                task.cancel()
        if self._ws_tasks:
            await asyncio.gather(*self._ws_tasks, return_exceptions=True)
            self._ws_tasks.clear()
        await self._close_hyperliquid_ws()
        await self._webhook_sender.close()
        if self.observer:
            self.observer.stop()
            self.observer.join()
        logger.info("Service stopped")

    def warn(self, msg: str, context: str = "") -> None:
        """Send warning log (no Feishu push)."""
        log_warning(f"Warning{f' ({context})' if context else ''}: {msg}")
