"""
Main notification service - coordinates all modules, manages WebSocket and signal detection.
"""

import asyncio
import threading
import time
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

import aiohttp
import numpy as np
import orjson
import toml  # type: ignore[import-untyped]

from hyperliquid.rest_client import (
    HyperliquidREST,
    fetch_meta,
    get_cached_klines,
    get_price_decimals,
    update_cache,
)
from indicators import (
    calculate_atr,
    run_atr_channel,
)
from logging_config import get_logger
from ml_common import MLConfig
from models import Kline
from notifications import log_warning
from notifications.webhook import send_webhook
from signals import (
    check_signals,
    check_signals_clustering,
    check_trailing_stop,
    recalculate_states,
    recalculate_states_clustering,
    update_klines,
)
from strategy.signal_selector import MLSignalStrategy

logger = get_logger(__name__)

WEBHOOK_LOG_FILE = "webhook.log"


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

    def __init__(self, config_path: str = "config.toml", debug: bool = False):
        self.config_path = config_path
        self.debug = debug
        self.config = self._load_config(config_path)
        sym_config = self.config.get("symbols", {})
        self.single_list: list[str] = sym_config.get("single_list", [])
        self.pair_list: list[str] = sym_config.get("pair_list", [])
        self._pair_components: dict[str, tuple[str, str]] = {}
        for p in self.pair_list:
            if "-" in p:
                parts = p.split("-", 1)
                self._pair_components[p] = (parts[0], parts[1])
        self.symbols: list[str] = self.single_list + self.pair_list

        self.single_strategy: str = sym_config.get("single_strategy", "atr_channel")
        self.pair_strategy: str = sym_config.get("pair_strategy", "clustering_st")
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
        self.atr15m_ma_type = self.config.get("atr_15m", {}).get("ma_type", "HMA")
        self.atr15m_period = self.config.get("atr_15m", {}).get("period", 14)
        self.atr15m_mult = self.config.get("atr_15m", {}).get("mult", 1.3)

        cs_config = self.config.get("clustering_st", {})
        self.clustering_min_mult = cs_config.get("min_mult", 1.0)
        self.clustering_max_mult = cs_config.get("max_mult", 5.0)
        self.clustering_step = cs_config.get("step", 0.5)
        self.clustering_perf_alpha = cs_config.get("perf_alpha", 10.0)
        self.clustering_from_cluster = cs_config.get("from_cluster", "Best")
        self.clustering_max_iter = cs_config.get("max_iter", 1000)
        self.clustering_history_klines = cs_config.get("history_klines", 500)

        self.heartbeat_file = self.config["service"]["heartbeat_file"]
        self.proxy_enable = self.config.get("proxy", {}).get("enable", False)
        self.proxy_url = self.config.get("proxy", {}).get("url", "")
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
        self.benchmark: dict[str, dict[str, Any]] = {}
        self.last_st_state: dict[str, str] = {}
        self.last_atr_state: dict[str, dict[str, Any]] = {}
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

        ml_config = self.config.get("ml", {})
        self.ml_enable = ml_config.get("enable", False)
        self.ml_model_path = ml_config.get("model_path", "models/ml/")
        self.ml_config = MLConfig(
            neutral_scale=ml_config.get("labeling", {}).get("neutral_scale", 0.5),
            lookforward_bars=ml_config.get("labeling", {}).get("lookforward_bars", 1),
            train_ratio=ml_config.get("training", {}).get("train_ratio", 0.7),
            valid_ratio=ml_config.get("training", {}).get("valid_ratio", 0.15),
            use_lstm_residual=ml_config.get("model", {}).get("use_lstm_residual", True),
            primary_model=ml_config.get("model", {}).get("primary", "catboost"),
        )
        self.ml_strategy: MLSignalStrategy | None = None

    def _is_pair_trading(self, symbol: str) -> bool:
        """Check if symbol is part of any pair (e.g. BTC in BTC-ETH -> True)."""
        return any(symbol in comp for comp in self._pair_components.values())

    def _get_pair_for_symbol(self, symbol: str) -> tuple[str, str] | None:
        """Get components for a pair symbol (e.g. BTC-ETH -> (BTC, ETH))."""
        return self._pair_components.get(symbol)

    def _is_pair_symbol(self, symbol: str) -> bool:
        """Check if symbol is a pair symbol itself (e.g. BTC-ETH -> True, BTC -> False)."""
        return symbol in self.pair_list

    def _get_strategy_for_symbol(self, symbol: str) -> str:
        """Get the strategy for a symbol based on its type (single or pair)."""
        if self._is_pair_symbol(symbol):
            return self.pair_strategy
        return self.single_strategy

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
        if not Path(config_path).exists():
            raise FileNotFoundError("Config file not found")
        return cast(dict[str, Any], toml.load(config_path))

    def _increment_alert_count(self) -> None:
        """Increment alert count thread-safely."""
        with self._lock:
            self._alert_count += 1

    async def _send_webhook(self, alert_type: str, message: str, extra: dict[str, Any] | None = None) -> None:
        """Send webhook notification."""
        await send_webhook(
            self.webhook_url,
            self.webhook_format,
            alert_type,
            message,
            extra,
            self.max_log_lines,
            self._get_timestamp,
        )

    async def connect(self) -> None:
        """Connect WebSocket and subscribe to all trading pairs."""
        try:
            await self._check_hyperliquid_connection()
            await self._connect_hyperliquid_ws()
            self.connected = True
            await self._send_webhook("SYSTEM", f"hypurr-monitor connected to {self._exchange_id}")
        except Exception as e:
            await self._send_webhook("ERROR", f"Connection failed: {e}")
            raise

    async def _check_hyperliquid_connection(self) -> None:
        """Check Hyperliquid REST API connectivity with retry."""
        max_retries = 5
        base_delay = 2
        http_ok = 200

        for attempt in range(1, max_retries + 1):
            try:
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        "https://api.hyperliquid.xyz/info",
                        json={"type": "meta"},
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as resp,
                ):
                    if resp.status == http_ok:
                        logger.info("Hyperliquid API connectivity check passed")
                        return
                    logger.warning(
                        "Hyperliquid API returned status %s, attempt %s/%s", resp.status, attempt, max_retries
                    )
            except Exception as e:
                logger.warning("Hyperliquid connectivity check failed: %s, attempt %s/%s", e, attempt, max_retries)

            if attempt < max_retries:
                delay = base_delay * (2 ** (attempt - 1))
                logger.info("Retrying in %ss...", delay)
                await asyncio.sleep(delay)

        msg = f"Hyperliquid API unreachable after {max_retries} attempts"
        logger.error(msg)
        raise ConnectionError(msg)

    async def _connect_hyperliquid_ws(self) -> None:
        """Connect to Hyperliquid native WebSocket for mark prices."""
        self._hl_ws: aiohttp.ClientWebSocketResponse | None = None
        self._hl_session: aiohttp.ClientSession | None = None
        self._hl_ws_running = True

        self._hl_session = aiohttp.ClientSession()
        self._hl_ws = await self._hl_session.ws_connect(
            "wss://api.hyperliquid.xyz/ws",
            timeout=aiohttp.ClientWSTimeout(ws_receive=30),
        )

        for dex in ["", "xyz", "hyna", "flx", "vntl", "km", "cash", "para"]:
            if dex:
                sub = {"method": "subscribe", "subscription": {"type": "allMids", "dex": dex}}
            else:
                sub = {"method": "subscribe", "subscription": {"type": "allMids"}}
            await self._hl_ws.send_json(sub)

        self._ws_tasks = [asyncio.create_task(self._watch_hyperliquid_marks())]

    async def _watch_hyperliquid_marks(self) -> None:  # noqa: PLR0912
        """Receive mark price updates from Hyperliquid WebSocket."""
        while self._hl_ws_running and self._hl_ws:
            try:
                msg = await self._hl_ws.receive()
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = orjson.loads(msg.data)
                    if data.get("channel") == "allMids":
                        mids = data.get("data", {}).get("mids", {})
                        updated_symbols: set[str] = set()
                        for sym in self.symbols:
                            if sym in mids:
                                price = float(mids[sym])
                                self.mark_prices[sym] = price
                                self.mark_price_times[sym] = time.time()
                                updated_symbols.add(sym)
                        for c1, c2 in self._pair_components.values():
                            if c1 in mids and c1 not in updated_symbols:
                                self.mark_prices[c1] = float(mids[c1])
                                self.mark_price_times[c1] = time.time()
                                updated_symbols.add(c1)
                            if c2 in mids and c2 not in updated_symbols:
                                self.mark_prices[c2] = float(mids[c2])
                                self.mark_price_times[c2] = time.time()
                                updated_symbols.add(c2)
                        for sym in updated_symbols:
                            if sym not in self._logged_initial_price:
                                self._logged_initial_price.add(sym)
                                self._log_symbol_state(sym)
                            price = self.mark_prices.get(sym, 0)
                            if price <= 0:
                                continue
                            await self._ct_check_trailing_stop(sym, price)
                            if not self._is_pair_trading(sym):
                                await self._ct_check_signals_by_strategy(sym)
                        for pair_sym, (c1, c2) in self._pair_components.items():
                            p1 = self.mark_prices.get(c1, 0)
                            p2 = self.mark_prices.get(c2, 0)
                            if p1 > 0 and p2 > 0:
                                pair_price = p1 / p2
                                self.mark_prices[pair_sym] = pair_price
                                self.mark_price_times[pair_sym] = time.time()
                                if pair_sym not in self._logged_initial_price:
                                    self._logged_initial_price.add(pair_sym)
                                    self._log_symbol_state(pair_sym)
                                await self._ct_check_trailing_stop(pair_sym, pair_price)
                                await self._ct_check_signals_by_strategy(pair_sym)
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break
            except Exception:
                logger.exception("Hyperliquid WS error")

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
        if is_pair and self.disable_pair_trailing:
            return
        if not is_pair and self.disable_single_trailing:
            return
        await check_trailing_stop(
            symbol,
            price,
            self.trailing_stop,
            self._send_webhook,
            self._increment_alert_count,
            self.last_alert_time,
        )

    async def _ct_check_signals(self, symbol: str) -> None:
        """Connector: bridge check_signals to instance method."""
        await check_signals(
            symbol,
            self.mark_prices,
            self.mark_price_times,
            self.benchmark,
            self.trailing_stop,
            self.last_atr_state,
            self.last_alert_time,
            self._initialized,
            self.last_st_state,
            self.atr1h_ma_type,
            self.atr1h_period,
            self.atr1h_mult,
            self.atr15m_ma_type,
            self.atr15m_period,
            self.atr15m_mult,
            self._send_webhook,
            self._increment_alert_count,
        )

    async def _ct_check_signals_clustering(self, symbol: str) -> None:
        """Connector: bridge check_signals_clustering to instance method."""
        await check_signals_clustering(
            symbol,
            self.mark_prices,
            self.mark_price_times,
            self.benchmark,
            self.trailing_stop,
            self.last_clustering_state,
            self.last_alert_time,
            self._initialized,
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
            self._send_webhook,
            self._increment_alert_count,
        )

    async def _ct_check_signals_by_strategy(self, symbol: str) -> None:
        """Dispatch signal check based on configured strategy for the symbol."""
        strategy = self._get_strategy_for_symbol(symbol)
        if strategy == "clustering_st":
            await self._ct_check_signals_clustering(symbol)
        elif strategy == "ml":
            await self._ct_check_ml_signals(symbol)
        else:
            await self._ct_check_signals(symbol)

    async def _ct_check_ml_signals(self, symbol: str) -> None:
        """ML signal detection via trained CatBoost+LSTM model."""
        if not self.ml_enable:
            return
        if self.ml_strategy is None or not self.ml_strategy.is_ready():
            return

        klines = self.kline_cache.get(symbol, [])
        if len(klines) < 50:
            return

        signal = self.ml_strategy.predict(klines)
        if signal is None:
            return

        last_alert = self.last_alert_time.get(symbol, 0)
        if time.time() - last_alert < 60:
            return

        direction_map = {
            "UP": "涨",
            "DOWN": "跌",
            "NEUTRAL": "平",
        }
        direction = direction_map.get(signal.signal_type.name, signal.signal_type.name)

        extra = {
            "symbol": symbol,
            "signal": signal.signal_type.name,
            "direction_cn": direction,
            "confidence": f"{signal.confidence:.2%}",
            "prob_down": f"{signal.prob_down:.2%}",
            "prob_neutral": f"{signal.prob_neutral:.2%}",
            "prob_up": f"{signal.prob_up:.2%}",
            "price": f"{signal.price:.6f}",
            "atr": f"{signal.atr:.6f}",
            "model": "CatBoost+LSTM",
        }

        await self._send_webhook(
            "ML_SIGNAL",
            f"[{symbol}] ML 信号: {direction} ({signal.confidence:.2%}) | P(跌)={signal.prob_down:.2%} P(平)={signal.prob_neutral:.2%} P(涨)={signal.prob_up:.2%}",
            extra,
        )
        self.last_alert_time[symbol] = time.time()
        self._increment_alert_count()

    async def _get_component_klines(
        self,
        sym: str,
        interval: str,
        limit: int,
        proxy: str | None,
        kline_cache: dict[str, Any] | None,
    ) -> list[Kline]:
        """Get klines for a component symbol, checking global cache first."""
        _min_klines = 200
        cached = get_cached_klines(sym, interval)
        if cached and len(cached) >= _min_klines:
            return cached
        local: list[Kline] | None = kline_cache.get(sym) if kline_cache else None
        if local and len(local) >= _min_klines:
            return local
        client = HyperliquidREST(proxy=proxy)
        try:
            klines = await client.fetch_klines(sym, interval=interval, limit=limit)
        finally:
            await client.close()
        if klines:
            update_cache(sym, interval, klines)
        return klines

    async def _hl_fetch_klines(self, symbol: str, proxy: str | None = None) -> list[Kline]:
        """Fetch K-lines using native Hyperliquid REST API."""
        client = HyperliquidREST(proxy=proxy)
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

        klines1 = await self._get_component_klines(sym1, interval, limit, proxy, kline_cache)
        klines2 = await self._get_component_klines(sym2, interval, limit, proxy, kline_cache)

        if not klines1 or not klines2:
            return []

        if kline_cache is not None:
            kline_cache[sym1] = klines1
            kline_cache[sym2] = klines2

        k2_by_time = {int(k.open_time): k for k in klines2}
        merged: list[Kline] = []
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
                Kline(
                    symbol=symbol,
                    interval=interval,
                    open_time=t,
                    open=ratio_o,
                    high=max(ratio_o, ratio_c, ratio_h),
                    low=min(ratio_o, ratio_c, ratio_l),
                    close=ratio_c,
                    volume=float(k1.volume),
                    close_time=int(k1.close_time) if k1.close_time else 0,
                    is_closed=k1.is_closed,
                )
            )
        return sorted(merged, key=lambda x: x.open_time)

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
        if symbol not in self.trailing_stop or not self.trailing_stop[symbol].get("active"):
            return
        try:
            h = float(kline.get("h", 0))
            low_price = float(kline.get("l", 0))
            c = float(kline.get("c", 0))
            if c == 0:
                return
            atr = self._calc_atr_single(h, low_price, c, self.atr15m_period, self.atr15m_ma_type)
            if atr <= 0:
                return
            ts = self.trailing_stop[symbol]
            prev_state = ts.get("atr15m_state", (float("nan"), float("nan"), 0))
            upper, lower, ch = run_atr_channel(c, atr, ts["atr_mult"], prev_state)
            ts["atr15m_upper"] = upper
            ts["atr15m_lower"] = lower
            ts["atr15m_state"] = (upper, lower, ch)
        except Exception as e:
            self.warn(f"update_15m_atr error for {symbol}: {e}")

    def _calc_atr_single(self, h: float, low_price: float, c: float, period: int, ma_type: int) -> float:
        """Calculate ATR for single bar."""
        if self._is_pair_trading:  # type: ignore[truthy-function]
            return 0
        high_arr = np.array([h])
        low_arr = np.array([low_price])
        close_arr = np.array([c])
        atr_arr = calculate_atr(high_arr, low_arr, close_arr, period, ma_type)
        return float(atr_arr[-1]) if len(atr_arr) > 0 else 0

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

    def _initialize_ml_strategy(self) -> None:
        """Initialize ML strategy by loading pre-trained models."""
        if not self.ml_enable:
            return
        try:
            self.ml_strategy = MLSignalStrategy(self.ml_config)
            loaded = self.ml_strategy.load_models(self.ml_model_path)
            if loaded:
                logger.info(f"ML strategy loaded from {self.ml_model_path}")
                logger.info(f"ML model info: {self.ml_strategy.model_info}")
            else:
                logger.warning(f"Failed to load ML models from {self.ml_model_path}")
        except Exception:
            logger.exception("ML strategy initialization failed")
            self.ml_strategy = None

    async def initialize(self) -> None:
        """Initialize service: fetch all K-lines and calculate indicators."""
        logger.info(f"Initializing klines for {len(self.symbols)} symbols...")
        self._initialized = False

        await fetch_meta(proxy=self.proxy_url if self.proxy_enable else None)

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

        if self.ml_enable:
            self._initialize_ml_strategy()

        for symbol in self.single_list:
            await self._recalculate_states(symbol)
        for symbol in self.pair_list:
            await self._ct_recalculate_states_clustering(symbol)

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
        await self._send_webhook("SYSTEM", msg)

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
        if hasattr(self, "_hl_ws") and self._hl_ws:
            await self._hl_ws.close()
        if hasattr(self, "_hl_session") and self._hl_session:
            await self._hl_session.close()
        if self.observer:
            self.observer.stop()
            self.observer.join()
        logger.info("Service stopped")

    def warn(self, msg: str, context: str = "") -> None:
        """Send warning log (no Feishu push)."""
        log_warning(f"Warning{f' ({context})' if context else ''}: {msg}")
