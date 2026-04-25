"""High-level orchestration helpers for NotificationService startup and run loop."""

from __future__ import annotations

import asyncio
from typing import Any

from hyperliquid.rest_client import get_price_decimals
from logging_config import get_logger
from notifications import ALERT_SYSTEM
from service.runtime_symbol_utils import seed_initial_signal_states

logger = get_logger(__name__)


class NotificationInitializationOrchestrator:
    """Own the initialize() workflow while keeping NotificationService thin."""

    def __init__(self, service: Any) -> None:
        """Bind orchestration to a concrete service instance."""
        self._service = service

    async def initialize(self) -> None:
        """Prepare runtime state, warm caches, and seed initial signals."""
        service = self._service
        service._initialized = False  # noqa: SLF001
        service._prune_runtime_state()  # noqa: SLF001

        await service.market_gateway.fetch_meta()

        for symbol in service.single_list:
            await service._ct_update_klines(symbol)  # noqa: SLF001
            klines = service.kline_cache.get(symbol, [])
            if klines:
                service.mark_prices[symbol] = float(klines[-1].close)

        for symbol in service.pair_list:
            await service._ct_update_klines(symbol)  # noqa: SLF001
            pair = service._get_pair_for_symbol(symbol)  # noqa: SLF001
            if pair:
                c1, c2 = pair
                klines1 = service.kline_cache.get(c1, [])
                klines2 = service.kline_cache.get(c2, [])
                if klines1 and c1 not in service.mark_prices:
                    service.mark_prices[c1] = float(klines1[-1].close)
                if klines2 and c2 not in service.mark_prices:
                    service.mark_prices[c2] = float(klines2[-1].close)
                p1 = service.mark_prices.get(c1, 0)
                p2 = service.mark_prices.get(c2, 0)
                if p1 > 0 and p2 > 0:
                    service.mark_prices[symbol] = p1 / p2

        for symbol in service.single_list:
            await service._recalculate_states(symbol)  # noqa: SLF001
            await service._recalculate_4h_breakout_state(symbol)  # noqa: SLF001
        for symbol in service.pair_list:
            if service._use_clustering_for_symbol(symbol):  # noqa: SLF001
                await service._ct_recalculate_states_clustering(symbol)  # noqa: SLF001
            else:
                await service._recalculate_states(symbol)  # noqa: SLF001
                await service._recalculate_4h_breakout_state(symbol)  # noqa: SLF001

        # Design note: seed current market states before enabling realtime edge detection.
        seed_initial_signal_states(
            service.symbols,
            service.benchmark,
            service.mark_prices,
            service.clustering_states,
            service._is_pair_symbol,  # noqa: SLF001
            service._use_clustering_for_symbol,  # noqa: SLF001
            service.last_atr_state,
            service.last_atr4h_state,
            service.last_clustering_state,
            service.last_st_state,
        )

        for symbol in service.symbols:
            service._log_symbol_state(symbol)  # noqa: SLF001

        service._initialized = True  # noqa: SLF001

    async def send_ready_summary(self) -> None:
        """Emit the READY summary after websocket connection is established."""
        service = self._service
        await asyncio.sleep(2)
        lines: list[str] = []
        for sym in service.symbols:
            is_pair = service._is_pair_symbol(sym)  # noqa: SLF001
            price = service.mark_prices.get(sym, 0)
            if price <= 0:
                klines = service.kline_cache.get(sym, [])
                if klines:
                    price = float(klines[-1].close)
            bm = service.benchmark.get(sym, {})
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
                service._log_symbol_state(sym)  # noqa: SLF001
                continue

            pd_val = get_price_decimals(sym)
            if price > 0 and atr_natrr > 0:
                natr = (atr_natrr / price) * 100
                natr_str = f"NATR {natr:.2f}%"
            else:
                natr_str = "NATR N/A"

            if is_pair:
                st_state = service.last_st_state.get(sym, "neutral")
                lines.append(
                    f"{sym} | {atr_dir}@{price:.{pd_val}f} | ATR_Ch[{atr_upper:.{pd_val}f}, {atr_lower:.{pd_val}f}] | {natr_str} | ST:{st_state}"
                )
            else:
                lines.append(
                    f"{sym} | {atr_dir}@{price:.{pd_val}f} | ATR_Ch[{atr_upper:.{pd_val}f}, {atr_lower:.{pd_val}f}] | {natr_str}"
                )

        msg = "READY\n" + "\n".join(lines)
        await service._send_webhook(ALERT_SYSTEM, msg)  # noqa: SLF001


class NotificationRunOrchestrator:
    """Own the high-level run loop orchestration."""

    def __init__(self, service: Any) -> None:
        """Bind orchestration to a concrete service instance."""
        self._service = service

    async def run(self) -> None:
        """Run initialize/connect/ready summary and the outer service loop."""
        service = self._service
        service.running = True
        await service.initialize()
        await service.connect()
        await service._ready_summary_orchestrator.send_ready_summary()  # noqa: SLF001
        try:
            while service.running:
                await asyncio.sleep(1)
        except (KeyboardInterrupt, asyncio.CancelledError):
            logger.info("Received shutdown signal, stopping service...")
            await service.stop()
