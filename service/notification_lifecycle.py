"""Lifecycle orchestration helpers for NotificationService."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from logging_config import get_logger
from notifications import (
    ALERT_ERROR,
    ALERT_SYSTEM,
    RECONNECT_ELAPSED_NOTIFY_THRESHOLD_SECONDS,
    format_connection_failed_message,
    format_connection_success_message,
)

logger = get_logger(__name__)


class NotificationLifecycleManager:
    """Own the thin lifecycle orchestration around NotificationService."""

    def __init__(self, service: Any) -> None:
        """Bind the lifecycle helper to a service instance."""
        self._service = service

    async def connect(self) -> None:
        """Connect websocket resources and emit lifecycle alerts."""
        try:
            await self._service._check_hyperliquid_connection()  # noqa: SLF001
            await self._service._connect_hyperliquid_ws()  # noqa: SLF001
            self._service.connected = True
            await self._service._send_webhook(  # noqa: SLF001
                ALERT_SYSTEM,
                format_connection_success_message(self._service._exchange_id),  # noqa: SLF001
            )
        except Exception as exc:
            await self._service._send_webhook(ALERT_ERROR, format_connection_failed_message(exc))  # noqa: SLF001
            raise

    async def reconnect_hyperliquid_ws(self, reason: str) -> bool:
        """Reconnect websocket resources with the existing retry policy."""
        self._service.connected = False
        delay = self._service.network.ws.reconnect_base_delay_seconds
        attempt = 0
        started_at = time.time()

        while self._service._hl_ws_running:  # noqa: SLF001
            attempt += 1
            elapsed = time.time() - started_at
            await self._service._notify_ws_reconnect_failure(reason, attempt)  # noqa: SLF001
            if attempt == 1 and elapsed <= RECONNECT_ELAPSED_NOTIFY_THRESHOLD_SECONDS:
                logger.info("Hyperliquid WS reconnected instantly (%s, attempt %s, %.1fs)", reason, attempt, elapsed)
            else:
                logger.warning("Hyperliquid WS reconnecting (%s), attempt %s", reason, attempt)
            try:
                await self._service._close_hyperliquid_ws()  # noqa: SLF001
                await self._service._connect_hyperliquid_ws(start_watch_task=False)  # noqa: SLF001
                self._service.connected = True
                await self._service._notify_ws_reconnect_success(reason, attempt, elapsed)  # noqa: SLF001
                return True
            except Exception as exc:
                logger.warning("Hyperliquid WS reconnect failed on attempt %s: %s", attempt, exc)
                if not self._service._hl_ws_running:  # noqa: SLF001
                    return False
                await asyncio.sleep(delay)
                delay = min(delay * 2, self._service.network.ws.reconnect_max_delay_seconds)

        return False

    async def stop(self) -> None:
        """Stop websocket, worker, sender, and observer resources."""
        self._service.running = False
        self._service._hl_ws_running = False  # noqa: SLF001
        for task in self._service._ws_tasks:  # noqa: SLF001
            if not task.done():
                task.cancel()
        if self._service._ws_tasks:  # noqa: SLF001
            await asyncio.gather(*self._service._ws_tasks, return_exceptions=True)  # noqa: SLF001
            self._service._ws_tasks.clear()  # noqa: SLF001
        await self._service._close_hyperliquid_ws()  # noqa: SLF001
        await self._service._webhook_sender.close()  # noqa: SLF001
        if self._service.observer:
            self._service.observer.stop()
            self._service.observer.join()
        logger.info("Service stopped")
