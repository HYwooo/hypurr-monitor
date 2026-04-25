"""Alert dispatching abstraction for structured and legacy notification calls."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from notifications import AlertEvent, WebhookSender, build_alert_event, send_alert_event
from notifications.webhook import WebhookDeliveryResult


@dataclass(slots=True)
class AlertDispatchStats:
    """Simple dispatch counters for operator visibility."""

    attempted: int = 0
    queued: int = 0
    dropped: int = 0
    sent: int = 0
    failed: int = 0
    deduped: int = 0


class AlertDispatcher:
    """Own the final alert delivery path to webhook transport."""

    def __init__(  # noqa: PLR0913
        self,
        webhook_url: str,
        webhook_format: str,
        log_file_path: str,
        max_log_lines: int,
        get_timestamp_fn: Callable[[], str],
        sender: WebhookSender,
        queue_maxsize: int | None = None,
    ) -> None:
        self._webhook_url = webhook_url
        self._webhook_format = webhook_format
        self._log_file_path = log_file_path
        self._max_log_lines = max_log_lines
        self._get_timestamp_fn = get_timestamp_fn
        self._sender = sender
        self._dedupe_ttl_seconds = 300
        self._dedupe_cache: dict[str, float] = {}
        self._stats = AlertDispatchStats()
        self._logger = logging.getLogger(__name__)
        self._queue: asyncio.Queue[AlertEvent | None] | None = None
        self._worker_task: asyncio.Task[None] | None = None
        self._closing = False
        if queue_maxsize is not None:
            self._queue = asyncio.Queue(maxsize=queue_maxsize)

    def _dedupe_ttl_for_event(self, event: AlertEvent) -> int:
        """Return dedupe TTL by alert type without changing legacy signal cooldowns."""
        # Design note: apply custom TTLs only to noisy/high-value alert types.
        ttl_by_type = {
            "SYSTEM": 30,
            "ERROR": 60,
            "BREAKOUT": 120,
        }
        return ttl_by_type.get(event.alert_type, self._dedupe_ttl_seconds)

    def _purge_expired_dedupe_entries(self, now: float) -> None:
        """Remove expired dedupe entries to keep cache bounded."""
        expired_keys = [key for key, expires_at in self._dedupe_cache.items() if expires_at <= now]
        for key in expired_keys:
            self._dedupe_cache.pop(key, None)

    def _should_send(self, event: AlertEvent) -> bool:
        """Check dedupe state before delivering an alert event."""
        dedupe_key = event.dedupe_key
        if not dedupe_key:
            return True
        now = time.time()
        self._purge_expired_dedupe_entries(now)
        expires_at = self._dedupe_cache.get(dedupe_key)
        if expires_at is not None and expires_at > now:
            return False
        self._dedupe_cache[dedupe_key] = now + self._dedupe_ttl_for_event(event)
        return True

    def _log_delivery_outcome(self, outcome: str, event: AlertEvent) -> None:
        """Log the final delivery outcome with stable alert context."""
        # Design note: log final outcomes at one exit point for easier diagnosis.
        self._logger.info(
            "Alert dispatch %s | alert_type=%s | symbol=%s | dedupe_key=%s",
            outcome,
            event.alert_type,
            event.symbol or "",
            event.dedupe_key or "",
        )

    def get_stats(self) -> AlertDispatchStats:
        """Return a snapshot of dispatch counters."""
        return AlertDispatchStats(
            attempted=self._stats.attempted,
            queued=self._stats.queued,
            dropped=self._stats.dropped,
            sent=self._stats.sent,
            failed=self._stats.failed,
            deduped=self._stats.deduped,
        )

    def _record_result(self, result: WebhookDeliveryResult) -> None:
        """Update counters from a structured delivery result."""
        self._stats.attempted += int(result.attempted)
        self._stats.sent += int(result.sent)
        self._stats.failed += int(result.failed)
        self._stats.deduped += int(result.deduped)

    async def start_worker(self) -> None:
        """Start the background queue worker when queueing is enabled."""
        if self._queue is None or self._worker_task is not None:
            return
        self._closing = False
        self._worker_task = asyncio.create_task(self._queue_worker(), name="alert-dispatcher-worker")

    async def stop_worker(self, drain_timeout_seconds: float = 1.0) -> None:
        """Stop the queue worker, draining briefly before forcing shutdown."""
        if self._worker_task is None or self._queue is None:
            return
        self._closing = True
        try:
            await asyncio.wait_for(self._queue.join(), timeout=drain_timeout_seconds)
        except TimeoutError:
            self._logger.warning("Alert dispatcher queue drain timed out; stopping worker")
        try:
            self._queue.put_nowait(None)
        except asyncio.QueueFull:
            self._logger.warning("Alert dispatcher queue full during shutdown; cancelling worker")
            self._worker_task.cancel()
        with suppress(asyncio.CancelledError):
            await asyncio.wait_for(self._worker_task, timeout=drain_timeout_seconds)
        self._worker_task = None

    async def _queue_worker(self) -> None:
        """Drain queued alerts and deliver them sequentially."""
        assert self._queue is not None
        while True:
            event = await self._queue.get()
            try:
                if event is None:
                    return
                result = await send_alert_event(
                    self._webhook_url,
                    self._webhook_format,
                    event,
                    log_file_path=self._log_file_path,
                    max_log_lines=self._max_log_lines,
                    get_timestamp_fn=self._get_timestamp_fn,
                    sender=self._sender,
                )
                self._record_result(result)
                self._log_delivery_outcome("send success" if result.sent else "send failed", event)
            finally:
                self._queue.task_done()

    async def send_event(self, event: AlertEvent) -> bool:
        """Deliver an already-structured alert event."""
        if not self._should_send(event):
            self._record_result(WebhookDeliveryResult(attempted=False, sent=False, failed=False, deduped=True))
            self._log_delivery_outcome("dedupe skipped", event)
            return False
        if self._queue is None:
            result = await send_alert_event(
                self._webhook_url,
                self._webhook_format,
                event,
                log_file_path=self._log_file_path,
                max_log_lines=self._max_log_lines,
                get_timestamp_fn=self._get_timestamp_fn,
                sender=self._sender,
            )
            self._record_result(result)
            self._log_delivery_outcome("send success" if result.sent else "send failed", event)
            return result.sent
        if self._closing:
            self._stats.dropped += 1
            self._logger.warning("Alert dispatcher is stopping; rejecting queued alert | alert_type=%s", event.alert_type)
            return False
        await self.start_worker()
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            self._stats.dropped += 1
            self._logger.warning(
                "Alert dispatcher queue full; dropping alert | alert_type=%s | symbol=%s | dedupe_key=%s",
                event.alert_type,
                event.symbol or "",
                event.dedupe_key or "",
            )
            return False
        self._stats.queued += 1
        return True

    async def send_alert(self, alert_type: str, message: str, extra: dict[str, Any] | None = None) -> bool:
        """Build and deliver a structured alert from legacy arguments."""
        return await self.send_event(build_alert_event(alert_type, message, extra))
