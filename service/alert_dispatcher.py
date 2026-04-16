"""Alert dispatching abstraction for structured and legacy notification calls."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from notifications import AlertEvent, WebhookSender, build_alert_event, send_alert_event


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
    ) -> None:
        self._webhook_url = webhook_url
        self._webhook_format = webhook_format
        self._log_file_path = log_file_path
        self._max_log_lines = max_log_lines
        self._get_timestamp_fn = get_timestamp_fn
        self._sender = sender

    async def send_event(self, event: AlertEvent) -> None:
        """Deliver an already-structured alert event."""
        await send_alert_event(
            self._webhook_url,
            self._webhook_format,
            event,
            log_file_path=self._log_file_path,
            max_log_lines=self._max_log_lines,
            get_timestamp_fn=self._get_timestamp_fn,
            sender=self._sender,
        )

    async def send_alert(self, alert_type: str, message: str, extra: dict[str, Any] | None = None) -> None:
        """Build and deliver a structured alert from legacy arguments."""
        await self.send_event(build_alert_event(alert_type, message, extra))
