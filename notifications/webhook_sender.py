"""Reusable webhook sender with shared aiohttp session and retry policy."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiohttp

from config.network import WebhookNetworkConfig

WEBHOOK_SUCCESS_STATUS_CODE = 200
HTTP_TOO_MANY_REQUESTS = 429
HTTP_SERVER_ERROR = 500


def _get_logger() -> logging.Logger:
    """Get logger instance."""
    return logging.getLogger(__name__)


def log_warning(msg: str) -> None:
    """Forward warning log to standard logger."""
    _get_logger().warning(msg)


def log_error(msg: str) -> None:
    """Forward error log to standard logger."""
    _get_logger().error(msg)


class WebhookSender:
    """Send webhook payloads through a shared aiohttp session."""

    def __init__(self, config: WebhookNetworkConfig) -> None:
        self._config = config
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self._config.timeout_seconds)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def send_json(self, webhook_url: str, payload: dict[str, Any]) -> None:
        """Send webhook payload with retry on transient transport failures."""
        attempt = 0
        max_attempts = self._config.retry.max_retries + 1
        last_error: Exception | None = None

        while attempt < max_attempts:
            attempt += 1
            session = await self._get_session()
            try:
                async with session.post(webhook_url, json=payload, proxy=self._config.proxy_url) as resp:
                    if resp.status == WEBHOOK_SUCCESS_STATUS_CODE:
                        return
                    is_transient_status = resp.status >= HTTP_SERVER_ERROR or resp.status == HTTP_TOO_MANY_REQUESTS
                    if is_transient_status and attempt < max_attempts:
                        log_warning(
                            f"Webhook transient failure: {resp.status}, retry {attempt}/{self._config.retry.max_retries}"
                        )
                        await asyncio.sleep(self._config.retry.base_delay_seconds * (2 ** (attempt - 1)))
                        continue
                    log_error(f"Webhook failed: {resp.status}")
                    return
            except (aiohttp.ClientError, TimeoutError, OSError) as exc:
                last_error = exc
                if attempt < max_attempts:
                    log_warning(f"Webhook transport error: {exc}, retry {attempt}/{self._config.retry.max_retries}")
                    await asyncio.sleep(self._config.retry.base_delay_seconds * (2 ** (attempt - 1)))
                    continue
                break

        if last_error is not None:
            log_error(f"Webhook error: {last_error}")

    async def close(self) -> None:
        """Close shared webhook session."""
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None
