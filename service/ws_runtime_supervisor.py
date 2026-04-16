"""Runtime supervisor for websocket receive/reconnect/ping lifecycle."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any

import aiohttp
import orjson

from logging_config import get_logger

logger = get_logger(__name__)


class WSRuntimeSupervisor:
    """Own websocket receive loop, silence checks, ping, and reconnect flow."""

    def __init__(  # noqa: PLR0913
        self,
        should_run_fn: Callable[[], bool],
        check_data_silence_fn: Callable[[], Awaitable[bool]],
        receive_message_fn: Callable[[], Awaitable[aiohttp.WSMessage]],
        send_ping_fn: Callable[[], Awaitable[None]],
        reconnect_fn: Callable[[str], Awaitable[bool]],
        mark_message_received_fn: Callable[[float], None],
        process_payload_fn: Callable[[dict[str, Any]], Awaitable[bool]],
    ) -> None:
        self._should_run_fn = should_run_fn
        self._check_data_silence_fn = check_data_silence_fn
        self._receive_message_fn = receive_message_fn
        self._send_ping_fn = send_ping_fn
        self._reconnect_fn = reconnect_fn
        self._mark_message_received_fn = mark_message_received_fn
        self._process_payload_fn = process_payload_fn

    async def run(self) -> None:  # noqa: PLR0912
        """Run websocket receive loop until supervisor is stopped."""
        while self._should_run_fn():
            if await self._check_data_silence_fn():
                continue
            try:
                msg = await self._receive_message_fn()
            except TimeoutError:
                if await self._check_data_silence_fn():
                    continue
                try:
                    await self._send_ping_fn()
                except Exception:
                    logger.exception("Hyperliquid WS ping failed")
                    if not await self._reconnect_fn("ping timeout"):
                        break
                continue
            except Exception:
                logger.exception("Hyperliquid WS receive error")
                if not await self._reconnect_fn("receive error"):
                    break
                continue

            try:
                self._mark_message_received_fn(time.time())
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = orjson.loads(msg.data)
                    if await self._process_payload_fn(data):
                        continue
                    if data.get("channel") == "pong":
                        logger.debug("Hyperliquid WS pong received")
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    logger.warning("Hyperliquid WS closed with message type %s", msg.type.name)
                    if not await self._reconnect_fn(f"message {msg.type.name}"):
                        break
            except Exception:
                logger.exception("Hyperliquid WS message handling error")
