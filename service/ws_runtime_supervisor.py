"""Runtime supervisor for websocket receive/reconnect/ping lifecycle."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

import aiohttp
import orjson

from logging_config import get_logger

logger = get_logger(__name__)

QUEUE_MAXSIZE = 100


class WSRuntimeSupervisor:
    """Own websocket receive loop, silence checks, ping, reconnect, and payload queue."""

    def __init__(  # noqa: PLR0913
        self,
        should_run_fn: Callable[[], bool],
        check_data_silence_fn: Callable[[], Awaitable[bool]],
        receive_message_fn: Callable[[], Awaitable[aiohttp.WSMessage]],
        send_ping_fn: Callable[[], Awaitable[None]],
        reconnect_fn: Callable[[str], Awaitable[bool]],
        mark_message_received_fn: Callable[[float], None],
        enqueue_payload_fn: Callable[[dict[str, Any]], None],
        process_payload_fn: Callable[[dict[str, Any]], Awaitable[bool]],
        queue_maxsize: int = QUEUE_MAXSIZE,
    ) -> None:
        self._should_run_fn = should_run_fn
        self._check_data_silence_fn = check_data_silence_fn
        self._receive_message_fn = receive_message_fn
        self._send_ping_fn = send_ping_fn
        self._reconnect_fn = reconnect_fn
        self._mark_message_received_fn = mark_message_received_fn
        self._enqueue_payload_fn = enqueue_payload_fn
        self._process_payload_fn = process_payload_fn
        self._queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=queue_maxsize)
        self._worker_task: asyncio.Task[None] | None = None

    def enqueue_payload(self, data: dict[str, Any]) -> None:
        """Enqueue a parsed payload for async processing."""
        try:
            self._queue.put_nowait(data)
        except asyncio.QueueFull:
            logger.warning("Hyperliquid WS payload queue full, dropping oldest market data")

    async def _queue_worker(self) -> None:
        """Drain payload queue and call process_payload_fn for each item."""
        while True:
            try:
                item = await self._queue.get()
            except asyncio.CancelledError:
                while not self._queue.empty():
                    try:
                        self._queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                break
            if item is None:
                while not self._queue.empty():
                    try:
                        self._queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                break
            try:
                await self._process_payload_fn(item)
            except Exception:
                logger.exception("Queue worker payload processing error")
            finally:
                self._queue.task_done()

    async def run(self) -> None:  # noqa: PLR0912
        """Run websocket receive loop and queue worker until stopped."""
        self._worker_task = asyncio.create_task(self._queue_worker())
        try:
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
                        if data.get("channel") == "pong":
                            logger.debug("Hyperliquid WS pong received")
                            continue
                        self._enqueue_payload_fn(data)
                    elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        logger.warning("Hyperliquid WS closed with message type %s", msg.type.name)
                        if not await self._reconnect_fn(f"message {msg.type.name}"):
                            break
                except asyncio.QueueFull:
                    logger.warning("Hyperliquid WS payload queue full, dropping oldest market data")
                except Exception:
                    logger.exception("Hyperliquid WS message handling error")
        finally:
            await self._queue.join()
            await self._queue.put(None)
            if self._worker_task is not None:
                await self._worker_task
