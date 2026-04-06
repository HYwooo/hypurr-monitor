"""
ICandlestickService - Abstract interface for candlestick data service.
This is a placeholder for future microservice architecture.
"""

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any


class ICandlestickService(ABC):
    """
    Abstract interface for CandleSticks microservice.

    Other modules interact with CandleSticks through this interface:
    - Sync get: get_klines / get_latest / get_price
    - Async subscribe: subscribe_klines / subscribe_prices (data push to callback)
    - Lifecycle: fetch_and_cache (startup warmup) / start / stop
    """

    @abstractmethod
    async def get_klines(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 500,
    ) -> list[Any]:
        """Get historical K-lines (from cache, priority L1 > L2)."""

    @abstractmethod
    async def get_latest(self, symbol: str, interval: str = "1h") -> Any:
        """Get latest K-line."""

    @abstractmethod
    async def get_price(self, symbol: str) -> float:
        """Get current price."""

    @abstractmethod
    async def subscribe_klines(
        self,
        symbols: list[str],
        intervals: list[str],
        callback: Callable[..., Any],
    ) -> list[asyncio.Task[Any]]:
        """Subscribe to real-time K-line updates."""

    @abstractmethod
    async def subscribe_prices(
        self,
        symbols: list[str],
        callback: Callable[[str, float], None],
    ) -> list[asyncio.Task[Any]]:
        """Subscribe to real-time price updates."""

    @abstractmethod
    async def fetch_and_cache(
        self,
        symbols: list[str],
        intervals: list[str],
    ) -> None:
        """Startup: prefetch K-lines from REST API."""

    @abstractmethod
    async def start(self) -> None:
        """Start service: connect WebSocket, start subscriptions."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop service: disconnect WebSocket, cancel subscriptions."""

    @abstractmethod
    async def get_all_prices(self) -> dict[str, float]:
        """Get all cached prices."""
