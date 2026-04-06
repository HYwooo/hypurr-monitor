"""
Hyperliquid REST client using native HTTP API.

API Reference: https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint

Rate Limits (per IP):
- 1200 weight per minute (shared across all requests)
- Weight = 1 + floor(batch_length / 40) for exchange actions
- candleSnapshot: weight 1 + 1 per 60 items returned
- meta: weight 1

Token bucket: 20 requests/min = 600 weight/min (50% of limit) to be safe.
"""

import asyncio
import logging
import time as time_module
from dataclasses import dataclass, field
from typing import Any

import aiohttp

from models import Kline

logger = logging.getLogger(__name__)

BASE_URL = "https://api.hyperliquid.xyz/info"
TIMEOUT = 30_000


@dataclass
class HyperliquidRateLimiter:
    """
    Token bucket rate limiter for Hyperliquid REST API.

    Docs: https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/rate-limits-and-user-limits
    - IP limit: 1200 weight per minute
    - candleSnapshot weight = 1 + ceil(items / 60)
    - We use 20 req/min = ~600 weight (50% headroom) to be safe
    """

    tokens: float = field(default=20.0)
    max_tokens: float = 20.0
    refill_rate: float = 20.0 / 60.0
    last_refill: float = field(default_factory=time_module.time)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def _refill(self) -> None:
        now = time_module.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.max_tokens, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    async def acquire(self, weight: float = 1.0) -> None:
        """Acquire tokens, waiting if necessary."""
        async with self._lock:
            while True:
                self._refill()
                if self.tokens >= weight:
                    self.tokens -= weight
                    return
                wait_time = (weight - self.tokens) / self.refill_rate
                await asyncio.sleep(wait_time)


_rate_limiter = HyperliquidRateLimiter()


class HyperliquidREST:
    """Hyperliquid REST client using native HTTP API."""

    def __init__(self, proxy: str | None = None) -> None:
        self.proxy = proxy
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=TIMEOUT / 1000)
            connector = aiohttp.TCPConnector()
            self._session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        return self._session

    async def _post(self, payload: dict[str, Any], weight: float = 1.0) -> Any:
        await _rate_limiter.acquire(weight)
        session = await self._get_session()
        headers = {"Content-Type": "application/json"}
        try:
            async with session.post(BASE_URL, json=payload, headers=headers, proxy=self.proxy) as resp:
                resp.raise_for_status()
                return await resp.json()
        except Exception as e:
            logger.warning(f"Hyperliquid REST request failed: {e}")
            raise

    async def fetch_klines(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 500,
    ) -> list[Kline]:
        """
        Fetch K-line (OHLCV) history via candleSnapshot.

        API: POST https://api.hyperliquid.xyz/info
        Body: {"type": "candleSnapshot", "req": {"coin": "BTC", "interval": "1h", "startTime": ..., "endTime": ...}}

        Response: [{"T": close_time, "c": "close", "h": "high", "i": "interval", "l": "low",
                    "n": n_trades, "o": "open", "s": "symbol", "t": open_time, "v": "volume"}]

        Supported intervals: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 8h, 12h, 1d, 3d, 1w, 1M
        Max 500 candles per request. Pagination via startTime.

        Weight: 1 + ceil(items / 60) per request.

        Args:
            symbol: Trading pair name, e.g. "BTC", "xyz:GOLD"
            interval: K-line interval (default "1h")
            limit: Number of K-lines to fetch (default 500)

        Returns:
            Kline list, oldest first
        """
        coin, _ = self._parse_symbol(symbol)
        all_klines: list[Kline] = []
        end_time = int(time_module.time() * 1000)
        fetched = 0

        while fetched < limit:
            batch_size = min(500, limit - fetched)
            start_time = end_time - (batch_size * self._interval_ms(interval))

            payload = {
                "type": "candleSnapshot",
                "req": {
                    "coin": coin,
                    "interval": interval,
                    "startTime": start_time,
                    "endTime": end_time,
                },
            }

            weight = 1.0 + batch_size / 60.0
            data = await self._post(payload, weight=weight)
            if not isinstance(data, list) or len(data) == 0:
                break

            for k in reversed(data):
                if fetched >= limit:
                    break
                all_klines.append(
                    Kline.from_dict(
                        symbol,
                        interval,
                        {
                            "open_time": int(k["t"]),
                            "open": float(k["o"]),
                            "high": float(k["h"]),
                            "low": float(k["l"]),
                            "close": float(k["c"]),
                            "volume": float(k["v"]),
                        },
                    )
                )
                fetched += 1

            if len(data) < 500:  # noqa: PLR2004
                break

            end_time = int(data[0]["t"])

        all_klines.sort(key=lambda x: x.open_time)
        return all_klines

    def _parse_symbol(self, symbol: str) -> tuple[str, dict[str, str]]:
        original = symbol.strip()
        if original.lower().startswith("spot:"):
            return original[5:].strip(), {"type": "spot"}
        if ":" in original:
            parts = original.split(":", 1)
            if len(parts) == 2:  # noqa: PLR2004
                return original, {}
        return original, {}
        """
        Parse user symbol to Hyperliquid API coin format.

        Args:
            symbol: e.g. "BTC", "xyz:GOLD", "BTC/USDC:USDC", "spot:HYPE"

        Returns:
            tuple of (coin, extra_params)
        """
        original = symbol.strip()

        if original.lower().startswith("spot:"):
            return original[5:].strip(), {"type": "spot"}

        if ":" in original:
            parts = original.split(":", 1)
            if len(parts) == 2:  # noqa: PLR2004
                return original, {}

        return original, {}

    def _interval_ms(self, interval: str) -> int:
        mapping = {
            "1m": 60_000,
            "3m": 180_000,
            "5m": 300_000,
            "15m": 900_000,
            "30m": 1_800_000,
            "1h": 3_600_000,
            "2h": 7_200_000,
            "4h": 14_400_000,
            "8h": 28_800_000,
            "12h": 43_200_000,
            "1d": 86_400_000,
            "3d": 259_200_000,
            "1w": 604_800_000,
            "1M": 2_592_000_000,
        }
        return mapping.get(interval, 3_600_000)

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None


async def fetch_klines(
    symbol: str,
    interval: str = "1h",
    limit: int = 500,
    proxy: str | None = None,
) -> list[Kline]:
    client = HyperliquidREST(proxy=proxy)
    try:
        return await client.fetch_klines(symbol, interval, limit)
    finally:
        await client.close()
