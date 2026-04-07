"""
Hyperliquid REST client using native HTTP API.

API Reference: https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint

Rate Limits (per IP):
- 1200 weight per minute (shared across all requests)
- candleSnapshot: weight = 1 + ceil(items / 60)
- Other info endpoints: weight = 2 or 20

Target: 85-95% utilization (1020-1140 weight/min).
Design: Sliding window (60s) + burst bucket.
"""

import asyncio
import bisect
import logging
import time as time_module
from dataclasses import dataclass, field
from typing import Any

import aiohttp

from models import Kline

logger = logging.getLogger(__name__)

BASE_URL = "https://api.hyperliquid.xyz/info"
TIMEOUT = 30_000

# Sliding window parameters
MAX_WEIGHT_PER_MINUTE = 1140.0  # 95% of 1200
WINDOW_SECONDS = 60.0
# Burst: allow up to 100 weight in a single burst at startup
BURST_MAX = 100.0
# Sustained: refill rate per second
SUSTAINED_RATE = 18.0  # weight/s  → 1080/min ≈ 90% utilization

# Cache parameters
MAX_CACHED_KLINES = 500  # per symbol per interval (sufficient for ATR period=14)
CACHE_EXPIRY_1H_SECONDS = 24 * 3600  # 24 hours for 1h klines
CACHE_EXPIRY_15M_SECONDS = 2 * 3600  # 2 hours for 15m klines

# Request constraints
MAX_REQUEST_KLINES = 2000  # max candles per single request
SERVER_MAX_KLINES = 5000  # server-side max historical candles

INTERVAL_SECONDS = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "8h": 28800,
    "12h": 43200,
    "1d": 86400,
    "3d": 259200,
    "1w": 604800,
    "1M": 2592000,
}

INTERVAL_MS = {k: v * 1000 for k, v in INTERVAL_SECONDS.items()}

MAX_PRICE_DECIMALS = 6  # max decimals for perpetuals price
MIN_PRICE_DECIMALS = 0

_META_CACHE_TTL = 3600.0  # 1 hour

_meta_cache: dict[str, Any] = {}
_price_decimals_map: dict[str, int] = {}
_meta_cache_time: float = 0.0


async def fetch_meta(
    proxy: str | None = None,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """
    Fetch meta info (szDecimals for each symbol) from Hyperliquid API.

    Caches result for _META_CACHE_TTL seconds.
    Populates _price_decimals_map with price_decimals per coin.
    """
    global _meta_cache, _price_decimals_map, _meta_cache_time  # noqa: PLW0602,PLW0603

    now = time_module.time()
    if not force_refresh and _meta_cache and (now - _meta_cache_time) < _META_CACHE_TTL:
        return _meta_cache

    client = HyperliquidREST(proxy=proxy)
    try:
        payload: dict[str, Any] = {"type": "meta"}
        data = await client._post(payload, weight=1.0)  # noqa: SLF001
        if isinstance(data, dict) and "universe" in data:
            _meta_cache = data
            _price_decimals_map.clear()
            for asset in data.get("universe", []):
                name = asset.get("name", "")
                sz = asset.get("szDecimals", 4)
                _price_decimals_map[name] = max(MIN_PRICE_DECIMALS, MAX_PRICE_DECIMALS - sz)
            _meta_cache_time = now
            return data
        return {}
    finally:
        await client.close()


def get_price_decimals(coin: str) -> int:
    """
    Get price decimal places for a given coin.

    Uses szDecimals from meta response: price_decimals = max(0, MAX_PRICE_DECIMALS - szDecimals).
    Returns 2 as default if coin not in cache.
    """
    return _price_decimals_map.get(coin, 2)


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

    async def acquire(self, weight: float = 1.0, timeout: float = 60.0) -> None:
        """
        Acquire tokens, waiting if necessary.

        Args:
            weight: Number of tokens needed
            timeout: Maximum seconds to wait before raising TimeoutError

        Raises:
            asyncio.TimeoutError: If tokens cannot be acquired within timeout
        """
        async with self._lock:
            total_wait = 0.0
            while True:
                self._refill()
                if self.tokens >= weight:
                    self.tokens -= weight
                    return
                wait_time = (weight - self.tokens) / self.refill_rate
                if total_wait + wait_time > timeout:
                    raise RateLimitError(wait_time, timeout)
                await asyncio.sleep(wait_time)
                total_wait += wait_time


class RateLimitError(TimeoutError):
    """Raised when rate limiter cannot acquire tokens within timeout."""

    def __init__(self, needed: float, limit: float) -> None:
        self.needed = needed
        self.limit = limit
        super().__init__(f"rate limiter timeout: needed {needed:.1f}s > limit {limit:.1f}s")


# Legacy rate limiter (kept for backward compat until all callers migrate)
_legacy_rate_limiter = HyperliquidRateLimiter()


# =============================================================================
# RateLimiter2: Sliding Window + Burst
# - Tracks every request timestamp with its weight in a 60s sliding window
# - Always respects 1140 weight/min (95% of 1200)
# - Burst bucket allows quick startup requests without hard staggering
# - Target utilization: 85-95%
# =============================================================================


@dataclass
class RateLimiter2:
    """
    Hyperliquid REST API rate limiter using sliding window + burst.

    - Sliding window: sum(weights in last 60s) <= MAX_WEIGHT_PER_MINUTE
    - Burst bucket: allows BURST_MAX weight instantly at startup
    - Sustained refill: SUSTAINED_RATE weight/s (after burst consumed)

    This gives ~90% sustained utilization (1080 weight/min) while
    supporting burst initialization requests (~10 concurrent 500-kline fetches).
    """

    max_weight: float = MAX_WEIGHT_PER_MINUTE
    window_seconds: float = WINDOW_SECONDS
    burst_max: float = BURST_MAX
    sustained_rate: float = SUSTAINED_RATE

    _timestamps: list[tuple[float, float]] = field(default_factory=list)
    _burst_tokens: float = field(default_factory=lambda: BURST_MAX)
    _last_refill: float = field(default_factory=time_module.time)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def _refill_burst(self, now: float) -> None:
        elapsed = now - self._last_refill
        self._burst_tokens = min(self.burst_max, self._burst_tokens + elapsed * self.sustained_rate)
        self._last_refill = now

    def _window_weight(self, now: float) -> float:
        cutoff = now - self.window_seconds
        return sum(w for ts, w in self._timestamps if ts > cutoff)

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        self._timestamps = [(ts, w) for ts, w in self._timestamps if ts > cutoff]

    def _can_emit(self, weight: float, now: float) -> tuple[bool, float]:
        """
        Check if we can emit a request of given weight.

        Returns:
            (can_emit, wait_seconds)  — wait_seconds=0 if can_emit=True
        """
        self._prune(now)
        window_w = self._window_weight(now)
        effective = self._burst_tokens + (self.max_weight - window_w)
        if effective >= weight:
            return True, 0.0
        deficit = weight - effective
        wait = deficit / self.sustained_rate
        return False, wait

    async def acquire(self, weight: float = 1.0, timeout: float = 120.0) -> None:
        """
        Acquire rate limit quota, waiting if necessary.

        Args:
            weight: Weight of the request (default 1.0)
            timeout: Maximum seconds to wait

        Raises:
            RateLimitError: If quota cannot be acquired within timeout
        """
        async with self._lock:
            total_wait = 0.0
            while True:
                now = time_module.time()
                self._refill_burst(now)
                can_emit, wait_time = self._can_emit(weight, now)
                if can_emit:
                    self._timestamps.append((now, weight))
                    self._burst_tokens = max(0, self._burst_tokens - weight)
                    return
                if total_wait + wait_time > timeout:
                    raise RateLimitError(wait_time, timeout)
                await asyncio.sleep(min(wait_time, 1.0))
                total_wait += min(wait_time, 1.0)


# =============================================================================
# KlineCache: Per-symbol per-interval K-line cache with expiry and merging
# - Max 500 klines per cache entry
# - Incremental fetch: only fetch missing/outdated portions
# - Auto-expiry based on interval
# =============================================================================


@dataclass
class KlineCacheEntry:
    """Single cache entry: one symbol + one interval."""

    symbol: str
    interval: str
    klines: list[Kline] = field(default_factory=list)
    fetched_at: float = 0.0

    def _expiry_seconds(self) -> int:
        if self.interval == "1h":
            return CACHE_EXPIRY_1H_SECONDS
        if self.interval == "15m":
            return CACHE_EXPIRY_15M_SECONDS
        return 24 * 3600

    def is_expired(self, now: float) -> bool:
        if not self.klines:
            return True
        return (now - self.fetched_at) > self._expiry_seconds()

    def latest_open_time(self) -> int:
        return self.klines[-1].open_time if self.klines else 0

    def earliest_open_time(self) -> int:
        return self.klines[0].open_time if self.klines else 0

    def trim_to_size(self) -> None:
        if len(self.klines) > MAX_CACHED_KLINES:
            self.klines = self.klines[-MAX_CACHED_KLINES:]


_global_kline_cache: dict[str, KlineCacheEntry] = {}


def _cache_key(symbol: str, interval: str) -> str:
    return f"{symbol}|{interval}"


def get_cached_klines(symbol: str, interval: str) -> list[Kline] | None:
    """Return cached klines if entry exists and not expired, else None."""
    entry = _global_kline_cache.get(_cache_key(symbol, interval))
    if entry is None:
        return None
    now = time_module.time()
    if entry.is_expired(now):
        del _global_kline_cache[_cache_key(symbol, interval)]
        return None
    return entry.klines


def merge_klines(existing: list[Kline], new: list[Kline]) -> list[Kline]:
    """
    Merge new klines into existing cache, avoiding duplicates by open_time.

    - existing: sorted by open_time, oldest first
    - new: sorted by open_time, oldest first
    - Returns merged list sorted by open_time, oldest first, max 500 items
    """
    if not existing:
        return new
    if not new:
        return existing
    seen_times = {k.open_time for k in existing}
    merged = list(existing)
    for k in new:
        if k.open_time not in seen_times:
            bisect.insort(merged, k, key=lambda x: x.open_time)
            seen_times.add(k.open_time)
    if len(merged) > MAX_CACHED_KLINES:
        merged = merged[-MAX_CACHED_KLINES:]
    return merged


def update_cache(symbol: str, interval: str, klines: list[Kline]) -> KlineCacheEntry:
    """Insert or update a cache entry."""
    key = _cache_key(symbol, interval)
    existing = _global_kline_cache.get(key)
    if existing:
        merged = merge_klines(existing.klines, klines)
        existing.klines = merged
        existing.fetched_at = time_module.time()
        existing.trim_to_size()
        return existing
    klines_sorted = sorted(klines, key=lambda x: x.open_time)
    entry = KlineCacheEntry(
        symbol=symbol,
        interval=interval,
        klines=klines_sorted[-MAX_CACHED_KLINES:],
        fetched_at=time_module.time(),
    )
    _global_kline_cache[key] = entry
    return entry


# Global rate limiter instance (v2)
_rate_limiter = RateLimiter2()


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
        start_time: int | None = None,
    ) -> list[Kline]:
        """
        Fetch K-line (OHLCV) history via candleSnapshot.

        API: POST https://api.hyperliquid.xyz/info
        Body: {"type": "candleSnapshot", "req": {"coin": "BTC", "interval": "1h", "startTime": ..., "endTime": ...}}

        Response: [{"T": close_time, "c": "close", "h": "high", "i": "interval", "l": "low",
                    "n": n_trades, "o": "open", "s": "symbol", "t": open_time, "v": "volume"}]

        Supported intervals: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 8h, 12h, 1d, 3d, 1w, 1M
        Max 2000 candles per request. Server caps at 5000 total. Cache stores last 500.

        Weight: 1 + ceil(items / 60) per request.

        Args:
            symbol: Trading pair name, e.g. "BTC", "xyz:GOLD"
            interval: K-line interval (default "1h")
            limit: Number of K-lines to fetch (default 500)
            start_time: Epoch ms to fetch from (exclusive). None = no filter.

        Returns:
            Kline list, oldest first
        """
        coin, _ = self._parse_symbol(symbol)
        all_klines: list[Kline] = []
        end_time = int(time_module.time() * 1000)
        fetched = 0

        while fetched < limit:
            batch_size = min(MAX_REQUEST_KLINES, limit - fetched)
            batch_start = end_time - (batch_size * self._interval_ms(interval))

            payload = {
                "type": "candleSnapshot",
                "req": {
                    "coin": coin,
                    "interval": interval,
                    "startTime": int(batch_start),
                    "endTime": int(end_time),
                },
            }

            weight = 1.0 + batch_size / 60.0
            data = await self._post(payload, weight=weight)
            if not isinstance(data, list) or len(data) == 0:
                break

            for k in reversed(data):
                if fetched >= limit:
                    break
                k_time = int(k["t"])
                if start_time is not None and k_time < start_time:
                    continue
                all_klines.append(
                    Kline.from_dict(
                        symbol,
                        interval,
                        {
                            "open_time": k_time,
                            "open": float(k["o"]),
                            "high": float(k["h"]),
                            "low": float(k["l"]),
                            "close": float(k["c"]),
                            "volume": float(k["v"]),
                        },
                    )
                )
                fetched += 1

            if len(data) < MAX_REQUEST_KLINES:
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
        return INTERVAL_MS.get(interval, 3_600_000)

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None


async def fetch_klines(
    symbol: str,
    interval: str = "1h",
    limit: int = 500,
    proxy: str | None = None,
    start_time: int | None = None,
) -> list[Kline]:
    client = HyperliquidREST(proxy=proxy)
    try:
        return await client.fetch_klines(symbol, interval, limit, start_time)
    finally:
        await client.close()
