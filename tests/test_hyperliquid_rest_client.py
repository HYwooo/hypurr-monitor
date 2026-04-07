"""
Unit tests for Hyperliquid REST client.

Covers:
- Rate limiting (HTTP 429)
- Network errors (timeout, connection failure)
- Error responses (HTTP 500, malformed JSON)
- Empty returns
- Malformed data handling
- Successful K-line fetching with pagination
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hyperliquid.rest_client import (
    HyperliquidRateLimiter,
    HyperliquidREST,
    fetch_klines,
)

pytestmark = pytest.mark.asyncio


class TestHyperliquidRateLimiter:
    """Test token bucket rate limiter."""

    async def test_acquire_decreases_tokens(self) -> None:
        """Acquiring tokens should decrease token count."""
        limiter = HyperliquidRateLimiter(max_tokens=10.0, refill_rate=1.0, tokens=10.0)
        await limiter.acquire(5.0)
        assert limiter.tokens == 5.0

    async def test_acquire_waits_when_empty(self) -> None:
        """Should wait when tokens insufficient."""
        limiter = HyperliquidRateLimiter(max_tokens=10.0, refill_rate=10.0, tokens=1.0)
        start = asyncio.get_event_loop().time()
        await limiter.acquire(5.0)
        elapsed = asyncio.get_event_loop().time() - start
        assert elapsed >= 0.35

    async def test_refill_adds_tokens(self) -> None:
        """Refill should add tokens over time."""
        limiter = HyperliquidRateLimiter(max_tokens=10.0, refill_rate=100.0, tokens=0.0)
        await asyncio.sleep(0.05)
        limiter._refill()
        assert limiter.tokens > 0

    async def test_tokens_capped_at_max(self) -> None:
        """Tokens should not exceed max."""
        limiter = HyperliquidRateLimiter(max_tokens=10.0, refill_rate=100.0, tokens=5.0)
        limiter.tokens = 15.0
        limiter._refill()
        assert limiter.tokens == 10.0


class TestHyperliquidRESTParseSymbol:
    """Test symbol parsing."""

    def test_simple_symbol_btc(self) -> None:
        """Simple symbol parsing."""
        client = HyperliquidREST()
        coin, params = client._parse_symbol("BTC")
        assert coin == "BTC"
        assert params == {}

    def test_xyz_gold(self) -> None:
        """HIP-3 symbol with prefix."""
        client = HyperliquidREST()
        coin, params = client._parse_symbol("xyz:GOLD")
        assert coin == "xyz:GOLD"
        assert params == {}

    def test_spot_prefix(self) -> None:
        """Spot symbol with prefix."""
        client = HyperliquidREST()
        coin, params = client._parse_symbol("spot:HYPE")
        assert coin == "HYPE"
        assert params == {"type": "spot"}

    def test_whitespace_trimmed(self) -> None:
        """Symbol whitespace should be trimmed."""
        client = HyperliquidREST()
        coin, params = client._parse_symbol("  BTC  ")
        assert coin == "BTC"


class TestHyperliquidRESTIntervalMs:
    """Test interval to milliseconds conversion."""

    @pytest.mark.parametrize(
        ("interval", "expected_ms"),
        [
            ("1m", 60_000),
            ("5m", 300_000),
            ("15m", 900_000),
            ("1h", 3_600_000),
            ("4h", 14_400_000),
            ("1d", 86_400_000),
            ("1w", 604_800_000),
        ],
    )
    def test_interval_mapping(self, interval: str, expected_ms: int) -> None:
        """All standard intervals should map correctly."""
        client = HyperliquidREST()
        assert client._interval_ms(interval) == expected_ms

    def test_unknown_interval_defaults_to_1h(self) -> None:
        """Unknown interval defaults to 1h."""
        client = HyperliquidREST()
        assert client._interval_ms("unknown") == 3_600_000


class TestHyperliquidRESTFetchKlines:
    """Test K-line fetching with mocked HTTP responses."""

    async def test_fetch_klines_empty_response(self) -> None:
        """Empty list response returns empty klines."""
        client = HyperliquidREST()
        client._post = AsyncMock(return_value=[])  # type: ignore[method-assign]
        result = await client.fetch_klines("BTC", limit=10)
        assert result == []

    async def test_fetch_klines_rate_limited(self) -> None:
        """HTTP 429 should raise exception."""
        from aiohttp import ClientResponseError

        client = HyperliquidREST()

        async def raise_429(*args: Any, **kwargs: Any) -> None:
            raise ClientResponseError(
                request_info=MagicMock(),
                history=(),
                status=429,
                message="Too Many Requests",
            )

        client._post = AsyncMock(side_effect=raise_429)  # type: ignore[method-assign]
        with pytest.raises(ClientResponseError) as exc_info:
            await client.fetch_klines("BTC", limit=10)
        assert exc_info.value.status == 429

    async def test_fetch_klines_network_error(self) -> None:
        """Network error should raise."""
        client = HyperliquidREST()
        client._post = AsyncMock(side_effect=OSError("Connection refused"))  # type: ignore[method-assign]
        with pytest.raises(OSError, match="Connection refused"):
            await client.fetch_klines("BTC", limit=10)

    async def test_fetch_klines_timeout(self) -> None:
        """Timeout should raise."""
        client = HyperliquidREST()
        client._post = AsyncMock(side_effect=TimeoutError("Request timed out"))  # type: ignore[method-assign]
        with pytest.raises(TimeoutError):
            await client.fetch_klines("BTC", limit=10)

    async def test_fetch_klines_server_error(self) -> None:
        """HTTP 500 should raise."""
        from aiohttp import ClientResponseError

        client = HyperliquidREST()

        async def raise_500(*args: Any, **kwargs: Any) -> None:
            raise ClientResponseError(
                request_info=MagicMock(),
                history=(),
                status=500,
                message="Internal Server Error",
            )

        client._post = AsyncMock(side_effect=raise_500)  # type: ignore[method-assign]
        with pytest.raises(ClientResponseError) as exc_info:
            await client.fetch_klines("BTC", limit=10)
        assert exc_info.value.status == 500

    async def test_fetch_klines_malformed_json(self) -> None:
        """Malformed JSON should raise."""
        client = HyperliquidREST()

        async def bad_json(*args: Any, **kwargs: Any) -> Any:
            raise ValueError("Expecting value")

        client._post = AsyncMock(side_effect=bad_json)  # type: ignore[method-assign]
        with pytest.raises(ValueError):
            await client.fetch_klines("BTC", limit=10)

    async def test_fetch_klines_non_list_response(self) -> None:
        """Non-list response treated as empty."""
        client = HyperliquidREST()
        client._post = AsyncMock(return_value={"error": "something went wrong"})  # type: ignore[method-assign]
        result = await client.fetch_klines("BTC", limit=10)
        assert result == []

    async def test_fetch_klines_success_single_batch(self) -> None:
        """Successful single-batch fetch."""
        client = HyperliquidREST()
        now_ms = 1704067200000
        mock_data = [
            {"t": now_ms - 3600000, "o": "65000", "h": "66000", "l": "64000", "c": "65500", "v": "100"},
            {"t": now_ms, "o": "65500", "h": "66500", "l": "64500", "c": "66000", "v": "150"},
        ]
        client._post = AsyncMock(return_value=mock_data)  # type: ignore[method-assign]
        result = await client.fetch_klines("BTC", interval="1h", limit=500)
        assert len(result) == 2
        assert result[0].open == 65000.0
        assert result[0].close == 65500.0
        assert result[1].high == 66500.0

    async def test_fetch_klines_pagination(self) -> None:
        """Pagination works when more than 2000 candles."""
        client = HyperliquidREST()
        now_ms = 1704067200000

        batch1 = [
            {"t": now_ms - 3600000 * i, "o": "65000", "h": "66000", "l": "64000", "c": "65500", "v": "100"}
            for i in range(2000)
        ]
        batch2 = [
            {"t": now_ms - 3600000 * (2000 + i), "o": "65000", "h": "66000", "l": "64000", "c": "65500", "v": "100"}
            for i in range(500)
        ]

        client._post = AsyncMock(side_effect=[batch1, batch2])  # type: ignore[method-assign]
        result = await client.fetch_klines("BTC", interval="1h", limit=2500)
        assert len(result) == 2500
        assert client._post.call_count == 2

    async def test_fetch_klines_xyz_gold(self) -> None:
        """xyz:GOLD symbol works."""
        client = HyperliquidREST()
        now_ms = 1704067200000
        mock_data = [
            {"t": now_ms - 3600000, "o": "2000", "h": "2100", "l": "1950", "c": "2050", "v": "50"},
        ]
        client._post = AsyncMock(return_value=mock_data)  # type: ignore[method-assign]
        result = await client.fetch_klines("xyz:GOLD", interval="1h", limit=10)
        assert len(result) == 1
        assert result[0].symbol == "xyz:GOLD"
        payload = client._post.call_args[0][0]
        assert payload["req"]["coin"] == "xyz:GOLD"


class TestFetchKlinesHelper:
    """Test the standalone fetch_klines helper."""

    async def test_helper_creates_and_closes_client(self) -> None:
        """Helper should create client and close after use."""
        client_instance = AsyncMock()
        client_instance.fetch_klines = AsyncMock(return_value=[])
        client_instance.close = AsyncMock()

        with patch("hyperliquid.rest_client.HyperliquidREST", return_value=client_instance):
            await fetch_klines("BTC")
            client_instance.close.assert_called_once()


class TestHyperliquidRESTClose:
    """Test client cleanup."""

    async def test_close_without_session(self) -> None:
        """Close without session should not error."""
        client = HyperliquidREST()
        await client.close()

    async def test_close_with_session(self) -> None:
        """Close with session should close it."""
        client = HyperliquidREST()
        mock_session = AsyncMock()
        mock_session.closed = False
        mock_session.close = AsyncMock()
        client._session = mock_session
        await client.close()
        mock_session.close.assert_called_once()
