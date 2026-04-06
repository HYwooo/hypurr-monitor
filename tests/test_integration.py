"""
Integration tests for Hyperliquid API.

Requires network access to Hyperliquid REST/WebSocket APIs.
Tests real K-line fetching and mark price subscriptions.

Run with: pytest tests/test_integration.py -v -s
"""

from __future__ import annotations

import asyncio
from contextlib import suppress

import pytest

from hyperliquid.rest_client import HyperliquidREST
from hyperliquid.ws_client import HyperliquidWS, get_mark_prices_once

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


class TestHyperliquidRESTKlines:
    """Test real K-line fetching from Hyperliquid API."""

    @pytest.mark.parametrize(
        ("symbol", "interval"),
        [
            ("BTC", "1h"),
            ("BTC", "4h"),
            ("BTC", "1d"),
            ("ETH", "1h"),
            ("SOL", "1h"),
        ],
    )
    async def test_fetch_klines_real(self, symbol: str, interval: str) -> None:
        """Fetch real K-lines for various symbols and intervals."""
        client = HyperliquidREST()
        try:
            result = await client.fetch_klines(symbol, interval=interval, limit=10)
            assert len(result) > 0, f"Should return klines for {symbol} {interval}"
            kline = result[-1]
            assert kline.symbol == symbol
            assert kline.interval == interval
            assert kline.close > 0, f"Close price should be positive for {symbol}"
            assert kline.high >= kline.low, "High should be >= low"
            print(f"  {symbol} {interval}: {len(result)} bars, close={kline.close}")
        finally:
            await client.close()

    @pytest.mark.parametrize(
        ("symbol", "interval"),
        [
            ("BTC", "1m"),
            ("BTC", "5m"),
            ("BTC", "15m"),
            ("BTC", "30m"),
            ("BTC", "2h"),
            ("BTC", "8h"),
            ("BTC", "12h"),
        ],
    )
    async def test_fetch_klines_short_intervals(self, symbol: str, interval: str) -> None:
        """Fetch real K-lines for short intervals."""
        client = HyperliquidREST()
        try:
            result = await client.fetch_klines(symbol, interval=interval, limit=20)
            assert len(result) > 0, f"Should return klines for {symbol} {interval}"
            kline = result[-1]
            assert kline.close > 0
            print(f"  {symbol} {interval}: {len(result)} bars, close={kline.close}")
        finally:
            await client.close()

    async def test_fetch_xyz_gold(self) -> None:
        """Fetch xyz:GOLD K-lines (HIP-3 asset)."""
        client = HyperliquidREST()
        try:
            result = await client.fetch_klines("xyz:GOLD", interval="1h", limit=10)
            assert len(result) > 0, "Should return klines for xyz:GOLD"
            kline = result[-1]
            assert kline.close > 0
            print(f"  xyz:GOLD 1h: {len(result)} bars, close={kline.close}")
        finally:
            await client.close()

    async def test_fetch_multiple_symbols_same_interval(self) -> None:
        """Fetch multiple symbols with same interval."""
        symbols = ["BTC", "ETH", "SOL"]
        client = HyperliquidREST()
        try:
            results = {}
            for sym in symbols:
                klines = await client.fetch_klines(sym, interval="1h", limit=5)
                assert len(klines) > 0
                results[sym] = klines[-1].close
                print(f"  {sym}: close={results[sym]}")

            assert results["BTC"] > results["ETH"] > 0
            assert results["ETH"] > results["SOL"] > 0
        finally:
            await client.close()

    async def test_klines_are_sorted_by_time(self) -> None:
        """K-lines should be sorted oldest first by open_time."""
        client = HyperliquidREST()
        try:
            result = await client.fetch_klines("BTC", interval="1h", limit=100)
            assert len(result) >= 2
            for i in range(len(result) - 1):
                assert result[i].open_time < result[i + 1].open_time
        finally:
            await client.close()


class TestHyperliquidWSMarkPrices:
    """Test real WebSocket mark price subscription."""

    async def test_ws_connect(self) -> None:
        """Connect to WebSocket and verify connection."""
        ws = HyperliquidWS()
        try:
            await ws.connect()
            assert ws._ws is not None
            assert ws._running is True
            print("  WebSocket connected successfully")
        finally:
            await ws.close()

    async def test_ws_receives_allmids(self) -> None:
        """WebSocket receives allMids updates."""
        ws = HyperliquidWS()
        try:
            await ws.connect()
            task = asyncio.create_task(ws._receive_loop())
            await asyncio.sleep(3)
            marks = await ws.get_marks()
            assert len(marks) > 100, f"Should receive many mark prices, got {len(marks)}"
            print(f"  Received {len(marks)} mark prices")
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        finally:
            await ws.close()

    async def test_ws_contains_btc_price(self) -> None:
        """BTC mark price should be present in allMids."""
        ws = HyperliquidWS()
        try:
            await ws.connect()
            task = asyncio.create_task(ws._receive_loop())
            await asyncio.sleep(3)
            marks = await ws.get_marks()
            assert "BTC" in marks, "BTC should be in mark prices"
            btc_price = float(marks["BTC"])
            assert btc_price > 0, "BTC price should be positive"
            print(f"  BTC = {btc_price}")
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        finally:
            await ws.close()

    async def test_ws_mark_price_updates(self) -> None:
        """Mark prices should update over time."""
        ws = HyperliquidWS()
        try:
            await ws.connect()
            task = asyncio.create_task(ws._receive_loop())
            await asyncio.sleep(2)
            marks_before = await ws.get_marks()
            await asyncio.sleep(2)
            marks_after = await ws.get_marks()
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            assert marks_after is not marks_before
            assert len(marks_after) >= len(marks_before)
        finally:
            await ws.close()

    @pytest.mark.parametrize(
        "expected_contains",
        ["BTC", "ETH", "SOL"],
    )
    async def test_ws_various_symbols(self, expected_contains: str) -> None:
        """Various symbols should appear in mark prices."""
        marks = await get_mark_prices_once()
        found = False
        price = None
        for sym, p in marks.items():
            if expected_contains in sym:
                found = True
                price = float(p)
                break
        assert found, f"{expected_contains} not found in mark prices"
        assert price is not None and price > 0
        print(f"  {expected_contains} = {price}")


class TestHyperliquidOneShot:
    """Test one-shot mark price fetch (convenience function)."""

    async def test_oneshot_returns_many_prices(self) -> None:
        """One-shot should return many mark prices."""
        marks = await get_mark_prices_once()
        assert len(marks) > 100, f"Should return many prices, got {len(marks)}"
        print(f"  One-shot returned {len(marks)} prices")

    async def test_oneshot_btc_eth_sol(self) -> None:
        """One-shot should include BTC, ETH, SOL."""
        marks = await get_mark_prices_once()
        for sym in ["BTC", "ETH", "SOL"]:
            assert sym in marks, f"{sym} should be in one-shot prices"
            price = float(marks[sym])
            assert price > 0, f"{sym} price should be positive"
            print(f"  {sym} = {price}")


class TestHyperliquidErrorHandling:
    """Test real API error handling (with invalid inputs)."""

    async def test_invalid_symbol_returns_empty_or_error(self) -> None:
        """Invalid symbol may return empty or raise."""
        client = HyperliquidREST()
        try:
            klines = await client.fetch_klines("INVALID_SYMBOL_XYZ_123", interval="1h", limit=10)
            assert isinstance(klines, list)
            print(f"  Invalid symbol returned {len(klines)} klines")
        except Exception as e:
            print(f"  Invalid symbol raised: {type(e).__name__}: {e}")
        finally:
            await client.close()
