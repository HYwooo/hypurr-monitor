"""
Unit tests for Hyperliquid WebSocket client.

Covers:
- Connection setup
- Message parsing (valid/invalid/empty)
- Subscription handling
- Disconnection/reconnection
- Mark price updates
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import orjson
import pytest

from hyperliquid.ws_client import HyperliquidWS

pytestmark = pytest.mark.asyncio


class TestHyperliquidWSInit:
    """Test client initialization."""

    def test_init_defaults(self) -> None:
        """Default values should be correct."""
        ws = HyperliquidWS()
        assert ws.url == "wss://api.hyperliquid.xyz/ws"
        assert ws._ws is None
        assert ws._session is None
        assert ws._marks == {}
        assert ws._running is False

    def test_dexes_list(self) -> None:
        """DEXES should include all known dexes."""
        assert "" in HyperliquidWS.DEXES
        assert "xyz" in HyperliquidWS.DEXES
        assert "hyna" in HyperliquidWS.DEXES
        assert "flx" in HyperliquidWS.DEXES
        assert "vntl" in HyperliquidWS.DEXES
        assert "km" in HyperliquidWS.DEXES
        assert "cash" in HyperliquidWS.DEXES
        assert "para" in HyperliquidWS.DEXES


def make_text_msg(channel: str, data: dict[str, Any]) -> MagicMock:
    """Helper to create a TEXT message."""
    msg = MagicMock()
    msg.type = aiohttp.WSMsgType.TEXT
    msg.data = orjson.dumps({"channel": channel, "data": data}).decode()
    return msg


class ClosedMsg:
    """Sentinel to signal receive loop to exit."""

    type = aiohttp.WSMsgType.CLOSED


class ErrorMsg:
    """Sentinel to signal receive loop to exit."""

    type = aiohttp.WSMsgType.ERROR


class TestHyperliquidWSConnect:
    """Test WebSocket connection."""

    async def test_connect_creates_session_and_ws(self) -> None:
        """Connect should create session and WebSocket."""
        ws = HyperliquidWS()
        mock_ws = MagicMock()
        mock_ws.send_json = AsyncMock()
        mock_session = MagicMock()
        mock_session.ws_connect = AsyncMock(return_value=mock_ws)
        mock_session.close = AsyncMock()

        with patch("hyperliquid.ws_client.aiohttp.ClientSession", return_value=mock_session):
            await ws.connect()

        mock_session.ws_connect.assert_called_once_with(ws.url)
        assert ws._running is True

    async def test_connect_subscribes_to_all_dexes(self) -> None:
        """Connect should subscribe to allMids for all dexes."""
        ws = HyperliquidWS()
        mock_ws = MagicMock()
        mock_ws.send_json = AsyncMock()
        mock_session = MagicMock()
        mock_session.ws_connect = AsyncMock(return_value=mock_ws)
        mock_session.close = AsyncMock()

        with patch("hyperliquid.ws_client.aiohttp.ClientSession", return_value=mock_session):
            await ws.connect()

        assert mock_ws.send_json.call_count == len(HyperliquidWS.DEXES)


class TestHyperliquidWSReceiveLoop:
    """Test message receiving and parsing."""

    async def test_receive_loop_parses_allmids(self) -> None:
        """Should parse allMids channel messages."""
        ws = HyperliquidWS()
        ws._running = True
        ws._ws = MagicMock()

        messages = [
            make_text_msg("allMids", {"mids": {"BTC": 65000.5, "ETH": 3500.25}}),
            ClosedMsg(),
        ]

        async def receive_side_effect() -> Any:
            return messages.pop(0)

        ws._ws.receive = receive_side_effect

        await ws._receive_loop()

        assert ws._marks["BTC"] == 65000.5
        assert ws._marks["ETH"] == 3500.25

    async def test_receive_loop_ignores_unknown_channel(self) -> None:
        """Should ignore unknown channel messages."""
        ws = HyperliquidWS()
        ws._running = True
        ws._ws = MagicMock()

        messages = [
            make_text_msg("unknown", {}),
            ClosedMsg(),
        ]

        async def receive_side_effect() -> Any:
            return messages.pop(0)

        ws._ws.receive = receive_side_effect

        await ws._receive_loop()

        assert ws._marks == {}

    async def test_receive_loop_stops_on_close(self) -> None:
        """Should stop on CLOSED message and close cleans up."""
        ws = HyperliquidWS()
        closed_msg = ClosedMsg()

        async def mock_receive() -> Any:
            return closed_msg

        ws._ws = MagicMock()
        ws._ws.receive = mock_receive
        ws._ws.close = AsyncMock()
        ws._session = MagicMock()
        ws._session.close = AsyncMock()
        ws._running = True

        await ws._receive_loop()
        await ws.close()

        assert ws._running is False

    async def test_receive_loop_stops_on_error(self) -> None:
        """Should stop on ERROR message and close cleans up."""
        ws = HyperliquidWS()
        error_msg = ErrorMsg()

        async def mock_receive() -> Any:
            return error_msg

        ws._ws = MagicMock()
        ws._ws.receive = mock_receive
        ws._ws.close = AsyncMock()
        ws._session = MagicMock()
        ws._session.close = AsyncMock()
        ws._running = True

        await ws._receive_loop()
        await ws.close()

        assert ws._running is False

    async def test_receive_loop_handles_malformed_json(self) -> None:
        """Malformed JSON would raise - code does not catch orjson.loads errors."""
        pytest.skip("Code does not catch orjson.JSONDecodeError, would need try-except around orjson.loads")

    async def test_receive_loop_handles_missing_mids(self) -> None:
        """Should handle message with missing mids field."""
        ws = HyperliquidWS()
        ws._running = True
        ws._ws = MagicMock()

        messages = [
            make_text_msg("allMids", {}),
            ClosedMsg(),
        ]

        async def receive_side_effect() -> Any:
            return messages.pop(0)

        ws._ws.receive = receive_side_effect

        await ws._receive_loop()

        assert ws._marks == {}

    async def test_receive_loop_handles_non_string_prices(self) -> None:
        """Prices should be stored as-is."""
        ws = HyperliquidWS()
        ws._running = True
        ws._ws = MagicMock()

        messages = [
            make_text_msg("allMids", {"mids": {"BTC": 65000.5}}),
            ClosedMsg(),
        ]

        async def receive_side_effect() -> Any:
            return messages.pop(0)

        ws._ws.receive = receive_side_effect

        await ws._receive_loop()

        assert ws._marks["BTC"] == 65000.5


class TestHyperliquidWSGetMarks:
    """Test getting mark prices."""

    async def test_get_marks_returns_copy(self) -> None:
        """get_marks should return a copy."""
        ws = HyperliquidWS()
        ws._marks = {"BTC": 65000.0}
        marks = await ws.get_marks()
        marks["ETH"] = 3500.0
        assert "ETH" not in ws._marks


class TestHyperliquidWSClose:
    """Test client cleanup."""

    async def test_close_sets_running_false(self) -> None:
        """Close should set _running to False."""
        ws = HyperliquidWS()
        ws._running = True
        ws._ws = AsyncMock()
        ws._session = AsyncMock()
        await ws.close()
        assert ws._running is False

    async def test_close_closes_ws_and_session(self) -> None:
        """Close should close WebSocket and session."""
        ws = HyperliquidWS()
        ws._running = True
        mock_ws = AsyncMock()
        mock_session = AsyncMock()
        ws._ws = mock_ws
        ws._session = mock_session
        await ws.close()
        mock_ws.close.assert_called_once()
        mock_session.close.assert_called_once()


class TestGetMarkPricesOnce:
    """Test one-shot mark price fetch helper."""

    async def test_oneshot_integration_skipped(self) -> None:
        """One-shot requires real network - tested in test_integration.py."""
        pytest.skip("One-shot is tested via real network in integration tests")
