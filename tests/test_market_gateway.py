"""Tests for Hyperliquid market gateway transport abstraction."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from config.network import RestNetworkConfig, RetryConfig, WsNetworkConfig
from hyperliquid.market_gateway import MarketGateway


def build_gateway() -> MarketGateway:
    """Create gateway with explicit non-default values for assertions."""
    return MarketGateway(
        RestNetworkConfig(
            proxy_url="http://127.0.0.1:7890",
            timeout_seconds=12.5,
            retry=RetryConfig(max_retries=3, base_delay_seconds=0.5),
        ),
        WsNetworkConfig(
            proxy_url="http://127.0.0.1:7890",
            connect_timeout_seconds=33.0,
            receive_timeout_seconds=44.0,
            idle_timeout_seconds=22.0,
            reconnect_base_delay_seconds=2.0,
            reconnect_max_delay_seconds=30.0,
        ),
    )


class TestMarketGateway:
    """Test unified market transport abstraction."""

    def test_create_rest_client_applies_unified_rest_config(self) -> None:
        """Gateway-created REST client should inherit unified network settings."""
        gateway = build_gateway()

        client = gateway.create_rest_client()

        assert client.proxy == "http://127.0.0.1:7890"
        assert client.timeout_seconds == 12.5
        assert client.max_retries == 3
        assert client.retry_base_delay_seconds == 0.5

    @pytest.mark.asyncio
    async def test_check_connectivity_uses_rest_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Connectivity check should delegate to REST client and close it."""
        gateway = build_gateway()
        fake_client = SimpleNamespace(
            _post=AsyncMock(return_value={"ok": True}),
            close=AsyncMock(),
        )
        monkeypatch.setattr(gateway, "create_rest_client", lambda: fake_client)

        result = await gateway.check_connectivity()

        assert result == {"ok": True}
        fake_client._post.assert_awaited_once_with({"type": "meta"}, weight=1.0)
        fake_client.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_connect_mark_price_ws_uses_unified_ws_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """WS connect should apply shared proxy and timeout settings."""
        gateway = build_gateway()
        fake_ws = MagicMock(spec=aiohttp.ClientWebSocketResponse)
        fake_session = MagicMock()
        fake_session.ws_connect = AsyncMock(return_value=fake_ws)
        fake_session.close = AsyncMock()

        def fake_client_session(*args: object, **kwargs: object) -> MagicMock:
            timeout = kwargs.get("timeout")
            assert isinstance(timeout, aiohttp.ClientTimeout)
            assert timeout.total == 33.0
            return fake_session

        monkeypatch.setattr("hyperliquid.market_gateway.aiohttp.ClientSession", fake_client_session)

        session, ws = await gateway.connect_mark_price_ws("wss://api.hyperliquid.xyz/ws")

        assert session is fake_session
        assert ws is fake_ws
        await_args = fake_session.ws_connect.await_args
        assert await_args is not None
        kwargs = await_args.kwargs
        assert kwargs["proxy"] == "http://127.0.0.1:7890"
        assert kwargs["timeout"].ws_receive == 44.0

    @pytest.mark.asyncio
    async def test_open_mark_price_stream_subscribes_after_connect(self) -> None:
        """Opening mark price stream should connect and subscribe in one gateway call."""
        gateway = build_gateway()
        fake_session = MagicMock(spec=aiohttp.ClientSession)
        fake_ws = MagicMock(spec=aiohttp.ClientWebSocketResponse)
        gateway.connect_mark_price_ws = AsyncMock(return_value=(fake_session, fake_ws))  # type: ignore[method-assign]
        gateway.subscribe_all_mids = AsyncMock()  # type: ignore[method-assign]

        session, ws = await gateway.open_mark_price_stream("wss://api.hyperliquid.xyz/ws")

        assert session is fake_session
        assert ws is fake_ws
        gateway.connect_mark_price_ws.assert_awaited_once_with("wss://api.hyperliquid.xyz/ws")
        gateway.subscribe_all_mids.assert_awaited_once_with(fake_ws)

    @pytest.mark.asyncio
    async def test_receive_ws_message_uses_idle_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Gateway receive helper should apply unified idle timeout via asyncio.wait_for."""
        gateway = build_gateway()
        fake_ws = MagicMock(spec=aiohttp.ClientWebSocketResponse)
        fake_message = MagicMock(spec=aiohttp.WSMessage)

        async def fake_wait_for(awaitable: object, timeout: float) -> aiohttp.WSMessage:
            assert timeout == 22.0
            coro = awaitable
            try:
                return fake_message
            finally:
                if hasattr(coro, "close"):
                    coro.close()

        monkeypatch.setattr("hyperliquid.market_gateway.asyncio.wait_for", fake_wait_for)

        result = await gateway.receive_ws_message(fake_ws)

        assert result is fake_message

    @pytest.mark.asyncio
    async def test_close_ws_resources_closes_ws_and_session(self) -> None:
        """Gateway close helper should release both websocket and owning session."""
        gateway = build_gateway()
        fake_ws = MagicMock(spec=aiohttp.ClientWebSocketResponse)
        fake_ws.closed = False
        fake_ws.close = AsyncMock()
        fake_session = MagicMock(spec=aiohttp.ClientSession)
        fake_session.closed = False
        fake_session.close = AsyncMock()

        await gateway.close_ws_resources(fake_session, fake_ws)

        fake_ws.close.assert_awaited_once()
        fake_session.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_subscribe_all_mids_sends_all_dex_subscriptions(self) -> None:
        """Gateway should own allMids subscription payload generation."""
        gateway = build_gateway()
        ws = MagicMock()
        ws.send_json = AsyncMock()

        await gateway.subscribe_all_mids(ws)

        assert ws.send_json.await_count == 8
        first_call = ws.send_json.await_args_list[0].args[0]
        last_call = ws.send_json.await_args_list[-1].args[0]
        assert first_call == {"method": "subscribe", "subscription": {"type": "allMids"}}
        assert last_call == {"method": "subscribe", "subscription": {"type": "allMids", "dex": "para"}}

    @pytest.mark.asyncio
    async def test_send_ws_ping_uses_application_ping(self) -> None:
        """Gateway ping helper should send the documented websocket heartbeat payload."""
        gateway = build_gateway()
        ws = MagicMock()
        ws.send_json = AsyncMock()

        await gateway.send_ws_ping(ws)

        ws.send_json.assert_awaited_once_with({"method": "ping"})
