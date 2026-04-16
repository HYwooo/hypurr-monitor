"""Unified Hyperliquid transport gateway for REST metadata and WS connections."""

from __future__ import annotations

import asyncio
from typing import Any, cast

import aiohttp

from config.network import RestNetworkConfig, WsNetworkConfig

from .rest_client import HyperliquidREST, fetch_meta


class MarketGateway:
    """Own transport configuration and resource creation for Hyperliquid access."""

    def __init__(self, rest: RestNetworkConfig, ws: WsNetworkConfig) -> None:
        self._rest = rest
        self._ws = ws

    def create_rest_client(self) -> HyperliquidREST:
        """Create a REST client using the unified network config."""
        return HyperliquidREST(network=self._rest)

    async def fetch_meta(self) -> dict[str, Any]:
        """Fetch exchange metadata using the unified REST config."""
        return await fetch_meta(
            proxy=self._rest.proxy_url,
            timeout_seconds=self._rest.timeout_seconds,
            max_retries=self._rest.retry.max_retries,
            retry_base_delay_seconds=self._rest.retry.base_delay_seconds,
        )

    async def check_connectivity(self) -> dict[str, Any]:
        """Perform a lightweight REST connectivity check through the REST client."""
        client = self.create_rest_client()
        try:
            result = await client._post({"type": "meta"}, weight=1.0)  # noqa: SLF001
            return cast(dict[str, Any], result)
        finally:
            await client.close()

    async def connect_mark_price_ws(self, url: str) -> tuple[aiohttp.ClientSession, aiohttp.ClientWebSocketResponse]:
        """Create websocket session and connect using unified WS config."""
        session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self._ws.connect_timeout_seconds))
        try:
            ws = await session.ws_connect(
                url,
                timeout=aiohttp.ClientWSTimeout(ws_receive=self._ws.receive_timeout_seconds),
                proxy=self._ws.proxy_url,
            )
            return session, ws
        except Exception:
            await session.close()
            raise

    async def open_mark_price_stream(self, url: str) -> tuple[aiohttp.ClientSession, aiohttp.ClientWebSocketResponse]:
        """Connect websocket and subscribe it to allMids feeds."""
        session, ws = await self.connect_mark_price_ws(url)
        try:
            await self.subscribe_all_mids(ws)
            return session, ws
        except Exception:
            await self.close_ws_resources(session, ws)
            raise

    async def receive_ws_message(self, ws: aiohttp.ClientWebSocketResponse) -> aiohttp.WSMessage:
        """Receive next websocket message using unified idle-timeout config."""
        return await asyncio.wait_for(ws.receive(), timeout=self._ws.idle_timeout_seconds)

    async def close_ws_resources(
        self,
        session: aiohttp.ClientSession | None,
        ws: aiohttp.ClientWebSocketResponse | None,
    ) -> None:
        """Close websocket and session resources owned by the market stream."""
        if ws and not ws.closed:
            await ws.close()
        if session and not session.closed:
            await session.close()

    async def subscribe_all_mids(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        """Subscribe the websocket to all configured allMids feeds."""
        for dex in ["", "xyz", "hyna", "flx", "vntl", "km", "cash", "para"]:
            if dex:
                sub = {"method": "subscribe", "subscription": {"type": "allMids", "dex": dex}}
            else:
                sub = {"method": "subscribe", "subscription": {"type": "allMids"}}
            await ws.send_json(sub)

    async def send_ws_ping(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        """Send application-level ping for low-traffic websocket channels."""
        await ws.send_json({"method": "ping"})
