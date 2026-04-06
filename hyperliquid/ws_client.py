"""
Hyperliquid WebSocket client for mark prices.

Uses native WebSocket API to batch subscribe to all mark prices:
- Main dex: allMids (536 symbols)
- HIP-3 dexes: allMids with dex parameter (xyz, hyna, flx, vntl, km, cash, para)
- Total: ~690 mark prices updated every second
"""

from dataclasses import dataclass
from typing import ClassVar

import aiohttp
import orjson


@dataclass
class MarkPrice:
    """Mark price data."""

    coin: str
    price: float
    dex: str


class HyperliquidWS:
    """Hyperliquid WebSocket client for mark prices."""

    DEXES: ClassVar[list[str]] = ["", "xyz", "hyna", "flx", "vntl", "km", "cash", "para"]

    def __init__(self) -> None:
        self.url: str = "wss://api.hyperliquid.xyz/ws"
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._session: aiohttp.ClientSession | None = None
        self._marks: dict[str, float] = {}
        self._running: bool = False

    async def connect(self) -> None:
        """Connect to WebSocket."""
        self._session = aiohttp.ClientSession()
        self._ws = await self._session.ws_connect(self.url)
        self._running = True

        for dex in self.DEXES:
            if dex:
                sub = {"method": "subscribe", "subscription": {"type": "allMids", "dex": dex}}
            else:
                sub = {"method": "subscribe", "subscription": {"type": "allMids"}}
            await self._ws.send_json(sub)

    async def _receive_loop(self) -> None:
        """Receive and process messages."""
        while self._running and self._ws:
            msg = await self._ws.receive()
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = orjson.loads(msg.data)
                if data.get("channel") == "allMids":
                    mids = data.get("data", {}).get("mids", {})
                    self._marks.update(mids)
            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                break

    async def get_marks(self) -> dict[str, float]:
        """Get current mark prices."""
        return self._marks.copy()

    async def close(self) -> None:
        """Close the connection."""
        self._running = False
        if self._ws:
            await self._ws.close()
        if self._session:
            await self._session.close()


async def get_mark_prices_once() -> dict[str, float]:
    """Get all mark prices in one shot (for testing)."""
    marks: dict[str, float] = {}

    async with aiohttp.ClientSession() as session, session.ws_connect("wss://api.hyperliquid.xyz/ws") as ws:
        for dex in HyperliquidWS.DEXES:
            if dex:
                sub = {"method": "subscribe", "subscription": {"type": "allMids", "dex": dex}}
            else:
                sub = {"method": "subscribe", "subscription": {"type": "allMids"}}
            await ws.send_json(sub)

        for _ in range(20):
            msg = await ws.receive()
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = orjson.loads(msg.data)
                if data.get("channel") == "allMids":
                    mids = data.get("data", {}).get("mids", {})
                    marks.update(mids)

    return marks
