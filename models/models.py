"""
Data models for hypurr-monitor.

Contains:
- Kline: Single candlestick data structure
- Ticker: Latest price data structure
- PairState: Pair trading state tracking
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class Kline:
    """
    Single candlestick data structure.

    Attributes correspond to exchange REST API format:
    [open_time, open, high, low, close, volume, close_time, ...]
    WebSocket format: {t: open_time, o: open, h: high, l: low, c: close, v: volume}
    """

    symbol: str
    interval: str
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int | None = 0
    is_closed: bool = True

    REST_MIN_FIELDS: int = 6

    @classmethod
    def from_rest(cls, symbol: str, interval: str, data: list[Any]) -> "Kline":
        """
        Create Kline from REST API list.

        Args:
            symbol: Trading pair name
            interval: Kline interval
            data: REST list, format: [open_time, open, high, low, close, volume, close_time, ...]
        """
        return cls(
            symbol=symbol,
            interval=interval,
            open_time=int(data[0]),
            open=float(data[1]),
            high=float(data[2]),
            low=float(data[3]),
            close=float(data[4]),
            volume=float(data[5]),
            close_time=int(data[6]) if len(data) > cls.REST_MIN_FIELDS else 0,
            is_closed=True,
        )

    @classmethod
    def from_dict(cls, symbol: str, interval: str, data: dict[str, Any]) -> "Kline":
        """
        Create Kline from dict (native Hyperliquid REST API response).

        Args:
            symbol: Trading pair name
            interval: Kline interval
            data: Dict with keys: open_time, open, high, low, close, volume
        """
        return cls(
            symbol=symbol,
            interval=interval,
            open_time=int(data["open_time"]),
            open=float(data["open"]),
            high=float(data["high"]),
            low=float(data["low"]),
            close=float(data["close"]),
            volume=float(data["volume"]),
            close_time=int(data.get("close_time", 0)),
            is_closed=data.get("is_closed", True),
        )

    @classmethod
    def from_ws(cls, symbol: str, interval: str, data: dict[str, Any], is_closed: bool = True) -> "Kline":
        """
        Create Kline from WebSocket push dict.

        Args:
            symbol: Trading pair name
            interval: Kline interval
            data: WebSocket kline dict, {t, o, h, l, c, v}
            is_closed: Whether this is a closed kline (WebSocket x field)
        """
        return cls(
            symbol=symbol,
            interval=interval,
            open_time=int(data.get("t", 0)),
            open=float(data.get("o", 0)),
            high=float(data.get("h", 0)),
            low=float(data.get("l", 0)),
            close=float(data.get("c", 0)),
            volume=float(data.get("v", 0)),
            close_time=int(data.get("T", 0)),
            is_closed=is_closed,
        )

    def to_list(self) -> list[Any]:
        """Convert to REST API format list."""
        return [
            self.open_time,
            self.open,
            self.high,
            self.low,
            self.close,
            self.volume,
            self.close_time or 0,
        ]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict."""
        return {
            "symbol": self.symbol,
            "interval": self.interval,
            "open_time": self.open_time,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "close_time": self.close_time,
            "is_closed": self.is_closed,
        }


@dataclass
class Ticker:
    """
    Latest price data structure.
    """

    symbol: str
    price: float
    update_time: float

    @classmethod
    def from_ws(cls, symbol: str, data: dict[str, Any]) -> "Ticker":
        """
        Create Ticker from WebSocket ticker data.

        WebSocket ticker format: {s: "BTCUSDT", c: "66000.0"} or {s: "BTCUSDT", lastPrice: "66000.0"}
        """
        price_str = data.get("c") or data.get("lastPrice") or "0"
        return cls(
            symbol=symbol,
            price=float(price_str),
            update_time=data.get("E", 0) / 1000.0 if data.get("E") else 0.0,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "price": self.price,
            "update_time": self.update_time,
        }


@dataclass
class PairState:
    """
    Pair trading state tracking.

    Used for calculating pair prices (e.g. BTCUSDT/ETHUSDT = BTC_price / ETH_price).
    Wait for both components ticker to arrive before calculating ratio.
    """

    symbol: str
    component1: str
    component2: str
    price1: float = 0.0
    price2: float = 0.0
    last_kline1: Kline | None = None
    last_kline2: Kline | None = None
    last_ratio_kline: Kline | None = None
    ratio: float = 0.0

    def update_price(self, comp_symbol: str, price: float) -> None:
        """Update price for a component."""
        if comp_symbol == self.component1:
            self.price1 = price
        elif comp_symbol == self.component2:
            self.price2 = price

        if self.price1 > 0 and self.price2 > 0:
            self.ratio = self.price1 / self.price2

    def is_ready(self) -> bool:
        """Are both component prices available?"""
        return self.price1 > 0 and self.price2 > 0

    def make_ratio_kline(self, new_kline: Kline, comp: int) -> Kline:
        """
        Calculate pair ratio Kline from newly arrived Kline and another component's Kline.

        Args:
            new_kline: Newly arrived Kline (component1 or component2 1h Kline)
            comp: Which component the new Kline belongs to, 1 or 2

        Returns:
            Pair ratio Kline:
            - open = open1 / open2
            - high = max(open, ratio)
            - low = min(open, ratio)
            - close = ratio
            - volume = new_kline.volume
        """
        other = self.last_kline2 if comp == 1 else self.last_kline1

        if other is None:
            return Kline(
                symbol=self.symbol,
                interval=new_kline.interval,
                open_time=new_kline.open_time,
                open=self.ratio,
                high=self.ratio,
                low=self.ratio,
                close=self.ratio,
                volume=new_kline.volume,
                close_time=new_kline.close_time,
                is_closed=new_kline.is_closed,
            )

        open_ratio = new_kline.open / other.open if other.open > 0 else self.ratio
        close_ratio = self.ratio
        high_ratio = max(open_ratio, close_ratio)
        low_ratio = min(open_ratio, close_ratio)

        if comp == 1:
            self.last_kline1 = new_kline
        else:
            self.last_kline2 = new_kline

        k = Kline(
            symbol=self.symbol,
            interval=new_kline.interval,
            open_time=new_kline.open_time,
            open=open_ratio,
            high=high_ratio,
            low=low_ratio,
            close=close_ratio,
            volume=new_kline.volume,
            close_time=new_kline.close_time,
            is_closed=new_kline.is_closed,
        )
        self.last_ratio_kline = k
        return k
