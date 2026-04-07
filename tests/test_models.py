"""Tests for data models."""

from typing import Any

import pytest

from models import Kline, PairState, Ticker


class TestKline:
    """Test Kline dataclass."""

    def test_from_rest(self) -> None:
        """Create Kline from REST list format."""
        data = [1000000, 65000.0, 66000.0, 64000.0, 65500.0, 100.5, 1001000]
        kline = Kline.from_rest("BTCUSDT", "1h", data)
        assert kline.symbol == "BTCUSDT"
        assert kline.interval == "1h"
        assert kline.open_time == 1000000
        assert kline.open == 65000.0
        assert kline.high == 66000.0
        assert kline.low == 64000.0
        assert kline.close == 65500.0
        assert kline.volume == 100.5
        assert kline.close_time == 1001000
        assert kline.is_closed is True

    def test_from_rest_minimal(self) -> None:
        """REST with minimal fields (no close_time)."""
        data = [1000000, 65000.0, 66000.0, 64000.0, 65500.0, 100.5]
        kline = Kline.from_rest("BTCUSDT", "1h", data)
        assert kline.close_time == 0

    def test_from_ws(self) -> None:
        """Create Kline from WebSocket dict format."""
        data = {
            "t": 1000000,
            "o": 65000.0,
            "h": 66000.0,
            "l": 64000.0,
            "c": 65500.0,
            "v": 100.5,
            "T": 1001000,
        }
        kline = Kline.from_ws("BTCUSDT", "1h", data, is_closed=True)
        assert kline.symbol == "BTCUSDT"
        assert kline.open_time == 1000000
        assert kline.open == 65000.0
        assert kline.is_closed is True

    def test_from_ws_with_defaults(self) -> None:
        """WS data with missing fields defaults to 0."""
        data: dict[str, Any] = {}
        kline = Kline.from_ws("BTCUSDT", "1h", data)
        assert kline.open == 0.0
        assert kline.high == 0.0

    def test_to_list(self) -> None:
        """Convert Kline to REST list format."""
        kline = Kline(
            symbol="BTCUSDT",
            interval="1h",
            open_time=1000000,
            open=65000.0,
            high=66000.0,
            low=64000.0,
            close=65500.0,
            volume=100.5,
            close_time=1001000,
        )
        lst = kline.to_list()
        assert lst == [1000000, 65000.0, 66000.0, 64000.0, 65500.0, 100.5, 1001000]

    def test_to_dict(self) -> None:
        """Convert Kline to dict."""
        kline = Kline(
            symbol="BTCUSDT",
            interval="1h",
            open_time=1000000,
            open=65000.0,
            high=66000.0,
            low=64000.0,
            close=65500.0,
            volume=100.5,
        )
        d = kline.to_dict()
        assert d["symbol"] == "BTCUSDT"
        assert d["open"] == 65000.0


class TestTicker:
    """Test Ticker dataclass."""

    def test_from_ws_with_c(self) -> None:
        """From WS with 'c' (close price) field."""
        data = {"s": "BTCUSDT", "c": "66000.0", "E": 1000000}
        ticker = Ticker.from_ws("BTCUSDT", data)
        assert ticker.symbol == "BTCUSDT"
        assert ticker.price == 66000.0

    def test_from_ws_with_lastprice(self) -> None:
        """From WS with 'lastPrice' field."""
        data = {"s": "BTCUSDT", "lastPrice": "67000.0"}
        ticker = Ticker.from_ws("BTCUSDT", data)
        assert ticker.price == 67000.0

    def test_from_ws_missing_price(self) -> None:
        """Missing price returns 0."""
        data = {"s": "BTCUSDT"}
        ticker = Ticker.from_ws("BTCUSDT", data)
        assert ticker.price == 0.0

    def test_to_dict(self) -> None:
        """Convert Ticker to dict."""
        ticker = Ticker(symbol="BTCUSDT", price=66000.0, update_time=1234567890.0)
        d = ticker.to_dict()
        assert d["symbol"] == "BTCUSDT"
        assert d["price"] == 66000.0


class TestPairState:
    """Test PairState dataclass."""

    def test_update_price(self) -> None:
        """Update component prices."""
        state = PairState(symbol="BTCETH", component1="BTC", component2="ETH")
        state.update_price("BTC", 50000.0)
        assert state.price1 == 50000.0
        assert state.ratio == 0.0
        state.update_price("ETH", 3000.0)
        assert state.price2 == 3000.0
        assert state.ratio == pytest.approx(50000.0 / 3000.0)

    def test_is_ready(self) -> None:
        """is_ready checks both prices."""
        state = PairState(symbol="BTCETH", component1="BTC", component2="ETH")
        assert state.is_ready() is False
        state.price1 = 50000.0
        assert state.is_ready() is False
        state.price2 = 3000.0
        assert state.is_ready() is True

    def test_make_ratio_kline_single_component(self) -> None:
        """Ratio kline with only one component."""
        state = PairState(symbol="BTCETH", component1="BTC", component2="ETH")
        state.price1 = 50000.0
        state.price2 = 3000.0
        state.ratio = 50000.0 / 3000.0
        kline = Kline(
            symbol="BTC",
            interval="1h",
            open_time=1000000,
            open=49000.0,
            high=51000.0,
            low=48000.0,
            close=50000.0,
            volume=100.0,
        )
        result = state.make_ratio_kline(kline, comp=1)
        assert result.symbol == "BTCETH"

    def test_make_ratio_kline_both_components(self) -> None:
        """Ratio kline with both components."""
        state = PairState(symbol="BTCETH", component1="BTC", component2="ETH")
        state.last_kline2 = Kline(
            symbol="ETH",
            interval="1h",
            open_time=1000000,
            open=2950.0,
            high=3050.0,
            low=2900.0,
            close=3000.0,
            volume=50.0,
        )
        state.price1 = 50000.0
        state.price2 = 3000.0
        state.ratio = 50000.0 / 3000.0
        kline = Kline(
            symbol="BTC",
            interval="1h",
            open_time=1000000,
            open=49000.0,
            high=51000.0,
            low=48000.0,
            close=50000.0,
            volume=100.0,
        )
        result = state.make_ratio_kline(kline, comp=1)
        assert result.open == pytest.approx(49000.0 / 2950.0)
        assert result.high >= result.low
