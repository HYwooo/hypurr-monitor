"""Tests for signal detection module - price comparisons and state machines."""

import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from models import Kline
from signals.detection import (
    PRECISION_EPSILON,
    check_signals,
    check_trailing_stop,
    price_ge,
    price_gt,
    price_le,
    price_lt,
    recalculate_states,
    update_klines,
)


class TestPriceComparisons:
    """Test IEEE 754 safe price comparisons."""

    def test_price_ge_true(self) -> None:
        """Greater than or close returns True when a > b."""
        assert price_ge(10.0, 5.0) is True

    def test_price_ge_false(self) -> None:
        """Greater than or close returns False when a << b."""
        assert price_ge(5.0, 10.0) is False

    def test_price_ge_within_epsilon(self) -> None:
        """Within epsilon tolerance should be considered equal."""
        a = 10.0
        b = 10.0 + PRECISION_EPSILON / 2
        assert price_ge(a, b) is True

    def test_price_ge_exactly_equal(self) -> None:
        """Exactly equal values."""
        assert price_ge(10.0, 10.0) is True

    def test_price_le_true(self) -> None:
        """Less than or close returns True when a < b."""
        assert price_le(5.0, 10.0) is True

    def test_price_le_false(self) -> None:
        """Less than or close returns False when a >> b."""
        assert price_le(10.0, 5.0) is False

    def test_price_le_within_epsilon(self) -> None:
        """Within epsilon tolerance should be considered equal."""
        a = 10.0 - PRECISION_EPSILON / 2
        b = 10.0
        assert price_le(a, b) is True

    def test_price_gt_true(self) -> None:
        """Strict greater than returns True when a > b."""
        assert price_gt(10.0, 5.0) is True

    def test_price_gt_false_equal(self) -> None:
        """Strict greater than returns False when a == b."""
        assert price_gt(10.0, 10.0) is False

    def test_price_gt_false_within_epsilon(self) -> None:
        """Strict greater than returns False when difference < epsilon."""
        a = 10.0
        b = 10.0 + PRECISION_EPSILON / 2
        assert price_gt(a, b) is False

    def test_price_lt_true(self) -> None:
        """Strict less than returns True when a < b."""
        assert price_lt(5.0, 10.0) is True

    def test_price_lt_false_equal(self) -> None:
        """Strict less than returns False when a == b."""
        assert price_lt(10.0, 10.0) is False

    def test_very_small_difference(self) -> None:
        """Very small price differences at decimal boundary."""
        a = 0.00000001
        b = 0.00000002
        assert price_lt(a, b) is True
        assert price_gt(b, a) is True


class TestCheckSignals:
    """Test signal detection logic."""

    @pytest.fixture
    def mock_webhook(self) -> AsyncMock:
        """Mock webhook sender."""
        return AsyncMock()

    @pytest.fixture
    def mock_increment(self) -> MagicMock:
        """Mock alert counter."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_no_signal_when_not_initialized(self, mock_webhook: AsyncMock, mock_increment: MagicMock) -> None:
        """During initialization, should not send signals."""
        mark_prices: dict[str, float] = {"BTC": 50000.0}
        mark_price_times: dict[str, float] = {"BTC": time.time()}
        benchmark: dict[str, dict[str, Any]] = {
            "BTC": {
                "st1": 49000.0,
                "st2": 48000.0,
                "atr1h_upper": 51000.0,
                "atr1h_lower": 49000.0,
            }
        }
        trailing_stop: dict[str, Any] = {}
        last_atr_state: dict[str, Any] = {}
        last_alert_time: dict[str, Any] = {}
        last_st_state: dict[str, Any] = {}

        await check_signals(
            "BTC",
            mark_prices,
            mark_price_times,
            benchmark,
            trailing_stop,
            last_atr_state,
            last_alert_time,
            False,
            last_st_state,
            "DEMA",
            14,
            1.618,
            "HMA",
            14,
            1.3,
            mock_webhook,
            mock_increment,
        )
        mock_webhook.assert_not_called()

    @pytest.mark.asyncio
    async def test_long_signal_on_upper_break(self, mock_webhook: AsyncMock, mock_increment: MagicMock) -> None:
        """Price breaks above ATR upper should trigger LONG."""
        mark_prices: dict[str, float] = {"BTC": 52000.0}
        mark_price_times: dict[str, float] = {"BTC": time.time()}
        benchmark: dict[str, dict[str, Any]] = {
            "BTC": {
                "st1": 49000.0,
                "st2": 48000.0,
                "atr1h_upper": 51000.0,
                "atr1h_lower": 49000.0,
            }
        }
        trailing_stop: dict[str, Any] = {}
        last_atr_state: dict[str, Any] = {}
        last_alert_time: dict[str, Any] = {}
        last_st_state: dict[str, Any] = {}

        await check_signals(
            "BTC",
            mark_prices,
            mark_price_times,
            benchmark,
            trailing_stop,
            last_atr_state,
            last_alert_time,
            True,
            last_st_state,
            "DEMA",
            14,
            1.618,
            "HMA",
            14,
            1.3,
            mock_webhook,
            mock_increment,
        )
        mock_webhook.assert_called_once()
        args, kwargs = mock_webhook.call_args
        assert args[2]["direction"] == "LONG"

    @pytest.mark.asyncio
    async def test_short_signal_on_lower_break(self, mock_webhook: AsyncMock, mock_increment: MagicMock) -> None:
        """Price breaks below ATR lower should trigger SHORT."""
        mark_prices: dict[str, float] = {"BTC": 48000.0}
        mark_price_times: dict[str, float] = {"BTC": time.time()}
        benchmark: dict[str, dict[str, Any]] = {
            "BTC": {
                "st1": 49000.0,
                "st2": 48000.0,
                "atr1h_upper": 51000.0,
                "atr1h_lower": 49000.0,
            }
        }
        trailing_stop: dict[str, Any] = {}
        last_atr_state: dict[str, Any] = {"BTC": {"ch": 0}}
        last_alert_time: dict[str, Any] = {}
        last_st_state: dict[str, Any] = {}

        await check_signals(
            "BTC",
            mark_prices,
            mark_price_times,
            benchmark,
            trailing_stop,
            last_atr_state,
            last_alert_time,
            True,
            last_st_state,
            "DEMA",
            14,
            1.618,
            "HMA",
            14,
            1.3,
            mock_webhook,
            mock_increment,
        )
        mock_webhook.assert_called_once()
        args, kwargs = mock_webhook.call_args
        assert args[2]["direction"] == "SHORT"

    @pytest.mark.asyncio
    async def test_no_duplicate_signal_same_direction(self, mock_webhook: AsyncMock, mock_increment: MagicMock) -> None:
        """Already in LONG state, price above upper should NOT trigger again."""
        mark_prices: dict[str, float] = {"BTC": 52000.0}
        mark_price_times: dict[str, float] = {"BTC": time.time()}
        benchmark: dict[str, dict[str, Any]] = {
            "BTC": {
                "st1": 49000.0,
                "st2": 48000.0,
                "atr1h_upper": 51000.0,
                "atr1h_lower": 49000.0,
            }
        }
        trailing_stop: dict[str, Any] = {}
        last_atr_state: dict[str, Any] = {"BTC": {"ch": 1, "sent": "LONG"}}
        last_alert_time: dict[str, Any] = {}
        last_st_state: dict[str, Any] = {}

        await check_signals(
            "BTC",
            mark_prices,
            mark_price_times,
            benchmark,
            trailing_stop,
            last_atr_state,
            last_alert_time,
            True,
            last_st_state,
            "DEMA",
            14,
            1.618,
            "HMA",
            14,
            1.3,
            mock_webhook,
            mock_increment,
        )
        mock_webhook.assert_not_called()

    @pytest.mark.asyncio
    async def test_stale_price_ignored(self, mock_webhook: AsyncMock, mock_increment: MagicMock) -> None:
        """Stale price (> 300s old) should not trigger signals."""
        mark_prices: dict[str, float] = {"BTC": 52000.0}
        mark_price_times: dict[str, float] = {"BTC": time.time() - 400}
        benchmark: dict[str, dict[str, Any]] = {
            "BTC": {
                "st1": 49000.0,
                "st2": 48000.0,
                "atr1h_upper": 51000.0,
                "atr1h_lower": 49000.0,
            }
        }
        trailing_stop: dict[str, Any] = {}
        last_atr_state: dict[str, Any] = {}
        last_alert_time: dict[str, Any] = {}
        last_st_state: dict[str, Any] = {}

        await check_signals(
            "BTC",
            mark_prices,
            mark_price_times,
            benchmark,
            trailing_stop,
            last_atr_state,
            last_alert_time,
            True,
            last_st_state,
            "DEMA",
            14,
            1.618,
            "HMA",
            14,
            1.3,
            mock_webhook,
            mock_increment,
        )
        mock_webhook.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_signal_no_benchmark(self, mock_webhook: AsyncMock, mock_increment: MagicMock) -> None:
        """Symbol not in benchmark should return early."""
        mark_prices: dict[str, float] = {"BTC": 50000.0}
        mark_price_times: dict[str, float] = {"BTC": time.time()}
        benchmark: dict[str, dict[str, Any]] = {}
        trailing_stop: dict[str, Any] = {}
        last_atr_state: dict[str, Any] = {}
        last_alert_time: dict[str, Any] = {}
        last_st_state: dict[str, Any] = {}

        await check_signals(
            "BTC",
            mark_prices,
            mark_price_times,
            benchmark,
            trailing_stop,
            last_atr_state,
            last_alert_time,
            True,
            last_st_state,
            "DEMA",
            14,
            1.618,
            "HMA",
            14,
            1.3,
            mock_webhook,
            mock_increment,
        )
        mock_webhook.assert_not_called()

    @pytest.mark.asyncio
    async def test_trailing_stop_established_on_signal(
        self, mock_webhook: AsyncMock, mock_increment: MagicMock
    ) -> None:
        """Signal should establish trailing stop state."""
        mark_prices: dict[str, float] = {"BTC": 52000.0}
        mark_price_times: dict[str, float] = {"BTC": time.time()}
        benchmark: dict[str, dict[str, Any]] = {
            "BTC": {
                "st1": 49000.0,
                "st2": 48000.0,
                "atr1h_upper": 51000.0,
                "atr1h_lower": 49000.0,
            }
        }
        trailing_stop: dict[str, Any] = {}
        last_atr_state: dict[str, Any] = {}
        last_alert_time: dict[str, Any] = {}
        last_st_state: dict[str, Any] = {}

        await check_signals(
            "BTC",
            mark_prices,
            mark_price_times,
            benchmark,
            trailing_stop,
            last_atr_state,
            last_alert_time,
            True,
            last_st_state,
            "DEMA",
            14,
            1.618,
            "HMA",
            14,
            1.3,
            mock_webhook,
            mock_increment,
        )
        assert "BTC" in trailing_stop
        assert trailing_stop["BTC"]["direction"] == "LONG"
        assert trailing_stop["BTC"]["active"] is True


class TestCheckTrailingStop:
    """Test trailing stop logic."""

    @pytest.fixture
    def mock_webhook(self) -> AsyncMock:
        """Mock webhook sender."""
        return AsyncMock()

    @pytest.fixture
    def mock_increment(self) -> MagicMock:
        """Mock alert counter."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_no_signal_inactive_trailing_stop(self, mock_webhook: AsyncMock, mock_increment: MagicMock) -> None:
        """Inactive trailing stop should not trigger."""
        trailing_stop: dict[str, Any] = {"BTC": {"direction": "LONG", "active": False}}
        await check_trailing_stop("BTC", 50000.0, trailing_stop, mock_webhook, mock_increment)
        mock_webhook.assert_not_called()

    @pytest.mark.asyncio
    async def test_long_stop_triggered(self, mock_webhook: AsyncMock, mock_increment: MagicMock) -> None:
        """LONG trailing stop triggered when price < lower band."""
        trailing_stop: dict[str, Any] = {
            "BTC": {
                "direction": "LONG",
                "entry_price": 50000.0,
                "atr15m_upper": 52000.0,
                "atr15m_lower": 49000.0,
                "active": True,
            }
        }
        await check_trailing_stop("BTC", 48500.0, trailing_stop, mock_webhook, mock_increment)
        mock_webhook.assert_called_once()
        args, kwargs = mock_webhook.call_args
        assert "TRAILING STOP" in args[1]

    @pytest.mark.asyncio
    async def test_short_stop_triggered(self, mock_webhook: AsyncMock, mock_increment: MagicMock) -> None:
        """SHORT trailing stop triggered when price > upper band."""
        trailing_stop: dict[str, Any] = {
            "BTC": {
                "direction": "SHORT",
                "entry_price": 50000.0,
                "atr15m_upper": 51000.0,
                "atr15m_lower": 49000.0,
                "active": True,
            }
        }
        await check_trailing_stop("BTC", 51500.0, trailing_stop, mock_webhook, mock_increment)
        mock_webhook.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_deactivated_after_trigger(self, mock_webhook: AsyncMock, mock_increment: MagicMock) -> None:
        """Trailing stop should be deactivated after triggering."""
        trailing_stop: dict[str, Any] = {
            "BTC": {
                "direction": "LONG",
                "entry_price": 50000.0,
                "atr15m_upper": 52000.0,
                "atr15m_lower": 49000.0,
                "active": True,
            }
        }
        await check_trailing_stop("BTC", 48500.0, trailing_stop, mock_webhook, mock_increment)
        assert trailing_stop["BTC"]["active"] is False

    @pytest.mark.asyncio
    async def test_zero_price_returns_early(self) -> None:
        """Zero price should return immediately."""
        trailing_stop: dict[str, Any] = {"BTC": {"direction": "LONG", "active": True}}
        mock_webhook = AsyncMock()
        mock_increment = MagicMock()
        await check_trailing_stop("BTC", 0.0, trailing_stop, mock_webhook, mock_increment)
        mock_webhook.assert_not_called()

    @pytest.mark.asyncio
    async def test_clustering_ts_stop(self, mock_webhook: AsyncMock, mock_increment: MagicMock) -> None:
        """Clustering TS trailing stop uses clustering_ts line."""
        trailing_stop: dict[str, Any] = {
            "BTC": {
                "direction": "LONG",
                "entry_price": 50000.0,
                "active": True,
                "use_clustering_ts": True,
                "clustering_ts": 49500.0,
            }
        }
        await check_trailing_stop("BTC", 49000.0, trailing_stop, mock_webhook, mock_increment)
        mock_webhook.assert_called_once()


class TestRecalculateStates:
    """Test indicator recalculation."""

    @pytest.mark.asyncio
    async def test_insufficient_klines(self) -> None:
        """Less than MIN_KLINES should return early."""
        kline_cache: dict[str, Any] = {"BTC": [[1, 2, 3, 4, 5, 6]] * 50}
        benchmark: dict[str, Any] = {}
        await recalculate_states(
            "BTC",
            kline_cache,
            benchmark,
            False,
            9,
            2.5,
            14,
            1.7,
            9,
            144,
            169,
            14,
            "DEMA",
            1.618,
            14,
            "HMA",
            1.3,
        )
        assert "BTC" not in benchmark

    @pytest.mark.asyncio
    async def test_recalculates_indicators(self) -> None:
        """Should calculate all indicators and store in benchmark."""
        n = 300
        now = int(time.time() * 1000)
        klines = [
            Kline(
                symbol="BTC",
                interval="1h",
                open_time=now - (n - i) * 3600000,
                open=65000.0 + i,
                high=66000.0 + i,
                low=64000.0 + i,
                close=65500.0 + i,
                volume=100.0,
            )
            for i in range(n)
        ]
        kline_cache: dict[str, list[Kline]] = {"BTC": klines}
        benchmark: dict[str, Any] = {}
        await recalculate_states(
            "BTC",
            kline_cache,
            benchmark,
            False,
            9,
            2.5,
            14,
            1.7,
            9,
            144,
            169,
            14,
            "DEMA",
            1.618,
            14,
            "HMA",
            1.3,
        )
        assert "BTC" in benchmark
        bm = benchmark["BTC"]
        assert "st1" in bm
        assert "st2" in bm
        assert "atr1h_upper" in bm
        assert "atr1h_lower" in bm


class TestUpdateKlines:
    """Test K-line update logic."""

    @pytest.mark.asyncio
    async def test_update_klines_creates_client_on_error(self) -> None:
        """Should handle errors gracefully."""
        kline_cache: dict[str, Any] = {}
        last_kline_time: dict[str, Any] = {}

        async def false_is_pair(_sym: str) -> bool:
            return False

        async def false_fetch(_sym: str, *, proxy: str | None = None) -> None:
            raise Exception("Network error")

        await update_klines(
            "BTC",
            kline_cache,
            last_kline_time,
            false_is_pair,
            proxy=None,
            fetch_klines_fn=false_fetch,
        )
        assert "BTC" not in kline_cache
