"""Tests for signal detection module - price comparisons and state machines."""

import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import orjson
import pytest

from models import Kline
from service.notification_service import (
    NotificationService,
    aggregate_pair_15m_to_1h,
    aggregate_pair_15m_to_4h,
    build_pair_15m_klines,
)
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


class TestPairATRKlines:
    """Test pair ATR synthetic kline generation."""

    def test_build_pair_15m_klines(self) -> None:
        """Pair 15m klines should be built from aligned leg open/close ratios."""
        klines1 = [
            Kline("LEG1", "15m", 0, 100.0, 101.0, 99.0, 102.0, 10.0, 899999, True),
            Kline("LEG1", "15m", 900000, 102.0, 103.0, 101.0, 104.0, 12.0, 1799999, True),
        ]
        klines2 = [
            Kline("LEG2", "15m", 0, 50.0, 51.0, 49.0, 51.0, 20.0, 899999, True),
            Kline("LEG2", "15m", 900000, 51.0, 52.0, 50.0, 52.0, 22.0, 1799999, True),
        ]

        result = build_pair_15m_klines("LEG1-LEG2", klines1, klines2)

        assert len(result) == 2
        assert result[0].open == 2.0
        assert result[0].close == 2.0
        assert result[1].open == 2.0
        assert result[1].close == 2.0
        assert result[0].high == 2.0
        assert result[0].low == 2.0

    def test_aggregate_pair_15m_to_1h(self) -> None:
        """Four 15m pair klines should aggregate into one 1h kline."""
        klines_15m = [
            Kline("PAIR", "15m", 0, 2.0, 2.1, 1.9, 2.05, 1.0, 899999, True),
            Kline("PAIR", "15m", 900000, 2.05, 2.2, 2.0, 2.1, 1.0, 1799999, True),
            Kline("PAIR", "15m", 1800000, 2.1, 2.15, 2.05, 2.08, 1.0, 2699999, True),
            Kline("PAIR", "15m", 2700000, 2.08, 2.3, 2.0, 2.25, 1.0, 3599999, True),
        ]

        result = aggregate_pair_15m_to_1h("PAIR", klines_15m)

        assert len(result) == 1
        assert result[0].open == 2.0
        assert result[0].close == 2.25
        assert result[0].high == 2.3
        assert result[0].low == 1.9

    def test_aggregate_pair_15m_to_4h(self) -> None:
        """Sixteen 15m pair klines should aggregate into one 4h kline."""
        klines_15m = [
            Kline(
                "PAIR",
                "15m",
                i * 900000,
                2.0 + i * 0.01,
                2.05 + i * 0.01,
                1.95 + i * 0.01,
                2.02 + i * 0.01,
                1.0,
                (i + 1) * 900000 - 1,
                True,
            )
            for i in range(16)
        ]

        result = aggregate_pair_15m_to_4h("PAIR", klines_15m)

        assert len(result) == 1
        assert result[0].interval == "4h"
        assert result[0].open == 2.0
        assert result[0].close == pytest.approx(2.17)
        assert result[0].high == pytest.approx(2.2)
        assert result[0].low == 1.95


class TestNotificationServiceATRMode:
    """Test ATR mode switching and trailing refresh."""

    def _write_config(self, tmp_path: Any, clustering_enabled: bool, heartbeat_timeout: int = 120) -> str:
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            "\n".join(
                [
                    "[webhook]",
                    'url = "https://example.com/hook"',
                    'format = "card"',
                    "",
                    "[symbols]",
                    "single_list = []",
                    'pair_list = ["AAA-BBB"]',
                    "",
                    "[service]",
                    'heartbeat_file = "heartbeat"',
                    f"heartbeat_timeout = {heartbeat_timeout}",
                    "",
                    "[clustering_st]",
                    f"enabled = {str(clustering_enabled).lower()}",
                    "",
                    "[settings]",
                    'timezone = "+08:00"',
                ]
            ),
            encoding="utf-8",
        )
        return str(config_path)

    def test_pair_uses_atr_path_when_clustering_disabled(self, tmp_path: Any) -> None:
        """Pair should stop using clustering path when config disables it."""
        service = NotificationService(self._write_config(tmp_path, clustering_enabled=False))
        assert service._use_clustering_for_symbol("AAA-BBB") is False

    @pytest.mark.asyncio
    async def test_refresh_trailing_stop_channel_uses_full_15m_history(self, tmp_path: Any) -> None:
        """Trailing stop refresh should rebuild ATR channel from fetched 15m history."""
        service = NotificationService(self._write_config(tmp_path, clustering_enabled=False))
        service.atr15m_period = 2
        service.atr15m_ma_type = "EMA"
        service.trailing_stop["AAA-BBB"] = {
            "direction": "LONG",
            "entry_price": 2.0,
            "atr_mult": 1.3,
            "atr15m_upper": 0.0,
            "atr15m_lower": 0.0,
            "atr15m_state": (float("nan"), float("nan"), 0),
            "active": True,
        }

        async def mock_fetch(_symbol: str, limit: int = 500) -> list[Kline]:
            return [
                Kline("AAA-BBB", "15m", 0, 2.0, 2.1, 1.9, 2.05, 1.0, 899999, True),
                Kline("AAA-BBB", "15m", 900000, 2.05, 2.2, 2.0, 2.1, 1.0, 1799999, True),
                Kline("AAA-BBB", "15m", 1800000, 2.1, 2.2, 2.05, 2.18, 1.0, 2699999, True),
                Kline("AAA-BBB", "15m", 2700000, 2.18, 2.25, 2.1, 2.2, 1.0, 3599999, True),
                Kline("AAA-BBB", "15m", 3600000, 2.2, 2.3, 2.15, 2.28, 1.0, 4499999, True),
            ]

        service._fetch_15m_klines = mock_fetch  # type: ignore[method-assign]
        await service._refresh_trailing_stop_channel("AAA-BBB", force=True)

        assert service.trailing_stop["AAA-BBB"]["atr15m_upper"] > 0
        assert service.trailing_stop["AAA-BBB"]["atr15m_lower"] > 0

    @pytest.mark.asyncio
    async def test_pair_symbol_not_checked_twice_in_ws_loop(self, tmp_path: Any) -> None:
        """Pair symbol should only run signal detection once per WS batch."""
        service = NotificationService(self._write_config(tmp_path, clustering_enabled=False))

        class FakeWSMessage:
            def __init__(self, msg_type: Any, data: str = "") -> None:
                self.type = msg_type
                self.data = data

        class FakeWS:
            def __init__(self, messages: list[FakeWSMessage]) -> None:
                self._messages = messages

            async def receive(self) -> FakeWSMessage:
                return self._messages.pop(0)

        payload = {
            "channel": "allMids",
            "data": {
                "mids": {
                    "AAA": "2.0",
                    "BBB": "1.0",
                    "AAA-BBB": "2.0",
                }
            },
        }
        service._hl_ws = FakeWS(
            [
                FakeWSMessage(msg_type=aiohttp.WSMsgType.TEXT, data=orjson.dumps(payload).decode()),
                FakeWSMessage(msg_type=aiohttp.WSMsgType.CLOSED),
            ]
        )
        service._hl_ws_running = True
        service._log_symbol_state = MagicMock()
        service._maybe_refresh_runtime_atr = AsyncMock()
        service._maybe_refresh_runtime_atr_4h = AsyncMock()
        service._refresh_trailing_stop_channel = AsyncMock()
        service._ct_check_trailing_stop = AsyncMock()
        service._ct_check_signals = AsyncMock()
        service._ct_check_signals_clustering = AsyncMock()
        service._ct_check_signals_4h = AsyncMock()
        service._reconnect_hyperliquid_ws = AsyncMock(return_value=False)

        await service._watch_hyperliquid_marks()

        service._ct_check_signals.assert_called_once_with("AAA-BBB")

    @pytest.mark.asyncio
    async def test_watch_marks_sends_ping_on_idle_timeout(self, tmp_path: Any, monkeypatch: Any) -> None:
        """Watcher should send ping when receive loop is idle."""
        service = NotificationService(self._write_config(tmp_path, clustering_enabled=False))

        class FakeWSMessage:
            def __init__(self, msg_type: Any) -> None:
                self.type = msg_type
                self.data = ""

        service._hl_ws_running = True
        fake_ws = MagicMock()
        fake_ws.receive = AsyncMock()
        fake_ws.send_json = AsyncMock()
        service._hl_ws = fake_ws
        service._reconnect_hyperliquid_ws = AsyncMock(return_value=False)

        calls = {"count": 0}

        async def fake_wait_for(_awaitable: Any, timeout: float) -> Any:
            _ = timeout
            calls["count"] += 1
            close_coro = getattr(_awaitable, "close", None)
            if callable(close_coro):
                close_coro()
            if calls["count"] == 1:
                raise TimeoutError
            return FakeWSMessage(aiohttp.WSMsgType.CLOSED)

        monkeypatch.setattr("service.notification_service.asyncio.wait_for", fake_wait_for)

        await service._watch_hyperliquid_marks()

        fake_ws.send_json.assert_called_once_with({"method": "ping"})

    @pytest.mark.asyncio
    async def test_reconnect_ws_retries_until_success(self, tmp_path: Any, monkeypatch: Any) -> None:
        """Reconnect helper should retry with backoff until websocket reconnects."""
        service = NotificationService(self._write_config(tmp_path, clustering_enabled=False))
        service._hl_ws_running = True
        service._close_hyperliquid_ws = AsyncMock()
        service._send_webhook = AsyncMock()

        attempts = {"count": 0}

        async def fake_connect(*, start_watch_task: bool = True) -> None:
            _ = start_watch_task
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise ConnectionError("temporary")

        monkeypatch.setattr(service, "_connect_hyperliquid_ws", fake_connect)
        monkeypatch.setattr("service.notification_service.asyncio.sleep", AsyncMock())

        assert await service._reconnect_hyperliquid_ws("test") is True
        assert attempts["count"] == 2
        service._send_webhook.assert_any_call("ERROR", "Hyperliquid WS disconnected: test. Reconnecting...")
        service._send_webhook.assert_any_call("SYSTEM", "Hyperliquid WS reconnected after test (attempt 2)")

    @pytest.mark.asyncio
    async def test_check_signals_4h_sends_alert(self, tmp_path: Any) -> None:
        """4H ATR breakout should emit webhook without creating trailing stop."""
        service = NotificationService(self._write_config(tmp_path, clustering_enabled=False))
        service._initialized = True
        service.mark_prices["AAA-BBB"] = 2.3
        service.mark_price_times["AAA-BBB"] = time.time()
        service.benchmark["AAA-BBB"] = {
            "atr4h_upper": 2.2,
            "atr4h_lower": 2.0,
            "atr4h_natrr": 0.05,
        }
        service._send_webhook = AsyncMock()

        await service._ct_check_signals_4h("AAA-BBB")

        service._send_webhook.assert_called_once()
        args, _ = service._send_webhook.call_args
        assert args[0] == "ATR_Ch"
        assert args[1] == "[AAA-BBB] 4H LONG"
        assert args[2]["timeframe"] == "4H"
        assert "AAA-BBB" not in service.trailing_stop

    @pytest.mark.asyncio
    async def test_check_ws_data_silence_triggers_reconnect(self, tmp_path: Any) -> None:
        """Data silence beyond heartbeat_timeout should trigger reconnect."""
        service = NotificationService(self._write_config(tmp_path, clustering_enabled=False, heartbeat_timeout=90))
        service._last_ws_data_time = 100.0
        service._reconnect_hyperliquid_ws = AsyncMock(return_value=True)
        service._send_webhook = AsyncMock()

        result = await service._check_ws_data_silence(now=191.0)

        assert result is True
        service._send_webhook.assert_any_call("ERROR", "Hyperliquid market data silent for 91s. Reconnecting...")
        service._reconnect_hyperliquid_ws.assert_awaited_once_with("market data silence > 90s (last 91s ago)")

    @pytest.mark.asyncio
    async def test_notify_ws_data_recovered_sends_system_webhook(self, tmp_path: Any) -> None:
        """Data recovery after silence should send a SYSTEM webhook once."""
        service = NotificationService(self._write_config(tmp_path, clustering_enabled=False))
        service._send_webhook = AsyncMock()
        service._ws_silence_alert_active = True
        service._ws_silence_started_at = 100.0

        await service._notify_ws_data_recovered(recovered_at=145.0)

        service._send_webhook.assert_awaited_once_with(
            "SYSTEM",
            "Hyperliquid market data resumed after 45s silence",
        )
        assert service._ws_silence_alert_active is False

    @pytest.mark.asyncio
    async def test_watch_marks_updates_heartbeat_on_allmids(self, tmp_path: Any) -> None:
        """Receiving allMids data should refresh heartbeat file and data timestamp."""
        service = NotificationService(self._write_config(tmp_path, clustering_enabled=False))
        heartbeat_path = tmp_path / "hb.txt"
        service.heartbeat_file = str(heartbeat_path)

        class FakeWSMessage:
            def __init__(self, msg_type: Any, data: str = "") -> None:
                self.type = msg_type
                self.data = data

        class FakeWS:
            def __init__(self, messages: list[FakeWSMessage]) -> None:
                self._messages = messages

            async def receive(self) -> FakeWSMessage:
                return self._messages.pop(0)

        payload = {
            "channel": "allMids",
            "data": {
                "mids": {
                    "AAA": "2.0",
                    "BBB": "1.0",
                }
            },
        }
        service._hl_ws = FakeWS(
            [
                FakeWSMessage(msg_type=aiohttp.WSMsgType.TEXT, data=orjson.dumps(payload).decode()),
                FakeWSMessage(msg_type=aiohttp.WSMsgType.CLOSED),
            ]
        )
        service._hl_ws_running = True
        service._log_symbol_state = MagicMock()
        service._maybe_refresh_runtime_atr = AsyncMock()
        service._maybe_refresh_runtime_atr_4h = AsyncMock()
        service._refresh_trailing_stop_channel = AsyncMock()
        service._ct_check_trailing_stop = AsyncMock()
        service._ct_check_signals = AsyncMock()
        service._ct_check_signals_clustering = AsyncMock()
        service._ct_check_signals_4h = AsyncMock()
        service._reconnect_hyperliquid_ws = AsyncMock(return_value=False)

        await service._watch_hyperliquid_marks()

        assert heartbeat_path.exists()
        assert int(heartbeat_path.read_text(encoding="utf-8").strip()) > 0
        assert service._last_ws_data_time > 0


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

        def false_is_pair(_sym: str) -> bool:
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
