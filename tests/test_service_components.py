"""Tests for extracted service-layer components."""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from models import Kline
from notifications.alert_event import build_alert_event
from service.alert_dispatcher import AlertDispatcher
from service.market_data_processor import MarketDataProcessor
from service.notification_lifecycle import NotificationLifecycleManager
from service.notification_orchestration import NotificationInitializationOrchestrator, NotificationRunOrchestrator
from service.notification_service import NotificationService
from service.runtime_symbol_utils import cleanup_symbol_state, parse_pair_components, seed_initial_signal_states
from service.signal_coordinator import SignalCoordinator
from service.state import RuntimeState
from service.ws_runtime_supervisor import WSRuntimeSupervisor
from signals.state import AtrChannelState, BreakoutMonitorState, ClusteringSignalState


class TestAlertDispatcher:
    """Test alert dispatcher delivery facade."""

    @pytest.mark.asyncio
    async def test_send_alert_builds_and_dispatches_event(self, tmp_path: Path) -> None:
        """Legacy alert arguments should be converted and sent through shared sender."""
        log_path = tmp_path / "dispatcher-webhook.log"
        sender = SimpleNamespace(send_json=AsyncMock())
        dispatcher = AlertDispatcher(
            "https://example.com/webhook",
            "text",
            str(log_path),
            100,
            lambda: "2026-04-14T12:00:00+0800",
            sender,
        )

        await dispatcher.send_alert("SYSTEM", "dispatcher ok", {"symbol": "BTC"})

        sender.send_json.assert_awaited_once()
        assert "[SYSTEM] dispatcher ok" in log_path.read_text(encoding="utf-8")

    @pytest.mark.asyncio
    async def test_send_alert_dedupes_with_ttl(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Dispatcher should suppress repeated delivery within dedupe TTL."""
        log_path = tmp_path / "dispatcher-webhook.log"
        sender = SimpleNamespace(send_json=AsyncMock(return_value=True))
        dispatcher = AlertDispatcher(
            "https://example.com/webhook",
            "text",
            str(log_path),
            100,
            lambda: "2026-04-14T12:00:00+0800",
            sender,
        )
        event = build_alert_event("SYSTEM", "dup", {"symbol": "BTC"})
        monkeypatch.setattr(dispatcher, "_dedupe_ttl_seconds", 60)
        monkeypatch.setattr("service.alert_dispatcher.time.time", lambda: 1000.0)

        first = await dispatcher.send_event(event)
        second = await dispatcher.send_event(event)

        assert first is True
        assert second is False
        sender.send_json.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_send_alert_allows_retry_after_ttl_expiry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Dispatcher should send again after dedupe TTL expires."""
        log_path = tmp_path / "dispatcher-webhook.log"
        sender = SimpleNamespace(send_json=AsyncMock(return_value=True))
        dispatcher = AlertDispatcher(
            "https://example.com/webhook",
            "text",
            str(log_path),
            100,
            lambda: "2026-04-14T12:00:00+0800",
            sender,
        )
        event = build_alert_event("ATR_Ch", "boom", {"symbol": "BTC"})
        monkeypatch.setattr(dispatcher, "_dedupe_ttl_seconds", 10)

        times = iter([1000.0, 1011.0])
        monkeypatch.setattr("service.alert_dispatcher.time.time", lambda: next(times))

        first = await dispatcher.send_event(event)
        second = await dispatcher.send_event(event)

        assert first is True
        assert second is True
        assert sender.send_json.await_count == 2

    def test_dedupe_ttl_varies_by_alert_type(self, tmp_path: Path) -> None:
        """Different alert types should use dedicated dedupe TTL policies."""
        dispatcher = AlertDispatcher(
            "https://example.com/webhook",
            "text",
            str(tmp_path / "dispatcher-webhook.log"),
            100,
            lambda: "2026-04-14T12:00:00+0800",
            SimpleNamespace(send_json=AsyncMock()),
        )

        assert dispatcher._dedupe_ttl_for_event(build_alert_event("SYSTEM", "sys")).__class__ is int
        assert dispatcher._dedupe_ttl_for_event(build_alert_event("SYSTEM", "sys")) == 30
        assert dispatcher._dedupe_ttl_for_event(build_alert_event("ERROR", "err")) == 60
        assert dispatcher._dedupe_ttl_for_event(build_alert_event("BREAKOUT", "brk")) == 120
        assert dispatcher._dedupe_ttl_for_event(build_alert_event("ATR_Ch", "atr")) == 300

    @pytest.mark.asyncio
    async def test_send_event_logs_dedupe_skip(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Dedupe hits should emit an explicit skip log and avoid transport calls."""
        log_path = tmp_path / "dispatcher-webhook.log"
        sender = SimpleNamespace(send_json=AsyncMock(return_value=True))
        dispatcher = AlertDispatcher(
            "https://example.com/webhook",
            "text",
            str(log_path),
            100,
            lambda: "2026-04-14T12:00:00+0800",
            sender,
        )
        event = build_alert_event("SYSTEM", "dup", {"symbol": "BTC"})
        monkeypatch.setattr("service.alert_dispatcher.time.time", lambda: 1000.0)
        info_calls: list[tuple[object, ...]] = []
        monkeypatch.setattr(dispatcher._logger, "info", lambda *args: info_calls.append(args))

        first = await dispatcher.send_event(event)
        second = await dispatcher.send_event(event)

        assert first is True
        assert second is False
        sender.send_json.assert_awaited_once()
        assert any(call[1] == "dedupe skipped" for call in info_calls)

    @pytest.mark.asyncio
    async def test_send_event_logs_success_and_failure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Dispatcher should log both success and failure outcomes at the unified exit."""
        log_path = tmp_path / "dispatcher-webhook.log"
        sender = SimpleNamespace(send_json=AsyncMock(side_effect=[True, False]))
        dispatcher = AlertDispatcher(
            "https://example.com/webhook",
            "text",
            str(log_path),
            100,
            lambda: "2026-04-14T12:00:00+0800",
            sender,
        )
        info_calls: list[tuple[object, ...]] = []
        monkeypatch.setattr(dispatcher._logger, "info", lambda *args: info_calls.append(args))
        monkeypatch.setattr("service.alert_dispatcher.time.time", lambda: 1000.0)

        ok = await dispatcher.send_event(build_alert_event("ERROR", "boom", {"symbol": "BTC1"}))
        failed = await dispatcher.send_event(build_alert_event("ERROR", "boom2", {"symbol": "BTC2"}))

        assert ok is True
        assert failed is False
        assert any(call[1] == "send success" for call in info_calls)
        assert any(call[1] == "send failed" for call in info_calls)

    @pytest.mark.asyncio
    async def test_dispatcher_tracks_stats(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Dispatcher should expose attempted/sent/failed/deduped counters."""
        log_path = tmp_path / "dispatcher-webhook.log"
        sender = SimpleNamespace(send_json=AsyncMock(side_effect=[True, False]))
        dispatcher = AlertDispatcher(
            "https://example.com/webhook",
            "text",
            str(log_path),
            100,
            lambda: "2026-04-14T12:00:00+0800",
            sender,
        )
        event = build_alert_event("SYSTEM", "dup", {"symbol": "BTC"})
        monkeypatch.setattr("service.alert_dispatcher.time.time", lambda: 1000.0)

        await dispatcher.send_event(event)
        await dispatcher.send_event(event)

        stats = dispatcher.get_stats()
        assert stats.attempted == 1
        assert stats.sent == 1
        assert stats.failed == 0
        assert stats.deduped == 1

    @pytest.mark.asyncio
    async def test_queue_dispatcher_enqueues_and_worker_sends(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Queued dispatcher should accept alerts and deliver them in the worker."""
        log_path = tmp_path / "dispatcher-webhook.log"
        sender = SimpleNamespace(send_json=AsyncMock(return_value=True))
        dispatcher = AlertDispatcher(
            "https://example.com/webhook",
            "text",
            str(log_path),
            100,
            lambda: "2026-04-14T12:00:00+0800",
            sender,
            queue_maxsize=2,
        )
        monkeypatch.setattr("service.alert_dispatcher.time.time", lambda: 1000.0)

        queued = await dispatcher.send_event(build_alert_event("ERROR", "boom", {"symbol": "BTC"}))
        await dispatcher.stop_worker()

        assert queued is True
        sender.send_json.assert_awaited_once()
        stats = dispatcher.get_stats()
        assert stats.queued == 1
        assert stats.sent == 1
        assert stats.attempted == 1

    @pytest.mark.asyncio
    async def test_queue_dispatcher_returns_failure_when_full(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Full queues should warn and return a visible failure."""
        log_path = tmp_path / "dispatcher-webhook.log"
        sender = SimpleNamespace(send_json=AsyncMock(return_value=True))
        dispatcher = AlertDispatcher(
            "https://example.com/webhook",
            "text",
            str(log_path),
            100,
            lambda: "2026-04-14T12:00:00+0800",
            sender,
            queue_maxsize=1,
        )
        monkeypatch.setattr("service.alert_dispatcher.time.time", lambda: 1000.0)
        warnings: list[str] = []
        monkeypatch.setattr(dispatcher._logger, "warning", lambda msg, *args: warnings.append(msg % args if args else msg))

        first = await dispatcher.send_event(build_alert_event("ERROR", "one", {"symbol": "BTC1"}))
        second = await dispatcher.send_event(build_alert_event("ERROR", "two", {"symbol": "BTC2"}))
        await dispatcher.stop_worker()

        assert first is True
        assert second is False
        assert any("queue full" in warning for warning in warnings)
        stats = dispatcher.get_stats()
        assert stats.queued == 1
        assert stats.dropped == 1

    @pytest.mark.asyncio
    async def test_queue_dispatcher_dedupes_before_enqueue(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Dedupe should happen before queued alerts count as enqueued."""
        log_path = tmp_path / "dispatcher-webhook.log"
        sender = SimpleNamespace(send_json=AsyncMock(return_value=True))
        dispatcher = AlertDispatcher(
            "https://example.com/webhook",
            "text",
            str(log_path),
            100,
            lambda: "2026-04-14T12:00:00+0800",
            sender,
            queue_maxsize=2,
        )
        event = build_alert_event("SYSTEM", "dup", {"symbol": "BTC"})
        monkeypatch.setattr("service.alert_dispatcher.time.time", lambda: 1000.0)

        first = await dispatcher.send_event(event)
        second = await dispatcher.send_event(event)
        await dispatcher.stop_worker()

        assert first is True
        assert second is False
        stats = dispatcher.get_stats()
        assert stats.queued == 1
        assert stats.deduped == 1

    @pytest.mark.asyncio
    async def test_queue_dispatcher_records_failed_send(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Worker should surface failed transport results in stats."""
        log_path = tmp_path / "dispatcher-webhook.log"
        sender = SimpleNamespace(send_json=AsyncMock(return_value=False))
        dispatcher = AlertDispatcher(
            "https://example.com/webhook",
            "text",
            str(log_path),
            100,
            lambda: "2026-04-14T12:00:00+0800",
            sender,
            queue_maxsize=2,
        )
        monkeypatch.setattr("service.alert_dispatcher.time.time", lambda: 1000.0)

        queued = await dispatcher.send_event(build_alert_event("ERROR", "boom", {"symbol": "BTC"}))
        await dispatcher.stop_worker()

        assert queued is True
        stats = dispatcher.get_stats()
        assert stats.failed == 1

    @pytest.mark.asyncio
    async def test_queue_dispatcher_stop_rejects_new_alerts(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Stopping should close the queue and reject new alerts."""
        log_path = tmp_path / "dispatcher-webhook.log"
        sender = SimpleNamespace(send_json=AsyncMock(return_value=True))
        dispatcher = AlertDispatcher(
            "https://example.com/webhook",
            "text",
            str(log_path),
            100,
            lambda: "2026-04-14T12:00:00+0800",
            sender,
            queue_maxsize=2,
        )
        monkeypatch.setattr("service.alert_dispatcher.time.time", lambda: 1000.0)

        assert await dispatcher.send_event(build_alert_event("ERROR", "boom", {"symbol": "BTC"})) is True
        await dispatcher.stop_worker()
        rejected = await dispatcher.send_event(build_alert_event("ERROR", "later", {"symbol": "ETH"}))

        assert rejected is False
        assert dispatcher.get_stats().dropped >= 1


class TestMarketDataProcessor:
    """Test market data processing extracted from websocket loop."""

    @pytest.mark.asyncio
    async def test_process_payload_updates_pair_price_and_runs_callbacks(self) -> None:
        """allMids payload should update prices and trigger pair processing callbacks."""
        mark_prices: dict[str, float] = {}
        mark_price_times: dict[str, float] = {}
        logged_initial_price: set[str] = set()
        record_ws_data_activity = AsyncMock()
        log_symbol_state = MagicMock()
        maybe_refresh_runtime_atr = AsyncMock()
        maybe_refresh_runtime_atr_4h = AsyncMock()
        refresh_trailing_stop_channel = AsyncMock()
        check_trailing_stop = AsyncMock()
        check_signals_clustering = AsyncMock()
        check_signals = AsyncMock()
        check_signals_4h = AsyncMock()
        check_breakout = AsyncMock()

        processor = MarketDataProcessor(
            symbols_fn=lambda: ["AAA-BBB"],
            pair_components_fn=lambda: {"AAA-BBB": ("AAA", "BBB")},
            mark_prices=mark_prices,
            mark_price_times=mark_price_times,
            logged_initial_price=logged_initial_price,
            record_ws_data_activity_fn=record_ws_data_activity,
            log_symbol_state_fn=log_symbol_state,
            maybe_refresh_runtime_atr_fn=maybe_refresh_runtime_atr,
            maybe_refresh_runtime_atr_4h_fn=maybe_refresh_runtime_atr_4h,
            refresh_trailing_stop_channel_fn=refresh_trailing_stop_channel,
            check_trailing_stop_fn=check_trailing_stop,
            use_clustering_for_symbol_fn=lambda _symbol: False,
            check_signals_clustering_fn=check_signals_clustering,
            is_pair_trading_fn=lambda symbol: symbol in {"AAA", "BBB"},
            is_pair_symbol_fn=lambda symbol: symbol == "AAA-BBB",
            check_signals_fn=check_signals,
            check_signals_4h_fn=check_signals_4h,
            check_breakout_fn=check_breakout,
        )

        handled = await processor.process_payload(
            {
                "channel": "allMids",
                "data": {"mids": {"AAA": "2.0", "BBB": "1.0"}},
            }
        )

        assert handled is True
        assert mark_prices["AAA"] == 2.0
        assert mark_prices["BBB"] == 1.0
        assert mark_prices["AAA-BBB"] == 2.0
        record_ws_data_activity.assert_awaited_once()
        maybe_refresh_runtime_atr.assert_any_await("AAA-BBB")
        check_trailing_stop.assert_any_await("AAA-BBB", 2.0)
        check_signals.assert_any_await("AAA-BBB")
        check_signals_4h.assert_any_await("AAA-BBB")
        check_breakout.assert_any_await("AAA-BBB")
        assert "AAA-BBB" in logged_initial_price


class TestRuntimeState:
    """Test runtime state aggregation helpers."""

    def test_runtime_state_uses_independent_default_mappings(self) -> None:
        """Each runtime state instance should own its mutable mappings."""
        state = RuntimeState()

        assert state.mark_prices == {}
        assert state.mark_price_times == {}
        assert state.benchmark == {}
        assert state.trailing_stop == {}
        assert state.breakout_monitor == {}
        assert state.last_alert_time == {}
        assert state.last_atr_state == {}
        assert state.last_atr4h_state == {}
        assert state.last_clustering_state == {}


class TestNotificationOrchestration:
    """Test notification startup/run orchestration helpers."""

    @pytest.mark.asyncio
    async def test_initialize_orchestrator_seeds_state_and_logs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Initialization helper should perform the same boot sequence as the service."""
        service = SimpleNamespace()
        service.symbols = ["BTC", "BTC-ETH"]
        service.single_list = ["BTC"]
        service.pair_list = ["BTC-ETH"]
        service._initialized = True
        service.mark_prices = {}
        service.benchmark = {}
        service.clustering_states = {}
        service.last_atr_state = {}
        service.last_atr4h_state = {}
        service.last_clustering_state = {}
        service.last_st_state = {}
        service.kline_cache = {"BTC": [Kline(symbol="BTC", interval="1h", open_time=1, open=1, high=1, low=1, close=10, volume=1, close_time=1, is_closed=True)]}
        service._prune_runtime_state = MagicMock()
        service.market_gateway = SimpleNamespace(fetch_meta=AsyncMock())
        service._ct_update_klines = AsyncMock()
        service._recalculate_states = AsyncMock()
        service._recalculate_4h_breakout_state = AsyncMock()
        service._ct_recalculate_states_clustering = AsyncMock()
        service._use_clustering_for_symbol = MagicMock(return_value=False)
        service._get_pair_for_symbol = MagicMock(return_value=("BTC", "ETH"))
        service._is_pair_symbol = MagicMock(side_effect=lambda symbol: symbol == "BTC-ETH")
        service._log_symbol_state = MagicMock()

        orchestrator = NotificationInitializationOrchestrator(service)
        monkeypatch.setattr("service.notification_orchestration.seed_initial_signal_states", MagicMock())

        await orchestrator.initialize()

        assert service._initialized is True
        service._prune_runtime_state.assert_called_once()
        service.market_gateway.fetch_meta.assert_awaited_once()
        service._ct_update_klines.assert_any_await("BTC")
        service._ct_update_klines.assert_any_await("BTC-ETH")
        service._recalculate_states.assert_any_await("BTC")
        service._recalculate_4h_breakout_state.assert_any_await("BTC")
        service._log_symbol_state.assert_any_call("BTC")
        service._log_symbol_state.assert_any_call("BTC-ETH")

    @pytest.mark.asyncio
    async def test_run_orchestrator_delegates_to_service_lifecycle(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Run helper should preserve the service lifecycle call order."""
        service = SimpleNamespace()
        service.running = False
        service.initialize = AsyncMock()
        service.connect = AsyncMock()
        service.stop = AsyncMock()
        service._ready_summary_orchestrator = SimpleNamespace(send_ready_summary=AsyncMock())

        orchestrator = NotificationRunOrchestrator(service)
        sleep_calls = 0

        async def fake_sleep(_: float) -> None:
            nonlocal sleep_calls
            sleep_calls += 1
            if sleep_calls == 1:
                service.running = False

        monkeypatch.setattr("service.notification_orchestration.asyncio.sleep", fake_sleep)

        await orchestrator.run()

        service.initialize.assert_awaited_once()
        service.connect.assert_awaited_once()
        service._ready_summary_orchestrator.send_ready_summary.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_process_payload_skips_stale_pair_components(self) -> None:
        """Pair price should not update when one leg is stale."""
        mark_prices: dict[str, float] = {"AAA": 2.0, "BBB": 1.0}
        mark_price_times: dict[str, float] = {"AAA": time.time(), "BBB": time.time() - 400}
        logged_initial_price: set[str] = set()
        processor = MarketDataProcessor(
            symbols_fn=lambda: ["AAA-BBB"],
            pair_components_fn=lambda: {"AAA-BBB": ("AAA", "BBB")},
            mark_prices=mark_prices,
            mark_price_times=mark_price_times,
            logged_initial_price=logged_initial_price,
            record_ws_data_activity_fn=AsyncMock(),
            log_symbol_state_fn=MagicMock(),
            maybe_refresh_runtime_atr_fn=AsyncMock(),
            maybe_refresh_runtime_atr_4h_fn=AsyncMock(),
            refresh_trailing_stop_channel_fn=AsyncMock(),
            check_trailing_stop_fn=AsyncMock(),
            use_clustering_for_symbol_fn=lambda _symbol: False,
            check_signals_clustering_fn=AsyncMock(),
            is_pair_trading_fn=lambda symbol: symbol in {"AAA", "BBB"},
            is_pair_symbol_fn=lambda symbol: symbol == "AAA-BBB",
            check_signals_fn=AsyncMock(),
            check_signals_4h_fn=AsyncMock(),
            check_breakout_fn=AsyncMock(),
        )

        await processor.process_payload({"channel": "allMids", "data": {"mids": {"AAA": "2.1"}}})

        assert "AAA-BBB" not in mark_prices

    @pytest.mark.asyncio
    async def test_process_payload_keeps_processing_after_bad_symbol_payload(self, caplog: pytest.LogCaptureFixture) -> None:
        """A bad symbol payload should not stop other symbols in the same batch."""
        mark_prices: dict[str, float] = {}
        mark_price_times: dict[str, float] = {}
        processor = MarketDataProcessor(
            symbols_fn=lambda: ["AAA", "BBB"],
            pair_components_fn=dict,
            mark_prices=mark_prices,
            mark_price_times=mark_price_times,
            logged_initial_price=set(),
            record_ws_data_activity_fn=AsyncMock(),
            log_symbol_state_fn=MagicMock(),
            maybe_refresh_runtime_atr_fn=AsyncMock(),
            maybe_refresh_runtime_atr_4h_fn=AsyncMock(),
            refresh_trailing_stop_channel_fn=AsyncMock(),
            check_trailing_stop_fn=AsyncMock(),
            use_clustering_for_symbol_fn=lambda _symbol: False,
            check_signals_clustering_fn=AsyncMock(),
            is_pair_trading_fn=lambda _symbol: False,
            is_pair_symbol_fn=lambda _symbol: False,
            check_signals_fn=AsyncMock(),
            check_signals_4h_fn=AsyncMock(),
            check_breakout_fn=AsyncMock(),
        )

        caplog.set_level("WARNING")
        handled = await processor.process_payload({"channel": "allMids", "data": {"mids": {"AAA": "oops", "BBB": "2.0"}}})

        assert handled is True
        assert "AAA" not in mark_prices
        assert mark_prices["BBB"] == 2.0
        assert "stage=parse_price" in caplog.text


class TestRuntimeSymbolUtils:
    """Test extracted symbol/state helper functions."""

    def test_parse_pair_components_ignores_invalid_symbols(self) -> None:
        """Only hyphenated pair symbols should be parsed."""
        assert parse_pair_components(["BTC-ETH", "BTC", "SOL-USDC"]) == {
            "BTC-ETH": ("BTC", "ETH"),
            "SOL-USDC": ("SOL", "USDC"),
        }

    def test_cleanup_symbol_state_removes_symbol_keys(self) -> None:
        """Cleanup should drop symbol-scoped runtime entries and alert keys."""
        prices = {"BTC": 1.0, "ETH": 2.0}
        alerts = {"BTC": 1.0, "ATR_Ch_BTC": 2.0, "misc": 3.0}

        cleanup_symbol_state("BTC", prices, alert_times=alerts)

        assert prices == {"ETH": 2.0}
        assert alerts == {"misc": 3.0}

    def test_seed_initial_signal_states_populates_expected_state(self) -> None:
        """Seed helper should mirror initialization behavior for signal caches."""
        benchmark = {"BTC": {"atr1h_ch": 1, "atr4h_ch": -1, "st1": 10, "st2": 12}}
        mark_prices = {"BTC": 13.0}
        clustering_states = {"BTC-ETH": SimpleNamespace(trend=2)}
        last_atr_state: dict[str, dict[str, object]] = {}
        last_atr4h_state: dict[str, dict[str, object]] = {}
        last_clustering_state: dict[str, dict[str, object]] = {}
        last_st_state: dict[str, str] = {}

        seed_initial_signal_states(
            ["BTC", "BTC-ETH"],
            benchmark,
            mark_prices,
            clustering_states,
            lambda symbol: symbol == "BTC-ETH",
            lambda symbol: symbol == "BTC-ETH",
            last_atr_state,
            last_atr4h_state,
            last_clustering_state,
            last_st_state,
        )

        assert last_atr_state["BTC"] == {"ch": 1, "sent": "LONG"}
        assert last_atr4h_state["BTC"] == {"ch": -1, "sent": "SHORT"}
        assert last_st_state["BTC"] == "11"
        assert last_clustering_state["BTC-ETH"] == {"trend": 2, "sent": None}

    def test_runtime_state_objects_support_legacy_access(self) -> None:
        """Dataclass runtime states should remain compatible with dict-style callers."""
        atr = AtrChannelState(ch=1, sent="LONG")
        clustering = ClusteringSignalState(trend=-1, sent="SHORT")
        breakout = BreakoutMonitorState(direction="11", trigger_price=1.0, trigger_time=2.0)

        assert atr["ch"] == 1
        assert clustering.get("sent") == "SHORT"
        assert breakout.to_legacy_dict()["trigger_price"] == 1.0


class TestNotificationServiceCompatibility:
    """Test thin compatibility wrappers on NotificationService."""

    def test_pair_helpers_delegate_to_shared_utils(self) -> None:
        """NotificationService should expose the same pair helper behavior."""
        service = NotificationService.__new__(NotificationService)
        service._pair_components = {"BTC-ETH": ("BTC", "ETH")}

        assert service._is_pair_trading("BTC") is True
        assert service._get_pair_for_symbol("BTC-ETH") == ("BTC", "ETH")

    @pytest.mark.asyncio
    async def test_lifecycle_manager_delegates_connect_and_stop(self) -> None:
        """Lifecycle helper should orchestrate the same service callbacks."""
        service = SimpleNamespace(
            _check_hyperliquid_connection=AsyncMock(),
            _connect_hyperliquid_ws=AsyncMock(),
            _send_webhook=AsyncMock(),
            _close_hyperliquid_ws=AsyncMock(),
            _webhook_sender=SimpleNamespace(close=AsyncMock()),
            observer=SimpleNamespace(stop=MagicMock(), join=MagicMock()),
            connected=False,
            running=True,
            _hl_ws_running=True,
            _ws_tasks=[],
            _exchange_id="binance",
            network=SimpleNamespace(ws=SimpleNamespace(reconnect_base_delay_seconds=1, reconnect_max_delay_seconds=2)),
            _notify_ws_reconnect_failure=AsyncMock(),
            _notify_ws_reconnect_success=AsyncMock(),
            _hl_ws=None,
        )
        manager = NotificationLifecycleManager(service)

        await manager.connect()
        await manager.stop()

        service._check_hyperliquid_connection.assert_awaited_once()
        service._connect_hyperliquid_ws.assert_awaited_once()
        service._send_webhook.assert_awaited()
        service._close_hyperliquid_ws.assert_awaited_once()
        service._webhook_sender.close.assert_awaited_once()
        service.observer.stop.assert_called_once()
        service.observer.join.assert_called_once()

    @pytest.mark.asyncio
    async def test_lifecycle_manager_reconnect_delegates_to_service(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Reconnect helper should reuse existing service reconnect flow."""
        service = SimpleNamespace(
            connected=True,
            _hl_ws_running=True,
            _close_hyperliquid_ws=AsyncMock(),
            _connect_hyperliquid_ws=AsyncMock(),
            _notify_ws_reconnect_failure=AsyncMock(),
            _notify_ws_reconnect_success=AsyncMock(),
            network=SimpleNamespace(ws=SimpleNamespace(reconnect_base_delay_seconds=0, reconnect_max_delay_seconds=0)),
        )
        manager = NotificationLifecycleManager(service)
        monkeypatch.setattr("service.notification_lifecycle.time.time", lambda: 100.0)

        assert await manager.reconnect_hyperliquid_ws("reason") is True
        service._close_hyperliquid_ws.assert_awaited_once()
        service._connect_hyperliquid_ws.assert_awaited_once_with(start_watch_task=False)


class TestWSRuntimeSupervisor:
    """Test websocket runtime lifecycle supervisor."""

    @pytest.mark.asyncio
    async def test_timeout_sends_ping(self) -> None:
        """Idle timeout should trigger application ping before continuing."""
        should_run_state = {"count": 0}
        receive_message = AsyncMock(side_effect=TimeoutError)
        check_silence = AsyncMock(side_effect=[False, False, False])
        send_ping = AsyncMock()
        reconnect = AsyncMock(return_value=False)
        mark_message_received = MagicMock()
        process_payload = AsyncMock(return_value=False)

        def should_run() -> bool:
            should_run_state["count"] += 1
            return should_run_state["count"] == 1

        supervisor = WSRuntimeSupervisor(
            should_run_fn=should_run,
            check_data_silence_fn=check_silence,
            receive_message_fn=receive_message,
            send_ping_fn=send_ping,
            reconnect_fn=reconnect,
            mark_message_received_fn=mark_message_received,
            enqueue_payload_fn=MagicMock(),
            process_payload_fn=process_payload,
        )

        await supervisor.run()

        send_ping.assert_awaited_once()
        reconnect.assert_not_awaited()
        mark_message_received.assert_not_called()

    @pytest.mark.asyncio
    async def test_closed_message_triggers_reconnect(self) -> None:
        """Closed websocket message should route through reconnect callback."""
        message = SimpleNamespace(type=aiohttp.WSMsgType.CLOSED, data="")
        receive_message = AsyncMock(return_value=message)
        check_silence = AsyncMock(return_value=False)
        send_ping = AsyncMock()
        reconnect = AsyncMock(return_value=False)
        mark_message_received = MagicMock()
        process_payload = AsyncMock(return_value=False)

        supervisor = WSRuntimeSupervisor(
            should_run_fn=lambda: True,
            check_data_silence_fn=check_silence,
            receive_message_fn=receive_message,
            send_ping_fn=send_ping,
            reconnect_fn=reconnect,
            mark_message_received_fn=mark_message_received,
            enqueue_payload_fn=MagicMock(),
            process_payload_fn=process_payload,
        )

        await supervisor.run()

        reconnect.assert_awaited_once_with("message CLOSED")
        send_ping.assert_not_awaited()
        mark_message_received.assert_called_once()


class TestSignalCoordinator:
    """Test extracted signal coordination helpers."""

    @pytest.mark.asyncio
    async def test_sync_breakout_monitor_from_cache_appends_newer_bars(self) -> None:
        """Cached 15m bars should advance breakout monitor state."""
        symbol = "AAA-BBB"
        first = Kline(symbol=symbol, interval="15m", open_time=1, open=1.0, high=1.0, low=1.0, close=1.0, volume=1.0)
        second = Kline(symbol=symbol, interval="15m", open_time=2, open=1.1, high=1.1, low=1.0, close=1.1, volume=1.0)
        third = Kline(symbol=symbol, interval="15m", open_time=3, open=1.2, high=1.3, low=1.1, close=1.3, volume=1.0)

        coordinator = SignalCoordinator(
            mark_prices={},
            mark_price_times={},
            benchmark={},
            trailing_stop={},
            last_atr_state={},
            last_clustering_state={},
            last_alert_time={},
            last_st_state={},
            clustering_states={},
            breakout_monitor={symbol: {"kline_15m_count": 0, "klines_15m": [first, second]}},
            kline_cache_15m={symbol: [first, second, third]},
            send_webhook_fn=AsyncMock(),
            increment_alert_count_fn=MagicMock(),
            send_event_fn=AsyncMock(),
            refresh_trailing_stop_channel_fn=AsyncMock(),
            start_breakout_monitor_fn=AsyncMock(),
            stop_breakout_monitor_fn=AsyncMock(),
            is_pair_symbol_fn=lambda _symbol: True,
            get_ws_fn=lambda: None,
            update_15m_atr_fn=AsyncMock(),
            fetch_pair_klines_fn=AsyncMock(),
            atr1h_ma_type="EMA",
            atr1h_period=14,
            atr1h_mult=1.0,
            atr15m_ma_type="EMA",
            atr15m_period=14,
            atr15m_mult=1.0,
            clustering_min_mult=1.0,
            clustering_max_mult=2.0,
            clustering_step=0.5,
            clustering_perf_alpha=10.0,
            clustering_from_cluster="Best",
            clustering_max_iter=10,
            disable_single_trailing=False,
            disable_pair_trailing=False,
            proxy_enable=False,
            proxy_url="",
            breakout_direction_long="11",
            breakout_direction_short="00",
            min_trailing_klines=2,
        )

        coordinator.sync_breakout_monitor_from_cache(symbol)

        monitor = coordinator.breakout_monitor[symbol]
        assert hasattr(monitor, "klines_15m")
        assert monitor.kline_15m_count == 1
        assert len(monitor.klines_15m) == 3

    @pytest.mark.asyncio
    async def test_check_breakout_normalizes_legacy_state(self) -> None:
        """Legacy breakout dict should be normalized in place before evaluation."""
        from signals.breakout import check_breakout

        symbol = "AAA-BBB"
        monitor = {symbol: {"direction": "11", "trigger_price": 1.0, "trigger_time": 1.0, "klines_15m": []}}

        await check_breakout(symbol, monitor, AsyncMock(), MagicMock())

        assert hasattr(monitor[symbol], "kline_15m_count")

    def test_state_helpers_store_objects(self) -> None:
        """State helpers should keep object instances as the canonical path."""
        from signals.state import (
            BreakoutMonitorState,
            TrailingStopState,
            get_breakout_monitor_state,
            get_trailing_stop_state,
            set_breakout_monitor_state,
            set_trailing_stop_state,
        )

        trailing: dict[str, object] = {}
        breakout: dict[str, object] = {}
        ts = set_trailing_stop_state(trailing, "AAA", TrailingStopState("LONG", 1.0, 2.0, 3.0))
        bm = set_breakout_monitor_state(breakout, "AAA", BreakoutMonitorState("11", 1.0, 2.0))

        assert get_trailing_stop_state(trailing, "AAA") is ts
        assert get_breakout_monitor_state(breakout, "AAA") is bm

    @pytest.mark.asyncio
    async def test_check_trailing_stop_clears_source_key(self) -> None:
        """Trailing stop should clear the matching cooldown key only."""
        from signals.detection import check_trailing_stop

        trailing_stop = {
            "AAA": {
                "direction": "LONG",
                "entry_price": 10.0,
                "atr15m_lower": 9.0,
                "atr15m_upper": 11.0,
                "active": True,
                "source": "ATR_Ch",
            }
        }
        last_alert_time = {"ATR_Ch_AAA": 123.0, "ClusterST_AAA": 456.0}

        await check_trailing_stop(
            "AAA",
            8.5,
            trailing_stop,
            AsyncMock(),
            MagicMock(),
            last_alert_time,
        )

        assert last_alert_time["ATR_Ch_AAA"] == 0
        assert last_alert_time["ClusterST_AAA"] == 456.0
