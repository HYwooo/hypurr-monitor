"""Tests for extracted service-layer components."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from models import Kline
from service.alert_dispatcher import AlertDispatcher
from service.market_data_processor import MarketDataProcessor
from service.signal_coordinator import SignalCoordinator
from service.ws_runtime_supervisor import WSRuntimeSupervisor


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
        assert monitor["kline_15m_count"] == 1
        assert len(monitor["klines_15m"]) == 3
