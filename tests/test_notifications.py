"""Tests for notification formatters and webhook builders."""

from types import TracebackType
from typing import Any
from unittest.mock import AsyncMock

import pytest

from config.network import RetryConfig, WebhookNetworkConfig
from notifications.alert_constants import (
    BREAKOUT_CONFIRMED,
    DIRECTION_LONG,
    format_breakout_message,
    format_directional_signal_message,
    format_trailing_stop_message,
)
from notifications.alert_event import build_alert_event, emit_alert
from notifications.formatters import format_number
from notifications.webhook import _rotate_webhook_log_if_needed, build_feishu_card, send_alert_event, send_webhook
from notifications.webhook_sender import WebhookSender


class TestFormatNumber:
    """Test number formatting."""

    def test_zero(self) -> None:
        """Zero returns 0.00000000."""
        assert format_number(0.0) == "0.00000000"

    def test_nan(self) -> None:
        """NaN returns string representation."""
        result = format_number(float("nan"))
        assert result == "nan"

    def test_inf(self) -> None:
        """Inf returns string representation."""
        assert format_number(float("inf")) == "inf"

    def test_negative_inf(self) -> None:
        """-Inf returns string representation."""
        assert format_number(float("-inf")) == "-inf"

    def test_small_less_than_one(self) -> None:
        """Values < 1 keep 8 decimal places."""
        result = format_number(0.00001234)
        assert result == "0.00001234"

    def test_one(self) -> None:
        """Value of 1 returns 1."""
        assert format_number(1.0) == "1"

    def test_integer(self) -> None:
        """Large integers keep some decimal places."""
        result = format_number(123456789.0)
        assert result == "123456789"

    def test_truncates_trailing_zeros(self) -> None:
        """Trailing zeros after decimal are stripped."""
        result = format_number(1.50000000)
        assert result == "1.5"

    def test_negative(self) -> None:
        """Negative numbers work correctly."""
        result = format_number(-0.00001234)
        assert result == "-0.00001234"


class TestBuildFeishuCard:
    """Test Feishu card message building."""

    def test_atr_ch_long(self) -> None:
        """ATR LONG signal card."""
        card = build_feishu_card(
            "ATR_Ch",
            "[BTC] LONG",
            {
                "symbol": "BTC",
                "direction": "long",
                "price": "50000.0",
                "atr_upper": "51000.0",
                "atr_lower": "49000.0",
            },
            "2026-04-03T12:00:00+0800",
        )
        assert card["header"]["template"] == "green"
        assert "BTC" in card["header"]["title"]["content"]

    def test_atr_ch_short(self) -> None:
        """ATR SHORT signal card."""
        card = build_feishu_card(
            "ATR_Ch",
            "[BTC] SHORT",
            {
                "symbol": "BTC",
                "direction": "short",
                "price": "49000.0",
                "atr_upper": "51000.0",
                "atr_lower": "49000.0",
            },
            "2026-04-03T12:00:00+0800",
        )
        assert card["header"]["template"] == "red"

    def test_trailing_stop(self) -> None:
        """Trailing stop card has orange color."""
        card = build_feishu_card(
            "ATR_Ch",
            "[BTC] TRAILING STOP",
            {
                "symbol": "BTC",
                "direction": "long",
                "reason": "trailing_stop",
                "price": "48500.0",
                "stop_line": "49000.0",
                "entry_price": "50000.0",
            },
            "2026-04-03T12:00:00+0800",
        )
        assert card["header"]["template"] == "orange"
        elements_text = " ".join(str(e) for e in card["elements"])
        assert "TRAILING STOP" in elements_text

    def test_system_message(self) -> None:
        """SYSTEM message card."""
        card = build_feishu_card(
            "SYSTEM",
            "hypurr-monitor connected to hyperliquid",
            {},
            "2026-04-03T12:00:00+0800",
        )
        assert card["header"]["template"] == "blue"
        assert "System" in card["header"]["title"]["content"]

    def test_error_message(self) -> None:
        """ERROR message card."""
        card = build_feishu_card(
            "ERROR",
            "Connection failed",
            {},
            "2026-04-03T12:00:00+0800",
        )
        assert card["header"]["template"] == "red"
        elements_text = " ".join(str(e) for e in card["elements"])
        assert "Connection failed" in elements_text

    def test_config_message(self) -> None:
        """CONFIG message card."""
        card = build_feishu_card(
            "CONFIG",
            "Config reloaded",
            {},
            "2026-04-03T12:00:00+0800",
        )
        assert card["header"]["template"] == "purple"

    def test_report_message(self) -> None:
        """REPORT message card."""
        card = build_feishu_card(
            "REPORT",
            "Daily report content",
            {},
            "2026-04-03T12:00:00+0800",
        )
        assert card["header"]["template"] == "purple"

    def test_breakout_confirmed(self) -> None:
        """BREAKOUT confirmed card."""
        card = build_feishu_card(
            "BREAKOUT",
            "[BTC] BREAKOUT",
            {
                "symbol": "BTC",
                "direction": "long",
                "confirmed": True,
                "price": "52000.0",
                "trigger": "52000.0",
            },
            "2026-04-03T12:00:00+0800",
        )
        assert card["header"]["template"] == "orange"

    def test_clustering_signal(self) -> None:
        """ClusterST signal includes TS and perf_ama."""
        card = build_feishu_card(
            "ClusterST",
            "[BTCETH] LONG",
            {
                "symbol": "BTCETH",
                "direction": "long",
                "price": "16.5",
                "ts": "16.4",
                "perf_ama": "16.35",
                "target_factor": "2.5",
            },
            "2026-04-03T12:00:00+0800",
        )
        assert card["header"]["template"] == "green"
        elements_text = " ".join(str(e) for e in card["elements"])
        assert "TS" in elements_text
        assert "perf_ama" in elements_text

    def test_unknown_alert_type(self) -> None:
        """Unknown alert type defaults to blue."""
        card = build_feishu_card(
            "UNKNOWN",
            "Some message",
            {},
            "2026-04-03T12:00:00+0800",
        )
        assert card["header"]["template"] == "blue"

    def test_extra_with_none(self) -> None:
        """extra=None should be handled gracefully."""
        card = build_feishu_card("SYSTEM", "msg", None, "2026-04-03T12:00:00+0800")
        assert card["header"]["template"] == "blue"


class TestSendWebhook:
    """Test webhook sending and local log behavior."""

    @pytest.mark.asyncio
    async def test_send_webhook_writes_to_custom_log_path(self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        """Webhook log should respect the provided runtime path."""
        log_path = tmp_path / "runtime-webhook.log"

        class FakeResponse:
            status = 200

            async def __aenter__(self) -> "FakeResponse":
                return self

            async def __aexit__(
                self,
                exc_type: type[BaseException] | None,
                exc: BaseException | None,
                tb: TracebackType | None,
            ) -> None:
                _ = (exc_type, exc, tb)

        class FakeSession:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                _ = (args, kwargs)

            async def __aenter__(self) -> "FakeSession":
                return self

            async def __aexit__(
                self,
                exc_type: type[BaseException] | None,
                exc: BaseException | None,
                tb: TracebackType | None,
            ) -> None:
                _ = (exc_type, exc, tb)

            def post(self, *args: Any, **kwargs: Any) -> FakeResponse:
                _ = (args, kwargs)
                return FakeResponse()

        monkeypatch.setattr("aiohttp.ClientSession", FakeSession)

        await send_webhook(
            "https://example.com/hook",
            "text",
            "SYSTEM",
            "runtime ok",
            log_file_path=str(log_path),
            get_timestamp_fn=lambda: "2026-04-14T12:00:00+0800",
        )

        content = log_path.read_text(encoding="utf-8")
        assert "[2026-04-14T12:00:00+0800] [SYSTEM] runtime ok" in content

    @pytest.mark.asyncio
    async def test_send_webhook_logs_error_on_http_failure(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-200 responses should be logged as errors."""
        errors: list[str] = []

        class FakeResponse:
            status = 500

            async def __aenter__(self) -> "FakeResponse":
                return self

            async def __aexit__(
                self,
                exc_type: type[BaseException] | None,
                exc: BaseException | None,
                tb: TracebackType | None,
            ) -> None:
                _ = (exc_type, exc, tb)

        class FakeSession:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                _ = (args, kwargs)

            async def __aenter__(self) -> "FakeSession":
                return self

            async def __aexit__(
                self,
                exc_type: type[BaseException] | None,
                exc: BaseException | None,
                tb: TracebackType | None,
            ) -> None:
                _ = (exc_type, exc, tb)

            def post(self, *args: Any, **kwargs: Any) -> FakeResponse:
                _ = (args, kwargs)
                return FakeResponse()

        monkeypatch.setattr("aiohttp.ClientSession", FakeSession)
        monkeypatch.setattr("notifications.webhook.log_error", errors.append)

        await send_webhook(
            "https://example.com/hook",
            "text",
            "ERROR",
            "request failed",
            log_file_path=str(tmp_path / "hook.log"),
            get_timestamp_fn=lambda: "2026-04-14T12:00:00+0800",
        )

        assert errors == ["Webhook failed: 500"]

    def test_rotate_webhook_log_if_needed_uses_given_path(self, tmp_path: Any) -> None:
        """Rotation should trim the specified log file instead of a global path."""
        log_path = tmp_path / "custom.log"
        log_path.write_text("a\nb\nc\n", encoding="utf-8")

        _rotate_webhook_log_if_needed(str(log_path), max_log_lines=2)

        assert log_path.read_text(encoding="utf-8") == "b\nc\n"

    @pytest.mark.asyncio
    async def test_webhook_sender_reuses_single_session(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """WebhookSender should reuse one aiohttp session across sends."""
        created_sessions: list[Any] = []

        class FakeResponse:
            status = 200

            async def __aenter__(self) -> "FakeResponse":
                return self

            async def __aexit__(
                self,
                exc_type: type[BaseException] | None,
                exc: BaseException | None,
                tb: TracebackType | None,
            ) -> None:
                _ = (exc_type, exc, tb)

        class FakeSession:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                _ = (args, kwargs)
                self.closed = False
                self.post_calls = 0
                created_sessions.append(self)

            def post(self, *args: Any, **kwargs: Any) -> FakeResponse:
                _ = (args, kwargs)
                self.post_calls += 1
                return FakeResponse()

            async def close(self) -> None:
                self.closed = True

        monkeypatch.setattr("notifications.webhook_sender.aiohttp.ClientSession", FakeSession)
        sender = WebhookSender(
            WebhookNetworkConfig(
                proxy_url="http://127.0.0.1:7890",
                timeout_seconds=5.0,
                retry=RetryConfig(max_retries=1, base_delay_seconds=0.1),
            )
        )

        await sender.send_json("https://example.com/hook", {"msg_type": "text"})
        await sender.send_json("https://example.com/hook", {"msg_type": "text"})
        await sender.close()

        assert len(created_sessions) == 1
        assert created_sessions[0].post_calls == 2
        assert created_sessions[0].closed is True


class TestAlertEvent:
    """Test structured AlertEvent compatibility layer."""

    def test_build_alert_event_infers_structured_fields(self) -> None:
        """Legacy ATR args should map into a stable structured event."""
        event = build_alert_event(
            "ATR_Ch",
            "[BTC] TRAILING STOP",
            {
                "symbol": "BTC",
                "direction": "long",
                "reason": "trailing_stop",
                "timeframe": "4H",
            },
        )

        assert event.category == "signal"
        assert event.severity == "info"
        assert event.event == "trailing_stop"
        assert event.symbol == "BTC"
        assert event.direction == "long"
        assert event.timeframe == "4H"
        assert event.dedupe_key == "ATR_Ch:BTC:4H:trailing_stop"

    @pytest.mark.asyncio
    async def test_send_alert_event_preserves_existing_output(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Structured send path should keep the same webhook log format."""
        log_path = tmp_path / "structured-webhook.log"

        class FakeResponse:
            status = 200

            async def __aenter__(self) -> "FakeResponse":
                return self

            async def __aexit__(
                self,
                exc_type: type[BaseException] | None,
                exc: BaseException | None,
                tb: TracebackType | None,
            ) -> None:
                _ = (exc_type, exc, tb)

        class FakeSession:
            async def __aenter__(self) -> "FakeSession":
                return self

            async def __aexit__(
                self,
                exc_type: type[BaseException] | None,
                exc: BaseException | None,
                tb: TracebackType | None,
            ) -> None:
                _ = (exc_type, exc, tb)

            def post(self, *args: Any, **kwargs: Any) -> FakeResponse:
                _ = (args, kwargs)
                return FakeResponse()

        monkeypatch.setattr("aiohttp.ClientSession", FakeSession)
        event = build_alert_event("SYSTEM", "structured ok")

        await send_alert_event(
            "https://example.com/hook",
            "text",
            event,
            log_file_path=str(log_path),
            get_timestamp_fn=lambda: "2026-04-14T12:00:00+0800",
        )

        content = log_path.read_text(encoding="utf-8")
        assert "[2026-04-14T12:00:00+0800] [SYSTEM] structured ok" in content

    @pytest.mark.asyncio
    async def test_emit_alert_prefers_structured_callback(self) -> None:
        """emit_alert should build AlertEvent when a structured callback is available."""
        legacy = AsyncMock()
        structured = AsyncMock()

        await emit_alert(legacy, "SYSTEM", "hello", {"symbol": "BTC"}, structured)

        legacy.assert_not_called()
        structured.assert_awaited_once()
        await_args = structured.await_args
        assert await_args is not None
        event = await_args.args[0]
        assert event.alert_type == "SYSTEM"
        assert event.symbol == "BTC"


class TestAlertMessageHelpers:
    """Test reusable legacy-compatible alert message helpers."""

    def test_format_directional_signal_message_with_timeframe(self) -> None:
        """Directional helper should preserve legacy bracket style and timeframe prefix."""
        assert format_directional_signal_message("BTC", DIRECTION_LONG, "4H") == "[BTC] 4H LONG"

    def test_format_trailing_stop_message(self) -> None:
        """Trailing-stop helper should preserve legacy message format."""
        assert format_trailing_stop_message("BTC") == "[BTC] TRAILING STOP"

    def test_format_breakout_message(self) -> None:
        """Breakout helper should preserve legacy breakout message format."""
        assert format_breakout_message("BTC", DIRECTION_LONG, BREAKOUT_CONFIRMED) == "BTC LONG CONFIRMED"
