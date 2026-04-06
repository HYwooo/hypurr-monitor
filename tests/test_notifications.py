"""Tests for notification formatters and webhook builders."""

from notifications.formatters import format_number
from notifications.webhook import build_feishu_card


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
