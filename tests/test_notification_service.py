"""Tests for NotificationService - strategy configuration and symbol management."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from service import NotificationService


class TestNotificationServiceInit:
    """Test NotificationService initialization."""

    def test_init_loads_single_and_pair_lists(self, temp_config_file: Path) -> None:
        """Service loads single_list and pair_list from config."""
        with patch("service.notification_service.HyperliquidREST"):
            service = NotificationService(config_path=str(temp_config_file))
            assert service.single_list == ["BTC", "ETH"]
            assert service.pair_list == ["BTC-ETH"]

    def test_init_loads_strategy_config(self, temp_config_file: Path) -> None:
        """Service loads single_strategy and pair_strategy from config."""
        with patch("service.notification_service.HyperliquidREST"):
            service = NotificationService(config_path=str(temp_config_file))
            assert service.single_strategy == "atr_channel"
            assert service.pair_strategy == "clustering_st"

    def test_init_parses_pair_components(self, temp_config_file: Path) -> None:
        """Service correctly parses pair_list into components."""
        with patch("service.notification_service.HyperliquidREST"):
            service = NotificationService(config_path=str(temp_config_file))
            assert service._pair_components == {"BTC-ETH": ("BTC", "ETH")}
            assert service.symbols == ["BTC", "ETH", "BTC-ETH"]

    def test_default_strategies_when_not_in_config(self) -> None:
        """If strategy not in config, use defaults."""
        config_content = """
[webhook]
url = "https://test.example.com/webhook"

[symbols]
single_list = ["BTC"]
pair_list = []

[supertrend]
period1 = 9
multiplier1 = 2.5
period2 = 14
multiplier2 = 1.7

[vegas]
ema_signal = 9
ema_upper = 144
ema_lower = 169

[atr_1h]
ma_type = "DEMA"
period = 14
mult = 1.618

[atr_15m]
ma_type = "HMA"
period = 14
mult = 1.3

[clustering_st]
enabled = true
min_mult = 1.0
max_mult = 5.0
step = 0.5
perf_alpha = 10.0
from_cluster = "Best"
max_iter = 1000
history_klines = 500

[service]
heartbeat_file = "heartbeat"

[proxy]
enable = false
url = ""

[report]
enable = false
times = []

[settings]
timezone = "Z"
exchange = "hyperliquid"
"""
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_content)
            temp_path = Path(f.name)
        try:
            with patch("service.notification_service.HyperliquidREST"):
                service = NotificationService(config_path=str(temp_path))
                assert service.single_strategy == "atr_channel"
                assert service.pair_strategy == "clustering_st"
        finally:
            temp_path.unlink()


class TestSymbolClassification:
    """Test symbol classification methods."""

    @pytest.fixture
    def service(self, temp_config_file: Path) -> NotificationService:
        """Create service instance for testing."""
        with patch("service.notification_service.HyperliquidREST"):
            return NotificationService(config_path=str(temp_config_file))

    def test_is_pair_symbol_true(self, service: NotificationService) -> None:
        """BTC-ETH is correctly identified as a pair symbol."""
        assert service._is_pair_symbol("BTC-ETH") is True

    def test_is_pair_symbol_false(self, service: NotificationService) -> None:
        """BTC is not a pair symbol."""
        assert service._is_pair_symbol("BTC") is False

    def test_is_pair_trading_component(self, service: NotificationService) -> None:
        """BTC is identified as part of a pair (pair trading)."""
        assert service._is_pair_trading("BTC") is True
        assert service._is_pair_trading("ETH") is True

    def test_is_pair_trading_non_component(self, service: NotificationService) -> None:
        """SOL is not part of any pair."""
        assert service._is_pair_trading("SOL") is False

    def test_get_pair_for_symbol(self, service: NotificationService) -> None:
        """Get pair components for a pair symbol."""
        assert service._get_pair_for_symbol("BTC-ETH") == ("BTC", "ETH")

    def test_get_pair_for_symbol_not_pair(self, service: NotificationService) -> None:
        """Non-pair symbols return None."""
        assert service._get_pair_for_symbol("BTC") is None


class TestStrategyDispatch:
    """Test strategy dispatch based on configuration."""

    def test_single_symbol_uses_single_strategy(self, temp_config_file: Path) -> None:
        """Single symbol uses single_strategy from config."""
        with patch("service.notification_service.HyperliquidREST"):
            service = NotificationService(config_path=str(temp_config_file))
            strategy = service._get_strategy_for_symbol("BTC")
            assert strategy == "atr_channel"

    def test_pair_symbol_uses_pair_strategy(self, temp_config_file: Path) -> None:
        """Pair symbol uses pair_strategy from config."""
        with patch("service.notification_service.HyperliquidREST"):
            service = NotificationService(config_path=str(temp_config_file))
            strategy = service._get_strategy_for_symbol("BTC-ETH")
            assert strategy == "clustering_st"

    def test_swapped_strategies(self, temp_config_with_strategy_swap: Path) -> None:
        """When swapped, single uses clustering_st and pair uses atr_channel."""
        with patch("service.notification_service.HyperliquidREST"):
            service = NotificationService(config_path=str(temp_config_with_strategy_swap))
            assert service._get_strategy_for_symbol("BTC") == "clustering_st"
            assert service._get_strategy_for_symbol("BTC-ETH") == "atr_channel"


class TestGetTimestamp:
    """Test timestamp formatting based on timezone."""

    @pytest.fixture
    def service(self, temp_config_file: Path) -> NotificationService:
        """Create service with Z timezone."""
        with patch("service.notification_service.HyperliquidREST"):
            return NotificationService(config_path=str(temp_config_file))

    def test_get_timestamp_returns_iso_format(self, service: NotificationService) -> None:
        """Timestamp is in ISO format."""
        ts = service._get_timestamp()
        assert "T" in ts
        assert "+" in ts or "Z" in ts or "-" in ts

    def test_get_timestamp_different_timezones(self) -> None:
        """Different timezone configs produce different offsets."""
        config_content = """
[webhook]
url = "https://test.example.com/webhook"

[symbols]
single_list = []
pair_list = []

[supertrend]
period1 = 9
multiplier1 = 2.5
period2 = 14
multiplier2 = 1.7

[vegas]
ema_signal = 9
ema_upper = 144
ema_lower = 169

[atr_1h]
ma_type = "DEMA"
period = 14
mult = 1.618

[atr_15m]
ma_type = "HMA"
period = 14
mult = 1.3

[clustering_st]
enabled = true
min_mult = 1.0
max_mult = 5.0
step = 0.5
perf_alpha = 10.0
from_cluster = "Best"
max_iter = 1000
history_klines = 500

[service]
heartbeat_file = "heartbeat"

[proxy]
enable = false
url = ""

[report]
enable = false
times = []

[settings]
timezone = "-05:00"
exchange = "hyperliquid"
"""
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_content)
            temp_path = Path(f.name)
        try:
            with patch("service.notification_service.HyperliquidREST"):
                service = NotificationService(config_path=str(temp_path))
                ts = service._get_timestamp()
                assert "-05:00" in ts or "-0500" in ts
        finally:
            temp_path.unlink()


class TestIndicatorParameters:
    """Test indicator parameter loading from config."""

    def test_loads_supertrend_params(self, temp_config_file: Path) -> None:
        """Supertrend parameters are loaded correctly."""
        with patch("service.notification_service.HyperliquidREST"):
            service = NotificationService(config_path=str(temp_config_file))
            assert service.st_period1 == 9
            assert service.st_multiplier1 == 2.5
            assert service.st_period2 == 14
            assert service.st_multiplier2 == 1.7

    def test_loads_vegas_params(self, temp_config_file: Path) -> None:
        """Vegas tunnel parameters are loaded correctly."""
        with patch("service.notification_service.HyperliquidREST"):
            service = NotificationService(config_path=str(temp_config_file))
            assert service.vt_ema_signal == 9
            assert service.vt_ema_upper == 144
            assert service.vt_ema_lower == 169

    def test_loads_atr_params(self, temp_config_file: Path) -> None:
        """ATR parameters are loaded correctly."""
        with patch("service.notification_service.HyperliquidREST"):
            service = NotificationService(config_path=str(temp_config_file))
            assert service.atr1h_ma_type == "DEMA"
            assert service.atr1h_period == 14
            assert service.atr1h_mult == 1.618
            assert service.atr15m_ma_type == "HMA"
            assert service.atr15m_period == 14
            assert service.atr15m_mult == 1.3

    def test_loads_clustering_params(self, temp_config_file: Path) -> None:
        """Clustering SuperTrend parameters are loaded correctly."""
        with patch("service.notification_service.HyperliquidREST"):
            service = NotificationService(config_path=str(temp_config_file))
            assert service.clustering_min_mult == 1.0
            assert service.clustering_max_mult == 5.0
            assert service.clustering_step == 0.5
            assert service.clustering_perf_alpha == 10.0
            assert service.clustering_from_cluster == "Best"
            assert service.clustering_max_iter == 1000
            assert service.clustering_history_klines == 500


class TestPairKlinesMerging:
    """Test pair K-line merging logic."""

    @pytest.fixture
    def service(self, temp_config_file: Path) -> NotificationService:
        """Create service with pair BTC-ETH."""
        with patch("service.notification_service.HyperliquidREST"):
            return NotificationService(config_path=str(temp_config_file))

    @pytest.mark.asyncio
    async def test_hl_fetch_pair_klines_empty_for_unknown_pair(self, service: NotificationService) -> None:
        """Unknown pair returns empty list."""
        result = await service._hl_fetch_pair_klines("UNKNOWN-PAIR")
        assert result == []


class TestAlertCount:
    """Test alert count thread safety."""

    @pytest.fixture
    def service(self, temp_config_file: Path) -> NotificationService:
        """Create service instance."""
        with patch("service.notification_service.HyperliquidREST"):
            return NotificationService(config_path=str(temp_config_file))

    def test_increment_alert_count(self, service: NotificationService) -> None:
        """Alert count increments correctly."""
        initial = service._alert_count
        service._increment_alert_count()
        assert service._alert_count == initial + 1

    def test_increment_alert_count_thread_safe(self, service: NotificationService) -> None:
        """Alert count increments correctly under concurrent access."""
        import threading

        initial = service._alert_count
        threads = [threading.Thread(target=service._increment_alert_count) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert service._alert_count == initial + 10


class TestConfigLoading:
    """Test config loading edge cases."""

    def test_load_config_missing_file_raises(self) -> None:
        """Loading nonexistent config raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            NotificationService(config_path="nonexistent.toml")


class TestTrailingStop:
    """Test trailing stop functionality."""

    @pytest.fixture
    def service(self, temp_config_file: Path) -> NotificationService:
        """Create service instance."""
        with patch("service.notification_service.HyperliquidREST"):
            svc = NotificationService(config_path=str(temp_config_file))
            svc.trailing_stop = {}
            return svc

    @pytest.mark.asyncio
    async def test_update_15m_atr_no_trailing_stop(self, service: NotificationService) -> None:
        """update_15m_atr does nothing if symbol not in trailing_stop."""
        service.trailing_stop = {}
        await service.update_15m_atr("BTC", {"h": "50000", "l": "49000", "c": "49500"})
        assert "BTC" not in service.trailing_stop

    @pytest.mark.asyncio
    async def test_update_15m_atr_inactive_trailing_stop(self, service: NotificationService) -> None:
        """update_15m_atr does nothing if trailing stop inactive."""
        service.trailing_stop = {"BTC": {"active": False}}
        await service.update_15m_atr("BTC", {"h": "50000", "l": "49000", "c": "49500"})


class TestPairPriceUpdate:
    """Test pair price update logic."""

    @pytest.fixture
    def service(self, temp_config_file: Path) -> NotificationService:
        """Create service with pair BTC-ETH."""
        with patch("service.notification_service.HyperliquidREST"):
            return NotificationService(config_path=str(temp_config_file))

    @pytest.mark.asyncio
    async def test_update_pair_price_calculates_ratio(self, service: NotificationService) -> None:
        """Pair price is calculated as ratio of component prices."""
        service.mark_prices = {"BTC": 50000.0, "ETH": 2500.0}

        with (
            patch.object(service, "_ct_check_trailing_stop", new_callable=AsyncMock),
            patch.object(service, "_ct_check_signals_clustering", new_callable=AsyncMock),
        ):
            await service._update_pair_price("BTC-ETH", "BTC", "ETH")

        expected_ratio = 50000.0 / 2500.0
        assert abs(service.mark_prices.get("BTC-ETH", 0) - expected_ratio) < 0.01

    @pytest.mark.asyncio
    async def test_update_pair_price_skips_zero_price(self, service: NotificationService) -> None:
        """Pair price update skips when component price is zero."""
        service.mark_prices = {"BTC": 50000.0, "ETH": 0.0}

        with (
            patch.object(service, "_ct_check_trailing_stop", new_callable=AsyncMock),
            patch.object(service, "_ct_check_signals_clustering", new_callable=AsyncMock),
        ):
            await service._update_pair_price("BTC-ETH", "BTC", "ETH")

        assert "BTC-ETH" not in service.mark_prices
