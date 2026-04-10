"""Tests for config manager module."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from config.manager import (
    cleanup_old_logs,
    create_config,
    load_config,
    save_config,
    update_symbols,
)


class TestLoadConfig:
    """Test config loading."""

    def test_load_existing_config(self, temp_config_file: Path) -> None:
        """Load existing config file."""
        config = load_config(str(temp_config_file))
        assert config["webhook"]["url"] == "https://test.example.com/webhook"
        assert config["symbols"]["single_list"] == ["BTC", "ETH"]

    def test_load_nonexistent_config_raises(self) -> None:
        """Loading nonexistent config raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.toml")


class TestSaveConfig:
    """Test config saving."""

    def test_save_and_reload_config(self, temp_config_file: Path) -> None:
        """Save config and reload it."""
        original = load_config(str(temp_config_file))
        original["test_key"] = "test_value"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            save_config(f.name, original)

        reloaded = load_config(f.name)
        assert reloaded["test_key"] == "test_value"
        Path(f.name).unlink()


class TestUpdateSymbols:
    """Test symbol update operations."""

    def test_add_single_symbol(self, temp_config_file: Path) -> None:
        """Add single symbol to config."""
        update_symbols(str(temp_config_file), "add", ["SOL"], target="single_list")
        config = load_config(str(temp_config_file))
        assert "SOL" in config["symbols"]["single_list"]

    def test_remove_single_symbol(self, temp_config_file: Path) -> None:
        """Remove single symbol from config."""
        update_symbols(str(temp_config_file), "remove", ["ETH"], target="single_list")
        config = load_config(str(temp_config_file))
        assert "ETH" not in config["symbols"]["single_list"]

    def test_add_pair_symbol(self, temp_config_file: Path) -> None:
        """Add pair symbol to config."""
        update_symbols(str(temp_config_file), "add", ["SOL-MATIC"], target="pair_list")
        config = load_config(str(temp_config_file))
        assert "SOL-MATIC" in config["symbols"]["pair_list"]

    def test_auto_detect_single_vs_pair(self, temp_config_file: Path) -> None:
        """Auto-detect single vs pair based on '/' character."""
        update_symbols(str(temp_config_file), "add", ["AVAX", "BTC/USDC"])
        config = load_config(str(temp_config_file))
        assert "AVAX" in config["symbols"]["single_list"]
        assert "BTC/USDC" in config["symbols"]["pair_list"]

    def test_remove_from_both_lists(self, temp_config_file: Path) -> None:
        """Remove symbols from appropriate lists."""
        update_symbols(str(temp_config_file), "remove", ["BTC", "ETH"], target="single_list")
        config = load_config(str(temp_config_file))
        assert "BTC" not in config["symbols"]["single_list"]
        assert "ETH" not in config["symbols"]["single_list"]


class TestCreateConfig:
    """Test config creation."""

    def test_create_default_config(self) -> None:
        """Create config with defaults."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            config = create_config(f.name, "https://test.com/webhook")

        assert config["webhook"]["url"] == "https://test.com/webhook"
        assert config["symbols"]["single_list"] == ["BTCUSDT", "ETHUSDT", "SOLUSDT", "HYPEUSDT", "XAUUSDT"]
        assert config["symbols"]["pair_list"] == []
        assert config["atr_1h"]["ma_type"] == "DEMA"
        assert config["clustering_st"]["from_cluster"] == "Best"
        Path(f.name).unlink()

    def test_create_config_with_custom_symbols(self) -> None:
        """Create config with custom symbols."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            config = create_config(
                f.name,
                "https://test.com/webhook",
                single_list=["CUSTOM"],
                pair_list=["BTC-ETH"],
            )

        assert config["symbols"]["single_list"] == ["CUSTOM"]
        assert config["symbols"]["pair_list"] == ["BTC-ETH"]
        Path(f.name).unlink()


class TestCleanupOldLogs:
    """Test log cleanup functionality."""

    def test_cleanup_nonexistent_file(self) -> None:
        """Cleanup of nonexistent file does not raise."""
        with patch("config.manager.Path.exists", return_value=False):
            cleanup_old_logs()
