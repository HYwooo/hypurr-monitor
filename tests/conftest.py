"""Pytest configuration and fixtures for hypurr-monitor tests."""

import tempfile
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def temp_config_file() -> Path:
    """Create a temporary config file for testing."""
    config_content = """
[webhook]
url = "https://test.example.com/webhook"
format = "card"

[symbols]
single_list = ["BTC", "ETH"]
pair_list = ["BTC-ETH"]
single_strategy = "atr_channel"
pair_strategy = "clustering_st"

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
heartbeat_timeout = 120

[proxy]
enable = false
url = ""

[report]
enable = false
times = ["08:00", "20:00"]

[settings]
timezone = "+08:00"
max_log_lines = 1000
disable_single_trailing = false
disable_pair_trailing = false
exchange = "hyperliquid"
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(config_content)
        return Path(f.name)


@pytest.fixture
def temp_config_with_strategy_swap() -> Path:
    """Create a config file with swapped strategies for testing."""
    config_content = """
[webhook]
url = "https://test.example.com/webhook"
format = "card"

[symbols]
single_list = ["BTC", "ETH"]
pair_list = ["BTC-ETH"]
single_strategy = "clustering_st"
pair_strategy = "atr_channel"

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
heartbeat_timeout = 120

[proxy]
enable = false
url = ""

[report]
enable = false
times = ["08:00", "20:00"]

[settings]
timezone = "+08:00"
max_log_lines = 1000
disable_single_trailing = false
disable_pair_trailing = false
exchange = "hyperliquid"
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(config_content)
        return Path(f.name)


@pytest.fixture
def mock_klines() -> list[Any]:
    """Create mock klines for testing."""
    from models import Kline

    klines = []
    base_time = 1700000000000
    for i in range(100):
        klines.append(
            Kline(
                symbol="BTC",
                interval="1h",
                open_time=base_time + i * 3600000,
                open=50000.0 + i * 100,
                high=50500.0 + i * 100,
                low=49500.0 + i * 100,
                close=50200.0 + i * 100,
                volume=1000.0 + i * 10,
                close_time=base_time + i * 3600000 + 3600000 - 1,
                is_closed=True,
            )
        )
    return klines
