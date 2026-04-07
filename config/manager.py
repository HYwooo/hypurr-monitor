"""
Configuration management module - handles loading, saving, creating and hot-reloading TOML config files.

Main functions:
- load_config / save_config: read/write TOML config files
- create_config: create default config file
- update_symbols: add/remove monitored trading pairs
- cleanup_old_logs: clean up expired webhook history logs
"""

import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import toml  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

WEBHOOK_LOG_FILE = "webhook.log"
LOG_RETENTION_DAYS = 7


def cleanup_old_logs() -> None:
    """
    Clean up logs in WEBHOOK_LOG_FILE older than LOG_RETENTION_DAYS.
    Parses date from each line's [timestamp], deletes lines past cutoff time.
    """
    try:
        webhook_path = Path(WEBHOOK_LOG_FILE)
        if not webhook_path.exists():
            return
        cutoff_time = time.time() - (LOG_RETENTION_DAYS * 24 * 3600)
        temp_file = Path(WEBHOOK_LOG_FILE + ".tmp")
        with webhook_path.open(encoding="utf-8") as f_in, temp_file.open("w", encoding="utf-8") as f_out:
            for line in f_in:
                try:
                    log_time_str = line.split("]")[0].strip("[")
                    log_time = datetime.strptime(log_time_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
                    if log_time.timestamp() > cutoff_time:
                        f_out.write(line)
                except Exception:
                    f_out.write(line)
        temp_file.replace(webhook_path)
    except Exception as e:
        logger.warning(f"Log cleanup failed: {e}")


def load_config(config_path: str) -> dict[str, Any]:
    """
    Load configuration dictionary from TOML file.

    Args:
        config_path: Config file path

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: Config file does not exist
    """
    if not Path(config_path).exists():
        msg = f"Config file not found: {config_path}"
        raise FileNotFoundError(msg)
    return cast(dict[str, Any], toml.load(config_path))


def save_config(config_path: str, config: dict[str, Any]) -> None:
    """
    Save configuration dictionary to TOML file.

    Args:
        config_path: Config file path
        config: Configuration dictionary
    """
    config_path_obj = Path(config_path)
    toml.dump(config, config_path_obj.open("w", encoding="utf-8"))
    logger.info(f"Config saved: {config_path}")


def update_symbols(config_path: str, action: str, symbols: list[str], target: str | None = None) -> None:
    """
    Add or remove monitored trading pairs.

    Args:
        config_path: Config file path
        action: "add" or "remove"
        symbols: List of trading pair names
        target: "single_list" or "pair_list" or None (auto-detect)
               If None, auto-detect: pairs with "/" go to pair_list, others to single_list
    """
    config = load_config(config_path)
    sym_config = config.setdefault("symbols", {})

    if target is None:
        single = [s for s in symbols if "/" not in s]
        pair = [s for s in symbols if "/" in s]
    else:
        single = symbols if target == "single_list" else []
        pair = symbols if target == "pair_list" else []

    if single:
        current_single = set(sym_config.get("single_list", []))
        if action == "add":
            current_single.update(single)
            logger.info(f"Added (single): {single}")
        elif action == "remove":
            current_single -= set(single)
            logger.info(f"Removed (single): {single}")
        sym_config["single_list"] = sorted(current_single)

    if pair:
        current_pair = set(sym_config.get("pair_list", []))
        if action == "add":
            current_pair.update(pair)
            logger.info(f"Added (pair): {pair}")
        elif action == "remove":
            current_pair -= set(pair)
            logger.info(f"Removed (pair): {pair}")
        sym_config["pair_list"] = sorted(current_pair)

    save_config(config_path, config)
    logger.info(f"Current single_list: {sym_config.get('single_list', [])}")
    logger.info(f"Current pair_list: {sym_config.get('pair_list', [])}")


def create_config(
    config_path: str,
    webhook_url: str,
    single_list: list[str] | None = None,
    pair_list: list[str] | None = None,
) -> dict[str, Any]:
    """
    Create default config file.

    Args:
        config_path: Config file path
        webhook_url: Feishu Webhook URL
        single_list: Initial ATR Channel monitoring list, default ["BTCUSDT", "ETHUSDT", "SOLUSDT", "HYPEUSDT", "XAUUSDT"]
        pair_list: Initial Clustering SuperTrend monitoring list, default []

    Returns:
        Created config dictionary
    """
    if single_list is None:
        single_list = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "HYPEUSDT", "XAUUSDT"]
    if pair_list is None:
        pair_list = []
    config = {
        "webhook": {"url": webhook_url, "format": "card"},
        "symbols": {
            "single_list": single_list,
            "pair_list": pair_list,
        },
        "atr_1h": {
            "ma_type": "DEMA",
            "period": 14,
            "mult": 1.618,
        },
        "atr_15m": {
            "ma_type": "HMA",
            "period": 14,
            "mult": 1.3,
        },
        "clustering_st": {
            "min_mult": 1.0,
            "max_mult": 5.0,
            "step": 0.5,
            "perf_alpha": 10.0,
            "from_cluster": "Best",
            "max_iter": 1000,
            "history_klines": 500,
        },
        "service": {
            "heartbeat_file": "heartbeat",
            "heartbeat_timeout": 120,
        },
        "proxy": {
            "enable": False,
            "url": "",
        },
        "report": {
            "enable": False,
            "times": ["08:00", "20:00"],
        },
        "settings": {
            "timezone": "Z",
            "max_log_lines": 1000,
            "data_source": "binance",
            "exchange": "binance",
        },
    }
    config_path_obj = Path(config_path)
    toml.dump(config, config_path_obj.open("w", encoding="utf-8"))
    logger.info(f"Config created: {config_path}")
    return config
