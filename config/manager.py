"""
Configuration management module - handles loading, saving, creating and hot-reloading TOML config files.

Main functions:
- load_config / save_config: read/write TOML config files
- create_config: create default config file
- update_symbols: add/remove monitored trading pairs
- cleanup_old_logs: clean up expired webhook history logs
"""

import logging
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import toml  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

WEBHOOK_LOG_FILE = "webhook.log"
LOG_RETENTION_DAYS = 7
PID_FILE = "hypurr-monitor.pid"
LOG_FILE = "hypurr-monitor.log"
DEBUG_LOG_FILE = "debug.log"
ERROR_LOG_FILE = "error.log"

_SHORT_TZ_RE = re.compile(r"([+-]\d{2})(\d{2})$")


def resolve_path_from_config(config_path: str, raw_path: str) -> str:
    """Resolve relative runtime path against config directory."""
    path = Path(raw_path)
    if path.is_absolute():
        return str(path)
    return str(Path(config_path).resolve().parent / path)


def get_runtime_paths(config_path: str) -> dict[str, str]:
    """Resolve runtime files relative to config directory."""
    config_dir = Path(config_path).resolve().parent
    service_config: dict[str, Any] = {}
    config_file = Path(config_path)
    if config_file.exists():
        service_config = load_config(config_path).get("service", {})
    heartbeat_file = str(service_config.get("heartbeat_file", "heartbeat"))
    return {
        "pid": str(config_dir / PID_FILE),
        "log": str(config_dir / LOG_FILE),
        "debug": str(config_dir / DEBUG_LOG_FILE),
        "error": str(config_dir / ERROR_LOG_FILE),
        "webhook": str(config_dir / WEBHOOK_LOG_FILE),
        "heartbeat": resolve_path_from_config(config_path, heartbeat_file),
    }


def _parse_webhook_timestamp(log_time_str: str) -> datetime:
    """Parse webhook timestamp in both +0800 and +08:00 forms."""
    normalized = _SHORT_TZ_RE.sub(r"\1:\2", log_time_str)
    return datetime.fromisoformat(normalized).astimezone(UTC)


def cleanup_old_logs(log_file_path: str | None = None) -> None:
    """
    Clean up logs in WEBHOOK_LOG_FILE older than LOG_RETENTION_DAYS.
    Parses date from each line's [timestamp], deletes lines past cutoff time.
    """
    try:
        webhook_path = Path(log_file_path or WEBHOOK_LOG_FILE)
        if not webhook_path.exists():
            return
        cutoff_time = time.time() - (LOG_RETENTION_DAYS * 24 * 3600)
        temp_file = Path(str(webhook_path) + ".tmp")
        with webhook_path.open(encoding="utf-8") as f_in, temp_file.open("w", encoding="utf-8") as f_out:
            for line in f_in:
                try:
                    log_time_str = line.split("]")[0].strip("[")
                    log_time = _parse_webhook_timestamp(log_time_str)
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
        "network": {
            "rest": {
                "timeout_seconds": 30,
                "max_retries": 2,
                "base_delay_seconds": 1,
            },
            "ws": {
                "connect_timeout_seconds": 65,
                "receive_timeout_seconds": 65,
                "idle_timeout_seconds": 25,
                "reconnect_base_delay_seconds": 2,
                "reconnect_max_delay_seconds": 30,
            },
            "webhook": {
                "timeout_seconds": 10,
                "max_retries": 2,
                "base_delay_seconds": 1,
            },
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
