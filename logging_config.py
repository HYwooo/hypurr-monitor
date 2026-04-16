"""
Logging configuration module.

Provides ISO timestamp (millisecond) + level + location + message format.
Debug mode outputs to console + debug.log, normal mode outputs to console only,
and ERROR logs are always written to error.log.
"""

import logging
import re
import sys
from datetime import datetime, timedelta, timezone
from enum import Enum
from logging.handlers import RotatingFileHandler


class LogLevel(Enum):
    """Log level enum matching standard logging levels."""

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR


RESET = "\033[0m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
GRAY = "\033[90m"


class ISOFormatter(logging.Formatter):
    """
    ISO timestamp (millisecond) + full call location formatter.

    Format: 2026-04-03T12:34:56.789+0800 [LEVEL] [filename:function:line] message
    """

    def format(self, record: logging.LogRecord) -> str:
        tz = timezone(timedelta(hours=8))
        timestamp = datetime.now(tz).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "+0800"
        location = f"{record.filename}:{record.funcName}:{record.lineno}"
        return f"{timestamp} [{record.levelname}] [{location}] {record.getMessage()}"


_HTTP_METHOD_RE = re.compile(r"\b(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\b")
_HTTP_STATUS_RE = re.compile(r"\bHTTP/\d\.\d[\s]?(\d{3})\b")
_AIOHTTP_WS_RE = re.compile(r"(WebSocket|ws_connect|send_json|receive)")


def _get_timestamp() -> str:
    """Get ISO timestamp with +0800 timezone."""
    tz = timezone(timedelta(hours=8))
    return datetime.now(tz).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "+0800"


class CondensedHttpFormatter(ISOFormatter):
    """
    Condensed HTTP debug formatter for debug.log.

    - HTTP request/response lines: method URL -> status, bytes
    - WebSocket messages: action (subscribe/send/receive) channel
    - Everything else: normal format
    """

    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()

        if _AIOHTTP_WS_RE.search(msg):
            return self._condense_ws(record, msg)

        if _HTTP_METHOD_RE.search(msg) or "aiohttp" in record.name.lower():
            return self._condense_http(record, msg)

        return super().format(record)

    def _condense_http(self, record: logging.LogRecord, msg: str) -> str:
        ts = _get_timestamp()
        method_match = _HTTP_METHOD_RE.search(msg)
        status_match = _HTTP_STATUS_RE.search(msg)

        if method_match and status_match:
            method = method_match.group(1)
            status = status_match.group(1)
            return f"{ts} [DEBUG] [{record.filename}:{record.funcName}:{record.lineno}] HTTP {method} -> {status}"
        if method_match:
            method = method_match.group(1)
            url = msg.split(method)[1].split()[0] if method in msg else ""
            return f"{ts} [DEBUG] [{record.filename}:{record.funcName}:{record.lineno}] HTTP {method} {url[:80]}"
        return f"{ts} [DEBUG] [{record.filename}:{record.funcName}:{record.lineno}] {msg[:120]}"

    def _condense_ws(self, record: logging.LogRecord, msg: str) -> str:
        ts = _get_timestamp()
        return f"{ts} [DEBUG] [{record.filename}:{record.funcName}:{record.lineno}] WS {msg[:100]}"


class ColoredFormatter(logging.Formatter):
    """
    Colored ISO formatter for console output.

    Colors:
        DEBUG: gray
        INFO: cyan
        WARNING: yellow
        ERROR: red
    """

    def format(self, record: logging.LogRecord) -> str:
        tz = timezone(timedelta(hours=8))
        timestamp = datetime.now(tz).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "+0800"
        location = f"{record.filename}:{record.funcName}:{record.lineno}"

        level = record.levelname
        if record.levelno == logging.DEBUG:
            level_color = GRAY
        elif record.levelno == logging.INFO:
            level_color = CYAN
        elif record.levelno == logging.WARNING:
            level_color = YELLOW
        elif record.levelno == logging.ERROR:
            level_color = RED
        else:
            level_color = RESET

        bracket_colored = f"{level_color}[{level}]{RESET}"
        return f"{timestamp} {bracket_colored} [{location}] {record.getMessage()}"


_logger_initialized = False
ERROR_LOG_FILE = "error.log"
DEBUG_LOG_FILE = "debug.log"
LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUP_COUNT = 3


def setup_logging(
    debug: bool = False, debug_log_path: str = DEBUG_LOG_FILE, error_log_path: str = ERROR_LOG_FILE
) -> None:
    """
    Setup logging with ISO formatter.

    Args:
        debug: If True, DEBUG level to console and debug.log
               If False, INFO level to console only
    """
    global _logger_initialized  # noqa: PLW0603
    if _logger_initialized:
        return

    root = logging.getLogger()
    root.setLevel(logging.DEBUG if debug else logging.INFO)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if debug else logging.INFO)
    console.setFormatter(ColoredFormatter())
    root.addHandler(console)

    error_handler = RotatingFileHandler(
        error_log_path,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(ISOFormatter())
    root.addHandler(error_handler)

    if debug:
        file_handler = RotatingFileHandler(
            debug_log_path,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(CondensedHttpFormatter())
        root.addHandler(file_handler)

    _logger_initialized = True


def get_logger(name: str) -> logging.Logger:
    """
    Get logger instance.

    Args:
        name: Logger name, typically __name__

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def set_log_level(level: LogLevel) -> None:
    """
    Set global log level.

    Args:
        level: LogLevel enum value
    """
    logging.getLogger().setLevel(level.value)
