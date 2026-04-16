"""
Main entry point for hypurr-monitor.

Usage:
    uv run python main.py              # Default run (INFO)
    uv run python main.py --debug      # DEBUG mode
    uv run python main.py --daemon     # Run in background
    uv run python main.py --restart    # Restart daemon
    uv run python main.py --stop       # Stop daemon
    uv run python main.py --status     # Check daemon status
"""

import argparse
import asyncio
import os
import signal
import subprocess
import sys
import time
import traceback
from contextlib import suppress
from pathlib import Path

from config import cleanup_old_logs, create_config, load_config, resolve_path_from_config, update_symbols
from logging_config import get_logger, setup_logging
from notifications import ALERT_ERROR, build_alert_event
from service import NotificationService

logger = get_logger(__name__)

DEBUG_LOG_FILE = "debug.log"
PID_FILE = "hypurr-monitor.pid"
LOG_FILE = "hypurr-monitor.log"


def _runtime_path(config_path: str, raw_path: str) -> Path:
    """Resolve runtime file path relative to config directory."""
    return Path(resolve_path_from_config(config_path, raw_path))


def save_pid(pid: int, config_path: str = "config.toml") -> None:
    _runtime_path(config_path, PID_FILE).write_text(str(pid), encoding="utf-8")


def read_pid(config_path: str = "config.toml") -> int | None:
    pid_path = _runtime_path(config_path, PID_FILE)
    if pid_path.exists():
        try:
            return int(pid_path.read_text(encoding="utf-8").strip())
        except ValueError:
            return None
    return None


def is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _get_heartbeat_status(config_path: str) -> tuple[bool, str]:
    """Check heartbeat file freshness from config."""
    config = load_config(config_path)
    service_config = config.get("service", {})
    heartbeat_file = str(service_config.get("heartbeat_file", "heartbeat"))
    heartbeat_timeout = int(service_config.get("heartbeat_timeout", 120))
    heartbeat_path = _runtime_path(config_path, heartbeat_file)

    if not heartbeat_path.exists():
        return False, f"heartbeat file missing: {heartbeat_path}"

    try:
        heartbeat_ts = int(heartbeat_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return False, f"heartbeat file invalid: {heartbeat_path}"

    age = int(time.time()) - heartbeat_ts
    if age > heartbeat_timeout:
        return False, f"heartbeat stale: {age}s > {heartbeat_timeout}s"
    return True, f"heartbeat OK: {age}s <= {heartbeat_timeout}s"


def get_status_result(config_path: str) -> tuple[bool, str]:
    """Return daemon status message and success flag."""
    pid = read_pid(config_path)
    if not pid:
        return False, "hypurr-monitor is NOT running"

    if not is_running(pid):
        return False, f"hypurr-monitor is NOT running (stale PID file: {pid})"

    heartbeat_ok, heartbeat_message = _get_heartbeat_status(config_path)
    if heartbeat_ok:
        return True, f"hypurr-monitor is RUNNING (PID: {pid}), {heartbeat_message}"
    return False, f"hypurr-monitor is RUNNING (PID: {pid}), but {heartbeat_message}"


def cmd_daemon(config_path: str) -> None:
    pid = read_pid(config_path)
    if pid and is_running(pid):
        print(f"hypurr-monitor is already running (PID: {pid})")
        sys.exit(1)

    log_path = _runtime_path(config_path, LOG_FILE)
    log_path.write_text("", encoding="utf-8")
    with log_path.open("a", encoding="utf-8") as log:
        proc = subprocess.Popen(
            [sys.executable, __file__, "--config", config_path, "--debug"],
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
    save_pid(proc.pid, config_path)
    print(f"hypurr-monitor started (PID: {proc.pid})")
    print(f"log file: {log_path}")


def cmd_stop(config_path: str = "config.toml") -> None:
    pid = read_pid(config_path)
    pid_path = _runtime_path(config_path, PID_FILE)
    if not pid:
        print("hypurr-monitor is not running (no PID file)")
        sys.exit(1)

    if not is_running(pid):
        print(f"hypurr-monitor is not running (stale PID file: {pid})")
        pid_path.unlink(missing_ok=True)
        sys.exit(1)

    try:
        os.kill(pid, signal.SIGTERM)
        print(f"hypurr-monitor stopped (PID: {pid})")
        pid_path.unlink(missing_ok=True)
    except OSError as e:
        print(f"Failed to stop: {e}")
        sys.exit(1)


def cmd_restart(config_path: str) -> None:
    """Restart daemon with the same semantics as scripts/daemon.sh."""
    with suppress(SystemExit):
        cmd_stop(config_path)
    time.sleep(1)
    cmd_daemon(config_path)


def cmd_status(config_path: str) -> None:
    ok, message = get_status_result(config_path)
    print(message)
    if not ok:
        sys.exit(1)


def _rotate_debug_log(config_path: str) -> None:
    debug_path = _runtime_path(config_path, DEBUG_LOG_FILE)
    if not debug_path.exists():
        return
    try:
        size = debug_path.stat().st_size
        if size == 0:
            debug_path.unlink()
            return
        ts = "debug_backup"
        backup_name = f"{ts}.log"
        debug_path.rename(backup_name)
        logger.warning("[DEBUG LOG] Rotated to %s", backup_name)
    except Exception as e:
        logger.warning("[DEBUG LOG] Rotate failed: %s", e)


async def main() -> None:  # noqa: PLR0911
    parser = argparse.ArgumentParser(description="hypurr-monitor")
    parser.add_argument("--config", default="config.toml", help="Config file path")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--webhook", help="Feishu WebHook URL")
    parser.add_argument("--symbols", help="Comma-separated symbol list")
    parser.add_argument("--add-symbol", help="Add symbol(s)")
    parser.add_argument("--remove-symbol", help="Remove symbol(s)")
    parser.add_argument("--list-symbols", action="store_true", help="List symbols")
    parser.add_argument("--daemon", action="store_true", help="Run in background")
    parser.add_argument("--restart", action="store_true", help="Restart daemon")
    parser.add_argument("--stop", action="store_true", help="Stop daemon")
    parser.add_argument("--status", action="store_true", help="Check daemon status")

    args = parser.parse_args()

    if args.daemon:
        cmd_daemon(args.config)
        return
    if args.restart:
        cmd_restart(args.config)
        return
    if args.stop:
        cmd_stop(args.config)
        return
    if args.status:
        cmd_status(args.config)
        return

    setup_logging(
        debug=args.debug,
        debug_log_path=str(_runtime_path(args.config, DEBUG_LOG_FILE)),
        error_log_path=str(_runtime_path(args.config, "error.log")),
    )

    if args.webhook:
        single = args.symbols.split(",") if args.symbols else None
        create_config(args.config, args.webhook, single)
    elif args.add_symbol:
        update_symbols(args.config, "add", [s.strip().upper() for s in args.add_symbol.split(",")])
        return
    elif args.remove_symbol:
        update_symbols(
            args.config,
            "remove",
            [s.strip().upper() for s in args.remove_symbol.split(",")],
        )
        return
    elif args.list_symbols:
        config = load_config(args.config)
        sym = config.get("symbols", {})
        logger.info("single_list: %s", sym.get("single_list", []))
        logger.info("pair_list: %s", sym.get("pair_list", []))
        return

    cleanup_old_logs(str(_runtime_path(args.config, "webhook.log")))
    service = NotificationService(args.config, debug=args.debug)

    if args.debug:
        _rotate_debug_log(args.config)

    try:
        await service.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt, stopping...")
    except Exception as e:
        logger.exception("Main error")
        await service.send_event(
            build_alert_event(
                ALERT_ERROR,
                f"Main error: {type(e).__name__}: {e!s}\n{traceback.format_exc()}",
            )
        )
    finally:
        await service.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception:
        logger.exception("Fatal error")
        traceback.print_exc()
