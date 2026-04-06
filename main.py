"""
Main entry point for hypurr-monitor.

Usage:
    uv run python main.py              # Default run (INFO)
    uv run python main.py --debug      # DEBUG mode
    uv run python main.py --daemon     # Run in background
    uv run python main.py --stop       # Stop daemon
    uv run python main.py --status     # Check daemon status
"""

import argparse
import asyncio
import os
import signal
import subprocess
import sys
import traceback
from pathlib import Path

from config import cleanup_old_logs, create_config, load_config, update_symbols
from logging_config import get_logger, setup_logging
from service import NotificationService

logger = get_logger(__name__)

DEBUG_LOG_FILE = "debug.log"
PID_FILE = "hypurr-monitor.pid"
LOG_FILE = "hypurr-monitor.log"


def save_pid(pid: int) -> None:
    Path(PID_FILE).write_text(str(pid), encoding="utf-8")


def read_pid() -> int | None:
    if Path(PID_FILE).exists():
        try:
            return int(Path(PID_FILE).read_text(encoding="utf-8").strip())
        except ValueError:
            return None
    return None


def is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def cmd_daemon() -> None:
    pid = read_pid()
    if pid and is_running(pid):
        print(f"hypurr-monitor is already running (PID: {pid})")
        sys.exit(1)

    print("Starting hypurr-monitor in background...")
    with Path(LOG_FILE).open("a", encoding="utf-8") as log:
        proc = subprocess.Popen(
            [sys.executable, __file__, "--config", "config.toml", "--debug"],
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
    save_pid(proc.pid)
    print(f"hypurr-monitor started (PID: {proc.pid})")
    print(f"Log file: {LOG_FILE}")


def cmd_stop() -> None:
    pid = read_pid()
    if not pid:
        print("hypurr-monitor is not running (no PID file)")
        sys.exit(1)

    if not is_running(pid):
        print("hypurr-monitor is not running (stale PID file)")
        Path(PID_FILE).unlink(missing_ok=True)
        sys.exit(1)

    try:
        os.kill(pid, signal.SIGTERM)
        print(f"hypurr-monitor stopped (PID: {pid})")
        Path(PID_FILE).unlink(missing_ok=True)
    except OSError as e:
        print(f"Failed to stop: {e}")
        sys.exit(1)


def cmd_status() -> None:
    pid = read_pid()
    if not pid:
        print("hypurr-monitor is NOT running")
        sys.exit(1)

    if is_running(pid):
        print(f"hypurr-monitor is RUNNING (PID: {pid})")
    else:
        print("hypurr-monitor is NOT running (stale PID file)")
        sys.exit(1)


def _rotate_debug_log() -> None:
    debug_path = Path(DEBUG_LOG_FILE)
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


async def main() -> None:
    parser = argparse.ArgumentParser(description="hypurr-monitor")
    parser.add_argument("--config", default="config.toml", help="Config file path")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--webhook", help="Feishu WebHook URL")
    parser.add_argument("--symbols", help="Comma-separated symbol list")
    parser.add_argument("--add-symbol", help="Add symbol(s)")
    parser.add_argument("--remove-symbol", help="Remove symbol(s)")
    parser.add_argument("--list-symbols", action="store_true", help="List symbols")
    parser.add_argument("--daemon", action="store_true", help="Run in background")
    parser.add_argument("--stop", action="store_true", help="Stop daemon")
    parser.add_argument("--status", action="store_true", help="Check daemon status")

    args = parser.parse_args()

    if args.daemon:
        cmd_daemon()
        return
    if args.stop:
        cmd_stop()
        return
    if args.status:
        cmd_status()
        return

    setup_logging(debug=args.debug)

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

    cleanup_old_logs()
    service = NotificationService(args.config, debug=args.debug)

    if args.debug:
        _rotate_debug_log()

    try:
        await service.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt, stopping...")
    except Exception as e:
        logger.exception("Main error")
        await service._send_webhook(  # noqa: SLF001
            "ERROR",
            f"Main error: {type(e).__name__}: {e!s}\n{traceback.format_exc()}",
        )
    finally:
        await service.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception:
        logger.exception("Fatal error")
        traceback.print_exc()
