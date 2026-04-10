#!/bin/bash
# ============================================================
# hypurr-monitor Daemon Control Script
# 后台进程与守护程序管理（无需 root 权限）
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

PID_FILE="$PROJECT_DIR/hypurr-monitor.pid"
LOG_FILE="$PROJECT_DIR/hypurr-monitor.log"
CONFIG_FILE="$PROJECT_DIR/config.toml"

cd "$PROJECT_DIR"

usage() {
    cat <<EOF
Usage: $0 {start|stop|restart|status|log|test}

Commands:
    start   - Start hypurr-monitor in background (DEBUG mode)
    stop    - Stop running hypurr-monitor
    restart - Restart hypurr-monitor
    status  - Check if hypurr-monitor is running
    log     - Tail recent logs
    test    - Quick connectivity test (no daemon)

Examples:
    $0 start
    $0 status
    $0 stop
    $0 log
EOF
    exit 1
}

check_config() {
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "ERROR: config.toml not found at $CONFIG_FILE"
        echo "Run: cp config.example.toml config.toml"
        exit 1
    fi
}

is_running() {
    if [ ! -f "$PID_FILE" ]; then
        return 1
    fi
    local pid=$(cat "$PID_FILE" 2>/dev/null)
    if [ -z "$pid" ]; then
        return 1
    fi
    if kill -0 "$pid" 2>/dev/null; then
        return 0
    else
        rm -f "$PID_FILE"
        return 1
    fi
}

do_start() {
    if is_running; then
        local pid=$(cat "$PID_FILE")
        echo "hypurr-monitor is already running (PID: $pid)"
        exit 0
    fi

    check_config

    echo "Starting hypurr-monitor in background..."
    echo "Log file: $LOG_FILE"

    nohup uv run python main.py --config "$CONFIG_FILE" --debug >> "$LOG_FILE" 2>&1 &
    local pid=$!
    echo "$pid" > "$PID_FILE"

    sleep 1
    if kill -0 "$pid" 2>/dev/null; then
        echo "hypurr-monitor started (PID: $pid)"
    else
        echo "ERROR: Failed to start hypurr-monitor"
        rm -f "$PID_FILE"
        exit 1
    fi
}

do_stop() {
    if ! is_running; then
        echo "hypurr-monitor is not running"
        rm -f "$PID_FILE"
        exit 0
    fi

    local pid=$(cat "$PID_FILE")
    echo "Stopping hypurr-monitor (PID: $pid)..."

    kill "$pid" 2>/dev/null || true

    local count=0
    while kill -0 "$pid" 2>/dev/null; do
        sleep 0.5
        count=$((count + 1))
        if [ $count -ge 10 ]; then
            echo "Force killing..."
            kill -9 "$pid" 2>/dev/null || true
            break
        fi
    done

    rm -f "$PID_FILE"
    echo "hypurr-monitor stopped"
}

do_status() {
    if is_running; then
        local pid=$(cat "$PID_FILE")
        echo "hypurr-monitor is RUNNING (PID: $pid)"
    else
        echo "hypurr-monitor is NOT running"
    fi
}

do_log() {
    if [ -f "$LOG_FILE" ]; then
        tail -n 50 "$LOG_FILE"
    else
        echo "No log file found: $LOG_FILE"
    fi
}

do_test() {
    echo "Running connectivity test..."
    uv run python main.py --config "$CONFIG_FILE" --list-symbols
}

case "${1:-}" in
    start)
        do_start
        ;;
    stop)
        do_stop
        ;;
    restart)
        do_stop
        sleep 1
        do_start
        ;;
    status)
        do_status
        ;;
    log)
        do_log
        ;;
    test)
        check_config
        do_test
        ;;
    *)
        usage
        ;;
esac
