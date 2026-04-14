#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-config.toml}"

if [[ $# -ge 1 && "$1" == *.toml ]]; then
  CONFIG_FILE="$1"
  shift
fi

CONFIG_PATH="$ROOT_DIR/$CONFIG_FILE"
PID_FILE="$ROOT_DIR/hypurr-monitor.pid"
LOG_FILE="$ROOT_DIR/hypurr-monitor.log"

read_config_value() {
  local key="$1"
  local default_value="$2"
  uv run python - "$CONFIG_PATH" "$key" "$default_value" <<'PY'
import sys
from pathlib import Path
import tomllib

config_path = Path(sys.argv[1])
key = sys.argv[2]
default = sys.argv[3]

if not config_path.exists():
    print(default)
    raise SystemExit(0)

with config_path.open("rb") as f:
    data = tomllib.load(f)

current = data
for part in key.split("."):
    if not isinstance(current, dict) or part not in current:
        print(default)
        raise SystemExit(0)
    current = current[part]

print(current)
PY
}

HEARTBEAT_FILE_RAW="$(read_config_value "service.heartbeat_file" "heartbeat")"
HEARTBEAT_TIMEOUT="$(read_config_value "service.heartbeat_timeout" "120")"

if [[ "$HEARTBEAT_FILE_RAW" = /* ]]; then
  HEARTBEAT_FILE="$HEARTBEAT_FILE_RAW"
else
  HEARTBEAT_FILE="$ROOT_DIR/$HEARTBEAT_FILE_RAW"
fi

is_running() {
  local pid="$1"
  kill -0 "$pid" >/dev/null 2>&1
}

read_pid() {
  if [[ -f "$PID_FILE" ]]; then
    cat "$PID_FILE"
  fi
}

heartbeat_age() {
  if [[ ! -f "$HEARTBEAT_FILE" ]]; then
    echo "-1"
    return 0
  fi

  local ts now
  ts="$(tr -d '[:space:]' < "$HEARTBEAT_FILE")"
  if [[ ! "$ts" =~ ^[0-9]+$ ]]; then
    echo "-1"
    return 0
  fi

  now="$(date +%s)"
  echo $((now - ts))
}

print_status() {
  local pid age
  pid="$(read_pid)"
  if [[ -z "$pid" ]]; then
    echo "hypurr-monitor is NOT running (no PID file)"
    return 1
  fi

  if ! is_running "$pid"; then
    echo "hypurr-monitor is NOT running (stale PID file: $pid)"
    return 1
  fi

  age="$(heartbeat_age)"
  if [[ "$age" -lt 0 ]]; then
    echo "hypurr-monitor is RUNNING (PID: $pid), but heartbeat file is missing or invalid: $HEARTBEAT_FILE"
    return 1
  fi

  if [[ "$age" -gt "$HEARTBEAT_TIMEOUT" ]]; then
    echo "hypurr-monitor is RUNNING (PID: $pid), but heartbeat is STALE: ${age}s > ${HEARTBEAT_TIMEOUT}s"
    return 1
  fi

  echo "hypurr-monitor is RUNNING (PID: $pid), heartbeat OK: ${age}s <= ${HEARTBEAT_TIMEOUT}s"
  return 0
}

start() {
  local pid
  pid="$(read_pid)"
  if [[ -n "$pid" ]] && is_running "$pid"; then
    echo "hypurr-monitor is already running (PID: $pid)"
    return 1
  fi

  mkdir -p "$ROOT_DIR/scripts" >/dev/null 2>&1 || true
  : > "$LOG_FILE"
  nohup uv run python "$ROOT_DIR/main.py" --config "$CONFIG_PATH" --debug >> "$LOG_FILE" 2>&1 &
  local new_pid=$!
  echo "$new_pid" > "$PID_FILE"
  echo "hypurr-monitor started (PID: $new_pid)"
  echo "log file: $LOG_FILE"
}

stop() {
  local pid
  pid="$(read_pid)"
  if [[ -z "$pid" ]]; then
    echo "hypurr-monitor is not running (no PID file)"
    return 1
  fi

  if ! is_running "$pid"; then
    echo "hypurr-monitor is not running (stale PID file: $pid)"
    rm -f "$PID_FILE"
    return 1
  fi

  kill "$pid"
  rm -f "$PID_FILE"
  echo "hypurr-monitor stopped (PID: $pid)"
}

restart() {
  stop || true
  sleep 1
  start
}

log_output() {
  if [[ -f "$LOG_FILE" ]]; then
    tail -n 100 "$LOG_FILE"
  else
    echo "log file not found: $LOG_FILE"
    return 1
  fi
}

test_heartbeat() {
  print_status
}

COMMAND="${1:-status}"

case "$COMMAND" in
  start)
    start
    ;;
  stop)
    stop
    ;;
  restart)
    restart
    ;;
  status)
    print_status
    ;;
  log)
    log_output
    ;;
  test)
    test_heartbeat
    ;;
  *)
    echo "Usage: bash scripts/daemon.sh [config.toml] {start|stop|restart|status|log|test}"
    exit 1
    ;;
esac
