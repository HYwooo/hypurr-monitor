"""Tests for main process helpers."""

import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import main
from main import _get_heartbeat_status


def write_config(tmp_path: Path, heartbeat_file: str = "heartbeat", heartbeat_timeout: int = 120) -> Path:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                "[webhook]",
                'url = "https://example.com/hook"',
                'format = "card"',
                "",
                "[symbols]",
                "single_list = []",
                "pair_list = []",
                "",
                "[service]",
                f'heartbeat_file = "{heartbeat_file}"',
                f"heartbeat_timeout = {heartbeat_timeout}",
                "",
                "[settings]",
                'timezone = "+08:00"',
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def test_get_heartbeat_status_ok(tmp_path: Path) -> None:
    config_path = write_config(tmp_path, heartbeat_timeout=120)
    heartbeat_path = tmp_path / "heartbeat"
    heartbeat_path.write_text(str(int(time.time()) - 10), encoding="utf-8")

    ok, message = _get_heartbeat_status(str(config_path))

    assert ok is True
    assert "heartbeat OK" in message


def test_get_heartbeat_status_stale(tmp_path: Path) -> None:
    config_path = write_config(tmp_path, heartbeat_timeout=30)
    heartbeat_path = tmp_path / "heartbeat"
    heartbeat_path.write_text(str(int(time.time()) - 31), encoding="utf-8")

    ok, message = _get_heartbeat_status(str(config_path))

    assert ok is False
    assert "heartbeat stale" in message


def test_get_heartbeat_status_missing_file(tmp_path: Path) -> None:
    config_path = write_config(tmp_path)

    ok, message = _get_heartbeat_status(str(config_path))

    assert ok is False
    assert "heartbeat file missing" in message


def test_cmd_daemon_truncates_log_and_uses_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    log_path = tmp_path / "hypurr-monitor.log"
    log_path.write_text("old log", encoding="utf-8")
    save_pid_mock = MagicMock()

    monkeypatch.setattr(main, "LOG_FILE", str(log_path))
    monkeypatch.setattr(main, "read_pid", lambda: None)
    monkeypatch.setattr(main, "save_pid", save_pid_mock)

    popen_mock = MagicMock(return_value=SimpleNamespace(pid=4321))
    monkeypatch.setattr("main.subprocess.Popen", popen_mock)

    main.cmd_daemon("custom.toml")

    assert log_path.read_text(encoding="utf-8") == ""
    popen_args = popen_mock.call_args.args[0]
    assert "custom.toml" in popen_args
    save_pid_mock.assert_called_once_with(4321)


def test_cmd_stop_stale_pid_matches_daemon_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    stale_pid = 1234
    pid_file = tmp_path / "test-stale.pid"

    monkeypatch.setattr(main, "read_pid", lambda: stale_pid)
    monkeypatch.setattr(main, "is_running", lambda _pid: False)
    monkeypatch.setattr(main, "PID_FILE", str(pid_file))
    pid_file.write_text(str(stale_pid), encoding="utf-8")

    with pytest.raises(SystemExit):
        main.cmd_stop()

    captured = capsys.readouterr()
    assert f"hypurr-monitor is not running (stale PID file: {stale_pid})" in captured.out
    assert pid_file.exists() is False
