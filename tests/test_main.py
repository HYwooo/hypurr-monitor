"""Tests for main process helpers."""

import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import main
from config import cleanup_old_logs, get_runtime_paths, resolve_path_from_config
from main import _get_heartbeat_status, get_status_result


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


def test_get_status_result_returns_running_message(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = write_config(tmp_path)
    heartbeat_path = tmp_path / "heartbeat"
    heartbeat_path.write_text(str(int(time.time())), encoding="utf-8")

    monkeypatch.setattr(main, "read_pid", lambda _config_path="config.toml": 4321)
    monkeypatch.setattr(main, "is_running", lambda pid: pid == 4321)

    ok, message = get_status_result(str(config_path))

    assert ok is True
    assert "hypurr-monitor is RUNNING (PID: 4321), heartbeat OK:" in message


def test_cmd_status_prints_shared_status_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = write_config(tmp_path)
    monkeypatch.setattr(main, "get_status_result", lambda _config: (False, "shared status output"))

    with pytest.raises(SystemExit):
        main.cmd_status(str(config_path))

    captured = capsys.readouterr()
    assert captured.out.strip() == "shared status output"


def test_resolve_path_from_config_uses_config_directory(tmp_path: Path) -> None:
    config_path = tmp_path / "nested" / "config.toml"
    config_path.parent.mkdir(parents=True)

    resolved = resolve_path_from_config(str(config_path), "runtime/heartbeat")

    assert Path(resolved) == config_path.parent / "runtime" / "heartbeat"


def test_get_runtime_paths_reads_heartbeat_from_config(tmp_path: Path) -> None:
    config_path = write_config(tmp_path, heartbeat_file="runtime/hb.txt")

    paths = get_runtime_paths(str(config_path))

    assert Path(paths["pid"]) == tmp_path / "hypurr-monitor.pid"
    assert Path(paths["log"]) == tmp_path / "hypurr-monitor.log"
    assert Path(paths["heartbeat"]) == tmp_path / "runtime" / "hb.txt"


def test_cleanup_old_logs_parses_iso_timestamp(tmp_path: Path) -> None:
    webhook_log = tmp_path / "webhook.log"
    webhook_log.write_text(
        "[2026-04-14T12:00:00+0800] [SYSTEM] keep\n[2000-01-01T00:00:00+0800] [SYSTEM] drop\n",
        encoding="utf-8",
    )

    cleanup_old_logs(str(webhook_log))

    content = webhook_log.read_text(encoding="utf-8")
    assert "keep" in content
    assert "drop" not in content


def test_cmd_daemon_truncates_log_and_uses_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    log_path = tmp_path / "hypurr-monitor.log"
    config_path = tmp_path / "custom.toml"
    config_path.write_text('[service]\nheartbeat_file = "heartbeat"\n', encoding="utf-8")
    log_path.write_text("old log", encoding="utf-8")
    save_pid_mock = MagicMock()

    monkeypatch.setattr(main, "LOG_FILE", str(log_path))
    monkeypatch.setattr(main, "read_pid", lambda _config_path="config.toml": None)
    monkeypatch.setattr(main, "save_pid", save_pid_mock)

    popen_mock = MagicMock(return_value=SimpleNamespace(pid=4321))
    monkeypatch.setattr("main.subprocess.Popen", popen_mock)

    main.cmd_daemon(str(config_path))

    assert log_path.read_text(encoding="utf-8") == ""
    popen_args = popen_mock.call_args.args[0]
    assert str(config_path) in popen_args
    save_pid_mock.assert_called_once_with(4321, str(config_path))


def test_cmd_daemon_output_matches_daemon_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    log_path = tmp_path / "hypurr-monitor.log"
    config_path = tmp_path / "config.toml"
    config_path.write_text('[service]\nheartbeat_file = "heartbeat"\n', encoding="utf-8")

    monkeypatch.setattr(main, "LOG_FILE", str(log_path))
    monkeypatch.setattr(main, "read_pid", lambda _config_path="config.toml": None)
    monkeypatch.setattr(main, "save_pid", MagicMock())
    monkeypatch.setattr("main.subprocess.Popen", MagicMock(return_value=SimpleNamespace(pid=4321)))

    main.cmd_daemon(str(config_path))

    captured = capsys.readouterr()
    assert captured.out.splitlines() == [
        "hypurr-monitor started (PID: 4321)",
        f"log file: {log_path}",
    ]


def test_cmd_stop_stale_pid_matches_daemon_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    stale_pid = 1234
    pid_file = tmp_path / "test-stale.pid"
    config_path = tmp_path / "config.toml"
    config_path.write_text('[service]\nheartbeat_file = "heartbeat"\n', encoding="utf-8")

    monkeypatch.setattr(main, "read_pid", lambda _config_path="config.toml": stale_pid)
    monkeypatch.setattr(main, "is_running", lambda _pid: False)
    monkeypatch.setattr(main, "PID_FILE", str(pid_file))
    pid_file.write_text(str(stale_pid), encoding="utf-8")

    with pytest.raises(SystemExit):
        main.cmd_stop(str(config_path))

    captured = capsys.readouterr()
    assert f"hypurr-monitor is not running (stale PID file: {stale_pid})" in captured.out
    assert pid_file.exists() is False


def test_cmd_restart_matches_daemon_script_semantics(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str | None]] = []

    def fake_stop(config_path: str = "config.toml") -> None:
        calls.append(("stop", config_path))
        raise SystemExit(1)

    def fake_daemon(config_path: str) -> None:
        calls.append(("daemon", config_path))

    sleep_mock = MagicMock()
    monkeypatch.setattr(main, "cmd_stop", fake_stop)
    monkeypatch.setattr(main, "cmd_daemon", fake_daemon)
    monkeypatch.setattr("main.time.sleep", sleep_mock)

    main.cmd_restart("config.toml")

    assert calls == [("stop", "config.toml"), ("daemon", "config.toml")]
    sleep_mock.assert_called_once_with(1)
