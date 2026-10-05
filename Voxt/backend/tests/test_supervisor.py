# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil

BACKEND_ROOT = Path(__file__).resolve().parents[1]
FAKE_SERVER = Path(__file__).parent / "fake_omni_server.py"


def start_supervisor(
    tmp_path: Path, *fake_server_options: str, startup_timeout_s: float = 20.0
) -> tuple[subprocess.Popen[str], Path]:
    pid_file = tmp_path / "server-pids.json"
    command = [
        sys.executable,
        "-m",
        "voxt_omni_backend.supervisor",
        "--model-kind",
        "qwen3_asr",
        "--model-directory",
        str(tmp_path),
        "--derived-root",
        str(tmp_path / "derived"),
        "--startup-timeout-s",
        str(startup_timeout_s),
        "--server-command",
        json.dumps(
            [
                sys.executable,
                str(FAKE_SERVER),
                "--pid-file",
                str(pid_file),
                *fake_server_options,
            ]
        ),
    ]
    supervisor = subprocess.Popen(
        command,
        cwd=BACKEND_ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return supervisor, pid_file


def next_event(supervisor: subprocess.Popen[str]) -> dict[str, object]:
    line = supervisor.stdout.readline()
    assert line, supervisor.stderr.read()
    return json.loads(line)


def server_pids(pid_file: Path) -> list[int]:
    """The fake server, its stage child and that child's own children."""
    deadline = time.monotonic() + 10
    while not pid_file.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    server_pid, stage_pid = json.loads(pid_file.read_text())
    try:
        grandchildren = [child.pid for child in psutil.Process(stage_pid).children()]
        lifeline = [psutil.Process(server_pid).ppid()]
    except psutil.NoSuchProcess:
        grandchildren, lifeline = [], []
    return [*lifeline, server_pid, stage_pid, *grandchildren]


def wait_until_gone(pids: list[int], timeout_s: float = 10.0) -> list[int]:
    deadline = time.monotonic() + timeout_s
    alive = pids
    while alive and time.monotonic() < deadline:
        alive = [
            pid
            for pid in alive
            if psutil.pid_exists(pid)
            and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
        ]
        time.sleep(0.05)
    return alive


def test_ready_event_carries_a_verified_loopback_endpoint(tmp_path: Path) -> None:
    supervisor, pid_file = start_supervisor(tmp_path)
    ready = next_event(supervisor)
    owned = server_pids(pid_file)

    assert ready["event"] == "ready"
    assert ready["host"] == "127.0.0.1"
    assert isinstance(ready["port"], int)
    assert str(ready["model_name"]).startswith("voxt-qwen3_asr-")
    assert ready["server_pid"] == owned[0]
    assert json.loads(Path(f"{pid_file}.env").read_text()) == {
        "SGLANG_OMNI_STRICT_PORT": "1"
    }
    supervisor.stdin.close()
    supervisor.wait(timeout=15)
    assert wait_until_gone(owned) == []


def test_a_killed_supervisor_takes_the_server_tree_with_it(tmp_path: Path) -> None:
    supervisor, pid_file = start_supervisor(tmp_path)
    assert next_event(supervisor)["event"] == "ready"
    owned = server_pids(pid_file)

    supervisor.kill()

    supervisor.wait(timeout=10)
    assert wait_until_gone(owned) == []


def test_sighup_reaps_the_server_tree(tmp_path: Path) -> None:
    supervisor, pid_file = start_supervisor(tmp_path)
    assert next_event(supervisor)["event"] == "ready"
    owned = server_pids(pid_file)

    supervisor.send_signal(signal.SIGHUP)

    supervisor.wait(timeout=15)
    assert wait_until_gone(owned) == []


def test_control_pipe_eof_during_startup_stops_without_waiting_for_ready(
    tmp_path: Path,
) -> None:
    supervisor, pid_file = start_supervisor(tmp_path, "--startup-delay-s", "60")
    owned = server_pids(pid_file)
    closed_at = time.monotonic()

    supervisor.stdin.close()

    assert supervisor.wait(timeout=15) == 0
    assert time.monotonic() - closed_at < 10
    assert "ready" not in supervisor.stdout.read()
    assert wait_until_gone(owned) == []


def test_shutdown_during_startup_reports_stopped(tmp_path: Path) -> None:
    supervisor, pid_file = start_supervisor(tmp_path, "--startup-delay-s", "60")
    owned = server_pids(pid_file)

    supervisor.stdin.write(json.dumps({"command": "shutdown"}) + "\n")
    supervisor.stdin.flush()

    assert next_event(supervisor)["event"] == "stopped"
    assert supervisor.wait(timeout=15) == 0
    assert wait_until_gone(owned) == []


def test_malformed_control_lines_are_ignored(tmp_path: Path) -> None:
    supervisor, pid_file = start_supervisor(tmp_path)
    assert next_event(supervisor)["event"] == "ready"
    owned = server_pids(pid_file)

    supervisor.stdin.write("not json\n")
    supervisor.stdin.write(json.dumps({"command": "shutdown"}) + "\n")
    supervisor.stdin.flush()

    assert next_event(supervisor)["event"] == "stopped"
    assert wait_until_gone(owned) == []


def test_control_pipe_eof_reaps_only_the_owned_process_tree(tmp_path: Path) -> None:
    bystander = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    supervisor, pid_file = start_supervisor(tmp_path)
    try:
        assert next_event(supervisor)["event"] == "ready"
        owned = server_pids(pid_file)

        supervisor.stdin.close()

        assert supervisor.wait(timeout=15) == 0
        assert wait_until_gone(owned) == []
        assert bystander.poll() is None
    finally:
        bystander.kill()


def test_shutdown_command_reports_stopped_after_cleanup(tmp_path: Path) -> None:
    supervisor, pid_file = start_supervisor(tmp_path)
    assert next_event(supervisor)["event"] == "ready"
    owned = server_pids(pid_file)

    supervisor.stdin.write(json.dumps({"command": "shutdown"}) + "\n")
    supervisor.stdin.flush()

    assert next_event(supervisor)["event"] == "stopped"
    assert wait_until_gone(owned, timeout_s=0.5) == []
    assert supervisor.wait(timeout=10) == 0


def test_supervisor_termination_signal_also_reaps_the_tree(tmp_path: Path) -> None:
    supervisor, pid_file = start_supervisor(tmp_path)
    assert next_event(supervisor)["event"] == "ready"
    owned = server_pids(pid_file)

    supervisor.send_signal(signal.SIGTERM)

    supervisor.wait(timeout=15)
    assert wait_until_gone(owned) == []


def test_unexpected_server_exit_is_reported_and_reaped(tmp_path: Path) -> None:
    supervisor, pid_file = start_supervisor(tmp_path, "--exit-after-s", "0.5")
    events = [next_event(supervisor), next_event(supervisor)]

    assert [event["event"] for event in events] == ["ready", "exited"]
    assert events[1]["exit_code"] == 3
    assert supervisor.wait(timeout=10) != 0
    assert wait_until_gone(server_pids(pid_file)) == []


def test_a_server_answering_for_another_instance_is_rejected(tmp_path: Path) -> None:
    supervisor, pid_file = start_supervisor(
        tmp_path, "--report-model-name", "someone-else"
    )

    event = next_event(supervisor)

    assert event["event"] == "failed"
    assert "instance" in str(event["reason"])
    assert supervisor.wait(timeout=15) != 0
    assert wait_until_gone(server_pids(pid_file)) == []


def test_startup_timeout_reaps_a_server_that_never_becomes_healthy(
    tmp_path: Path,
) -> None:
    supervisor, pid_file = start_supervisor(
        tmp_path, "--startup-delay-s", "60", startup_timeout_s=2.0
    )
    event = next_event(supervisor)

    assert event["event"] == "failed"
    assert "timeout" in str(event["reason"])
    assert wait_until_gone(server_pids(pid_file)) == []
