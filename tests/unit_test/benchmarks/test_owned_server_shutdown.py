import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.benchmarker.utils import stop_server

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux process groups")


def live(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


@pytest.mark.parametrize("leader_mode", ["running", "exited", "ignores_term"])
def test_shutdown_stops_owned_worker_when_leader_exits(leader_mode):
    child = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print('ready',flush=True); time.sleep(60)"
    parent = "import subprocess,sys,time,signal; " f"p=subprocess.Popen([sys.executable,'-c',{child!r}],stdout=subprocess.PIPE,text=True); " + (
        "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        if leader_mode == "ignores_term"
        else ""
    ) + "p.stdout.readline(); print(p.pid,flush=True); " + (
        "sys.exit(0)" if leader_mode == "exited" else "time.sleep(60)"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", parent],
        start_new_session=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    unrelated = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    try:
        worker_pid = int(proc.stdout.readline())
        assert live(worker_pid)
        if leader_mode == "exited":
            proc.wait(timeout=5)
        stop_server(proc, terminate_timeout_s=0.1, kill_timeout_s=2)
        assert not live(worker_pid)
        assert unrelated.poll() is None
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=5)
        proc.stdout.close()
        unrelated.terminate()
        unrelated.wait(timeout=5)


def test_shutdown_rejects_a_process_in_the_callers_group():
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        with pytest.raises(ValueError, match="own process group"):
            stop_server(proc, terminate_timeout_s=0.1, kill_timeout_s=2)
        assert proc.poll() is None
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_shutdown_allows_leader_to_close_worker_before_group_signal(tmp_path):
    marker = tmp_path / "worker-closed"
    child = """
import pathlib,signal,sys
signal.signal(signal.SIGTERM, lambda *_: sys.exit(2))
print('ready', flush=True)
sys.stdin.readline()
pathlib.Path(sys.argv[1]).write_text('closed by leader')
"""
    parent = f"""
import signal,subprocess,sys,time
p = subprocess.Popen([sys.executable, '-c', {child!r}, {str(marker)!r}], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
p.stdout.readline()
def stop(*_):
    p.stdin.write('close\\n')
    p.stdin.flush()
    p.wait(timeout=3)
    sys.exit(0)
signal.signal(signal.SIGTERM, stop)
print('ready', flush=True)
time.sleep(60)
"""
    proc = subprocess.Popen(
        [sys.executable, "-c", parent],
        start_new_session=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert proc.stdout.readline().strip() == "ready"
        stop_server(proc, terminate_timeout_s=5, kill_timeout_s=2)
        assert marker.read_text() == "closed by leader"
        assert proc.returncode == 0
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=5)
        proc.stdout.close()
