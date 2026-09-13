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


@pytest.mark.parametrize("leader_exits_first", [False, True])
def test_shutdown_stops_owned_worker_when_leader_exits(leader_exits_first):
    child = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print('ready',flush=True); time.sleep(60)"
    parent = (
        "import subprocess,sys,time; "
        f"p=subprocess.Popen([sys.executable,'-c',{child!r}],stdout=subprocess.PIPE,text=True); "
        "p.stdout.readline(); print(p.pid,flush=True); "
        + ("sys.exit(0)" if leader_exits_first else "time.sleep(60)")
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
        if leader_exits_first:
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
