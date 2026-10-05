# SPDX-License-Identifier: Apache-2.0
"""Runs the server so that it cannot outlive the supervisor that started it.

    python -m voxt_omni_backend.lifeline -- <server argv...>

The supervisor starts this module as the leader of a new process group. It runs
the server as a child in that group and waits for it. A kqueue watch on the
supervisor fires when the supervisor exits for any reason, including SIGKILL,
and the whole group is then killed: the server, its stage processes and any
helper they started, even ones already reparented away from the server.
"""

from __future__ import annotations

import os
import select
import signal
import subprocess
import sys
import threading
from types import FrameType


def kill_group_when_parent_exits(parent_pid: int) -> None:
    queue = select.kqueue()
    watch = select.kevent(
        parent_pid,
        filter=select.KQ_FILTER_PROC,
        flags=select.KQ_EV_ADD,
        fflags=select.KQ_NOTE_EXIT,
    )
    try:
        queue.control([watch], 0, None)
    except OSError:
        os.killpg(os.getpgrp(), signal.SIGKILL)
        return
    if os.getppid() != parent_pid:
        os.killpg(os.getpgrp(), signal.SIGKILL)
    else:
        queue.control(None, 1, None)
        os.killpg(os.getpgrp(), signal.SIGKILL)


def main() -> int:
    if len(sys.argv) < 3 or sys.argv[1] != "--":
        sys.stderr.write(__doc__ or "")
        return 2
    else:
        pass
    threading.Thread(
        target=kill_group_when_parent_exits, args=(os.getppid(),), daemon=True
    ).start()
    server = subprocess.Popen(sys.argv[2:], stdin=subprocess.DEVNULL)

    def forward(signal_number: int, frame: FrameType | None) -> None:
        del frame
        server.send_signal(signal_number)

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    return server.wait()


if __name__ == "__main__":
    sys.exit(main())
