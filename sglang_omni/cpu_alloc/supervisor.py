# SPDX-License-Identifier: Apache-2.0
"""One-way CPU lease supervisor for ``--cpu-allocator dynamic``.

Exclusive holders are never re-pinned: only shared-pool processes have their
affinity widened when an exclusive process goes idle, and narrowed back when
it turns busy again. Lending is same-NUMA only. Hysteresis (idle/busy
watermarks plus an idle hold time) keeps affinity from flapping.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Callable

from sglang_omni.cpu_alloc.allocator import CpuAllocationPlan
from sglang_omni.cpu_alloc.topology import format_cpulist

logger = logging.getLogger(__name__)


def _read_proc_cpu_seconds(pid: int) -> float:
    with open(f"/proc/{pid}/stat", "rb") as f:
        data = f.read().decode("ascii", errors="replace")
    # Fields after the comm field, which may itself contain spaces/parens.
    fields = data.rsplit(")", 1)[1].split()
    utime_ticks = int(fields[11])
    stime_ticks = int(fields[12])
    return (utime_ticks + stime_ticks) / os.sysconf("SC_CLK_TCK")


def _set_process_affinity(pid: int, cpu_ids: set[int]) -> None:
    """Apply the mask to every task of *pid*, not just its main thread.

    sched_setaffinity is per-thread. Iterate ``/proc/<pid>/task`` until the
    task list is stable so threads spawned mid-replay are covered; a thread
    created in the final race window inherits its creator's already-updated
    mask, so the residual exposure is bounded to that window.
    """
    seen: set[int] = set()
    for _ in range(3):
        try:
            tids = [int(t) for t in os.listdir(f"/proc/{pid}/task")]
        except (OSError, ValueError):
            return
        fresh = [tid for tid in tids if tid not in seen]
        if not fresh:
            return
        for tid in fresh:
            seen.add(tid)
            try:
                os.sched_setaffinity(tid, cpu_ids)
            except OSError:
                continue


@dataclass
class _ExclusiveState:
    idle_since: float | None = None
    lent: bool = False
    last_cpu_seconds: float | None = None
    last_sample_time: float | None = None


class CpuLeaseSupervisor:
    """Lend idle exclusive cores to same-node shared processes."""

    def __init__(
        self,
        plan: CpuAllocationPlan,
        pids: dict[str, int],
        *,
        interval_s: float = 5.0,
        idle_below: float = 0.10,
        busy_above: float = 0.30,
        idle_hold_s: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        cpu_seconds: Callable[[int], float] = _read_proc_cpu_seconds,
        set_affinity: Callable[[int, set[int]], None] = _set_process_affinity,
    ):
        if not 0.0 <= idle_below < busy_above <= 1.0:
            raise ValueError(
                f"Watermarks must satisfy 0 <= idle_below < busy_above <= 1, "
                f"got idle_below={idle_below}, busy_above={busy_above}"
            )
        self._plan = plan
        self._pids = dict(pids)
        self._interval_s = interval_s
        self._idle_below = idle_below
        self._busy_above = busy_above
        self._idle_hold_s = idle_hold_s
        self._clock = clock
        self._cpu_seconds = cpu_seconds
        self._set_affinity = set_affinity

        self._exclusive = {
            name: _ExclusiveState()
            for name, a in plan.assignments.items()
            if a.exclusive and name in self._pids
        }
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Supervisor already started")
        self._thread = threading.Thread(
            target=self._run, name="cpu-lease-supervisor", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval_s + 5.0)
            self._thread = None
        # Fail closed: leave no widened shared mask behind.
        if any(state.lent for state in self._exclusive.values()):
            for state in self._exclusive.values():
                state.lent = False
            self._replay_shared_affinity()

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval_s):
            try:
                self.tick()
            except Exception:
                logger.exception("cpu_alloc supervisor tick failed")

    def tick(self) -> None:
        """One sampling round; public so tests can drive it synchronously."""
        now = self._clock()
        changed = False
        for name, state in self._exclusive.items():
            fraction = self._sample_busy_fraction(name, state, now)
            if fraction is None:
                continue
            if fraction >= self._busy_above:
                state.idle_since = None
                if state.lent:
                    state.lent = False
                    changed = True
                    logger.info(
                        "cpu_alloc: %s busy again (%.0f%%); reclaiming its cores",
                        name,
                        fraction * 100,
                    )
            elif fraction < self._idle_below and not state.lent:
                if state.idle_since is None:
                    state.idle_since = now
                elif now - state.idle_since >= self._idle_hold_s:
                    state.lent = True
                    changed = True
                    logger.info(
                        "cpu_alloc: %s idle for %.0fs; lending its cores to the "
                        "shared pool",
                        name,
                        now - state.idle_since,
                    )
        if changed:
            self._replay_shared_affinity()

    def _sample_busy_fraction(
        self, name: str, state: _ExclusiveState, now: float
    ) -> float | None:
        pid = self._pids[name]
        try:
            cpu_seconds = self._cpu_seconds(pid)
        except (OSError, ValueError, IndexError):
            return None
        last_seconds = state.last_cpu_seconds
        last_time = state.last_sample_time
        state.last_cpu_seconds = cpu_seconds
        state.last_sample_time = now
        if last_seconds is None or last_time is None or now <= last_time:
            return None
        n_cpus = len(self._plan.assignments[name].cpu_ids)
        return (cpu_seconds - last_seconds) / ((now - last_time) * n_cpus)

    def _replay_shared_affinity(self) -> None:
        lent_by_node: dict[int | None, set[int]] = {}
        for name, state in self._exclusive.items():
            if not state.lent:
                continue
            assignment = self._plan.assignments[name]
            lent_by_node.setdefault(assignment.numa_node, set()).update(
                assignment.cpu_ids
            )
        all_lent = set().union(*lent_by_node.values()) if lent_by_node else set()

        for name, assignment in self._plan.assignments.items():
            if assignment.exclusive or name not in self._pids:
                continue
            extra = (
                all_lent
                if assignment.numa_node is None
                else lent_by_node.get(assignment.numa_node, set())
            )
            target = set(assignment.cpu_ids) | extra
            try:
                self._set_affinity(self._pids[name], target)
            except (OSError, ValueError):
                continue
            logger.info("cpu_alloc: %s affinity -> %s", name, format_cpulist(target))
