# SPDX-License-Identifier: Apache-2.0
import os

import pytest

from sglang_omni.cpu_alloc.allocator import CpuAllocationPlan, ProcessCpuAssignment
from sglang_omni.cpu_alloc.supervisor import CpuLeaseSupervisor, _set_process_affinity


def make_plan():
    return CpuAllocationPlan(
        assignments={
            "ar": ProcessCpuAssignment("ar", (0, 8), True, 0),
            "shared0": ProcessCpuAssignment("shared0", (1, 2, 9, 10), False, 0),
            "shared1": ProcessCpuAssignment("shared1", (4, 12), False, 1),
        },
        shared_pools={0: (1, 2, 9, 10), 1: (4, 12), None: (1, 2, 4, 9, 10, 12)},
        events=(),
    )


class FakeHost:
    def __init__(self):
        self.now = 0.0
        self.cpu_seconds = {100: 0.0}
        self.affinity_calls: list[tuple[int, set[int]]] = []

    def clock(self):
        return self.now

    def read_cpu_seconds(self, pid):
        return self.cpu_seconds[pid]

    def set_affinity(self, pid, cpus):
        self.affinity_calls.append((pid, set(cpus)))


def make_supervisor(host, **kwargs):
    return CpuLeaseSupervisor(
        make_plan(),
        {"ar": 100, "shared0": 200, "shared1": 300},
        idle_hold_s=60.0,
        clock=host.clock,
        cpu_seconds=host.read_cpu_seconds,
        set_affinity=host.set_affinity,
        **kwargs,
    )


def advance(host, supervisor, seconds, busy_fraction):
    # busy_fraction is per allocated CPU (the "ar" grant is 2 CPUs wide).
    host.now += seconds
    host.cpu_seconds[100] += busy_fraction * seconds * 2
    supervisor.tick()


class TestLeaseLifecycle:
    def test_idle_hold_then_lend_same_node_only(self):
        host = FakeHost()
        supervisor = make_supervisor(host)
        advance(host, supervisor, 5, 0.0)  # baseline sample
        advance(host, supervisor, 5, 0.0)  # idle observed, hold starts
        assert host.affinity_calls == []
        advance(host, supervisor, 61, 0.0)  # hold expires -> lend
        assert (200, {0, 1, 2, 8, 9, 10}) in host.affinity_calls
        # shared1 lives on node 1: no lent cpus there, affinity unchanged.
        assert (300, {4, 12}) in host.affinity_calls

    def test_busy_reclaims_immediately(self):
        host = FakeHost()
        supervisor = make_supervisor(host)
        advance(host, supervisor, 5, 0.0)
        advance(host, supervisor, 5, 0.0)
        advance(host, supervisor, 61, 0.0)
        host.affinity_calls.clear()
        advance(host, supervisor, 5, 0.9)  # busy again
        assert (200, {1, 2, 9, 10}) in host.affinity_calls

    def test_short_idle_does_not_lend(self):
        host = FakeHost()
        supervisor = make_supervisor(host)
        advance(host, supervisor, 5, 0.0)
        advance(host, supervisor, 30, 0.0)  # < idle_hold_s
        assert host.affinity_calls == []

    def test_hysteresis_band_keeps_state(self):
        host = FakeHost()
        supervisor = make_supervisor(host)
        advance(host, supervisor, 5, 0.0)
        advance(host, supervisor, 5, 0.2)  # between watermarks: no hold start
        advance(host, supervisor, 61, 0.2)
        assert host.affinity_calls == []

    def test_busy_interruption_resets_hold(self):
        host = FakeHost()
        supervisor = make_supervisor(host)
        advance(host, supervisor, 5, 0.0)
        advance(host, supervisor, 30, 0.0)
        advance(host, supervisor, 5, 0.9)  # busy resets the hold
        advance(host, supervisor, 40, 0.0)  # only 40s of idle since reset
        assert host.affinity_calls == []

    def test_dead_pid_is_skipped(self):
        host = FakeHost()
        supervisor = make_supervisor(host)

        def raising(pid):
            raise OSError("gone")

        supervisor._cpu_seconds = raising
        advance(host, supervisor, 5, 0.0)
        assert host.affinity_calls == []

    def test_exclusive_holder_never_repinned(self):
        host = FakeHost()
        supervisor = make_supervisor(host)
        advance(host, supervisor, 5, 0.0)
        advance(host, supervisor, 5, 0.0)
        advance(host, supervisor, 61, 0.0)
        advance(host, supervisor, 5, 0.9)
        assert all(pid != 100 for pid, _ in host.affinity_calls)

    def test_watermark_validation(self):
        host = FakeHost()
        with pytest.raises(ValueError, match="Watermarks"):
            make_supervisor(host, idle_below=0.5, busy_above=0.4)

    def test_stop_restores_base_masks_after_lend(self):
        host = FakeHost()
        supervisor = make_supervisor(host)
        advance(host, supervisor, 5, 0.0)
        advance(host, supervisor, 5, 0.0)
        advance(host, supervisor, 61, 0.0)  # lent
        host.affinity_calls.clear()
        supervisor.stop()
        assert (200, {1, 2, 9, 10}) in host.affinity_calls

    def test_stop_without_lend_touches_nothing(self):
        host = FakeHost()
        supervisor = make_supervisor(host)
        advance(host, supervisor, 5, 0.0)
        supervisor.stop()
        assert host.affinity_calls == []


class TestSetProcessAffinity:
    def test_applies_to_every_task_until_stable(self, monkeypatch):
        pinned = []
        listings = [["10", "11"], ["10", "11", "12"], ["10", "11", "12"]]

        monkeypatch.setattr(os, "listdir", lambda path: listings.pop(0))
        monkeypatch.setattr(
            os,
            "sched_setaffinity",
            lambda tid, cpus: pinned.append(tid),
            raising=False,
        )
        _set_process_affinity(42, {0, 1})
        # First pass pins 10/11; the re-scan catches the new thread 12.
        assert pinned == [10, 11, 12]

    def test_dead_process_is_silent(self, monkeypatch):
        def gone(path):
            raise FileNotFoundError(path)

        monkeypatch.setattr(os, "listdir", gone)
        _set_process_affinity(42, {0})

    def test_one_task_failure_does_not_stop_others(self, monkeypatch):
        pinned = []
        listings = [["10", "11"], ["10", "11"]]

        def set_affinity(tid, cpus):
            if tid == 10:
                raise OSError("gone")
            pinned.append(tid)

        monkeypatch.setattr(os, "listdir", lambda path: listings.pop(0))
        monkeypatch.setattr(os, "sched_setaffinity", set_affinity, raising=False)
        _set_process_affinity(42, {0})
        assert pinned == [11]
