# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the cpuset contention sampler."""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

from tests.utils.ci_cpu_contention import (
    ContentionSampler,
    _parse_cpu_list,
    _Proc,
    _tree_pids,
    foreign_ticks,
)

_HAS_PROC = os.path.isdir("/proc")
_TWO_CPUS = (
    _HAS_PROC and hasattr(os, "sched_getaffinity") and len(os.sched_getaffinity(0)) >= 2
)


def test_parse_cpu_list_ranges_and_singles():
    assert _parse_cpu_list("48-51,112") == {48, 49, 50, 51, 112}


def test_tree_pids_follows_descendants_only():
    procs = {
        1: _Proc(ppid=0, ticks=0),
        10: _Proc(ppid=1, ticks=0),
        20: _Proc(ppid=10, ticks=0),
        30: _Proc(ppid=20, ticks=0),
        99: _Proc(ppid=1, ticks=0),
    }
    assert _tree_pids(procs, 10) == {10, 20, 30}


def test_foreign_ticks_deducts_tree_and_clamps():
    prev = {10: _Proc(ppid=1, ticks=100)}
    # note (Jiaxin Deng): pid 20 was born inside the window, so its full
    # tick count is deducted; outsiders never appear here because foreign
    # time comes from the per-CPU busy delta, not from process scans.
    cur = {10: _Proc(ppid=1, ticks=300), 20: _Proc(ppid=1, ticks=50)}
    assert foreign_ticks(500, prev, cur, tree={10, 20}) == 250
    assert foreign_ticks(100, prev, cur, tree={10, 20}) == 0


@pytest.mark.skipif(not _HAS_PROC, reason="requires Linux procfs")
def test_sampler_smoke_produces_summary():
    sampler = ContentionSampler({0, 1}, interval_s=0.2)
    sampler.start()
    time.sleep(0.7)
    sampler.stop()
    assert "[cpuset-contention]" in sampler.summary()


def _spin(core: int) -> subprocess.Popen:
    return subprocess.Popen(
        [
            sys.executable,
            "-c",
            f"import os\nos.sched_setaffinity(0, {{{core}}})\nwhile True: pass",
        ]
    )


@pytest.mark.skipif(not _TWO_CPUS, reason="requires Linux procfs and two allowed CPUs")
def test_controlled_load_inside_cpuset_counts_outside_does_not():
    """A busy outsider on the pinned core is charged; one on another core is not."""
    cores = sorted(os.sched_getaffinity(0))
    core_in, core_out = cores[0], cores[1]
    session_root = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"]
    )
    inside = _spin(core_in)
    outside = _spin(core_out)
    sampler = ContentionSampler({core_in}, interval_s=0.4, root_pid=session_root.pid)
    try:
        time.sleep(0.2)
        sampler.start()
        time.sleep(1.4)
        sampler.stop()
        peak = sampler.peak_foreign_cores()
        assert 0.5 <= peak <= 1.6, sampler.summary()
    finally:
        for proc in (inside, outside, session_root):
            proc.kill()
            proc.wait(timeout=5)
