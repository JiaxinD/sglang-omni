# SPDX-License-Identifier: Apache-2.0
"""Open-loop load generator: arrival scheduling, no-semaphore invariant, guard."""

from __future__ import annotations

import asyncio

from benchmarks.benchmarker.data import RequestResult
from benchmarks.benchmarker.runner import (
    BenchmarkRunner,
    RunConfig,
    planned_arrival_offsets,
)


def _tracking_send_fn(hold_s: float):
    """A fake send_fn that records peak concurrent in-flight calls."""
    state = {"active": 0, "peak": 0, "launch_order": []}

    async def send_fn(_session, sample) -> RequestResult:
        state["active"] += 1
        state["peak"] = max(state["peak"], state["active"])
        state["launch_order"].append(sample)
        try:
            await asyncio.sleep(hold_s)
            return RequestResult(request_id=str(sample), is_success=True)
        finally:
            state["active"] -= 1

    return send_fn, state


def test_openloop_fixed_offsets_are_evenly_spaced() -> None:
    offsets = planned_arrival_offsets(
        "openloop_fixed", request_count=4, request_rate=2.0, arrival_seed=0
    )
    assert offsets == [0.0, 0.5, 1.0, 1.5]


def test_openloop_poisson_offsets_are_seeded_and_monotonic() -> None:
    kw = dict(request_count=6, request_rate=5.0)
    a = planned_arrival_offsets("openloop_poisson", arrival_seed=123, **kw)
    b = planned_arrival_offsets("openloop_poisson", arrival_seed=123, **kw)
    c = planned_arrival_offsets("openloop_poisson", arrival_seed=999, **kw)
    assert a == b  # same seed reproduces the schedule
    assert a != c  # different seed yields a different schedule
    assert len(a) == 6
    assert a[0] == 0.0  # first arrival at t0
    assert all(x <= y for x, y in zip(a, a[1:]))  # nondecreasing


def test_openloop_never_caps_inflight_at_max_concurrency() -> None:
    # The open-loop path must NOT gate launches on a semaphore: requests arrive on
    # schedule even while earlier ones are still in flight. With 5 fast arrivals and
    # a slow send_fn, all 5 are concurrently in flight despite max_concurrency=2.
    send_fn, state = _tracking_send_fn(hold_s=0.2)
    cfg = RunConfig(
        max_concurrency=2,
        request_rate=1000.0,
        warmup=0,
        load_mode="openloop_fixed",
        disable_tqdm=True,
    )
    results = asyncio.run(BenchmarkRunner(cfg).run(list(range(5)), send_fn))
    assert len(results) == 5
    assert state["peak"] == 5  # not throttled to max_concurrency=2


def test_openloop_guard_aborts_and_flags_without_waiting_for_slot() -> None:
    # The hard backstop: when in-flight reaches max_inflight_guard it must abort and
    # mark the run invalid, NOT pause-until-a-slot-frees (which would be closed loop).
    send_fn, state = _tracking_send_fn(hold_s=0.3)
    cfg = RunConfig(
        request_rate=1000.0,
        warmup=0,
        load_mode="openloop_fixed",
        max_inflight_guard=3,
        disable_tqdm=True,
    )
    runner = BenchmarkRunner(cfg)
    asyncio.run(runner.run(list(range(8)), send_fn))
    meta = runner.dispatch_meta
    assert meta["guard_tripped"] is True
    assert meta["peak_inflight"] <= 3
    assert meta["launched_count"] < 8  # aborted early, did not launch all 8
    assert state["peak"] <= 3


def test_openloop_records_per_request_launch_metadata() -> None:
    # hold (0.2s) >> inter-arrival (0.01s) so all 4 arrivals overlap in flight.
    send_fn, _ = _tracking_send_fn(hold_s=0.2)
    cfg = RunConfig(
        request_rate=100.0,
        warmup=0,
        load_mode="openloop_fixed",
        disable_tqdm=True,
    )
    results = asyncio.run(BenchmarkRunner(cfg).run(list(range(4)), send_fn))
    # gather preserves task order, so results[i] is sample i (planned 0, .01, .02, .03)
    assert results[0].planned_start_s == 0.0
    planned = [r.planned_start_s for r in results]
    assert planned == sorted(planned)
    assert all(r.actual_start_s >= r.planned_start_s for r in results)
    assert all(r.start_delay_s >= 0.0 for r in results)
    assert [r.inflight_at_launch for r in results] == [1, 2, 3, 4]


def test_closed_loop_still_caps_inflight_at_max_concurrency() -> None:
    # Regression guard: the default path is unchanged and stays gated by the semaphore.
    send_fn, state = _tracking_send_fn(hold_s=0.1)
    cfg = RunConfig(
        max_concurrency=2, warmup=0, disable_tqdm=True
    )  # default closed_loop
    asyncio.run(BenchmarkRunner(cfg).run(list(range(6)), send_fn))
    assert state["peak"] == 2
