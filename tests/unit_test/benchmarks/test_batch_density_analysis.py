# SPDX-License-Identifier: Apache-2.0
"""Client-side analysis: window server bs snapshots, detect open-loop overload."""

from __future__ import annotations

from benchmarks.metrics.batch_density import overload_signal, windowed_batch_density


def test_windowed_density_is_end_minus_start_per_bucket() -> None:
    # JSON snapshots arrive with string histogram keys; the window is end - start.
    start = {"decode_frames_total": 10, "bs_histogram": {"1": 2, "4": 8}}
    end = {"decode_frames_total": 21, "bs_histogram": {"1": 5, "4": 12, "2": 4}}
    w = windowed_batch_density(start, end)
    assert w["bs_histogram"] == {1: 3, 4: 4, 2: 4}  # 5-2, 12-8, 4-0
    assert w["decode_frames_total"] == 11
    assert w["bs1_frame_ratio"] == 3 / 11


def test_overload_flags_guard_timeouts_and_rising_inflight() -> None:
    rising = list(range(1, 21))  # inflight climbs 1..20: server not keeping up
    flat = [4] * 20
    assert (
        overload_signal(flat, timeout_count=0, guard_tripped=False)["overloaded"]
        is False
    )
    assert (
        overload_signal(rising, timeout_count=0, guard_tripped=False)["overloaded"]
        is True
    )
    assert (
        overload_signal(flat, timeout_count=2, guard_tripped=False)["overloaded"]
        is True
    )
    assert (
        overload_signal(flat, timeout_count=0, guard_tripped=True)["overloaded"] is True
    )
