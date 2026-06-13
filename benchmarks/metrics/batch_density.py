# SPDX-License-Identifier: Apache-2.0
"""Client-side analysis of server batch-density snapshots and open-loop overload.

The server exposes a cumulative batch-density snapshot. The client takes one snapshot
after warmup (window start) and one after dispatch (window end) and diffs them, so the
measured bs distribution covers exactly the experiment window without a server reset.
"""

from __future__ import annotations


def _int_keyed(histogram: dict) -> dict[int, int]:
    return {int(bs): int(count) for bs, count in histogram.items()}


def windowed_batch_density(snap_start: dict, snap_end: dict) -> dict:
    """Frame-density metrics for the (start, end] window: end minus start per bucket."""
    start_hist = _int_keyed(snap_start.get("bs_histogram", {}))
    end_hist = _int_keyed(snap_end.get("bs_histogram", {}))
    hist: dict[int, int] = {}
    for bs, count in end_hist.items():
        delta = count - start_hist.get(bs, 0)
        if delta:
            hist[bs] = delta
    total = sum(hist.values())
    return {
        "decode_frames_total": total,
        "bs_histogram": hist,
        "bs1_frame_ratio": (hist.get(1, 0) / total) if total else 0.0,
    }


def _inflight_rising(
    trajectory: list[int], min_len: int = 8, factor: float = 1.5
) -> bool:
    """True if the back half of the in-flight trajectory is meaningfully higher than the
    front half, i.e. arrivals are outrunning completions (backlog building)."""
    if len(trajectory) < min_len:
        return False
    half = len(trajectory) // 2
    front = sum(trajectory[:half]) / half
    back = sum(trajectory[half:]) / (len(trajectory) - half)
    return back > front * factor


def overload_signal(
    inflight_trajectory: list[int],
    timeout_count: int,
    guard_tripped: bool,
) -> dict:
    """Divergence-based overload detector (replaces the p99-vs-baseline guard).

    Open-loop latency includes queue wait by design, so high p99 is not overload; a
    growing backlog, timeouts, or a tripped hard guard are.
    """
    reasons: list[str] = []
    if guard_tripped:
        reasons.append("guard_tripped")
    if timeout_count > 0:
        reasons.append(f"timeouts={timeout_count}")
    if _inflight_rising(inflight_trajectory):
        reasons.append("inflight_rising")
    return {"overloaded": bool(reasons), "reasons": reasons}
