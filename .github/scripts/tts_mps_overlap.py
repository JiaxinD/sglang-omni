# SPDX-License-Identifier: Apache-2.0
"""One-to-one overlap oracle for bounded same-GPU MPS validation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable

MAX_INTERVALS_PER_REPLICA = 20


@dataclass(frozen=True)
class Interval:
    replica_id: int
    request_id: str
    start_ns: int
    end_ns: int


def _intervals(
    events: Iterable[dict[str, Any]], *, expected_run_id: str
) -> list[Interval]:
    points: dict[tuple[int, str], dict[str, int]] = {}
    run_ids: set[str] = set()
    boot_ids: set[str] = set()
    for event in events:
        if event.get("clock") != "CLOCK_MONOTONIC":
            raise ValueError("activity events must use CLOCK_MONOTONIC")
        run_id = event.get("run_id")
        boot_id = event.get("host_boot_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("activity events must carry a run ID")
        if not isinstance(boot_id, str) or not boot_id:
            raise ValueError("activity events must carry a server host boot ID")
        run_ids.add(run_id)
        boot_ids.add(boot_id)
        kind = event.get("event")
        if kind not in {"model_path_start", "model_path_end"}:
            continue
        if kind == "model_path_end" and event.get("status") != "success":
            continue
        key = (int(event["replica_id"]), str(event["request_id"]))
        slot = points.setdefault(key, {})
        point = "start" if kind == "model_path_start" else "end"
        if point in slot:
            raise ValueError(f"duplicate {point} event for {key}")
        slot[point] = int(event["monotonic_ns"])
    if run_ids != {expected_run_id}:
        raise ValueError(
            f"activity events do not match expected run {expected_run_id!r}: {run_ids}"
        )
    if len(boot_ids) != 1:
        raise ValueError("activity events must carry one server host boot ID")

    result: list[Interval] = []
    for (replica_id, request_id), slot in points.items():
        if set(slot) != {"start", "end"}:
            continue
        if slot["end"] <= slot["start"]:
            raise ValueError(f"non-positive model-path interval for {request_id}")
        result.append(Interval(replica_id, request_id, slot["start"], slot["end"]))
    return result


def build_overlap_verdict(
    events: Iterable[dict[str, Any]],
    *,
    expected_run_id: str,
    min_successes_per_replica: int,
    min_matched_overlap_count: int,
    measurement_uncertainty_ns: int,
) -> dict[str, Any]:
    if min_successes_per_replica < 2 or min_matched_overlap_count < 2:
        raise ValueError("overlap proof requires at least two repeated intervals")
    if measurement_uncertainty_ns < 0:
        raise ValueError("measurement uncertainty must be non-negative")

    intervals = _intervals(events, expected_run_id=expected_run_id)
    left = [item for item in intervals if item.replica_id == 0]
    right = [item for item in intervals if item.replica_id == 1]
    if len(left) > MAX_INTERVALS_PER_REPLICA or len(right) > MAX_INTERVALS_PER_REPLICA:
        raise ValueError("bounded overlap canary exceeded its interval budget")
    counts = {"0": len(left), "1": len(right)}
    if min(counts.values()) < min_successes_per_replica:
        raise ValueError(
            f"insufficient successful requests per replica: {counts}; "
            f"required at least two"
        )

    @lru_cache(maxsize=None)
    def match(
        left_index: int, used: int
    ) -> tuple[int, int, tuple[tuple[int, int, int], ...]]:
        if left_index == len(left):
            return 0, 0, ()
        best = match(left_index + 1, used)
        for right_index, candidate in enumerate(right):
            if used & (1 << right_index):
                continue
            overlap_ns = min(left[left_index].end_ns, candidate.end_ns) - max(
                left[left_index].start_ns, candidate.start_ns
            )
            if overlap_ns <= measurement_uncertainty_ns:
                continue
            count, aggregate, pairs = match(left_index + 1, used | (1 << right_index))
            option = (
                count + 1,
                aggregate + overlap_ns,
                ((left_index, right_index, overlap_ns), *pairs),
            )
            if option[:2] > best[:2]:
                best = option
        return best

    matched_count, aggregate_ns, pairs = match(0, 0)
    if matched_count < min_matched_overlap_count:
        raise ValueError(
            f"insufficient repeated cross-replica overlap: {matched_count}; "
            f"required {min_matched_overlap_count}"
        )
    matches = [
        {
            "replica_0_request_id": left[left_index].request_id,
            "replica_1_request_id": right[right_index].request_id,
            "overlap_ns": overlap_ns,
        }
        for left_index, right_index, overlap_ns in pairs
    ]
    return {
        "schema_version": 1,
        "status": "pass",
        "clock": "CLOCK_MONOTONIC",
        "per_replica_successes": counts,
        "matched_overlap_count": matched_count,
        "aggregate_overlap_ns": aggregate_ns,
        "maximum_overlap_ns": max(item["overlap_ns"] for item in matches),
        "measurement_uncertainty_ns": measurement_uncertainty_ns,
        "matches": matches,
    }
