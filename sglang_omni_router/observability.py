# SPDX-License-Identifier: Apache-2.0
"""CP-side aggregation of per-DP worker counters.

DPs report monotonic cumulative totals (never deltas), keyed by
(dp_index, generation, counter_seq): anything not strictly newer is dropped,
so re-delivery and reordering are naturally idempotent.

Baseline protocol (since-CP-start semantics): the first report the CP sees
for a (dp_index, generation) establishes a per-worker BASELINE; a worker's
contribution is its high-water cumulative minus that baseline. A CP restart
therefore restarts the displayed window at zero coherently, regardless of
how long the surviving DPs have been running. The cost is bounded: requests
a DP served before its first post-(re)start flush (at most one flush
interval) are not counted.

Displayed totals never move backwards: cumulative values are clamped to a
per-worker high-water mark (a regressed report, e.g. after a worker URL was
deleted and re-added on the DP, cannot lower the display), workers missing
from a report keep their last contribution, and when a DP generation is
replaced its final contribution is folded into a retired accumulator.

current_active is a gauge, not a counter: the latest reported value, summed
over entries whose last report is recent (a dead DP's in-flight work is gone
with it).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

_COUNTER_KEYS = ("routed_total", "successful_total", "failed_total")


class WorkerCounters(BaseModel):
    worker_id: str
    # Note (Jiaxin Deng): a deleted-and-re-added URL is a fresh DP-side object
    # whose cumulatives restart at zero, so the ledger must retire the old
    # segment instead of clamping the new one to nothing.
    incarnation: str = ""
    routed_total: int = Field(default=0, ge=0)
    successful_total: int = Field(default=0, ge=0)
    failed_total: int = Field(default=0, ge=0)
    current_active: int = Field(default=0, ge=0)


class CounterReport(BaseModel):
    dp_index: int = Field(ge=0)
    generation: int = Field(ge=1)
    counter_seq: int = Field(ge=1)
    workers: list[WorkerCounters]


class StaleCounterGenerationError(Exception):
    pass


@dataclass
class _WorkerLedger:
    baseline: dict[str, int]
    high_water: dict[str, int]
    current_active: int = 0

    def contribution(self, key: str) -> int:
        return max(0, self.high_water[key] - self.baseline[key])


@dataclass
class _LedgerEntry:
    generation: int
    counter_seq: int
    reported_at: float
    # Note (Jiaxin Deng): keyed by (worker_id, incarnation), because one report
    # carries both the new object and the draining old one during a
    # replacement; a stable-id key would make each row retire the other.
    per_worker: dict[tuple[str, str], _WorkerLedger] = field(default_factory=dict)


class DataPlaneCounterLedger:
    def __init__(self, liveness_secs: float = 3.0) -> None:
        self._entries: dict[int, _LedgerEntry] = {}
        self._retired: dict[str, dict[str, int]] = {}
        self._liveness_secs = liveness_secs

    def apply(self, report: CounterReport, *, now: float | None = None) -> bool:
        """Returns True when applied, False when dropped as stale/duplicate."""
        entry = self._entries.get(report.dp_index)
        if entry is not None:
            if report.generation < entry.generation:
                raise StaleCounterGenerationError(
                    f"generation {report.generation} < {entry.generation}"
                )
            if (
                report.generation == entry.generation
                and report.counter_seq <= entry.counter_seq
            ):
                return False
            if report.generation > entry.generation:
                self._retire(entry)
                entry = None

        carried = dict(entry.per_worker) if entry is not None else {}
        first_contact = entry is None
        per_worker: dict[tuple[str, str], _WorkerLedger] = {}
        for item in report.workers:
            ledger = carried.pop((item.worker_id, item.incarnation), None)
            if ledger is None:
                # Note (Jiaxin Deng): first sight under this (dp, generation)
                # takes the whole cumulative as the since-CP-start baseline; a
                # replaced incarnation restarts at zero, so it counts in full.
                per_worker[(item.worker_id, item.incarnation)] = _WorkerLedger(
                    baseline={
                        key: getattr(item, key) if first_contact else 0
                        for key in _COUNTER_KEYS
                    },
                    high_water={key: getattr(item, key) for key in _COUNTER_KEYS},
                    current_active=item.current_active,
                )
                continue
            for key in _COUNTER_KEYS:
                # Note (Jiaxin Deng): clamp; a regressed cumulative must not
                # lower the display.
                ledger.high_water[key] = max(ledger.high_water[key], getattr(item, key))
            ledger.current_active = item.current_active
            per_worker[(item.worker_id, item.incarnation)] = ledger

        # Note (Jiaxin Deng): a report is a full snapshot for that DP, so a
        # segment missing from it has drained for good: fold it into the stable
        # worker's retired total and drop it, or the map grows per incarnation.
        for (worker_id, _), ledger in carried.items():
            self._retire_worker(worker_id, ledger)

        self._entries[report.dp_index] = _LedgerEntry(
            generation=report.generation,
            counter_seq=report.counter_seq,
            reported_at=now if now is not None else time.monotonic(),
            per_worker=per_worker,
        )
        return True

    def _retire(self, entry: _LedgerEntry) -> None:
        for (worker_id, _), ledger in entry.per_worker.items():
            self._retire_worker(worker_id, ledger)

    def _retire_worker(self, worker_id: str, ledger: _WorkerLedger) -> None:
        slot = self._retired.setdefault(worker_id, {key: 0 for key in _COUNTER_KEYS})
        for key in _COUNTER_KEYS:
            slot[key] += ledger.contribution(key)

    def totals(self, worker_id: str) -> dict[str, int]:
        totals = dict(self._retired.get(worker_id, {key: 0 for key in _COUNTER_KEYS}))
        for entry in self._entries.values():
            for (candidate, _), ledger in entry.per_worker.items():
                if candidate != worker_id:
                    continue
                for key in _COUNTER_KEYS:
                    totals[key] += ledger.contribution(key)
        return totals

    def active_gauge(self, worker_id: str, *, now: float | None = None) -> int:
        current = now if now is not None else time.monotonic()
        gauge = 0
        for entry in self._entries.values():
            if current - entry.reported_at > self._liveness_secs:
                continue
            for (candidate, _), ledger in entry.per_worker.items():
                if candidate == worker_id:
                    gauge += ledger.current_active
        return gauge

    def overlay(self, worker) -> dict[str, int]:
        """Counter fields merged over Worker.to_dict() when rendering /workers.

        active_requests is a best-effort instantaneous sum over live DPs;
        the totals follow the documented baseline (since-CP-start) protocol.
        """
        totals = self.totals(worker.worker_id)
        return {
            "active_requests": self.active_gauge(worker.worker_id),
            "routed_requests": totals["routed_total"],
            "successful_requests": totals["successful_total"],
            "failed_requests": totals["failed_total"],
        }
