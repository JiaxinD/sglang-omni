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
    per_worker: dict[str, _WorkerLedger] = field(default_factory=dict)


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

        per_worker = entry.per_worker if entry is not None else {}
        for item in report.workers:
            ledger = per_worker.get(item.worker_id)
            if ledger is None:
                # Note (Jiaxin Deng): first sight under this (dp, generation)
                # takes the whole cumulative as the since-CP-start baseline.
                first_contact = entry is None
                per_worker[item.worker_id] = _WorkerLedger(
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
        # Note (Jiaxin Deng): workers missing from the report keep their
        # last contribution.

        self._entries[report.dp_index] = _LedgerEntry(
            generation=report.generation,
            counter_seq=report.counter_seq,
            reported_at=now if now is not None else time.monotonic(),
            per_worker=per_worker,
        )
        return True

    def _retire(self, entry: _LedgerEntry) -> None:
        for worker_id, ledger in entry.per_worker.items():
            slot = self._retired.setdefault(
                worker_id, {key: 0 for key in _COUNTER_KEYS}
            )
            for key in _COUNTER_KEYS:
                slot[key] += ledger.contribution(key)

    def totals(self, worker_id: str) -> dict[str, int]:
        totals = dict(self._retired.get(worker_id, {key: 0 for key in _COUNTER_KEYS}))
        for entry in self._entries.values():
            ledger = entry.per_worker.get(worker_id)
            if ledger is None:
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
            ledger = entry.per_worker.get(worker_id)
            if ledger is not None:
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
