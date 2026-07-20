from __future__ import annotations

import pytest

from sglang_omni_router.observability import (
    CounterReport,
    DataPlaneCounterLedger,
    StaleCounterGenerationError,
    WorkerCounters,
)


def _report(
    seq: int,
    routed: int,
    *,
    dp: int = 0,
    generation: int = 1,
    active: int = 0,
    worker_id: str = "w0",
) -> CounterReport:
    return CounterReport(
        dp_index=dp,
        generation=generation,
        counter_seq=seq,
        workers=[
            WorkerCounters(
                worker_id=worker_id,
                routed_total=routed,
                successful_total=routed,
                failed_total=0,
                current_active=active,
            )
        ],
    )


def test_first_contact_establishes_the_baseline() -> None:
    # since-CP-start: the first report only sets the baseline; growth after
    # first contact is what gets displayed
    ledger = DataPlaneCounterLedger()
    assert ledger.apply(_report(1, 20), now=0.0) is True
    assert ledger.totals("w0")["routed_total"] == 0
    ledger.apply(_report(2, 25), now=0.0)
    assert ledger.totals("w0")["routed_total"] == 5


def test_cp_restart_restarts_the_window_coherently() -> None:
    # a fresh ledger (CP restart) takes a surviving DP's cumulative as
    # baseline: the display restarts at zero
    old_cp = DataPlaneCounterLedger()
    old_cp.apply(_report(1, 0), now=0.0)
    old_cp.apply(_report(9, 120), now=0.0)
    assert old_cp.totals("w0")["routed_total"] == 120

    new_cp = DataPlaneCounterLedger()
    new_cp.apply(_report(10, 120), now=0.0)
    assert new_cp.totals("w0")["routed_total"] == 0
    new_cp.apply(_report(11, 130), now=0.0)
    assert new_cp.totals("w0")["routed_total"] == 10


def test_cumulative_reports_are_idempotent_and_ordered() -> None:
    ledger = DataPlaneCounterLedger()
    ledger.apply(_report(1, 0), now=0.0)
    assert ledger.apply(_report(2, 5), now=0.0) is True
    # duplicate delivery of the same seq: dropped, totals unchanged
    assert ledger.apply(_report(2, 5), now=0.0) is False
    # out-of-order older seq: dropped even with different numbers
    assert ledger.apply(_report(1, 3), now=0.0) is False
    assert ledger.totals("w0")["routed_total"] == 5


def test_a_regressed_cumulative_cannot_lower_the_display() -> None:
    # e.g. a worker URL deleted and re-added on the DP resets its counters;
    # the display clamps to the high-water mark instead of falling
    ledger = DataPlaneCounterLedger()
    ledger.apply(_report(1, 0), now=0.0)
    ledger.apply(_report(2, 10), now=0.0)
    assert ledger.apply(_report(3, 3), now=0.0) is True  # newer seq, lower value
    assert ledger.totals("w0")["routed_total"] == 10


def test_workers_missing_from_a_report_keep_their_contribution() -> None:
    ledger = DataPlaneCounterLedger()
    ledger.apply(_report(1, 0), now=0.0)
    ledger.apply(_report(2, 7), now=0.0)
    # next report omits w0 entirely (e.g. deleted from the registry)
    ledger.apply(
        CounterReport(dp_index=0, generation=1, counter_seq=3, workers=[]),
        now=0.0,
    )
    assert ledger.totals("w0")["routed_total"] == 7


def test_generation_bump_retires_the_old_contribution() -> None:
    # displayed totals must never move backwards across a DP restart
    ledger = DataPlaneCounterLedger()
    ledger.apply(_report(1, 2, generation=1), now=0.0)  # baseline 2
    ledger.apply(_report(7, 10, generation=1), now=0.0)  # contribution 8
    ledger.apply(_report(1, 3, generation=2), now=0.0)  # new gen: baseline 3
    assert ledger.totals("w0")["routed_total"] == 8
    ledger.apply(_report(2, 4, generation=2), now=0.0)
    assert ledger.totals("w0")["routed_total"] == 9


def test_older_generation_reports_are_fenced() -> None:
    ledger = DataPlaneCounterLedger()
    ledger.apply(_report(1, 1, generation=2), now=0.0)
    with pytest.raises(StaleCounterGenerationError):
        ledger.apply(_report(9, 9, generation=1), now=0.0)


def test_active_gauge_sums_only_live_entries() -> None:
    ledger = DataPlaneCounterLedger(liveness_secs=3.0)
    ledger.apply(_report(1, 1, dp=0, active=2), now=100.0)
    ledger.apply(_report(1, 1, dp=1, active=3), now=102.0)
    assert ledger.active_gauge("w0", now=102.5) == 5
    # dp 0's report is now stale: its in-flight work is gone with it
    assert ledger.active_gauge("w0", now=103.5) == 3


def test_overlay_renders_counter_fields() -> None:
    class _Worker:
        worker_id = "w0"

    ledger = DataPlaneCounterLedger()
    ledger.apply(_report(1, 0, active=0))
    ledger.apply(_report(2, 6, active=1))  # real clock: entry counts as live
    overlay = ledger.overlay(_Worker())
    assert overlay == {
        "active_requests": 1,
        "routed_requests": 6,
        "successful_requests": 6,
        "failed_requests": 0,
    }
