# SPDX-License-Identifier: Apache-2.0
"""Server-side batch-density recorder: frame-weighted decode count and bs histogram."""

from __future__ import annotations

import json
import os

from sglang_omni.model_runner.batch_density import (
    BatchDensityRecorder,
    effective_decode_bs,
)


def test_dump_writes_snapshot_json(tmp_path) -> None:
    recorder = BatchDensityRecorder()
    for bs in [1, 2, 2, 1]:
        recorder.record_step(bs)
    path = str(tmp_path / "bd.json")
    recorder.dump(path)
    loaded = json.loads(open(path).read())
    # JSON object keys are strings; the windowing consumer re-ints them.
    assert loaded["decode_frames_total"] == recorder.decode_frames_total == 6
    assert loaded["bs_histogram"] == {"1": 2, "2": 4}
    assert loaded["bs1_frame_ratio"] == recorder.bs1_frame_ratio


def test_maybe_dump_is_throttled(tmp_path) -> None:
    recorder = BatchDensityRecorder()
    path = str(tmp_path / "bd.json")
    for _ in range(3):
        recorder.record_step(1)
        recorder.maybe_dump(path, every=5)
    assert not os.path.exists(path)  # below threshold, not written yet
    for _ in range(2):
        recorder.record_step(1)
        recorder.maybe_dump(path, every=5)
    assert os.path.exists(path)  # threshold reached, dumped


def test_effective_bs_is_count_of_non_skipped_requests() -> None:
    # The effective batch size at _finalize is how many requests committed a frame
    # this step, i.e. the batch minus the skipped (finished/retracted/overrun) rids.
    assert effective_decode_bs(["a", "b", "c"], {"b"}) == 2
    assert effective_decode_bs(["a", "b"], set()) == 2
    assert effective_decode_bs(["a", "b", "c"], {"a", "b", "c"}) == 0
    assert effective_decode_bs([], {"x"}) == 0


def test_frames_are_weighted_by_batch_size() -> None:
    # Steps of batch size 1,1,4,2,1 commit 1+1+4+2+1 = 9 frames. A bs=k step
    # contributes k frames at bucket k, so sum(bs_histogram) == decode_frames_total.
    recorder = BatchDensityRecorder()
    for effective_bs in [1, 1, 4, 2, 1]:
        recorder.record_step(effective_bs)
    assert recorder.decode_frames_total == 9
    assert recorder.bs_histogram == {1: 3, 2: 2, 4: 4}
    assert sum(recorder.bs_histogram.values()) == recorder.decode_frames_total
    assert recorder.bs1_frame_ratio == 3 / 9


def test_ignores_steps_that_committed_no_frame() -> None:
    # A step where every request was skipped (effective_bs == 0) produced no frame
    # and must not be counted, nor create a spurious {0: 0} bucket.
    recorder = BatchDensityRecorder()
    recorder.record_step(0)
    recorder.record_step(2)
    recorder.record_step(0)
    assert recorder.decode_frames_total == 2
    assert recorder.bs_histogram == {2: 2}
    assert recorder.bs1_frame_ratio == 0.0


def test_snapshot_reports_cumulative_counters() -> None:
    recorder = BatchDensityRecorder()
    for effective_bs in [1, 3, 1]:
        recorder.record_step(effective_bs)
    assert recorder.snapshot() == {
        "decode_frames_total": 5,
        "bs_histogram": {1: 2, 3: 3},
        "bs1_frame_ratio": 2 / 5,
        "iter_bs_waiting_empty": {},
        "iter_bs_waiting_nonempty": {},
    }


def test_records_decode_iteration_waiting_split() -> None:
    # Per decode iteration, split by whether the waiting queue had backlog. At bs=1 this
    # sizes the addressable admission opportunity: waiting>0 (a request was queued but not
    # admitted) is addressable; waiting==0 (nothing to batch with) is not.
    recorder = BatchDensityRecorder()
    recorder.record_iteration(1, waiting_len=0)  # bs=1, empty queue: unaddressable
    recorder.record_iteration(1, waiting_len=3)  # bs=1, backlog present: addressable
    recorder.record_iteration(1, waiting_len=0)
    recorder.record_iteration(4, waiting_len=0)
    recorder.record_iteration(0, waiting_len=2)  # no frame committed: ignored
    snap = recorder.snapshot()
    assert snap["iter_bs_waiting_empty"] == {1: 2, 4: 1}
    assert snap["iter_bs_waiting_nonempty"] == {1: 1}
    assert recorder.bs1_addressable_fraction == 1 / 3  # 1 of 3 bs=1 iters had backlog
