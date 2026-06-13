# SPDX-License-Identifier: Apache-2.0
"""Server-side batch-density recorder: frame-weighted decode count and bs histogram."""

from __future__ import annotations

from sglang_omni.model_runner.batch_density import BatchDensityRecorder


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
    }
