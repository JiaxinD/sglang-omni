# SPDX-License-Identifier: Apache-2.0
"""Server-side batch-density recorder for the open-loop batch-composition study.

Counts decode frames and their effective batch size at a single producer-agnostic
hook (model runner ``_finalize``). Dependency-free so it is cheap and unit-testable.

Frame-weighted by construction: a decode step of batch size ``k`` commits ``k`` frames
(one per request), so ``decode_frames_total += k`` and ``bs_histogram[k] += k``. This
makes ``sum(bs_histogram.values()) == decode_frames_total`` AND keeps the strong,
independent cross-check exact: ``decode_frames_total`` equals the sum over finished
requests of client ``completion_tokens`` (each ``= len(output_rows)``).
``bs1_frame_ratio`` is therefore the fraction of produced frames that were emitted
while the decode batch had shrunk to a single request.
"""

from __future__ import annotations


class BatchDensityRecorder:
    """Accumulate decode-frame counts keyed by the batch size that produced them."""

    def __init__(self) -> None:
        self.decode_frames_total = 0
        self.bs_histogram: dict[int, int] = {}

    def record_step(self, effective_bs: int) -> None:
        """Record one decode step that committed ``effective_bs`` frames."""
        if effective_bs <= 0:
            return  # no request committed a frame this step; not a decode frame
        self.decode_frames_total += effective_bs
        self.bs_histogram[effective_bs] = (
            self.bs_histogram.get(effective_bs, 0) + effective_bs
        )

    @property
    def bs1_frame_ratio(self) -> float:
        if self.decode_frames_total == 0:
            return 0.0
        return self.bs_histogram.get(1, 0) / self.decode_frames_total

    def snapshot(self) -> dict:
        """Cumulative snapshot for surfacing; the client windows by diffing two of these."""
        return {
            "decode_frames_total": self.decode_frames_total,
            "bs_histogram": dict(self.bs_histogram),
            "bs1_frame_ratio": self.bs1_frame_ratio,
        }
