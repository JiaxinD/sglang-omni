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

import json
import os
from collections.abc import Iterable

# Opt-in at server launch (MOSS_BATCH_DENSITY=1). Off by default so production serving
# and other benchmarks are untouched; the experiment server sets it explicitly.
_ENABLED = os.environ.get("MOSS_BATCH_DENSITY", "0").lower() not in ("0", "", "false")
# Surfacing: the worker periodically dumps the cumulative snapshot to this path; the
# same-host benchmark reads it at window start/end and diffs (no cross-process IPC).
_DUMP_PATH = os.environ.get("MOSS_BATCH_DENSITY_DUMP", "/tmp/moss_batch_density.json")
_DUMP_EVERY = 16


def enabled() -> bool:
    return _ENABLED


def effective_decode_bs(request_ids: Iterable[str], skip_rids: set[str]) -> int:
    """Effective batch size at ``_finalize``: requests that committed a frame this step.

    A request is skipped (no frame) when it is finished, retracted, or an overrun row
    in a lagged async-resolve batch. The rest each commit one frame, so the count is
    the per-step batch size for the bs histogram.
    """
    return sum(1 for rid in request_ids if rid not in skip_rids)


class BatchDensityRecorder:
    """Accumulate decode-frame counts keyed by the batch size that produced them."""

    def __init__(self) -> None:
        self.decode_frames_total = 0
        self.bs_histogram: dict[int, int] = {}
        self._since_dump = 0

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

    def dump(self, path: str) -> None:
        """Atomically write the cumulative snapshot to ``path`` (tmp + replace)."""
        tmp = path + ".tmp"
        with open(tmp, "w") as handle:
            json.dump(self.snapshot(), handle)
        os.replace(tmp, path)

    def maybe_dump(self, path: str | None = None, every: int = _DUMP_EVERY) -> None:
        """Dump once every ``every`` steps to bound hot-path file I/O."""
        self._since_dump += 1
        if self._since_dump >= every:
            self._since_dump = 0
            self.dump(path or _DUMP_PATH)


# Worker-process-global recorder. _finalize records into it; the per-request result
# adapter reads its snapshot onto the outgoing result dict (same worker process). The
# benchmark client windows by diffing the warmup-end and dispatch-end snapshots, so the
# cumulative counter is never reset in-process.
_RECORDER = BatchDensityRecorder()


def get_recorder() -> BatchDensityRecorder:
    return _RECORDER
