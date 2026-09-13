# SPDX-License-Identifier: Apache-2.0
"""Per-execution host timings, separate from request events and GPU timings."""

import json
import logging
import os
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)
_execution_local = threading.local()


def current_work_unit() -> dict | None:
    """Current encoder attempt, also available in capture-only sessions."""
    return getattr(_execution_local, "work_unit", None)


@contextmanager
def work_unit_scope(identity: dict):
    previous = current_work_unit()
    _execution_local.work_unit = identity
    try:
        yield identity
    finally:
        _execution_local.work_unit = previous


def current_stage_construction() -> dict | None:
    """Stage factory active on this thread; not physical device binding."""
    return getattr(_execution_local, "stage_construction", None)


@contextmanager
def stage_construction_scope(stage: dict):
    previous = current_stage_construction()
    _execution_local.stage_construction = stage
    try:
        yield
    finally:
        _execution_local.stage_construction = previous


def current_execution_observations() -> list[dict] | None:
    """Return this Python worker's collector, absent outside a profiled unit."""
    return getattr(_execution_local, "observations", None)


@contextmanager
def collect_execution_observations():
    previous = current_execution_observations()
    observations = []
    _execution_local.observations = observations
    try:
        yield observations
    finally:
        _execution_local.observations = previous


@contextmanager
def annotate_execution_device(device, observations: list[dict]):
    """Attach worker-side device properties to newly recorded executions.

    CUDA device properties can be cached by the runtime. They are not a query
    of context affinity, exclusive partitions, or actual MPS attachment.
    """
    first = len(observations)
    try:
        yield
    finally:
        if len(observations) > first:
            metadata = {"type": device.type, "index": device.index}
            try:
                if device.type == "cuda":
                    import torch

                    properties = torch.cuda.get_device_properties(device)
                    metadata.update(
                        name=properties.name,
                        device_reported_sm_count=properties.multi_processor_count,
                        mps_active_thread_percentage=os.environ.get(
                            "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"
                        ),
                        scope="device properties; not context affinity or MPS attachment",
                    )
            except Exception as exc:
                # Note (Jiaxin Deng): diagnostic queries must not replace the
                # encoder's original exception or invalidate a successful batch.
                metadata["query_error_type"] = type(exc).__name__
            for execution in observations[first:]:
                execution["device"] = dict(metadata)


class WorkUnitRecorder:
    """One profiling session; late completions never enter another session."""

    def __init__(self, directory: Path, run_id: str):
        self.run_id = run_id
        self.pid = os.getpid()
        self.path = directory / f"work_units_{self.pid}_{uuid.uuid4().hex}.jsonl"
        self._lock = threading.Lock()
        self._fp = self.path.open("x", buffering=1, encoding="utf-8")
        self._begun = self._ended = self._failures = 0
        self._write(
            "open",
            clock="perf_counter_ns",
            wall_ns=time.time_ns(),
            scope="host execution envelope; excludes future dispatch; not isolated GPU service time",
        )

    def is_active(self) -> bool:
        return self._fp is not None

    def _write(self, kind: str, **fields) -> bool:
        if self._fp is None:
            return False
        try:
            self._fp.write(
                json.dumps(
                    {
                        "kind": kind,
                        "run_id": self.run_id,
                        "pid": self.pid,
                        "recorded_ns": time.perf_counter_ns(),
                        **fields,
                    }
                )
                + "\n"
            )
            return True
        except Exception:
            self._failures += 1
            if self._failures == 1:
                logger.warning("Failed to write work-unit record", exc_info=True)
            return False

    def begin(self, **metadata) -> str | None:
        with self._lock:
            unit_id = uuid.uuid4().hex
            if self._write("begin", unit_id=unit_id, **metadata):
                self._begun += 1
                return unit_id
            return None

    def metadata_error(self, *, batch_id: str, attempt: int, error_type: str):
        with self._lock:
            self._write(
                "metadata_error",
                batch_id=batch_id,
                attempt=attempt,
                error_type=error_type,
            )

    def end(
        self,
        unit_id: str,
        *,
        start_ns: int,
        end_ns: int,
        error_type: str | None,
        executions: list[dict] | None = None,
    ):
        with self._lock:
            if self._write(
                "end",
                unit_id=unit_id,
                start_ns=start_ns,
                end_ns=end_ns,
                error_type=error_type,
                executions=executions,
            ):
                self._ended += 1

    def close(self):
        with self._lock:
            if self._fp is None:
                return
            self._write(
                "stop",
                begun_units=self._begun,
                ended_units=self._ended,
                write_failures=self._failures,
            )
            try:
                self._fp.close()
            except Exception:
                logger.warning("Failed to close work-unit file", exc_info=True)
            finally:
                self._fp = None
