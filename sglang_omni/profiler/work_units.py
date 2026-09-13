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
