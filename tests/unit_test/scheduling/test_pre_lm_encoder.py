# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import concurrent.futures
import contextlib
import json
import queue
import threading
from collections.abc import Iterator

import pytest

from sglang_omni.profiler.event_recorder import get_recorder
from sglang_omni.scheduling.pre_lm_encoder import PreLMEncoderService, QueueEntry

_STOP = object()


def _work_records(directory):
    return [
        json.loads(line)
        for path in directory.glob("work_units_*.jsonl")
        for line in path.read_text().splitlines()
    ]


def test_profile_records_each_retry_without_fake_request_ids(tmp_path):
    recorder = get_recorder()
    recorder.start("run", str(tmp_path), "asr")
    service = _Service(controlled_drain=True)
    service.fail_multi = service.retry = True
    try:
        first, second = service._submit(1), service._submit(2)
        service.drain_gate.set()
        assert first.result(timeout=2) == 2
        assert second.result(timeout=2) == 4
    finally:
        service.close()
        recorder.stop()
    records = _work_records(tmp_path)
    begins = [r for r in records if r["kind"] == "begin"]
    ends = [r for r in records if r["kind"] == "end"]
    assert [r["attempt"] for r in begins] == [0, 1, 2]
    assert [len(r["members"]) for r in begins] == [2, 1, 1]
    assert len({r["batch_id"] for r in begins}) == 1
    assert [r["error_type"] for r in ends] == ["RuntimeError", None, None]
    assert all(r["end_ns"] >= r["start_ns"] for r in ends)
    assert not any("request_id" in r for r in records)


def test_profile_finishes_execution_before_synchronous_future_callback(tmp_path):
    recorder = get_recorder()
    recorder.start("run", str(tmp_path), "asr")
    service = _Service()
    callback_entered, callback_release = threading.Event(), threading.Event()
    future = concurrent.futures.Future()

    def slow_callback(_):
        callback_entered.set()
        assert callback_release.wait(timeout=2)

    future.add_done_callback(slow_callback)
    try:
        service._submit(1, future)
        assert callback_entered.wait(timeout=2)
        ends = [r for r in _work_records(tmp_path) if r["kind"] == "end"]
        assert len(ends) == 1
        assert ends[0]["error_type"] is None
    finally:
        callback_release.set()
        service.close()
        recorder.stop()


def test_profile_off_never_reads_item_metadata():
    class Item(int):
        @property
        def feature(self):
            raise AssertionError("profile-off path accessed feature")

    get_recorder().stop()
    service = _Service()
    try:
        assert service._submit(Item(2)).result(timeout=2) == 4
    finally:
        service.close()


def test_profile_metadata_failure_does_not_fail_encoder(tmp_path):
    class Item(int):
        @property
        def feature(self):
            raise ValueError("unavailable shape")

    recorder = get_recorder()
    recorder.start("run", str(tmp_path), "asr")
    service = _Service()
    try:
        assert service._submit(Item(2)).result(timeout=2) == 4
    finally:
        service.close()
        recorder.stop()
    records = _work_records(tmp_path)
    error = next(r for r in records if r["kind"] == "metadata_error")
    assert error["error_type"] == "ValueError"
    assert not any(r["kind"] == "end" for r in records)


def test_profile_captures_shape_before_attachment_clears_feature(tmp_path):
    from types import SimpleNamespace

    class Item(int):
        pass

    class ClearingService(_Service):
        def attach_embedding(self, item, embedding):
            item.feature = None
            super().attach_embedding(item, embedding)

    item = Item(3)
    item.feature = SimpleNamespace(shape=(1, 80, 3000))
    item.num_audio_tokens = 1500
    item.audio_fingerprint = "fingerprint"
    recorder = get_recorder()
    recorder.start("run", str(tmp_path), "asr")
    service = ClearingService()
    try:
        assert service._submit(item).result(timeout=2) == 6
    finally:
        service.close()
        recorder.stop()
    begin = next(r for r in _work_records(tmp_path) if r["kind"] == "begin")
    assert begin["members"][0]["feature_shape"] == [1, 80, 3000]
    assert begin["members"][0]["num_audio_tokens"] == 1500
    assert item.feature is None


def test_profile_restart_does_not_move_inflight_completion_to_new_session(tmp_path):
    entered, release = threading.Event(), threading.Event()

    class BlockingService(_Service):
        def encode_batch(self, items):
            entered.set()
            assert release.wait(timeout=2)
            return super().encode_batch(items)

    recorder = get_recorder()
    recorder.start("run", str(tmp_path), "asr")
    old_file = recorder.work_unit_recorder().path
    service = BlockingService()
    try:
        future = service._submit(1)
        assert entered.wait(timeout=2)
        recorder.stop()
        recorder.start("run", str(tmp_path), "asr")
        new_file = recorder.work_unit_recorder().path
        release.set()
        assert future.result(timeout=2) == 2
        assert service._submit(2).result(timeout=2) == 4
    finally:
        release.set()
        service.close()
        recorder.stop()
    old = [json.loads(line) for line in old_file.read_text().splitlines()]
    new = [json.loads(line) for line in new_file.read_text().splitlines()]
    assert [r["kind"] for r in old] == ["open", "begin", "stop"]
    assert old[-1]["begun_units"] == 1 and old[-1]["ended_units"] == 0
    assert [r["kind"] for r in new] == ["open", "begin", "end", "stop"]


def test_execution_observations_are_serialized_per_retry(tmp_path):
    from sglang_omni.profiler.work_units import current_execution_observations

    class ObservedService(_Service):
        def encode_batch(self, items):
            observations = current_execution_observations()
            assert observations is not None
            observations.append({"input_shape": [len(items), 80, 3000]})
            return super().encode_batch(items)

    recorder = get_recorder()
    recorder.start("run", str(tmp_path), "asr")
    service = ObservedService(controlled_drain=True)
    service.fail_multi = service.retry = True
    try:
        first, second = service._submit(1), service._submit(2)
        service.drain_gate.set()
        assert first.result(timeout=2) == 2
        assert second.result(timeout=2) == 4
    finally:
        service.close()
        recorder.stop()
    ends = [r for r in _work_records(tmp_path) if r["kind"] == "end"]
    assert [e["executions"] for e in ends] == [
        [{"input_shape": [2, 80, 3000]}],
        [{"input_shape": [1, 80, 3000]}],
        [{"input_shape": [1, 80, 3000]}],
    ]
    assert ends[0]["error_type"] == "RuntimeError"
    assert current_execution_observations() is None


def test_service_keeps_construction_snapshot_after_factory_scope(tmp_path):
    from sglang_omni.profiler.work_units import stage_construction_scope

    origin = {"stage": "encoder", "tp_rank": 1, "gpu_id": 0, "placement_gpu_id": 3}
    with stage_construction_scope(origin):
        service = _Service()
    origin["stage"] = "changed"
    unknown = _Service()
    recorder = get_recorder()
    recorder.start("run", str(tmp_path), "asr")
    try:
        assert service._submit(1).result(timeout=2) == 2
        assert unknown._submit(2).result(timeout=2) == 4
    finally:
        service.close()
        unknown.close()
        recorder.stop()
    begins = [r for r in _work_records(tmp_path) if r["kind"] == "begin"]
    assert begins[0]["constructed_in"] == {
        "stage": "encoder",
        "tp_rank": 1,
        "gpu_id": 0,
        "placement_gpu_id": 3,
    }
    assert begins[1]["constructed_in"] is None


class _Service(PreLMEncoderService[int, list[int], int]):
    def __init__(self, *, controlled_drain: bool = False) -> None:
        self.attachments: list[tuple[int, int]] = []
        self.cached: list[tuple[int, int, object | None]] = []
        self.context_events: list[str] = []
        self.stage_host_copies = False
        self.fail_multi = False
        self.fail_items: set[int] = set()
        self.retry = False
        self.retry_hook_error = False
        self.start_hook_error = False
        self.split_mode = "exact"
        self.drain_gate = threading.Event() if controlled_drain else None
        super().__init__(worker_name="test-pre-lm-encoder")

    def close(self) -> None:
        if self._thread.is_alive():
            self._queue.put(_STOP)
            self._thread.join(timeout=2)

    def _next_batch(self) -> tuple[list[QueueEntry[int]], bool]:
        first = self._queue.get()
        if first is _STOP:
            return [], True
        if self.drain_gate is not None:
            assert self.drain_gate.wait(timeout=2)
        batch = [first]
        while True:
            try:
                batch.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return batch, False

    @contextlib.contextmanager
    def _batch_context(self) -> Iterator[None]:
        self.context_events.append("enter")
        try:
            yield
        finally:
            self.context_events.append("exit")

    def encode_batch(self, items: list[int]) -> list[int]:
        if self.fail_multi and len(items) > 1:
            raise RuntimeError("batch encode failed")
        if any(item in self.fail_items for item in items):
            raise RuntimeError("item encode failed")
        return [item * 2 for item in items]

    def split_embeddings(self, items: list[int], encoded: list[int]) -> list[int]:
        if self.split_mode == "too_few":
            return encoded[:-1]
        if self.split_mode == "too_many":
            return [*encoded, -1]
        return encoded

    def attach_embedding(self, item: int, embedding: int) -> None:
        self.attachments.append((item, embedding))

    def stage_host_copy(self, item: int, embedding: int) -> object | None:
        if not self.stage_host_copies:
            return None
        self.context_events.append("stage")
        return ("host", embedding)

    def cache_embedding(
        self, item: int, embedding: int, host_copy: object | None = None
    ) -> None:
        self.cached.append((item, embedding, host_copy))

    def _retry_batch(self, batch, exc):  # noqa: ANN001, ANN202
        if self.retry_hook_error:
            raise RuntimeError("retry policy failed")
        return self.retry

    def _on_batch_start(self, batch):  # noqa: ANN001, ANN202
        if self.start_hook_error:
            raise RuntimeError("start hook failed")


def test_successful_dispatch_attaches_and_caches() -> None:
    service = _Service()
    try:
        future = service._submit(3)

        assert future.result(timeout=2) == 6
        assert service.attachments == [(3, 6)]
        assert service.cached == [(3, 6, None)]
    finally:
        service.close()


def test_stage_host_copy_runs_in_batch_context_and_reaches_cache() -> None:
    service = _Service()
    service.stage_host_copies = True
    try:
        future = service._submit(4)

        assert future.result(timeout=2) == 8
        # note (Jeffro): staged inside the batch context (before its exit), delivered to
        # cache_embedding after the barrier.
        assert service.context_events == ["enter", "stage", "exit"]
        assert service.cached == [(4, 8, ("host", 8))]
    finally:
        service.close()


def test_batch_failure_recovers_each_item() -> None:
    service = _Service(controlled_drain=True)
    service.fail_multi = True
    service.retry = True
    try:
        first = service._submit(1)
        second = service._submit(2)
        service.drain_gate.set()

        assert first.result(timeout=2) == 2
        assert second.result(timeout=2) == 4
    finally:
        service.close()


@pytest.mark.parametrize("split_mode", ["too_few", "too_many"])
def test_wrong_embedding_cardinality_fails_future(split_mode: str) -> None:
    service = _Service()
    service.split_mode = split_mode
    try:
        future = service._submit(1)

        with pytest.raises(RuntimeError, match="split_embeddings returned"):
            future.result(timeout=2)
        assert service.context_events == ["enter", "exit"]
    finally:
        service.close()


def test_statistics_hook_failure_does_not_kill_worker() -> None:
    service = _Service()
    service.start_hook_error = True
    try:
        assert service._submit(1).result(timeout=2) == 2
        service.start_hook_error = False
        assert service._submit(2).result(timeout=2) == 4
    finally:
        service.close()


def test_fatal_policy_failure_completes_current_future_and_rejects_submits() -> None:
    service = _Service(controlled_drain=True)
    service.fail_items.add(1)
    service.retry_hook_error = True
    first = service._submit(1)
    second = service._submit(2)
    service.drain_gate.set()

    with pytest.raises(RuntimeError, match="retry policy failed"):
        first.result(timeout=2)
    with pytest.raises(RuntimeError, match="retry policy failed"):
        second.result(timeout=2)
    with pytest.raises(RuntimeError, match="worker has failed"):
        service._submit(3)

    service.close()
