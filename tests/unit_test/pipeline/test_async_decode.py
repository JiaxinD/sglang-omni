# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the async-decode (one-step lookahead) state machine.

The heavy sub-steps (_build_forward_batch / _prepare_and_forward / _finalize)
and the model-specific hooks are stubbed, and torch.cuda.Event is patched, so
these run CPU-only. The pinned ping-pong test is CUDA-guarded.

Pending ownership lives with the CALLER (execute_launch returns a handle,
execute_resolve takes it) because launch-first scheduling has two steps
momentarily in flight.
"""
from __future__ import annotations

import types
from unittest import mock

import pytest
import torch

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.types import ModelRunnerOutput


class _StubRunner(ModelRunner):
    """ModelRunner with mocked sub-steps; exercises only execute_launch/resolve."""

    def __init__(self):
        self._async_enabled = True
        self._staging_slot = 0
        self._host_staging_buffers = []
        self._async_query_hit = 0
        self._async_query_miss = 0
        self.launch_calls = 0
        self.resolve_calls = 0
        self.finalize_calls = 0
        self.last_resolved_buf = None

    def _build_forward_batch(self, scheduler_output):
        sb = types.SimpleNamespace(is_prefill_only=False, output_ids=None)
        return types.SimpleNamespace(), sb, types.SimpleNamespace(), False  # decode

    def _prepare_and_forward(self, forward_batch, schedule_batch, requests, is_prefill):
        return types.SimpleNamespace(
            next_token_ids=object(),
            logits_output=types.SimpleNamespace(next_token_logits=None),
            can_run_cuda_graph=False,
        )

    def post_decode_launch(self, result, forward_batch, requests):
        self.launch_calls += 1
        return f"hostbuf-{self.launch_calls}"

    def post_decode_resolve(self, host_buf, result, forward_batch, schedule_batch, requests):
        self.resolve_calls += 1
        self.last_resolved_buf = host_buf

    def _finalize(self, batch_result, forward_batch, schedule_batch, model_worker_batch, scheduler_output):
        self.finalize_calls += 1
        return ModelRunnerOutput(outputs={}, req_ids=[], req_id_to_index={})


def _patch_event(ready: bool):
    class _FakeEvent:
        def __init__(self):
            self.synced = False

        def record(self):
            pass

        def query(self):
            return ready

        def synchronize(self):
            self.synced = True

    return mock.patch("torch.cuda.Event", _FakeEvent)


def _sched_output(n):
    return types.SimpleNamespace(requests=list(range(n)), batch_data=object())


def test_launch_returns_handle_resolve_consumes_it():
    r = _StubRunner()
    with _patch_event(ready=True):
        step = r.execute_launch(_sched_output(2))
        assert step is not None and step.n_real == 2
        out = r.execute_resolve(step)
    assert out is not None
    assert (r.launch_calls, r.resolve_calls, r.finalize_calls) == (1, 1, 1)
    assert (r._async_query_hit, r._async_query_miss) == (1, 0)


def test_two_launches_return_distinct_handles():
    # launch-first keeps two steps in flight; both must be independent handles
    r = _StubRunner()
    with _patch_event(ready=True):
        s1 = r.execute_launch(_sched_output(1))
        s2 = r.execute_launch(_sched_output(1))
        assert s1 is not s2 and s1.host_buf != s2.host_buf
        # resolve in order N-1 then N
        r.execute_resolve(s1)
        assert r.last_resolved_buf == s1.host_buf
        r.execute_resolve(s2)
        assert r.last_resolved_buf == s2.host_buf


def test_resolve_none_returns_none():
    # Warmup / drained: nothing to resolve.
    r = _StubRunner()
    assert r.execute_resolve(None) is None
    assert r.finalize_calls == 0


def test_query_miss_falls_back_to_synchronize():
    r = _StubRunner()
    with _patch_event(ready=False):
        step = r.execute_launch(_sched_output(1))
        r.execute_resolve(step)
    assert step.event.synced is True
    assert (r._async_query_hit, r._async_query_miss) == (0, 1)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="pinned memory requires CUDA"
)
def test_host_staging_pingpong():
    r = _StubRunner()
    dev = torch.zeros(8, 18)
    b0 = r._next_host_staging(dev)
    b1 = r._next_host_staging(dev)
    b2 = r._next_host_staging(dev)
    assert len(r._host_staging_buffers) == 2
    assert b0 is b2 and b0 is not b1  # ping-pong between exactly 2 buffers
    assert b0.is_pinned() and tuple(b0.shape) == (8, 18) and b0.dtype == dev.dtype


def test_batch_is_decode():
    decode = types.SimpleNamespace(
        forward_mode=types.SimpleNamespace(is_decode=lambda: True, is_extend=lambda: False)
    )
    extend = types.SimpleNamespace(
        forward_mode=types.SimpleNamespace(is_decode=lambda: False, is_extend=lambda: True)
    )
    assert OmniScheduler._batch_is_decode(decode) is True
    assert OmniScheduler._batch_is_decode(extend) is False
    assert OmniScheduler._batch_is_decode(types.SimpleNamespace()) is False  # no mode


def test_async_pending_batch_getattr_safe():
    # OmniScheduler.__getattr__ raises for unset attrs; _async_pending_batch
    # must tolerate that (test fixtures may bypass __init__).
    s = OmniScheduler.__new__(OmniScheduler)
    assert s._async_pending_batch() is None
    s._async_pending = ("batchX", "sched_out", "pending_step")
    assert s._async_pending_batch() == "batchX"
