# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the async-decode (one-step lookahead) state machine.

The heavy sub-steps (_build_forward_batch / _prepare_and_forward / _finalize)
and the model-specific hooks are stubbed, and torch.cuda.Event is patched, so
these run CPU-only. The pinned ping-pong test is CUDA-guarded.
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
        self._pending = None
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


def test_launch_sets_pending_resolve_clears_it():
    r = _StubRunner()
    with _patch_event(ready=True):
        r.execute_launch(_sched_output(2))
        assert r._pending is not None
        assert r._pending.n_real == 2
        out = r.execute_resolve()
    assert r._pending is None
    assert out is not None
    assert (r.launch_calls, r.resolve_calls, r.finalize_calls) == (1, 1, 1)
    assert (r._async_query_hit, r._async_query_miss) == (1, 0)


def test_double_launch_without_resolve_asserts():
    r = _StubRunner()
    with _patch_event(ready=True):
        r.execute_launch(_sched_output(1))
        with pytest.raises(AssertionError):
            r.execute_launch(_sched_output(1))  # invariant: <=1 in flight


def test_resolve_without_pending_returns_none():
    # Warmup: first iteration has nothing to resolve.
    r = _StubRunner()
    assert r.execute_resolve() is None
    assert r.finalize_calls == 0


def test_query_miss_falls_back_to_synchronize():
    r = _StubRunner()
    with _patch_event(ready=False):
        r.execute_launch(_sched_output(1))
        ev = r._pending.event
        r.execute_resolve()
    assert ev.synced is True
    assert (r._async_query_hit, r._async_query_miss) == (0, 1)


def test_resolve_consumes_the_launched_steps_buffer():
    r = _StubRunner()
    with _patch_event(ready=True):
        r.execute_launch(_sched_output(1))
        r.execute_resolve()
        assert r.last_resolved_buf == "hostbuf-1"  # step N's buffer
        r.execute_launch(_sched_output(1))
        r.execute_resolve()
        assert r.last_resolved_buf == "hostbuf-2"  # step N+1's buffer


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
    s._async_pending = ("batchX", "sched_out")
    assert s._async_pending_batch() == "batchX"
