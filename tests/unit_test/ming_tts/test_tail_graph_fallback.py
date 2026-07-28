# SPDX-License-Identifier: Apache-2.0
"""An uncovered batch must fall back to eager, not fail the request.

``run_tail_step`` already owns an eager path; the tail-graph cache used to raise
when no captured bucket covered the live batch, turning a capacity miss into a
500 for every request in that batch.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sglang_omni.models.ming_tts.sglang_model import (
    MingTTSSGLangModel,
    _MingTTSTailGraphCache,
)


class _StubGraph:
    def __init__(self, bucket: int) -> None:
        self.bucket = bucket

    def replay(self, inputs, *, noise, sde_random):
        return ("graphed", self.bucket)


def _cache(buckets) -> _MingTTSTailGraphCache:
    cache = _MingTTSTailGraphCache(model=object())
    cache.buckets = tuple(sorted(buckets))
    cache.graphs = {bucket: _StubGraph(bucket) for bucket in cache.buckets}
    return cache


def _inputs(batch_size: int):
    return SimpleNamespace(
        hidden_states=SimpleNamespace(shape=(batch_size, 1, 8), device="cpu")
    )


@pytest.mark.parametrize("batch_size, expected_bucket", [(1, 1), (2, 2), (3, 4)])
def test_covered_batches_replay_the_smallest_fitting_bucket(
    batch_size, expected_bucket
):
    cache = _cache([1, 2, 4])
    assert cache.replay(_inputs(batch_size), noise=None, sde_random=None) == (
        "graphed",
        expected_bucket,
    )


def test_uncovered_batch_returns_none_instead_of_raising():
    cache = _cache([1, 2, 4])
    assert cache.replay(_inputs(8), noise=None, sde_random=None) is None


def test_run_tail_step_falls_back_to_eager_when_no_bucket_covers_the_batch():
    model = object.__new__(MingTTSSGLangModel)
    model._tail_graphs = _cache([1, 2])
    eager_calls = []

    def _eager(inputs, *, noise, timesteps, sde_random):
        eager_calls.append(int(inputs.hidden_states.shape[0]))
        return "eager"

    model._compute_tail_step = _eager
    model._make_tail_sampling_inputs = lambda *, batch_size, device: (None, None, None)

    assert model.run_tail_step(_inputs(8)) == "eager"
    assert eager_calls == [8], "the eager path must receive the uncovered batch"

    # a covered batch still replays the graph
    assert model.run_tail_step(_inputs(2)) == ("graphed", 2)
