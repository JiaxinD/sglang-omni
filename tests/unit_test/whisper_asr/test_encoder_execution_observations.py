# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.whisper_asr.encoder_cuda_graph import (
    WhisperEncoderCudaGraphRunner,
    _CapturedGraph,
)
from sglang_omni.models.whisper_asr.sglang_model import WhisperForConditionalGeneration
from sglang_omni.profiler.work_units import collect_execution_observations


def _model(encoder, runner=None):
    return SimpleNamespace(
        model=SimpleNamespace(encoder=encoder), _encoder_graph_runner=runner
    )


def test_no_runner_records_eager_return():
    features = torch.ones(3, 2, 5)
    observations = []
    output = WhisperForConditionalGeneration._run_encoder(
        _model(torch.nn.Identity()), features, observations=observations
    )
    assert output is features
    assert len(observations) == 1
    assert observations[0]["reason"] == "no_runner"
    assert observations[0]["outcome"] == "host_returned"


@pytest.mark.parametrize("graph_failure", [False, True])
def test_runner_failure_and_model_recovery_are_both_observed(graph_failure):
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.calls = 0

        def forward(self, features):
            self.calls += 1
            if not graph_failure and self.calls == 1:
                raise RuntimeError("first eager attempt")
            return features

    class FailedGraph:
        def replay(self):
            raise RuntimeError("graph attempt")

    encoder = Encoder()
    runner = WhisperEncoderCudaGraphRunner(encoder, num_mel_bins=2, input_feature_len=5)
    if graph_failure:
        runner._graphs[4] = _CapturedGraph(
            graph=FailedGraph(),
            input_features=torch.zeros(4, 2, 5),
            output=torch.zeros(4, 2, 5),
        )
    observations = []
    features = torch.ones(3, 2, 5)
    output = WhisperForConditionalGeneration._run_encoder(
        _model(encoder, runner), features, observations=observations
    )
    assert output is features
    assert len(observations) == 2
    assert observations[0]["outcome"] == "raised"
    assert observations[0]["path"] == ("cuda_graph" if graph_failure else "eager")
    assert observations[1]["reason"] == "runner_raised"
    assert observations[1]["outcome"] == "host_returned"


def test_default_encoder_call_compiles_without_reading_active_collector():
    model = _model(torch.nn.Linear(5, 5))

    def run(features):
        return WhisperForConditionalGeneration._run_encoder(model, features)

    features = torch.ones(3, 2, 5)
    compiled = torch.compile(run, backend="eager", fullgraph=True)
    with collect_execution_observations() as observations:
        torch.testing.assert_close(compiled(features), run(features))
        assert observations == []


def test_profiled_pre_lm_entry_retains_device_metadata():
    from functools import partial

    model = _model(torch.nn.Linear(5, 5))
    model._run_encoder = partial(WhisperForConditionalGeneration._run_encoder, model)
    features = torch.ones(1, 2, 5)
    with collect_execution_observations() as observations:
        output = WhisperForConditionalGeneration.encode_audio_features(
            model, [SimpleNamespace(feature=features)]
        )
    torch.testing.assert_close(output, model.model.encoder(features))
    assert observations[0]["device"] == {"type": "cpu", "index": None}
