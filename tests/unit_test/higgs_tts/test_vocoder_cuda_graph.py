# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Higgs vocoder CUDA-graph runner.

A fake codec model stands in for the real DAC so capture/replay machinery can be
exercised without the checkpoint; it keeps upstream's RVQ accumulation shape
(including the capture-hostile scalar seed) so the capture-safety rewrite is
exercised too. The real-DAC bit-identity gate lives in
``test_vocoder_cuda_graph_real_model.py``.
"""

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.higgs_tts.vocoder_cuda_graph import (
    VOCODER_CUDA_GRAPH_ENV,
    HiggsVocoderCudaGraphRunner,
    capture_safe_rvq_decode,
)

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA-graph capture requires a GPU"
)

_N = 8
_DIM = 16


class _FakeQuantizer:
    def __init__(self, table: torch.Tensor) -> None:
        self._table = table

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        return self._table[indices].permute(0, 2, 1).contiguous()


class _FakeRvq(torch.nn.Module):
    """Upstream's residual-VQ decode verbatim, scalar seed included."""

    def __init__(self, tables: list[torch.Tensor]) -> None:
        super().__init__()
        self.quantizers = [_FakeQuantizer(t) for t in tables]

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        quantized_out = torch.tensor(0.0, device=codes.device)
        for i, indices in enumerate(codes):
            quantized_out = quantized_out + self.quantizers[i].decode(indices).to(
                codes.device
            )
        return quantized_out


class _FakeCodecModel(torch.nn.Module):
    """``decode([B, N, T]) -> .audio_values [B, 1, T]``, shaped like upstream's."""

    def __init__(self, num_quantizers: int = _N) -> None:
        super().__init__()
        self.config = SimpleNamespace(num_quantizers=num_quantizers)
        torch.manual_seed(0)
        self.tables = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.randn(32, _DIM), requires_grad=False)
                for _ in range(num_quantizers)
            ]
        )
        self.quantizer = _FakeRvq(list(self.tables))
        self.proj = torch.nn.Conv1d(_DIM, 1, kernel_size=3, padding=1)

    def decode(self, codes_BNT: torch.Tensor) -> SimpleNamespace:
        quantized = self.quantizer.decode(codes_BNT.transpose(0, 1))
        return SimpleNamespace(audio_values=self.proj(quantized))


class _FlakyCodecModel(_FakeCodecModel):
    def __init__(self, bad_frames: int) -> None:
        super().__init__()
        self._bad_frames = bad_frames

    def decode(self, codes_BNT: torch.Tensor) -> SimpleNamespace:
        if codes_BNT.shape[-1] == self._bad_frames:
            raise RuntimeError("synthetic capture failure")
        return super().decode(codes_BNT)


def _model(cls=_FakeCodecModel, *args) -> _FakeCodecModel:
    return cls(*args).to("cuda").eval()


def _codes(batch: int, frames: int, *, num_codebooks: int = _N, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, 32, (batch, num_codebooks, frames), generator=generator).to(
        "cuda", torch.long
    )


def _runner(model, **kwargs) -> HiggsVocoderCudaGraphRunner:
    kwargs.setdefault("warmup_iters", 2)
    return HiggsVocoderCudaGraphRunner(model, **kwargs)


@cuda_only
@torch.no_grad()
def test_upstream_rvq_decode_is_not_capturable():
    """The rewrite is not cosmetic: upstream's scalar seed aborts capture."""
    model = _model()
    static_in = torch.zeros((1, _N, 8), dtype=torch.long, device="cuda")
    model.decode(static_in)
    torch.cuda.synchronize()
    with pytest.raises(RuntimeError, match="pinned"):
        with torch.cuda.graph(torch.cuda.CUDAGraph()):
            model.decode(static_in)


@torch.no_grad()
def test_capture_safe_rvq_decode_matches_upstream_bitwise():
    """CPU gate on the rewrite itself, against upstream's own implementation."""
    from transformers.models.higgs_audio_v2_tokenizer.modeling_higgs_audio_v2_tokenizer import (  # noqa: E501
        HiggsAudioV2TokenizerResidualVectorQuantization as UpstreamRvq,
    )

    for dtype in (torch.bfloat16, torch.float32):
        torch.manual_seed(0)
        rvq = SimpleNamespace(
            quantizers=[
                _FakeQuantizer(torch.randn(32, _DIM, dtype=dtype)) for _ in range(_N)
            ]
        )
        codes = torch.randint(0, 32, (_N, 2, 5))
        reference = UpstreamRvq.decode(rvq, codes)
        rewritten = capture_safe_rvq_decode(rvq, codes)
        assert rewritten.dtype == reference.dtype
        assert torch.equal(rewritten, reference)


@cuda_only
@torch.no_grad()
@pytest.mark.parametrize("batch,frames", [(1, 1), (1, 16), (2, 16), (4, 7)])
def test_replay_is_bit_identical_to_pre_graph_eager(batch, frames):
    model = _model()
    # Reference taken before any patch or capture: the verbatim pre-graph path.
    codes = _codes(batch, frames, seed=frames)
    reference = model.decode(codes).audio_values.clone()

    runner = _runner(model, batch_sizes=(1, 2, 4))
    runner.warmup([(batch, frames)])
    replayed = runner.decode(codes)
    assert replayed is not None
    assert torch.equal(replayed, reference)


@cuda_only
@torch.no_grad()
def test_padded_bucket_is_refused():
    """A request narrower than its bucket must fall back, never pad.

    Batch padding changes the kernels cuDNN/cuBLAS select, so a padded replay is
    not bit-identical (measured on the real DAC in the real-model test).
    """
    model = _model()
    runner = _runner(model, batch_sizes=(4,))
    runner.warmup([(4, 16)])
    assert runner.decode(_codes(4, 16)) is not None
    assert runner.decode(_codes(1, 16)) is None
    assert runner.decode(_codes(3, 16)) is None


@cuda_only
@torch.no_grad()
def test_misses_fall_back_to_eager():
    model = _model()
    runner = _runner(model, batch_sizes=(1,))
    runner.warmup([(1, 16)])
    assert runner.decode(_codes(1, 99)) is None  # frame count never captured
    assert runner.decode(_codes(2, 16)) is None  # batch outside the bucket list
    assert runner.decode(_codes(1, 16, num_codebooks=6)) is None  # codebook mismatch
    assert runner.decode(torch.zeros((1, _N, 16), dtype=torch.long)) is None  # on CPU


@cuda_only
@torch.no_grad()
def test_warmup_is_sealed_after_the_first_call():
    model = _model()
    runner = _runner(model, batch_sizes=(1,))
    runner.warmup([(1, 16)])
    runner.warmup([(1, 24)])
    assert runner.decode(_codes(1, 24)) is None


@cuda_only
@torch.no_grad()
def test_capture_failure_disables_only_that_shape():
    model = _model(_FlakyCodecModel, 13)
    runner = _runner(model, batch_sizes=(1,))
    runner.warmup([(1, 12), (1, 13), (1, 16)])
    assert runner.captured_shapes() == [(1, 12), (1, 16)]
    assert runner.decode(_codes(1, 13)) is None


@cuda_only
@torch.no_grad()
def test_env_kill_switch_leaves_the_model_untouched(monkeypatch):
    monkeypatch.setenv(VOCODER_CUDA_GRAPH_ENV, "0")
    model = _model()
    stock_decode = model.quantizer.decode
    runner = _runner(model, batch_sizes=(1,))
    runner.warmup([(1, 16)])
    assert not runner.captured_shapes()
    assert runner.decode(_codes(1, 16)) is None
    assert model.quantizer.decode is stock_decode


@cuda_only
@torch.no_grad()
def test_no_persistent_state_is_declared():
    """The codec decode is a pure function of the codes, so nothing is declared."""
    model = _model()
    runner = _runner(model, batch_sizes=(1,))
    runner.warmup([(1, 16)])
    assert runner.persistent_state.is_empty()
    assert runner.persistent_state.declared_names() == []


@cuda_only
@torch.no_grad()
def test_key_ceiling_bounds_the_captured_set():
    model = _model()
    runner = _runner(model, batch_sizes=(1,), max_keys=3)
    runner.warmup([(1, f) for f in (8, 12, 16, 20, 24)])
    assert len(runner.captured_shapes()) == 3


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
