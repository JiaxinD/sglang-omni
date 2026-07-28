# SPDX-License-Identifier: Apache-2.0
"""Bit-identity gate for the Higgs vocoder CUDA graph on the real DAC.

Every reference here is captured from the pristine codec *before* the runner
touches it, so it is the verbatim pre-graph path (upstream Transformers RVQ
decode included) rather than a helper shared with the graphed path.

Needs a GPU and the ~9 GB checkpoint, so it is gated on ``HIGGS_TTS_CKPT``:

    HIGGS_TTS_CKPT=bosonai/higgs-tts-3-4b python -m pytest \
        tests/unit_test/higgs_tts/test_vocoder_cuda_graph_real_model.py -v -s
"""

import os
import time

import pytest
import torch

from sglang_omni.models.higgs_tts.audio_codec import HiggsAudioCodec

_CKPT = os.environ.get("HIGGS_TTS_CKPT")

real_model = pytest.mark.skipif(
    not torch.cuda.is_available() or not _CKPT,
    reason="needs a GPU and HIGGS_TTS_CKPT pointing at a Higgs TTS checkpoint",
)

# Default streaming params emit windows in [1, max(stride, followup+holdback+overlap)].
_MAX_FRAMES = 87
_FRAME_CASES = [1, 7, 16, 64, 83, _MAX_FRAMES]
_BATCH_CASES = [1, 2, 4]
_BATCH_FRAMES = 32


def _codes(codec, batch, frames, *, seed):
    vocab = int(codec.model.config.codebook_size)
    num_codebooks = int(codec.model.config.num_quantizers)
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(
        0, vocab, (batch, num_codebooks, frames), generator=generator
    ).to(codec.device, torch.long)


@pytest.fixture(scope="module")
def graphed_codec():
    """Codec with sealed graphs, plus pre-graph references for every case."""
    codec = HiggsAudioCodec.from_pretrained(_CKPT, device="cuda", dtype=torch.bfloat16)
    references = {}
    with torch.no_grad():
        for frames in _FRAME_CASES:
            codes = _codes(codec, 1, frames, seed=frames)
            references[(1, frames)] = (
                codes,
                codec.model.decode(codes).audio_values.clone(),
            )
        for batch in _BATCH_CASES:
            codes = _codes(codec, batch, _BATCH_FRAMES, seed=1000 + batch)
            references[(batch, _BATCH_FRAMES)] = (
                codes,
                codec.model.decode(codes).audio_values.clone(),
            )
        codes_TN = _codes(codec, 1, 83, seed=7)[0].transpose(0, 1).cpu()
        call_site = (codes_TN, codec.decode(codes_TN).clone())

    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    started = time.time()
    codec.warmup_cuda_graph(
        [(1, frames) for frames in range(1, _MAX_FRAMES + 1)]
        + [(batch, _BATCH_FRAMES) for batch in _BATCH_CASES]
    )
    torch.cuda.synchronize()
    cost = {
        "seconds": time.time() - started,
        "gib": (torch.cuda.memory_allocated() - before) / 1024**3,
    }
    assert codec._cg_runner is not None, "warmup captured nothing"
    return codec, references, call_site_reference, cost


@real_model
@torch.no_grad()
@pytest.mark.parametrize("frames", _FRAME_CASES)
def test_replay_is_bit_identical_to_pre_graph_eager(graphed_codec, frames):
    codec, references, _, _ = graphed_codec
    codes, reference = references[(1, frames)]
    replayed = codec._cg_runner.decode(codes)
    assert replayed is not None, f"(1, {frames}) should have been captured"
    assert replayed.dtype == reference.dtype
    assert torch.equal(replayed, reference)


@real_model
@torch.no_grad()
@pytest.mark.parametrize("batch", _BATCH_CASES)
def test_replay_is_bit_identical_across_batch_sizes(graphed_codec, batch):
    codec, references, _, _ = graphed_codec
    codes, reference = references[(batch, _BATCH_FRAMES)]
    replayed = codec._cg_runner.decode(codes)
    assert replayed is not None, f"({batch}, {_BATCH_FRAMES}) should have been captured"
    assert torch.equal(replayed, reference)


@real_model
@torch.no_grad()
def test_codec_decode_call_site_is_bit_identical(graphed_codec):
    """The gate on the real entry point, not just the runner."""
    codec, references, call_site_reference, _ = graphed_codec
    codes = references[(1, 83)][0]
    codes_TN = codes[0].transpose(0, 1).cpu()
    assert torch.equal(codec.decode(codes_TN), call_site_reference)


@real_model
@torch.no_grad()
def test_padded_bucket_diverges_and_is_therefore_refused(graphed_codec):
    """Why the runner never pads: a padded replay is measurably not bit-identical."""
    codec, references, _, _ = graphed_codec
    narrow_codes, narrow_reference = references[(1, _BATCH_FRAMES)]
    entry = codec._cg_runner.captured_entry(4, _BATCH_FRAMES)
    assert entry is not None

    padded = torch.zeros_like(entry.static_codes)
    padded[:1] = narrow_codes
    entry.static_codes.copy_(padded)
    entry.graph.replay()
    torch.cuda.synchronize()
    delta = (entry.static_audio[:1].float() - narrow_reference.float()).abs().max()
    assert delta > 0, "padding happened to be exact here; re-check the guard's premise"
    print(f"\npadded-bucket B=1 into B=4: max delta {delta.item():.6g}")

    codes_b3 = _codes(codec, 3, _BATCH_FRAMES, seed=3)
    assert codec._cg_runner.decode(codes_b3) is None


@real_model
@torch.no_grad()
def test_decode_carries_no_state_between_calls(graphed_codec):
    """Why nothing is declared in the persistent-state registry: replaying other
    shapes in between cannot change a window's output."""
    codec, references, _, _ = graphed_codec
    codes, reference = references[(1, 83)]
    assert codec._cg_runner.persistent_state.is_empty()
    for other in (1, 16, _MAX_FRAMES):
        codec._cg_runner.decode(references[(1, other)][0])
    assert torch.equal(codec._cg_runner.decode(codes), reference)


@real_model
@torch.no_grad()
def test_uncaptured_windows_fall_back_to_eager(graphed_codec):
    codec, _, _, _ = graphed_codec
    assert codec._cg_runner.decode(_codes(codec, 1, _MAX_FRAMES + 40, seed=11)) is None
    assert codec._cg_runner.decode(_codes(codec, 8, _BATCH_FRAMES, seed=12)) is None


@real_model
def test_warmup_cost_is_bounded(graphed_codec):
    codec, _, _, cost = graphed_codec
    print(
        f"\nwarmup: {len(codec._cg_runner.captured_frames())} frame graphs, "
        f"{cost['seconds']:.1f}s, +{cost['gib']:.2f} GiB"
    )
    assert cost["seconds"] < 180
    assert cost["gib"] < 4.0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
