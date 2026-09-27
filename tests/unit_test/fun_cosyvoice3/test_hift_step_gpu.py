# SPDX-License-Identifier: Apache-2.0
"""The batched, windowed streaming HiFT step reproduces the per-request
whole-history chain on the real CausalHiFTGenerator."""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.fun_cosyvoice3 import stages

pytestmark = pytest.mark.accelerator

cosyvoice_generator = pytest.importorskip("cosyvoice.hifigan.generator")
cosyvoice_f0 = pytest.importorskip("cosyvoice.hifigan.f0_predictor")

HOP = 480
# note (Jiaxin Deng): a non-final call holds back the F0 look-right (3), the
# conv_pre look-right (4) and the trailing ISTFT frame (1).
HOLD = 8


def make_hift() -> torch.nn.Module:
    torch.manual_seed(0)
    hift = cosyvoice_generator.CausalHiFTGenerator(
        in_channels=80,
        base_channels=512,
        nb_harmonics=8,
        sampling_rate=24000,
        nsf_alpha=0.1,
        nsf_sigma=0.003,
        nsf_voiced_threshold=10,
        upsample_rates=[8, 5, 3],
        upsample_kernel_sizes=[16, 11, 7],
        istft_params={"n_fft": 16, "hop_len": 4},
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        source_resblock_kernel_sizes=[7, 7, 11],
        source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        lrelu_slope=0.1,
        audio_limit=0.99,
        conv_pre_look_right=4,
        f0_predictor=cosyvoice_f0.CausalConvRNNF0Predictor(
            num_class=1, in_channels=80, cond_channels=512
        ),
    )
    hift = hift.cuda().eval()
    stages.keep_hift_constants_on_device(hift, "cuda")
    stages.patch_causal_conv_cache()
    return hift


class FlowStub:
    output_size = 80

    def parameters(self):
        yield torch.zeros(1, device="cuda")

    @property
    def decoder(self):
        return type("Decoder", (), {"estimator": torch.nn.Identity()})()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_hift_step_matches_per_request_chain() -> None:
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    hift = make_hift()
    vocoder = stages.CosyVoice3Vocoder(FlowStub(), hift)
    assert vocoder.hift_geometry() == (HOP, HOLD)
    torch.manual_seed(1)
    # note (Jiaxin Deng): F0 must cross the voiced threshold so the sine source
    # carries phase; random weights leave it there for some frames.
    mels = [torch.randn(1, 80, frames, device="cuda") * 3 for frames in (180, 240, 300)]
    hops = (56, 156)
    with torch.inference_mode():
        for step in range(len(hops) + 1):
            expected: list[torch.Tensor] = []
            rows: list[tuple[torch.Tensor, int, bool]] = []
            for mel in mels:
                ends = (
                    [*hops[:step], mel.shape[2]]
                    if step == len(hops)
                    else list(hops[: step + 1])
                )
                final = step == len(hops)
                hift_mel, offset = None, 0
                for index, end in enumerate(ends):
                    delta, hift_mel, offset = vocoder.hift_delta(
                        mel[
                            :,
                            :,
                            (hift_mel.shape[2] if hift_mel is not None else 0) : end,
                        ],
                        hift_mel=hift_mel,
                        speech_offset=offset,
                        finalize=final and index == len(ends) - 1,
                    )
                expected.append(delta)
                previous = (ends[-2] - HOLD) * HOP if len(ends) > 1 else 0
                rows.append((mel[:, :, : ends[-1]], previous, final))
            for (delta, offset), reference, (history, _, final) in zip(
                vocoder.hift_step(rows), expected, rows, strict=True
            ):
                assert delta.shape == reference.shape
                assert offset == history.shape[2] * HOP - (0 if final else HOLD * HOP)
                # note (Jiaxin Deng): finals of unequal length share one padded
                # batch, whose ISTFT edge differs from the single-row call in
                # the last frame; hops discard that region as hold-back.
                compare = delta.shape[1] - (HOP if final else 0)
                torch.testing.assert_close(
                    delta[:, :compare], reference[:, :compare], atol=1e-4, rtol=0
                )
