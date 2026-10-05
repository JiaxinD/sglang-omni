# SPDX-License-Identifier: Apache-2.0
"""Voxt's Swift Qwen3-ASR audio layout, reproduced on request."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from transformers import WhisperFeatureExtractor

import sglang_omni.preprocessing.transcription as transcription
from sglang_omni.models.qwen3_asr.audio_lengths import qwen3_asr_num_audio_tokens
from sglang_omni.models.qwen3_asr.request_builders import (
    make_qwen3_asr_scheduler_adapters,
)
from sglang_omni.models.qwen3_asr.swift_layout import (
    AUDIO_LAYOUT_PARAM,
    VOXT_SWIFT_LAYOUT,
    swift_log_mel,
    swift_output_length,
)
from sglang_omni.proto import OmniRequest, StagePayload

from .test_request_builders import FakeTokenizer, unwrap_built

mx = pytest.importorskip("mlx.core")

from sglang_omni.models.qwen3_asr.mlx.config import (  # noqa: E402
    AudioEncoderConfig,
    ModelConfig,
    TextConfig,
)
from sglang_omni.models.qwen3_asr.mlx.model import Qwen3ASRModel  # noqa: E402


def whisper_extractor() -> WhisperFeatureExtractor:
    return WhisperFeatureExtractor(
        feature_size=128, sampling_rate=16000, hop_length=160, chunk_length=30, n_fft=400
    )


@pytest.mark.parametrize(
    ("frames", "swift_tokens"),
    # Placeholder counts Voxt's Swift port printed for these mel lengths.
    [(679, 98), (553, 78), (306, 40)],
)
def test_swift_output_length_matches_the_voxt_swift_port(
    frames: int, swift_tokens: int
) -> None:
    assert swift_output_length(frames) == swift_tokens


def test_swift_output_length_agrees_with_the_reference_on_whole_chunks() -> None:
    for frames in (100, 600, 3000):
        assert swift_output_length(frames) == qwen3_asr_num_audio_tokens(frames)


def test_swift_log_mel_keeps_the_final_stft_frame() -> None:
    sample_rate = 16000
    t = np.arange(int(sample_rate * 1.234)) / sample_rate
    # Loudest in the middle, so the per-clip maximum is not in the final frame.
    audio = (np.sin(2 * np.pi * 440 * t) * np.exp(-((t - 0.6) ** 2) / 0.02)).astype(np.float32)
    extractor = whisper_extractor()

    swift = swift_log_mel(audio, extractor)
    reference = extractor(
        audio, sampling_rate=sample_rate, return_tensors="pt", padding="longest", truncation=False
    ).input_features

    assert swift.shape == (1, 128, len(audio) // 160 + 1)
    assert reference.shape[-1] == len(audio) // 160
    assert torch.allclose(swift[..., :-1], reference, atol=1e-4)


def chunk100_model() -> Qwen3ASRModel:
    mx.random.seed(0)
    audio = AudioEncoderConfig(
        num_mel_bins=8,
        encoder_layers=1,
        encoder_attention_heads=2,
        encoder_ffn_dim=16,
        d_model=8,
        max_source_positions=200,
        n_window=50,
        n_window_infer=800,
        conv_chunksize=500,
        downsample_hidden_size=4,
        output_dim=8,
    )
    text = TextConfig(
        vocab_size=64,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=256,
    )
    return Qwen3ASRModel(ModelConfig(audio_config=audio, text_config=text, audio_token_id=10))


def test_swift_layout_keeps_the_padded_final_chunk_rows_swift_keeps() -> None:
    model = chunk100_model()
    features = mx.random.normal((1, 8, 179))
    mask = mx.ones((1, 179))

    reference = model.get_audio_features(features, mask)
    swift = model.get_audio_features(features, mask, layout=VOXT_SWIFT_LAYOUT)

    # 100 + 79 frames: 13 + 10 rows by the reference formula; Swift credits
    # the 79-frame chunk with 20 and keeps all 13 rows of its padded conv.
    assert reference.shape[0] == 23
    assert swift.shape[0] == 26


def test_swift_layout_leaves_placeholders_beyond_the_encoder_rows_as_audio_pads() -> None:
    model = chunk100_model()
    input_ids = mx.array([[1, *([10] * 5), 2]], dtype=mx.int32)
    audio_features = mx.ones((3, 8))

    with pytest.raises(ValueError):
        model.build_inputs_embeds(input_ids, audio_features, audio_start=1, num_audio_tokens=5)
    embeds = model.build_inputs_embeds(
        input_ids, audio_features, audio_start=1, num_audio_tokens=5, layout=VOXT_SWIFT_LAYOUT
    )

    pad_embedding = model.model.embed_tokens(mx.array([10]))[0]
    assert mx.allclose(embeds[0, 1:4], mx.ones((3, 8)).astype(embeds.dtype)).item()
    assert mx.allclose(embeds[0, 4], pad_embedding).item()
    assert mx.allclose(embeds[0, 5], pad_embedding).item()


def test_request_builder_applies_the_swift_layout_only_when_asked(monkeypatch) -> None:
    samples = 678 * 160 + 80  # 679 mel frames with the final frame kept
    monkeypatch.setattr(
        transcription,
        "load_audio",
        lambda source, **kwargs: np.random.default_rng(0).normal(0, 0.1, samples).astype(np.float32),
    )
    request_builder, _ = make_qwen3_asr_scheduler_adapters(
        tokenizer=FakeTokenizer(),
        max_new_tokens=32,
        feature_extractor=whisper_extractor(),
    )

    def build(params: dict[str, object]):
        return unwrap_built(
            request_builder(
                StagePayload(
                    request_id="req-layout",
                    request=OmniRequest(inputs={"audio_bytes": b"wav"}, params=params),
                    data={},
                )
            )
        )

    swift = build({AUDIO_LAYOUT_PARAM: VOXT_SWIFT_LAYOUT}).req.multimodal_inputs.mm_items[0]
    default = build({}).req.multimodal_inputs.mm_items[0]

    assert swift.feature.shape[-1] == 679
    assert swift.model_specific_data["num_audio_tokens"] == 98
    assert swift.model_specific_data[AUDIO_LAYOUT_PARAM] == VOXT_SWIFT_LAYOUT
    assert default.feature.shape[-1] == 678
    assert default.model_specific_data["num_audio_tokens"] == qwen3_asr_num_audio_tokens(678)
    assert AUDIO_LAYOUT_PARAM not in default.model_specific_data


def test_request_builder_rejects_an_unknown_layout(monkeypatch) -> None:
    monkeypatch.setattr(
        transcription,
        "load_audio",
        lambda source, **kwargs: np.zeros(1600, dtype=np.float32),
    )
    request_builder, _ = make_qwen3_asr_scheduler_adapters(
        tokenizer=FakeTokenizer(),
        max_new_tokens=32,
        feature_extractor=whisper_extractor(),
    )
    with pytest.raises(ValueError, match="audio_layout"):
        request_builder(
            StagePayload(
                request_id="req-bad-layout",
                request=OmniRequest(inputs={"audio_bytes": b"wav"}, params={AUDIO_LAYOUT_PARAM: "other"}),
                data={},
            )
        )
