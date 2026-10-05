# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace

import numpy as np
import pytest
import torch

mx = pytest.importorskip("mlx.core")

from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: E402
from transformers.models.whisper.configuration_whisper import (  # noqa: E402
    WhisperConfig,
)
from transformers.models.whisper.modeling_whisper import (  # noqa: E402
    WhisperEncoder as TorchWhisperEncoder,
)

from sglang_omni.models.moss_transcribe_diarize.mlx.model import (  # noqa: E402
    ModelConfig,
    MossTranscribeDiarizeModel,
)
from sglang_omni.models.moss_transcribe_diarize.mlx.runner import (  # noqa: E402
    MossTranscribeDiarizeMlxModelRunner,
)

AUDIO_TOKEN_ID = 60
MERGE_SIZE = 4
MEL_BINS = 8
SOURCE_POSITIONS = 16
CHUNK_FRAMES = 2 * SOURCE_POSITIONS


@pytest.fixture(autouse=True)
def exact_mlx_device() -> Iterator[None]:
    # Metal float32 kernels accumulate differently from Torch CPU; the CPU
    # device isolates graph parity from that precision difference.
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def tiny_configs() -> tuple[dict[str, object], dict[str, object]]:
    audio_config = {
        "num_mel_bins": MEL_BINS,
        "d_model": 8,
        "encoder_layers": 2,
        "encoder_attention_heads": 2,
        "encoder_ffn_dim": 16,
        "max_source_positions": SOURCE_POSITIONS,
    }
    text_config = {
        "model_type": "qwen3",
        "vocab_size": 64,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1_000_000,
        "tie_word_embeddings": True,
    }
    return audio_config, text_config


class TorchReference(torch.nn.Module):
    """The checkpoint's reference graph assembled from transformers modules."""

    def __init__(self) -> None:
        super().__init__()
        audio_config, text_config = tiny_configs()
        self.whisper_encoder = TorchWhisperEncoder(WhisperConfig(**audio_config))
        hidden_size = int(text_config["hidden_size"])
        self.vq_adaptor = torch.nn.Module()
        self.vq_adaptor.layers = torch.nn.Sequential(
            torch.nn.Linear(int(audio_config["d_model"]) * MERGE_SIZE, hidden_size),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.LayerNorm(hidden_size, eps=1e-6),
        )
        self.language_model = Qwen3ForCausalLM(
            Qwen3Config(**{k: v for k, v in text_config.items() if k != "model_type"})
        )

    def audio_features(
        self, input_features: torch.Tensor, token_lengths: list[int]
    ) -> torch.Tensor:
        encoded = self.whisper_encoder(input_features).last_hidden_state
        trimmed = torch.cat(
            [
                encoded[index : index + 1, : length * MERGE_SIZE]
                for index, length in enumerate(token_lengths)
            ],
            dim=1,
        )
        batch_size, frame_count, width = trimmed.shape
        kept = (frame_count // MERGE_SIZE) * MERGE_SIZE
        merged = trimmed[:, :kept].reshape(
            batch_size, kept // MERGE_SIZE, width * MERGE_SIZE
        )
        return self.vq_adaptor.layers(merged).squeeze(0)

    def checkpoint_weights(self) -> dict[str, mx.array]:
        weights: dict[str, mx.array] = {}
        for name, tensor in self.state_dict().items():
            if name.startswith("language_model.lm_head."):
                continue
            elif name.startswith("language_model.model."):
                key = "model.language_model." + name[len("language_model.model.") :]
            else:
                key = "model." + name
            weights[key] = mx.array(tensor.detach().float().numpy())
        return weights


def build_pair() -> tuple[TorchReference, MossTranscribeDiarizeModel]:
    torch.manual_seed(0)
    reference = TorchReference().eval()
    audio_config, text_config = tiny_configs()
    model = MossTranscribeDiarizeModel(
        ModelConfig(
            audio_config=audio_config,
            text_config=text_config,
            audio_merge_size=MERGE_SIZE,
            adaptor_input_dim=int(audio_config["d_model"]) * MERGE_SIZE,
            audio_token_id=AUDIO_TOKEN_ID,
        )
    )
    weights = model.sanitize(reference.checkpoint_weights())
    model.load_weights(list(weights.items()), strict=True)
    return reference, model


def test_checkpoint_sanitize_maps_prefixes_and_converts_conv_layout() -> None:
    reference, model = build_pair()
    sanitized = model.sanitize(reference.checkpoint_weights())

    torch_conv = reference.whisper_encoder.conv1.weight.detach().numpy()
    assert sanitized["whisper_encoder.conv1.weight"].shape == (8, 3, MEL_BINS)
    np.testing.assert_allclose(
        np.array(sanitized["whisper_encoder.conv1.weight"]),
        torch_conv.transpose(0, 2, 1),
    )
    assert "language_model.model.embed_tokens.weight" in sanitized
    assert "vq_adaptor.layers.3.weight" in sanitized
    assert not any(key.startswith("model.") for key in sanitized)


def test_audio_features_match_the_reference_graph() -> None:
    reference, model = build_pair()
    rng = np.random.default_rng(1)
    input_features = rng.standard_normal((2, MEL_BINS, CHUNK_FRAMES)).astype(np.float32)
    token_lengths = [4, 2]

    with torch.no_grad():
        expected = reference.audio_features(
            torch.from_numpy(input_features), token_lengths
        ).numpy()
    actual = model.get_audio_features(
        mx.array(input_features),
        audio_feature_lengths=token_lengths,
        audio_chunk_mapping=[0, 0],
    )

    assert actual.shape == expected.shape == (6, 16)
    np.testing.assert_allclose(np.array(actual), expected, rtol=1e-5, atol=1e-5)


def test_prefill_logits_match_the_reference_with_split_audio_spans() -> None:
    reference, model = build_pair()
    rng = np.random.default_rng(2)
    input_features = rng.standard_normal((1, MEL_BINS, CHUNK_FRAMES)).astype(np.float32)
    # A time marker splits the four audio rows, as in the checkpoint's template.
    input_ids = [1, 2, AUDIO_TOKEN_ID, AUDIO_TOKEN_ID, 7, 8]
    input_ids += [AUDIO_TOKEN_ID, AUDIO_TOKEN_ID, 3]

    with torch.no_grad():
        features = reference.audio_features(torch.from_numpy(input_features), [4])
        embeddings = reference.language_model.get_input_embeddings()(
            torch.tensor([input_ids])
        )
        audio_mask = torch.tensor(input_ids) == AUDIO_TOKEN_ID
        embeddings[0, audio_mask] = features
        expected = reference.language_model(inputs_embeds=embeddings).logits[0, -1]

    audio_features = model.get_audio_features(
        mx.array(input_features), audio_feature_lengths=[4], audio_chunk_mapping=[0]
    )
    inputs_embeds = model.build_inputs_embeds(
        mx.array([input_ids], dtype=mx.int32), audio_features
    )
    actual = model.forward_last_logits(inputs_embeds, cache=model.make_cache())

    np.testing.assert_allclose(
        np.array(actual[0, -1]), expected.numpy(), rtol=1e-5, atol=1e-5
    )


def test_cached_decode_matches_the_uncached_forward() -> None:
    _reference, model = build_pair()
    input_ids = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
    full_logits = model(input_ids)

    cache = model.make_cache()
    model(input_ids[:, :4], cache=cache)
    step_logits = model(input_ids[:, 4:], cache=cache)

    np.testing.assert_allclose(
        np.array(step_logits[0, -1]), np.array(full_logits[0, -1]), atol=1e-4
    )


def test_build_inputs_embeds_rejects_a_placeholder_count_mismatch() -> None:
    _reference, model = build_pair()
    with pytest.raises(ValueError, match="3 placeholders, 2 features"):
        model.build_inputs_embeds(
            mx.array([[1, AUDIO_TOKEN_ID, 5, AUDIO_TOKEN_ID, AUDIO_TOKEN_ID]]),
            mx.zeros((2, 16)),
        )


def test_runner_restores_placeholders_and_reads_chunk_metadata() -> None:
    _reference, model = build_pair()
    runner = object.__new__(MossTranscribeDiarizeMlxModelRunner)
    runner.model = model
    pad_value = 1_000_001
    item = SimpleNamespace(
        feature=torch.zeros((2, MEL_BINS, CHUNK_FRAMES)),
        model_specific_data={
            "audio_feature_lengths": torch.tensor([4, 1]),
            "audio_chunk_mapping": torch.tensor([0, 0]),
        },
        pad_value=pad_value,
    )
    req = SimpleNamespace(
        multimodal_inputs=SimpleNamespace(
            audio_token_id=AUDIO_TOKEN_ID, mm_items=[item]
        )
    )
    token_ids = [1, pad_value, pad_value, pad_value, 9, pad_value, pad_value, 2]

    input_ids, embeddings = runner.audio_prefill_inputs(req, token_ids)

    mx.eval(embeddings)
    assert input_ids.tolist() == [
        [1, AUDIO_TOKEN_ID, AUDIO_TOKEN_ID, AUDIO_TOKEN_ID, 9]
        + [AUDIO_TOKEN_ID, AUDIO_TOKEN_ID, 2]
    ]
    assert embeddings.shape == (1, 8, 16)


def test_runner_rejects_audio_without_feature_lengths() -> None:
    _reference, model = build_pair()
    runner = object.__new__(MossTranscribeDiarizeMlxModelRunner)
    runner.model = model
    item = SimpleNamespace(
        feature=torch.zeros((1, MEL_BINS, CHUNK_FRAMES)),
        model_specific_data={},
        pad_value=1_000_001,
    )
    req = SimpleNamespace(
        multimodal_inputs=SimpleNamespace(
            audio_token_id=AUDIO_TOKEN_ID, mm_items=[item]
        )
    )

    with pytest.raises(ValueError, match="audio_feature_lengths"):
        runner.audio_prefill_inputs(req, [1, 1_000_001, 2])
