# SPDX-License-Identifier: Apache-2.0
"""Native MLX MOSS-Transcribe-Diarize: Whisper encoder, adaptor, Qwen3 decoder."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import qwen3
from mlx_lm.models.base import BaseModelArgs
from mlx_lm.models.cache import KVCache

from sglang_omni.model_runner.whisper_encoder_mlx import (
    WhisperEncoder,
    WhisperEncoderConfig,
)

CHECKPOINT_PREFIX_RENAMES = (
    ("model.language_model.", "language_model.model."),
    ("model.whisper_encoder.", "whisper_encoder."),
    ("model.vq_adaptor.", "vq_adaptor."),
)
# Bounds the transient attention memory of long recordings on unified memory.
ENCODER_CHUNKS_PER_EVAL = 4


@dataclass
class ModelConfig(BaseModelArgs):
    audio_config: dict[str, object]
    text_config: dict[str, object]
    audio_merge_size: int
    adaptor_input_dim: int
    audio_token_id: int


class VQAdaptor(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, norm_eps: float) -> None:
        super().__init__()
        # A plain list keeps the checkpoint's layers.{0,2,3} parameter paths.
        self.layers = [
            nn.Linear(input_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size, eps=norm_eps),
        ]

    def __call__(self, merged_features: mx.array) -> mx.array:
        for layer in self.layers:
            merged_features = layer(merged_features)
        return merged_features


class MossTranscribeDiarizeModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        text_args = qwen3.ModelArgs.from_dict(config.text_config)
        self.whisper_encoder = WhisperEncoder(
            # The reference casts MOSS features to the encoder weight dtype.
            WhisperEncoderConfig.from_hf_config(
                config.audio_config, casts_input_to_weight_dtype=True
            )
        )
        self.vq_adaptor = VQAdaptor(
            config.adaptor_input_dim, text_args.hidden_size, text_args.rms_norm_eps
        )
        self.language_model = qwen3.Model(text_args)

    def get_audio_features(
        self,
        input_features: mx.array,
        *,
        audio_feature_lengths: list[int],
        audio_chunk_mapping: list[int],
    ) -> mx.array:
        """Encode 30-second chunks and return one adapted row per audio token."""
        chunk_count = input_features.shape[0]
        if len(audio_feature_lengths) != chunk_count:
            raise ValueError(
                "audio_feature_lengths must contain one length per input_features "
                f"chunk: got {len(audio_feature_lengths)} for {chunk_count} chunks"
            )
        else:
            pass
        if len(audio_chunk_mapping) != chunk_count:
            raise ValueError(
                "audio_chunk_mapping must contain one index per input_features "
                f"chunk: got {len(audio_chunk_mapping)} for {chunk_count} chunks"
            )
        else:
            pass
        merge_size = self.config.audio_merge_size
        encoded_chunks: list[mx.array] = []
        for start in range(0, chunk_count, ENCODER_CHUNKS_PER_EVAL):
            encoded = self.whisper_encoder(
                input_features[start : start + ENCODER_CHUNKS_PER_EVAL]
            )
            mx.eval(encoded)
            encoded_chunks.extend(
                encoded[offset, : audio_feature_lengths[start + offset] * merge_size]
                for offset in range(encoded.shape[0])
            )

        adapted: list[mx.array] = []
        for audio_index in sorted(set(audio_chunk_mapping)):
            frames = mx.concatenate(
                [
                    encoded_chunks[chunk_index]
                    for chunk_index, owner in enumerate(audio_chunk_mapping)
                    if owner == audio_index
                ],
                axis=0,
            )
            kept_frames = (frames.shape[0] // merge_size) * merge_size
            merged = frames[:kept_frames].reshape(
                kept_frames // merge_size, frames.shape[1] * merge_size
            )
            adapted.append(self.vq_adaptor(merged))
        return mx.concatenate(adapted, axis=0)

    def build_inputs_embeds(
        self, input_ids: mx.array, audio_features: mx.array
    ) -> mx.array:
        """Scatter audio rows into every placeholder; time markers split the span."""
        if input_ids.shape[0] != 1:
            raise ValueError(
                "MOSS-Transcribe-Diarize MLX audio prefill supports one request"
            )
        else:
            pass
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)
        token_ids = input_ids[0].tolist()
        audio_positions = mx.array(
            [
                index
                for index, token_id in enumerate(token_ids)
                if token_id == self.config.audio_token_id
            ],
            dtype=mx.int32,
        )
        if audio_positions.size != audio_features.shape[0]:
            raise ValueError(
                "MOSS-Transcribe-Diarize audio placeholder and feature counts "
                f"differ: {audio_positions.size} placeholders, "
                f"{audio_features.shape[0]} features"
            )
        else:
            pass
        inputs_embeds[0, audio_positions] = audio_features.astype(inputs_embeds.dtype)
        return inputs_embeds

    def forward_last_logits(
        self,
        inputs_embeds: mx.array,
        cache: Sequence[KVCache | None] | None = None,
    ) -> mx.array:
        hidden_states = self.language_model.model(
            None, cache=cache, input_embeddings=inputs_embeds
        )
        return self.language_model.model.embed_tokens.as_linear(
            hidden_states[:, -1:, :]
        )

    def __call__(
        self,
        input_ids: mx.array,
        cache: Sequence[KVCache | None] | None = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        return self.language_model(
            input_ids, cache=cache, input_embeddings=input_embeddings
        )

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.language_model.layers]

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        sanitized: dict[str, mx.array] = {}
        for name, weight in weights.items():
            renamed = name
            for checkpoint_prefix, module_prefix in CHECKPOINT_PREFIX_RENAMES:
                if name.startswith(checkpoint_prefix):
                    renamed = module_prefix + name[len(checkpoint_prefix) :]
                    break
                else:
                    pass
            if renamed == name:
                raise ValueError(f"Unexpected MOSS-Transcribe-Diarize weight {name}")
            elif renamed in (
                "whisper_encoder.conv1.weight",
                "whisper_encoder.conv2.weight",
            ):
                sanitized[renamed] = WhisperEncoder.torch_conv_weight(weight)
            else:
                sanitized[renamed] = weight
        return sanitized


Model = MossTranscribeDiarizeModel
ModelArgs = ModelConfig
