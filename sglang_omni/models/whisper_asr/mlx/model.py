# SPDX-License-Identifier: Apache-2.0
"""Native MLX Whisper encoder-decoder for the SGLang MLX runner."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import KVCache
from transformers import WhisperConfig

from sglang_omni.model_runner.whisper_encoder_mlx import (
    WHISPER_LAYER_NORM_EPS,
    WhisperEncoder,
    WhisperEncoderConfig,
)

# mlx-community checkpoints use the original OpenAI module names.
MLX_COMMUNITY_RENAMES = (
    ("encoder.blocks.", "encoder.layers."),
    ("decoder.blocks.", "decoder.layers."),
    ("encoder.ln_post.", "encoder.layer_norm."),
    ("decoder.ln.", "decoder.layer_norm."),
    ("decoder.token_embedding.", "decoder.embed_tokens."),
    ("decoder.positional_embedding", "decoder.embed_positions.weight"),
    (".cross_attn_ln.", ".encoder_attn_layer_norm."),
    (".attn_ln.", ".self_attn_layer_norm."),
    (".mlp_ln.", ".final_layer_norm."),
    (".cross_attn.", ".encoder_attn."),
    (".attn.", ".self_attn."),
    (".query.", ".q_proj."),
    (".key.", ".k_proj."),
    (".value.", ".v_proj."),
    (".out.", ".out_proj."),
    (".mlp1.", ".fc1."),
    (".mlp2.", ".fc2."),
)
SINUSOID_MAX_TIMESCALE = 10000.0


@dataclass(kw_only=True, frozen=True)
class WhisperDimensions:
    encoder: WhisperEncoderConfig
    vocab_size: int
    decoder_layers: int
    decoder_attention_heads: int
    decoder_ffn_dim: int
    max_target_positions: int


def decoder_dimensions_from_hf_config(config: WhisperConfig) -> WhisperDimensions:
    return WhisperDimensions(
        encoder=WhisperEncoderConfig(
            num_mel_bins=config.num_mel_bins,
            d_model=config.d_model,
            encoder_layers=config.encoder_layers,
            encoder_attention_heads=config.encoder_attention_heads,
            encoder_ffn_dim=config.encoder_ffn_dim,
            max_source_positions=config.max_source_positions,
            # The reference feeds float32 log-mel to float16 weights, so its
            # activations run in float32; keep that precision.
            casts_input_to_weight_dtype=False,
        ),
        vocab_size=config.vocab_size,
        decoder_layers=config.decoder_layers,
        decoder_attention_heads=config.decoder_attention_heads,
        decoder_ffn_dim=config.decoder_ffn_dim,
        max_target_positions=config.max_target_positions,
    )


def whisper_sinusoids(length: int, channels: int) -> mx.array:
    """The fixed encoder position table that OpenAI checkpoints omit."""
    timescale_step = math.log(SINUSOID_MAX_TIMESCALE) / (channels // 2 - 1)
    inverse_timescales = mx.exp(-timescale_step * mx.arange(channels // 2))
    angles = mx.arange(length)[:, None] * inverse_timescales[None, :]
    return mx.concatenate([mx.sin(angles), mx.cos(angles)], axis=1)


def identity_rope(x: mx.array, offset: int = 0) -> mx.array:
    """Whisper decoders use learned absolute positions instead of rotation."""
    del offset
    return x


class WhisperDecoderSelfAttention(nn.Module):
    """Causal attention shaped to the SGLang MLX attention-module contract."""

    def __init__(self, width: int, num_heads: int) -> None:
        super().__init__()
        self.n_heads = num_heads
        self.n_kv_heads = num_heads
        self.head_dim = width // num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(width, width, bias=True)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=True)
        self.o_proj = nn.Linear(width, width, bias=True)
        self.rope = identity_rope

    def __call__(
        self,
        hidden_states: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        batch_size, token_count, width = hidden_states.shape

        def split_heads(projected: mx.array) -> mx.array:
            return projected.reshape(
                batch_size, token_count, self.n_heads, self.head_dim
            ).transpose(0, 2, 1, 3)

        queries = split_heads(self.q_proj(hidden_states))
        keys = split_heads(self.k_proj(hidden_states))
        values = split_heads(self.v_proj(hidden_states))
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)
        else:
            pass
        attended = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        return self.o_proj(
            attended.transpose(0, 2, 1, 3).reshape(batch_size, token_count, width)
        )


class WhisperCrossAttention(nn.Module):
    def __init__(self, width: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.q_proj = nn.Linear(width, width, bias=True)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=True)
        self.out_proj = nn.Linear(width, width, bias=True)

    def split_heads(self, projected: mx.array) -> mx.array:
        batch_size, token_count, _width = projected.shape
        return projected.reshape(
            batch_size, token_count, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

    def encoder_states(self, encoder_output: mx.array) -> tuple[mx.array, mx.array]:
        return (
            self.split_heads(self.k_proj(encoder_output)),
            self.split_heads(self.v_proj(encoder_output)),
        )

    def __call__(
        self, hidden_states: mx.array, encoder_states: tuple[mx.array, mx.array]
    ) -> mx.array:
        batch_size, token_count, width = hidden_states.shape
        keys, values = encoder_states
        attended = mx.fast.scaled_dot_product_attention(
            self.split_heads(self.q_proj(hidden_states)),
            keys,
            values,
            scale=self.head_dim**-0.5,
        )
        return self.out_proj(
            attended.transpose(0, 2, 1, 3).reshape(batch_size, token_count, width)
        )


class WhisperDecoderLayer(nn.Module):
    def __init__(self, width: int, num_heads: int, ffn_width: int) -> None:
        super().__init__()
        self.self_attn = WhisperDecoderSelfAttention(width, num_heads)
        self.self_attn_layer_norm = nn.LayerNorm(width, eps=WHISPER_LAYER_NORM_EPS)
        self.encoder_attn = WhisperCrossAttention(width, num_heads)
        self.encoder_attn_layer_norm = nn.LayerNorm(width, eps=WHISPER_LAYER_NORM_EPS)
        self.fc1 = nn.Linear(width, ffn_width)
        self.fc2 = nn.Linear(ffn_width, width)
        self.final_layer_norm = nn.LayerNorm(width, eps=WHISPER_LAYER_NORM_EPS)

    def __call__(
        self,
        hidden_states: mx.array,
        mask: mx.array | str | None,
        cache: KVCache | None,
        encoder_states: tuple[mx.array, mx.array],
    ) -> mx.array:
        hidden_states = hidden_states + self.self_attn(
            self.self_attn_layer_norm(hidden_states), mask=mask, cache=cache
        )
        hidden_states = hidden_states + self.encoder_attn(
            self.encoder_attn_layer_norm(hidden_states), encoder_states
        )
        return hidden_states + self.fc2(
            nn.gelu(self.fc1(self.final_layer_norm(hidden_states)))
        )


class WhisperTextDecoder(nn.Module):
    def __init__(self, dimensions: WhisperDimensions) -> None:
        super().__init__()
        width = dimensions.encoder.d_model
        self.embed_tokens = nn.Embedding(dimensions.vocab_size, width)
        self.embed_positions = nn.Embedding(dimensions.max_target_positions, width)
        self.layers = [
            WhisperDecoderLayer(
                width, dimensions.decoder_attention_heads, dimensions.decoder_ffn_dim
            )
            for _ in range(dimensions.decoder_layers)
        ]
        self.layer_norm = nn.LayerNorm(width, eps=WHISPER_LAYER_NORM_EPS)


class WhisperMlxModel(nn.Module):
    def __init__(self, dimensions: WhisperDimensions) -> None:
        super().__init__()
        self.dimensions = dimensions
        self.encoder = WhisperEncoder(dimensions.encoder)
        self.decoder = WhisperTextDecoder(dimensions)

    @property
    def layers(self) -> list[WhisperDecoderLayer]:
        """Decoder layers, where the SGLang MLX runner discovers attention."""
        return self.decoder.layers

    def cross_attention_states(
        self, encoder_output: mx.array
    ) -> list[tuple[mx.array, mx.array]]:
        return [
            layer.encoder_attn.encoder_states(encoder_output)
            for layer in self.decoder.layers
        ]

    def decode(
        self,
        input_ids: mx.array,
        *,
        cache: Sequence[KVCache],
        cross_states: list[tuple[mx.array, mx.array]],
    ) -> mx.array:
        """Logits for every input position, continuing the self-attention cache."""
        position_offset = cache[0].offset
        token_count = input_ids.shape[1]
        hidden_states = self.decoder.embed_tokens(input_ids) + (
            self.decoder.embed_positions.weight[
                position_offset : position_offset + token_count
            ]
        )
        mask = create_attention_mask(hidden_states, cache[0])
        for layer, layer_cache, encoder_states in zip(
            self.decoder.layers, cache, cross_states
        ):
            hidden_states = layer(hidden_states, mask, layer_cache, encoder_states)
        return self.decoder.embed_tokens.as_linear(
            self.decoder.layer_norm(hidden_states)
        )

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.decoder.layers]

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Accept both the HF and the mlx-community tensor naming."""
        is_hf_layout = any(name.startswith("model.") for name in weights)
        sanitized: dict[str, mx.array] = {}
        for name, weight in weights.items():
            if name in ("alignment_heads", "proj_out.weight"):
                continue
            elif is_hf_layout:
                renamed = name.removeprefix("model.")
                if renamed in ("encoder.conv1.weight", "encoder.conv2.weight"):
                    weight = WhisperEncoder.torch_conv_weight(weight)
                else:
                    pass
            else:
                renamed = name
                for checkpoint_name, module_name in MLX_COMMUNITY_RENAMES:
                    renamed = renamed.replace(checkpoint_name, module_name)
            if (
                renamed.startswith("decoder.layers.")
                and ".self_attn.out_proj." in renamed
            ):
                renamed = renamed.replace(".self_attn.out_proj.", ".self_attn.o_proj.")
            else:
                pass
            sanitized[renamed] = weight
        if "encoder.embed_positions.weight" not in sanitized:
            encoder = self.dimensions.encoder
            sanitized["encoder.embed_positions.weight"] = whisper_sinusoids(
                encoder.max_source_positions, encoder.d_model
            )
        else:
            pass
        return sanitized
