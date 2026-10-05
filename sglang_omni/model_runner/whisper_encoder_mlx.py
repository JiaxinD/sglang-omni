# SPDX-License-Identifier: Apache-2.0
"""Whisper audio encoder in MLX, shared by the native MLX ASR models."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

WHISPER_LAYER_NORM_EPS = 1e-5


@dataclass(kw_only=True, frozen=True)
class WhisperEncoderConfig:
    num_mel_bins: int
    d_model: int
    encoder_layers: int
    encoder_attention_heads: int
    encoder_ffn_dim: int
    max_source_positions: int

    @classmethod
    def from_hf_config(cls, config: dict[str, object]) -> "WhisperEncoderConfig":
        return cls(
            num_mel_bins=int(config["num_mel_bins"]),
            d_model=int(config["d_model"]),
            encoder_layers=int(config["encoder_layers"]),
            encoder_attention_heads=int(config["encoder_attention_heads"]),
            encoder_ffn_dim=int(config["encoder_ffn_dim"]),
            max_source_positions=int(config["max_source_positions"]),
        )


class WhisperEncoderAttention(nn.Module):
    def __init__(self, config: WhisperEncoderConfig) -> None:
        super().__init__()
        self.num_heads = config.encoder_attention_heads
        self.head_dim = config.d_model // self.num_heads
        assert self.head_dim * self.num_heads == config.d_model
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=True)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=True)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=True)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        batch_size, frame_count, width = hidden_states.shape

        def split_heads(projected: mx.array) -> mx.array:
            return projected.reshape(
                batch_size, frame_count, self.num_heads, self.head_dim
            ).transpose(0, 2, 1, 3)

        attended = mx.fast.scaled_dot_product_attention(
            split_heads(self.q_proj(hidden_states)),
            split_heads(self.k_proj(hidden_states)),
            split_heads(self.v_proj(hidden_states)),
            scale=self.scale,
        )
        return self.out_proj(
            attended.transpose(0, 2, 1, 3).reshape(batch_size, frame_count, width)
        )


class WhisperEncoderLayer(nn.Module):
    def __init__(self, config: WhisperEncoderConfig) -> None:
        super().__init__()
        self.self_attn = WhisperEncoderAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(
            config.d_model, eps=WHISPER_LAYER_NORM_EPS
        )
        self.fc1 = nn.Linear(config.d_model, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, config.d_model)
        self.final_layer_norm = nn.LayerNorm(config.d_model, eps=WHISPER_LAYER_NORM_EPS)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = hidden_states + self.self_attn(
            self.self_attn_layer_norm(hidden_states)
        )
        return hidden_states + self.fc2(
            nn.gelu(self.fc1(self.final_layer_norm(hidden_states)))
        )


class WhisperEncoder(nn.Module):
    """Encodes fixed 30-second log-mel windows; parameter names follow HF."""

    def __init__(self, config: WhisperEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.conv1 = nn.Conv1d(config.num_mel_bins, config.d_model, 3, padding=1)
        self.conv2 = nn.Conv1d(config.d_model, config.d_model, 3, stride=2, padding=1)
        self.embed_positions = nn.Embedding(config.max_source_positions, config.d_model)
        self.layers = [
            WhisperEncoderLayer(config) for _ in range(config.encoder_layers)
        ]
        self.layer_norm = nn.LayerNorm(config.d_model, eps=WHISPER_LAYER_NORM_EPS)

    def __call__(self, input_features: mx.array) -> mx.array:
        """Map [chunks, mel_bins, frames] features to [chunks, frames / 2, d_model]."""
        expected_frames = 2 * self.config.max_source_positions
        if input_features.shape[-1] != expected_frames:
            raise ValueError(
                f"Whisper encoder expects {expected_frames} mel frames, got "
                f"{input_features.shape[-1]}"
            )
        else:
            pass
        weight_dtype = self.conv1.weight.dtype
        hidden_states = input_features.transpose(0, 2, 1).astype(weight_dtype)
        hidden_states = nn.gelu(self.conv1(hidden_states))
        hidden_states = nn.gelu(self.conv2(hidden_states))
        hidden_states = hidden_states + self.embed_positions.weight
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.layer_norm(hidden_states)

    @staticmethod
    def torch_conv_weight(weight: mx.array) -> mx.array:
        """Reorder a PyTorch Conv1d [out, in, kernel] weight for MLX."""
        return weight.transpose(0, 2, 1)
