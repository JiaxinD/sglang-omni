# SPDX-License-Identifier: Apache-2.0
"""Server-readable views of Voxt model directories, built without copying weights."""

from __future__ import annotations

import json
from pathlib import Path

from transformers import GenerationConfig, WhisperConfig, WhisperFeatureExtractor

# Voxt installs the MLX weights plus these tokenizer assets in one directory.
WHISPER_LINKED_FILES = (
    "weights.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "normalizer.json",
    "generation_config.json",
)
# Every Whisper size uses a feed-forward width of four times the model width.
WHISPER_FEED_FORWARD_MULTIPLIER = 4


def build_whisper_hf_view(model_directory: Path, view_directory: Path) -> Path:
    """Write HF config files beside links to an mlx-whisper directory's files.

    The MLX config only carries layer sizes; the special token ids come from the
    generation config Voxt installed next to it.
    """
    missing = [
        name for name in WHISPER_LINKED_FILES if not (model_directory / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Whisper model directory {model_directory} is missing {missing}"
        )
    else:
        pass
    mlx_config = json.loads((model_directory / "config.json").read_text())
    generation_config = GenerationConfig.from_pretrained(model_directory)
    hf_config = WhisperConfig(
        vocab_size=mlx_config["n_vocab"],
        num_mel_bins=mlx_config["n_mels"],
        d_model=mlx_config["n_audio_state"],
        encoder_layers=mlx_config["n_audio_layer"],
        encoder_attention_heads=mlx_config["n_audio_head"],
        encoder_ffn_dim=WHISPER_FEED_FORWARD_MULTIPLIER * mlx_config["n_audio_state"],
        decoder_layers=mlx_config["n_text_layer"],
        decoder_attention_heads=mlx_config["n_text_head"],
        decoder_ffn_dim=WHISPER_FEED_FORWARD_MULTIPLIER * mlx_config["n_text_state"],
        max_source_positions=mlx_config["n_audio_ctx"],
        max_target_positions=mlx_config["n_text_ctx"],
        decoder_start_token_id=generation_config.decoder_start_token_id,
        bos_token_id=generation_config.bos_token_id,
        eos_token_id=generation_config.eos_token_id,
        pad_token_id=generation_config.pad_token_id,
        architectures=["WhisperForConditionalGeneration"],
        torch_dtype="float16",
    )
    if mlx_config["n_audio_state"] != mlx_config["n_text_state"]:
        raise ValueError("Whisper encoder and decoder widths must match")
    else:
        pass

    view_directory.mkdir(parents=True, exist_ok=True)
    hf_config.save_pretrained(view_directory)
    feature_extractor = WhisperFeatureExtractor(feature_size=mlx_config["n_mels"])
    feature_extractor.processor_class = "WhisperProcessor"
    feature_extractor.save_pretrained(view_directory)
    for name in WHISPER_LINKED_FILES:
        link = view_directory / name
        if link.is_symlink() or link.exists():
            link.unlink()
        else:
            pass
        link.symlink_to((model_directory / name).resolve())
    return view_directory
