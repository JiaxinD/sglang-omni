# SPDX-License-Identifier: Apache-2.0
"""Server-readable views of Voxt model directories, built without copying weights.

Plain JSON only: the supervisor builds views at every launch and must not pay
for importing a model library first.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

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
# Whisper's fixed front end: 30 s windows of 16 kHz audio, 10 ms hops.
WHISPER_SAMPLE_RATE = 16000
WHISPER_CHUNK_SECONDS = 30
WHISPER_HOP_LENGTH = 160
WHISPER_FFT_SIZE = 400


def build_whisper_hf_view(model_directory: Path, view_directory: Path) -> Path:
    """Write HF config files beside links to an mlx-whisper directory's files.

    The MLX config only carries layer sizes; the special token ids come from the
    generation config Voxt installed next to it. The view is rebuilt from
    scratch and swapped in, so a stale file never survives.
    """
    missing = [
        name
        for name in ("config.json", *WHISPER_LINKED_FILES)
        if not (model_directory / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Whisper model directory {model_directory} is missing {missing}"
        )
    else:
        pass
    mlx_config = json.loads((model_directory / "config.json").read_text())
    generation_config = json.loads(
        (model_directory / "generation_config.json").read_text()
    )
    if mlx_config["n_audio_state"] != mlx_config["n_text_state"]:
        raise ValueError("Whisper encoder and decoder widths must match")
    else:
        pass
    hf_config = {
        "model_type": "whisper",
        "architectures": ["WhisperForConditionalGeneration"],
        "torch_dtype": "float16",
        "activation_function": "gelu",
        "scale_embedding": False,
        "vocab_size": mlx_config["n_vocab"],
        "num_mel_bins": mlx_config["n_mels"],
        "d_model": mlx_config["n_audio_state"],
        "encoder_layers": mlx_config["n_audio_layer"],
        "encoder_attention_heads": mlx_config["n_audio_head"],
        "encoder_ffn_dim": WHISPER_FEED_FORWARD_MULTIPLIER
        * mlx_config["n_audio_state"],
        "decoder_layers": mlx_config["n_text_layer"],
        "decoder_attention_heads": mlx_config["n_text_head"],
        "decoder_ffn_dim": WHISPER_FEED_FORWARD_MULTIPLIER * mlx_config["n_text_state"],
        "max_source_positions": mlx_config["n_audio_ctx"],
        "max_target_positions": mlx_config["n_text_ctx"],
        "decoder_start_token_id": generation_config["decoder_start_token_id"],
        "bos_token_id": generation_config["bos_token_id"],
        "eos_token_id": generation_config["eos_token_id"],
        "pad_token_id": generation_config["pad_token_id"],
    }
    chunk_samples = WHISPER_CHUNK_SECONDS * WHISPER_SAMPLE_RATE
    preprocessor_config = {
        "feature_extractor_type": "WhisperFeatureExtractor",
        "processor_class": "WhisperProcessor",
        "feature_size": mlx_config["n_mels"],
        "sampling_rate": WHISPER_SAMPLE_RATE,
        "hop_length": WHISPER_HOP_LENGTH,
        "chunk_length": WHISPER_CHUNK_SECONDS,
        "n_fft": WHISPER_FFT_SIZE,
        "n_samples": chunk_samples,
        "nb_max_frames": chunk_samples // WHISPER_HOP_LENGTH,
        "padding_side": "right",
        "padding_value": 0.0,
        "return_attention_mask": False,
    }

    building = view_directory.with_name(f"{view_directory.name}.building-{os.getpid()}")
    shutil.rmtree(building, ignore_errors=True)
    building.mkdir(parents=True)
    (building / "config.json").write_text(json.dumps(hf_config, indent=2))
    (building / "preprocessor_config.json").write_text(
        json.dumps(preprocessor_config, indent=2)
    )
    for name in WHISPER_LINKED_FILES:
        (building / name).symlink_to((model_directory / name).resolve())
    shutil.rmtree(view_directory, ignore_errors=True)
    os.replace(building, view_directory)
    return view_directory
