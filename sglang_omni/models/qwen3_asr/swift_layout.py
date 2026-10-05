# SPDX-License-Identifier: Apache-2.0
"""Voxt's Swift Qwen3-ASR audio layout, reproduced on request.

Voxt's original MLXAudio Swift port builds the audio part of the prompt in two
ways that differ from the reference processor:

* its log-mel keeps the final centered STFT frame, which WhisperFeatureExtractor
  drops, so every clip has one more mel frame;
* its output-length formula divides the frame count by 100 with true division
  (``(lengths / 100) * 13`` on an integer MLX array), so a partial final
  100-frame chunk is credited with extra tokens. The prompt reserves that many
  placeholders, the encoder keeps every row of the padded final chunk up to that
  credit, and placeholders beyond the encoder rows stay plain ``<|audio_pad|>``.

Clients that must reproduce Voxt's original transcripts ask for this layout
with ``audio_layout=voxt_swift``; the default layout is the reference one.
"""

from __future__ import annotations

import numpy as np
import torch

from sglang_omni.client.types import (
    AUDIO_LAYOUT_PARAM,
    AUDIO_LAYOUTS,
    VOXT_SWIFT_AUDIO_LAYOUT,
)

VOXT_SWIFT_LAYOUT = VOXT_SWIFT_AUDIO_LAYOUT

# The Swift port hardcodes the 100-frame conv chunk and its 13 output rows.
_SWIFT_CHUNK_FRAMES = 100
_SWIFT_CHUNK_TOKENS = 13


def swift_output_length(frames: int) -> int:
    """Voxt's Swift getFeatExtractOutputLengths for one length, float32 as there."""
    leave = frames % _SWIFT_CHUNK_FRAMES
    feat = (leave - 1) // 2 + 1
    base = ((feat - 1) // 2 + 1 - 1) // 2 + 1
    chunks = np.float32(frames) / np.float32(_SWIFT_CHUNK_FRAMES)
    total = np.float32(base) + chunks * np.float32(_SWIFT_CHUNK_TOKENS)
    # MLXArray.item(Int32.self) truncates toward zero.
    return int(np.trunc(total))


def swift_log_mel(audio: np.ndarray, feature_extractor) -> torch.Tensor:
    """WhisperFeatureExtractor's log-mel, keeping the final STFT frame.

    Returns ``[1, n_mels, len(audio) // hop_length + 1]``.
    """
    waveform = torch.as_tensor(np.asarray(audio, dtype=np.float32))
    n_fft = int(feature_extractor.n_fft)
    stft = torch.stft(
        waveform,
        n_fft,
        int(feature_extractor.hop_length),
        window=torch.hann_window(n_fft),
        return_complex=True,
    )
    mel_filters = torch.from_numpy(np.asarray(feature_extractor.mel_filters)).float()
    log_spec = torch.clamp(mel_filters.T @ (stft.abs() ** 2), min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    return ((log_spec + 4.0) / 4.0).unsqueeze(0)


def validate_audio_layout(layout: object) -> str | None:
    if layout is None:
        return None
    else:
        pass
    if layout not in AUDIO_LAYOUTS:
        raise ValueError(
            f"Unsupported Qwen3-ASR audio_layout {layout!r}; "
            f"expected one of {sorted(AUDIO_LAYOUTS)}"
        )
    else:
        pass
    return str(layout)
