# SPDX-License-Identifier: Apache-2.0
"""SGLang MLX runner extension for MOSS-Transcribe-Diarize audio prefill."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import mlx.core as mx

from sglang_omni.model_runner.audio_mlx import AudioMlxModelRunner

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
else:
    pass

logger = logging.getLogger(__name__)


class MossTranscribeDiarizeMlxModelRunner(AudioMlxModelRunner):
    """Encodes every 30-second chunk inside prefill and fills split audio spans."""

    model_name = "MOSS-Transcribe-Diarize"

    def audio_prefill_inputs(
        self,
        req: "Req",
        token_ids: list[int],
    ) -> tuple[mx.array, mx.array]:
        item = self.audio_item(req)
        if item.feature is None:
            raise ValueError(f"{self.model_name} MLX prefill requires audio features")
        else:
            pass
        chunk_metadata = item.model_specific_data
        if "audio_feature_lengths" not in chunk_metadata:
            raise ValueError(
                f"{self.model_name} MLX prefill requires audio_feature_lengths"
            )
        else:
            pass
        audio_feature_lengths = [
            int(length) for length in chunk_metadata["audio_feature_lengths"].tolist()
        ]
        chunk_mapping = chunk_metadata.get("audio_chunk_mapping")
        audio_chunk_mapping = (
            [0] * len(audio_feature_lengths)
            if chunk_mapping is None
            else [int(index) for index in chunk_mapping.tolist()]
        )
        input_ids = mx.array(
            [self.normalize_audio_token_ids(req, token_ids)], dtype=mx.int32
        )
        audio_features = self.model.get_audio_features(
            mx.array(self.to_numpy(item.feature)),
            audio_feature_lengths=audio_feature_lengths,
            audio_chunk_mapping=audio_chunk_mapping,
        )
        return input_ids, self.model.build_inputs_embeds(input_ids, audio_features)

    def _load_model(self) -> None:  # noqa: leading-underscore  # SGLang hook name
        from mlx_lm.utils import load_model
        from sglang.srt.hardware_backend.mlx.remote_code_gate import (
            ensure_remote_code_allowed,
            resolve_model_directory,
        )

        from .model import ModelConfig, MossTranscribeDiarizeModel

        model_path = resolve_model_directory(self.model_path, revision=self.revision)
        ensure_remote_code_allowed(model_path, self.trust_remote_code)
        logger.info(f"Loading native MLX MOSS-Transcribe-Diarize model: {model_path}")
        started = time.perf_counter()
        self.model, _config = load_model(
            model_path,
            get_model_classes=lambda config: (MossTranscribeDiarizeModel, ModelConfig),
        )
        logger.info(
            "Loaded native MLX MOSS-Transcribe-Diarize model in "
            f"{time.perf_counter() - started:.2f}s"
        )


def make_moss_transcribe_diarize_mlx_runner_class():
    """Build the extension class after the MLX backend has been selected."""
    from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner

    class MossTranscribeDiarizeMlxRunner(
        MossTranscribeDiarizeMlxModelRunner, MlxModelRunner
    ):
        pass

    return MossTranscribeDiarizeMlxRunner
