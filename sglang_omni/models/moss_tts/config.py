# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for MOSS-TTS Delay."""

from __future__ import annotations

from typing import Any, ClassVar

from sglang_omni.config import (
    EngineStageConfig,
    ModelGroup,
    PipelineConfig,
    StageConfig,
)

_PKG = "sglang_omni.models.moss_tts"
_REF_AUDIO_CACHE_MAX_ITEMS = 8192
_REF_AUDIO_CACHE_MAX_BYTES = 64 * 1024 * 1024


class MossTTSPipelineConfig(PipelineConfig):
    """MOSS-TTS Delay pipeline: preprocessing -> AR engine -> vocoder."""

    architecture: ClassVar[str] = "MossTTSDelayModel"
    requires_model_capabilities: ClassVar[bool] = True
    architecture_aliases: ClassVar[tuple[str, ...]] = (
        "MossTTSDelay",
        "MossTTSDelayForConditionalGeneration",
        "MossTTSDelayWithCodec",
        "MossTTSDelayWithCodecModel",
    )

    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "tts_engine": EngineStageConfig,
    }

    model_path: str
    stages: list[StageConfig] = [
        StageConfig(
            name="preprocessing",
            process="pipeline",
            factory=f"{_PKG}.stages.create_preprocessing_executor",
# Keep the standalone reference encoder off GPU. MOSS-TTS loads a
            # second audio-tokenizer instance for vocoding, so colocating both
            # FP32 codec copies leaves no credible runtime margin on 32 GB.
            model=ModelGroup(device="cpu", dtype="float32"),
            scheduler=SchedulerConfig(
                ref_audio_cache=True,
                ref_audio_cache_max_items=_REF_AUDIO_CACHE_MAX_ITEMS,
                ref_audio_cache_max_bytes=_REF_AUDIO_CACHE_MAX_BYTES,
            ),
            gpu=0,
            next="tts_engine",
        ),
        EngineStageConfig(
            name="tts_engine",
            process="pipeline",
            factory=f"{_PKG}.stages.create_sglang_tts_engine_executor",
            model=ModelGroup(dtype="bfloat16"),
            gpu=0,
            next="vocoder",
            stream_to=["vocoder"],
        ),
        StageConfig(
            name="vocoder",
            process="pipeline",
            factory=f"{_PKG}.stages.create_vocoder_executor",
model=ModelGroup(dtype="float32", compute_dtype="bfloat16"),
            gpu=0,
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
    ]

    def stage_factory_kwargs(self, stage_name: str) -> dict[str, Any]:
        if stage_name == "preprocessing":
            return {
                "ref_audio_cache": True,
                "ref_audio_cache_max_items": _REF_AUDIO_CACHE_MAX_ITEMS,
                "ref_audio_cache_max_bytes": _REF_AUDIO_CACHE_MAX_BYTES,
            }
        return {}

    def supports_uploaded_voice_references(self) -> bool:
        return True


EntryClass = MossTTSPipelineConfig
