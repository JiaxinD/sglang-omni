# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Higgs TTS (V1)."""

from __future__ import annotations

from typing import Any, ClassVar

from sglang_omni.config import (
    EngineStageConfig,
    ModelGroup,
    PipelineConfig,
    SchedulerConfig,
    StageConfig,
)
from sglang_omni.utils.cpu import bounded_intraop_threads

_PKG = "sglang_omni.models.higgs_tts"
_PREPROCESS_MAX_WORKERS = 2


class HiggsTtsPipelineConfig(PipelineConfig):
    """4-stage TTS pipeline: preprocessing → audio_encoder → tts_engine → vocoder.

    Preprocessing normalizes text/reference inputs; audio_encoder codec-encodes
    raw reference audio and builds the prompt; tts_engine composes the delayed
    reference-code embeddings at ``-100`` placeholder positions and drives the
    SGLang AR loop; vocoder reverses the delay pattern and decodes the waveform.
    """

    architecture: ClassVar[str] = "HiggsMultimodalQwen3ForConditionalGeneration"
    requires_model_capabilities: ClassVar[bool] = True

    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "tts_engine": EngineStageConfig,
    }

    model_path: str
    stages: list[StageConfig] = [
        StageConfig(
            name="preprocessing",
            process="tts_frontend",
            factory=f"{_PKG}.stages.create_preprocessing_executor",
            scheduler=SchedulerConfig(max_concurrency=_PREPROCESS_MAX_WORKERS),
            next="audio_encoder",
        ),
        StageConfig(
            name="audio_encoder",
            process="tts_frontend",
            factory=f"{_PKG}.stages.create_audio_encoder_executor",
            model=ModelGroup(device="cuda"),
            gpu=0,
            gpu_memory_fraction=0.03,
            next="tts_engine",
        ),
        EngineStageConfig(
            name="tts_engine",
            process="pipeline",
            factory=f"{_PKG}.stages.create_sglang_tts_engine_executor",
            model=ModelGroup(device="cuda", max_new_tokens=2048),
            scheduler=SchedulerConfig(enable_async_decode=True),
            gpu=0,
            gpu_memory_fraction=0.85,
            next="vocoder",
            stream_to=["vocoder"],
        ),
        StageConfig(
            name="vocoder",
            # Keep the LM and vocoder in one CUDA context by default.  Splitting
            # them into same-GPU processes time-slices the H100 at ordinary
            # serving concurrency and prevents decode/vocoder overlap.
            process="pipeline",
            factory=f"{_PKG}.stages.create_vocoder_executor",
            model=ModelGroup(device="cuda"),
            gpu=0,
            gpu_memory_fraction=0.10,
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
    ]

    def stage_factory_kwargs(self, stage_name: str) -> dict[str, Any]:
        if stage_name == "vocoder":
            return {
                "compile_decode": False,
                # Before the steady cursor is established, a decode window is
                # bounded by the default 75-row stride plus its 75-row
                # follow-up. Capture that complete finite domain so terminal
                # flushes cannot silently fall back to eager execution.
                "decode_cuda_graph_frame_counts": tuple(range(1, 151)),
            }
        return {}

    def model_post_init(self, __context: Any = None) -> None:
        super().model_post_init(__context)
        stages = {stage.name: stage for stage in self.stages}
        preprocessing = stages["preprocessing"]
        if "OMP_NUM_THREADS" not in self.env_defaults:
            preprocessing.env.setdefault(
                "OMP_NUM_THREADS",
                str(
                    bounded_intraop_threads(
                        worker_count=_PREPROCESS_MAX_WORKERS,
                        max_threads=8,
                    )
                ),
            )

    def requires_uploaded_voice_for_named_voice(self) -> bool:
        return True

    def supports_uploaded_voice_references(self) -> bool:
        return True


EntryClass = HiggsTtsPipelineConfig
