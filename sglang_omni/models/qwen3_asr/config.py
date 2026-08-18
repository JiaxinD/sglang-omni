# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Qwen3-ASR."""

from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import (
    AudioChunkingConfig,
    EngineArgs,
    EngineStageConfig,
    ModelGroup,
    PipelineConfig,
    SchedulerConfig,
    StageConfig,
)
from sglang_omni.models.qwen3_asr.audio_lengths import QWEN3_ASR_MAX_INPUT_SECONDS

_PKG = "sglang_omni.models.qwen3_asr"


class Qwen3ASRPipelineConfig(PipelineConfig):
    """Single-stage batched ASR pipeline for Qwen3-ASR checkpoints."""

    architecture: ClassVar[str] = "Qwen3ASRForConditionalGeneration"
    audio_chunking: ClassVar[AudioChunkingConfig] = AudioChunkingConfig(
        allow_audio_chunking=True,
        max_audio_clip_s=60.0,
        max_native_clip_s=float(QWEN3_ASR_MAX_INPUT_SECONDS),
    )

    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "asr": EngineStageConfig,
    }

    model_path: str
    entry_stage: str = "asr"
    stages: list[StageConfig] = [
        EngineStageConfig(
            name="asr",
            process="asr",
            factory=f"{_PKG}.stages.create_sglang_qwen3_asr_executor",
            # Note (Jeffro): max_new_tokens is the floor for the per-request
            # output budget. The request builder will scale the actual budget
            # with audio duration.
            model=ModelGroup(
                device=None,
                max_new_tokens=128,
                enable_pre_lm_encoder=True,
                pre_lm_cache_max_entries=4096,
                pre_lm_cache_size_bytes=2 * 1024**3,
                pre_lm_max_batch_size=8,
                pre_lm_max_batch_wait_ms=0,
            ),
            engine=EngineArgs(
                max_running_requests=64,
                enable_torch_compile=True,
                torch_compile_max_bs=2,
            ),
            scheduler=SchedulerConfig(
                request_build_max_workers=8,
                request_build_max_pending=32,
                prefill_coalesce_requests=16,
                prefill_coalesce_wait_ms=40,
                prefill_coalesce_when_idle=True,
                prefill_coalesce_requires_pending_builds=True,
                prefill_coalesce_after_builds_during_decode=True,
            ),
            gpu=0,
            terminal=True,
        )
    ]


EntryClass = Qwen3ASRPipelineConfig
