# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for ARK-ASR-3B."""

from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import (
    EngineArgs,
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    StageConfig,
)

_PKG = "sglang_omni.models.arkasr"


class ArkasrPipelineConfig(PipelineConfig):
    """Single-stage batched ASR pipeline for ARK-ASR-3B checkpoints."""

    architecture: ClassVar[str] = "ArkasrForConditionalGeneration"

    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "asr": EngineStageConfig,
    }

    model_path: str
    entry_stage: str = "asr"
    stages: list[StageConfig] = [
        EngineStageConfig(
            name="asr",
            process="asr",
            factory_path=f"{_PKG}.stages.create_sglang_arkasr_executor",
            factory=FactoryArgs(
                device="cuda:0",
                max_new_tokens=256,
                encoder_max_batch_size=8,
                enable_pre_lm_encoder=True,
                pre_lm_cache_max_entries=4096,
                pre_lm_cache_size_bytes=2 * 1024**3,
                # Note (Akazaakane): One drained group maps to exactly one
                # encoder microbatch at the matching encoder_max_batch_size.
                pre_lm_max_batch_size=8,
                pre_lm_max_batch_wait_ms=0,
                pre_lm_max_pending=32,
                request_build_max_workers=2,
                request_build_max_pending=16,
            ),
            engine=EngineArgs(max_running_requests=32),
            gpu=0,
            terminal=True,
        )
    ]


EntryClass = ArkasrPipelineConfig
