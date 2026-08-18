# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for dots.tts."""

from __future__ import annotations

from typing import Any, ClassVar

from sglang_omni.config import EngineStageConfig, PipelineConfig, StageConfig

_PKG = "sglang_omni.models.dots_tts"


class DotsTTSPipelineConfig(PipelineConfig):
    """preprocess -> reference encode -> SGLang latent AR -> AudioVAE."""

    architecture: ClassVar[str] = "DotsTTSForConditionalGeneration"
    requires_model_capabilities: ClassVar[bool] = True
    required_speech_reference_count: ClassVar[int | None] = 1
    speech_reference_text_required: ClassVar[bool] = True
    architecture_aliases: ClassVar[tuple[str, ...]] = (
        "dots_tts",
        "DotsTtsModel",
    )
    additional_speech_languages: ClassVar[frozenset[str]] = frozenset({"auto_detect"})

    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "latent_engine": EngineStageConfig,
    }

    model_path: str
    stages: list[StageConfig] = [
        StageConfig(
            name="preprocessing",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_preprocessing_executor",
            next="reference_encode",
        ),
        StageConfig(
            name="reference_encode",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_reference_encode_executor",
            gpu=0,
            next="latent_engine",
        ),
        EngineStageConfig(
            name="latent_engine",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_sglang_latent_engine_executor",
            gpu=0,
            next="vocoder",
            stream_to=["vocoder"],
        ),
        StageConfig(
            name="vocoder",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_vocoder_executor",
            gpu=0,
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
    ]

    def model_post_init(self, __context: Any = None) -> None:
        super().model_post_init(__context)
        if any(stage.tp_size != 1 for stage in self.stages):
            raise ValueError("dots.tts currently supports tp_size=1 only")

    def stage_factory_kwargs(self, stage_name: str) -> dict[str, Any]:
        if stage_name == "vocoder":
            # note (guozhihao-224): stream_slots must match backbone
            # concurrency so a max_running_requests override cannot outrun
            # vocoder admission after readiness. Derived at launch, after
            # every configuration source has been resolved; an explicit
            # vocoder stream_slots that disagrees is refused.
            latent = self.stage_named("latent_engine")
            max_running_requests = self._latent_max_running_requests(latent)
            explicit = (self.stage_named("vocoder").model.model_extra or {}).get(
                "stream_slots"
            )
            if explicit is not None and int(explicit) != max_running_requests:
                raise ValueError(
                    "dots.tts vocoder stream_slots "
                    f"({int(explicit)}) must equal latent_engine "
                    f"max_running_requests ({max_running_requests}); "
                    "omit stream_slots to derive it from the latent engine"
                )
            return {"stream_slots": max_running_requests}
        return {}

    @staticmethod
    def _latent_max_running_requests(latent_engine: StageConfig) -> int:
        engine = latent_engine.engine
        value = engine.max_running_requests if engine is not None else None
        if value is None:
            # Match DotsTTSEngineBuilder default when unset.
            return 16
        if int(value) < 1:
            raise ValueError(
                f"dots.tts max_running_requests must be positive, got {value}"
            )
        return int(value)

    def supports_uploaded_voice_references(self) -> bool:
        return True


EntryClass = DotsTTSPipelineConfig
