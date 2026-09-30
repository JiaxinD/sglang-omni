# SPDX-License-Identifier: Apache-2.0
"""MOSS-TD's model state and wiring for the shared PD schedulers."""

from __future__ import annotations

from typing import Any

from sglang_omni.models.moss_transcribe_diarize.engine_builder import (
    MossTranscribeDiarizeEngineBuilder,
)
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.pd_scheduler import (
    OmniDecodeScheduler,
    OmniPrefillScheduler,
)
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData

MOSS_TD_PD_RESUME_SCHEMA = "moss-td-pd-v1"


def build_decode_state(req: Any) -> tuple[dict, dict, list[int]]:
    """Keep audio/embeddings on P; only small result metadata accompanies KV."""
    data = req.omni_data
    payload = data.stage_payload
    continuation_payload = StagePayload(
        request_id=payload.request_id,
        request=OmniRequest(inputs=None, params=dict(payload.request.params or {})),
        data=None,
    )
    resume = {
        "schema": MOSS_TD_PD_RESUME_SCHEMA,
        "prompt_token_ids": list(data.prompt_token_ids or req.origin_input_ids),
        "audio_duration_s": float(data.audio_duration_s),
        "language": str(data.language),
        "engine_start_s": float(data.engine_start_s),
        "enforce_request_limits": bool(data.enforce_request_limits),
    }
    return continuation_payload.to_dict(), resume, list(req.origin_input_ids)


def restore_decode_state(
    req: Any, data: SGLangARRequestData, resume: dict[str, Any] | None
) -> None:
    if resume is None or resume.get("schema") != MOSS_TD_PD_RESUME_SCHEMA:
        raise ValueError("invalid MOSS-TD PD resume state")
    else:
        pass
    data.prompt_token_ids = list(resume["prompt_token_ids"])
    data.audio_duration_s = float(resume["audio_duration_s"])
    data.language = str(resume["language"])
    data.engine_start_s = float(resume["engine_start_s"])
    data.enforce_request_limits = bool(resume["enforce_request_limits"])
    req.multimodal_inputs = None
    req._codec_suppress_tokens = (
        None  # noqa: leading-underscore  # SGLang request extension
    )
    # The model's request builder uses token-id stops only.
    req.tokenizer = None


class MossTranscribeDiarizePDEngineBuilder(MossTranscribeDiarizeEngineBuilder):
    def __init__(self, *, pd_role: str, stage_name: str, **kwargs: Any) -> None:
        if pd_role not in ("prefill", "decode") or not stage_name:
            raise ValueError("MOSS-TD PD requires a role and a concrete stage_name")
        else:
            pass
        super().__init__(**kwargs)
        self.pd_role = pd_role
        self.stage_name = stage_name

    def setup_model_resources(self, model: Any, server_args: Any, **kwargs) -> None:
        if self.pd_role == "prefill":
            super().setup_model_resources(model, server_args, **kwargs)
        else:
            # D still loads the checkpoint, but never runs or captures the encoder.
            model.init_encoder_cache(0)

    def setup_runtime_resources(self, model: Any, server_args: Any) -> None:
        if self.pd_role == "prefill":
            super().setup_runtime_resources(model, server_args)
        else:
            pass

    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        request_builder, result_adapter = super().make_adapters(model)

        def build_request(payload: StagePayload) -> Any:
            if (payload.request.params or {}).get("stream"):
                raise ValueError("MOSS-Transcribe-Diarize PD requires stream=false")
            elif self.pd_role == "decode":
                raise ValueError("MOSS-TD Decode requires a KV continuation")
            else:
                return request_builder(payload)

        return build_request, result_adapter

    def extra_scheduler_kwargs(self) -> dict[str, Any]:
        kwargs = super().extra_scheduler_kwargs()
        kwargs.pop("stream_output_builder")
        return kwargs

    def make_scheduler(
        self, *, model_worker: Any, extra_scheduler_kwargs: dict, **kwargs: Any
    ) -> Any:
        kwargs.update(extra_scheduler_kwargs)
        kwargs.update(self.extra_scheduler_callbacks())
        kwargs.update(
            tp_worker=model_worker,
            stage_name=self.stage_name,
            abort_callback=self.make_abort_callback(),
            request_finished_callback=self.make_request_finished_callback(),
        )
        if self.pd_role == "prefill":
            return OmniPrefillScheduler(
                partner_stage="asr_decode", state_builder=build_decode_state, **kwargs
            )
        else:
            return OmniDecodeScheduler(
                state_restorer=restore_decode_state,
                resume_schema=MOSS_TD_PD_RESUME_SCHEMA,
                **kwargs,
            )
