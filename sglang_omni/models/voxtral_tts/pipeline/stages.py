"""Stage executor factories for the Voxtral TTS pipeline."""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from sglang_omni.models.voxtral_tts.acoustic_transformer import AudioSpecialTokens
from sglang_omni.models.voxtral_tts.io import VoxtralTTSState
from sglang_omni.models.voxtral_tts.pipeline.state_io import load_state, store_state
from sglang_omni.platforms import current_platform
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.scheduling.streaming_vocoder import (
    StreamingVocoderBase,
    vocoder_decode_stream_priority,
)
from sglang_omni.scheduling.vocoder_base import BatchVocoderBase
from sglang_omni.utils.audio_payload import audio_waveform_payload
from sglang_omni.utils.checkpoint import resolve_checkpoint as _resolve_checkpoint

if TYPE_CHECKING:
    from sglang_omni.models.voxtral_tts.audio_tokenizer import VoxtralTTSAudioTokenizer
    from sglang_omni.models.voxtral_tts.request_builders import VoxtralSGLangRequestData
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler
else:
    pass

logger = logging.getLogger(__name__)
_VOXTRAL_MISTRAL_COMMON_HINT = "Voxtral TTS requires the `mistral_common` package (speech / Tekken tokenizer). Please install it in your active environment, for example:\n  pip install 'mistral_common[audio]>=1.11.0'\n  uv pip install 'mistral_common[audio]>=1.11.0'"


def import_mistral_common_for_voxtral():
    """Lazy import so the rest of sglang-omni does not depend on mistral-common."""
    try:
        from mistral_common.protocol.speech.request import SpeechRequest
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
    except ImportError as exc:
        raise RuntimeError(_VOXTRAL_MISTRAL_COMMON_HINT) from exc
    return (SpeechRequest, MistralTokenizer)


def validate_voxtral_speech_params(
    *,
    inputs: object,
    params: Mapping[str, object],
    tts_params: Mapping[str, object],
) -> None:
    explicit_generation_params = tts_params.get("explicit_generation_params")
    if isinstance(explicit_generation_params, (list, tuple, set)):
        explicit_fields = {str(field) for field in explicit_generation_params}
    else:
        explicit_fields = set()
    unsupported: set[str] = set()
    for field in explicit_fields:
        if field != "max_new_tokens":
            unsupported.add(field)
        else:
            pass
    if params.get("seed") is not None:
        unsupported.add("seed")
    else:
        pass
    if params.get("stage_sampling"):
        unsupported.add("stage_sampling")
    else:
        pass
    if params.get("stage_params"):
        unsupported.add("stage_params")
    else:
        pass
    for field in ("task_type", "language", "instructions", "ref_audio", "ref_text"):
        if tts_params.get(field) not in (None, ""):
            unsupported.add(field)
        else:
            pass
    if isinstance(inputs, dict) and inputs.get("references"):
        unsupported.add("references")
    else:
        pass
    if unsupported:
        fields = ", ".join(sorted(unsupported))
        raise ValueError(
            f"Voxtral TTS does not support these /v1/audio/speech fields: {fields}. Supported model-specific fields are voice and max_new_tokens."
        )
    else:
        pass


def ensure_non_empty_audio_codes(audio_codes: object) -> None:
    if audio_codes is None:
        raise ValueError("Voxtral TTS generated no audio codes")
    else:
        pass
    if isinstance(audio_codes, torch.Tensor) and audio_codes.numel() == 0:
        raise ValueError("Voxtral TTS generated no audio codes")
    else:
        pass


def create_preprocessing_executor(
    model_path: str,
) -> SimpleScheduler[StagePayload, StagePayload]:
    """Factory for the preprocessing stage."""
    checkpoint_dir = _resolve_checkpoint(model_path)
    SpeechRequest, MistralTokenizer = import_mistral_common_for_voxtral()
    tekken_path = os.path.join(checkpoint_dir, "tekken.json")
    tokenizer = MistralTokenizer.from_file(tekken_path)

    def _preprocess(payload: StagePayload) -> StagePayload:
        inputs = payload.request.inputs
        params = payload.request.params or {}
        metadata = payload.request.metadata or {}
        tts_params = metadata.get("tts_params", {})
        if not isinstance(tts_params, dict):
            tts_params = {}
        else:
            pass
        validate_voxtral_speech_params(
            inputs=inputs, params=params, tts_params=tts_params
        )
        if isinstance(inputs, str):
            text = inputs
        elif isinstance(inputs, dict):
            text = inputs.get("text", "")
        else:
            text = str(inputs) if inputs else ""
        voice = tts_params.get("voice") or params.get("voice")
        if voice in (None, "", "default"):
            voice = "cheerful_female"
        else:
            pass
        encoded = tokenizer.encode_speech_request(
            SpeechRequest(input=text, voice=voice)
        )
        max_new_tokens = params.get("max_new_tokens", 4096)
        if isinstance(max_new_tokens, dict):
            max_new_tokens = max_new_tokens.get("max_new_tokens", 4096)
        else:
            pass
        input_ids = list(encoded.tokens)
        state = VoxtralTTSState(
            input_ids=input_ids, voice=voice, max_new_tokens=max_new_tokens
        )
        return store_state(payload, state)

    return SimpleScheduler(_preprocess)


def enable_inductor_gemm_autotune() -> None:
    try:
        from torch._inductor import config as inductor_config
    except Exception:
        return
    if hasattr(inductor_config, "max_autotune_gemm"):
        inductor_config.max_autotune_gemm = True
    else:
        pass
    if hasattr(inductor_config, "max_autotune_gemm_backends"):
        inductor_config.max_autotune_gemm_backends = "TRITON,ATEN"
    else:
        pass
    logger.info(
        "Voxtral: enabled inductor per-shape GEMM autotuning (TRITON,ATEN); adds one-time startup autotune cost."
    )


def create_generation_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    max_new_tokens: int = 4096,
    server_args_overrides: Mapping[str, object] | None = None,
) -> OmniScheduler[VoxtralSGLangRequestData]:
    """Factory for the SGLang-backed AR generation stage."""
    del max_new_tokens
    from sglang_omni.models.voxtral_tts.pipeline.engine_builder import (
        VoxtralTtsEngineBuilder,
    )

    return VoxtralTtsEngineBuilder().build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        server_args_overrides=server_args_overrides,
    )


def write_voxtral_sglang_config(checkpoint_dir: str) -> str:
    from sglang_omni.models.voxtral_tts.model_config import VoxtralModelConfig

    cfg = VoxtralModelConfig.from_model_path(checkpoint_dir).text_config
    path = os.path.join(
        tempfile.gettempdir(), f"voxtral_sglang_config_{abs(hash(checkpoint_dir))}.json"
    )
    data = {
        "model_type": "llama",
        "architectures": ["VoxtralSGLangTTSModel"],
        "hidden_size": cfg.dim,
        "intermediate_size": cfg.hidden_dim,
        "num_hidden_layers": cfg.n_layers,
        "num_attention_heads": cfg.n_heads,
        "num_key_value_heads": cfg.n_kv_heads,
        "head_dim": cfg.head_dim,
        "vocab_size": cfg.vocab_size,
        "max_position_embeddings": cfg.max_seq_len,
        "rope_theta": cfg.rope_theta,
        "rms_norm_eps": cfg.norm_eps,
        "tie_word_embeddings": cfg.tied_embeddings,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return path


def load_voxtral_voice_embeddings(
    checkpoint_dir: str, device: str
) -> dict[str, torch.Tensor]:
    voice_embeddings: dict[str, torch.Tensor] = {}
    voice_dir = os.path.join(checkpoint_dir, "voice_embedding")
    if not os.path.isdir(voice_dir):
        return voice_embeddings
    else:
        pass
    map_location = "cpu" if current_platform.is_musa() else device
    for fname in sorted(os.listdir(voice_dir)):
        if not fname.endswith(".pt"):
            continue
        else:
            pass
        name = fname.removesuffix(".pt")
        emb = torch.load(
            os.path.join(voice_dir, fname), map_location=map_location, weights_only=True
        )
        voice_embeddings[name] = emb.to(device=device, dtype=torch.bfloat16)
    return voice_embeddings


def load_audio_tokenizer(
    checkpoint_dir: str, audio_config: dict, device: str
) -> "VoxtralTTSAudioTokenizer":
    """Load the VoxtralTTSAudioTokenizer (decoder) from checkpoint."""
    import glob

    from sglang.srt.model_loader.weight_utils import safetensors_weights_iterator

    from sglang_omni.models.voxtral_tts.audio_tokenizer import VoxtralTTSAudioTokenizer
    from sglang_omni.models.voxtral_tts.model_config import VoxtralModelConfig

    config = VoxtralModelConfig.from_model_path(checkpoint_dir)
    tokenizer = VoxtralTTSAudioTokenizer(
        audio_tokenizer_args=config.audio_tokenizer_args,
        audio_config={
            "audio_model_args": config.audio_model_args.acoustic_transformer_args
        },
    )
    safetensors_files = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
    if not safetensors_files:
        raise RuntimeError(f"No .safetensors files found in {checkpoint_dir}")
    else:
        pass
    logger.info("Loading audio tokenizer weights...")
    t0 = time.perf_counter()
    remapping_rules = [
        ("^audio_tokenizer\\.(.*)$", "\\1"),
        (
            "^mm_audio_embeddings\\.audio_codebook_embeddings\\.embeddings\\.(weight|bias)",
            "audio_token_embedding.embeddings.\\1",
        ),
    ]
    for name, tensor in safetensors_weights_iterator(safetensors_files):
        is_audio_tokenizer = name.startswith(
            "mm_audio_embeddings.audio_codebook_embeddings"
        ) or name.startswith("audio_tokenizer.")
        if not is_audio_tokenizer:
            continue
        else:
            pass
        remapped = name
        for pattern, repl in remapping_rules:
            if re.fullmatch(pattern, remapped):
                remapped = re.sub(pattern, repl, remapped)
            else:
                pass
        tokenizer.load_weight((remapped, tensor))
    tokenizer = tokenizer.to(dtype=torch.bfloat16, device=device).eval()
    logger.info(f"Audio tokenizer loaded in {time.perf_counter() - t0:.2f}s")
    return tokenizer


class VoxtralTTSVocoder(BatchVocoderBase[VoxtralTTSState, torch.Tensor]):
    """Decode audio codes with repeated initial frames as warmup context."""

    N_WARMUP = 2
    FADE_IN_MS = 10

    def __init__(self, audio_tokenizer: "VoxtralTTSAudioTokenizer") -> None:
        self.audio_tokenizer = audio_tokenizer

    def prepare_item(
        self, payload: StagePayload
    ) -> tuple[VoxtralTTSState, torch.Tensor]:
        state = load_state(payload)
        audio_codes = state.audio_codes
        ensure_non_empty_audio_codes(audio_codes)
        if not isinstance(audio_codes, torch.Tensor):
            audio_codes = torch.tensor(audio_codes)
        else:
            pass
        if audio_codes.shape[0] > 0:
            first_frame = audio_codes[0:1]
            warmup = first_frame.repeat(self.N_WARMUP, 1)
            codes_with_warmup = torch.cat([warmup, audio_codes], dim=0)
        else:
            codes_with_warmup = audio_codes
        return (state, codes_with_warmup)

    async def decode_batch(
        self, items: list[tuple[VoxtralTTSState, torch.Tensor]]
    ) -> list[tuple[torch.Tensor, int]]:
        codes_list = [codes for _, codes in items]
        results = self.audio_tokenizer.decode_helper_batch_async(codes_list)
        sample_rate = self.audio_tokenizer.sampling_rate
        return [(audio_np, sample_rate) for audio_np in results]

    def trim_warmup(self, state: VoxtralTTSState, wav: torch.Tensor) -> torch.Tensor:
        original_codes = state.audio_codes
        original_len = (
            original_codes.shape[0]
            if isinstance(original_codes, torch.Tensor)
            else len(original_codes)
        )
        warmup_samples = (
            self.N_WARMUP * self.audio_tokenizer.downsample_factor
            if original_len > 0
            else 0
        )
        if warmup_samples > 0 and len(wav) > warmup_samples:
            return wav[warmup_samples:]
        else:
            pass
        return wav

    def apply_fade_in(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        fade_samples = min(int(self.FADE_IN_MS * sample_rate / 1000), len(audio))
        if fade_samples > 0:
            fade_in = torch.linspace(
                0, 1, fade_samples, device=audio.device, dtype=audio.dtype
            )
            audio[:fade_samples] = audio[:fade_samples] * fade_in
        else:
            pass
        return audio

    def store_result(
        self,
        payload: StagePayload,
        state: VoxtralTTSState,
        wav: torch.Tensor,
        sample_rate: int,
    ) -> StagePayload:
        audio = self.apply_fade_in(self.trim_warmup(state, wav), sample_rate)
        return self.build_result(payload, state, audio, sample_rate)

    def build_result(
        self,
        payload: StagePayload,
        state: VoxtralTTSState,
        audio_np: torch.Tensor,
        sample_rate: int,
    ) -> StagePayload:
        audio_payload = audio_waveform_payload(audio_np, source_hint="Voxtral TTS")
        state.audio_samples = None
        state.sample_rate = sample_rate
        payload = store_state(payload, state)
        payload.data.update(audio_payload)
        payload.data["sample_rate"] = sample_rate
        payload.data["modality"] = "audio"
        if state.prompt_tokens or state.completion_tokens:
            payload.data["usage"] = {
                "prompt_tokens": state.prompt_tokens,
                "completion_tokens": state.completion_tokens,
                "total_tokens": state.prompt_tokens + state.completion_tokens,
            }
        else:
            pass
        return payload


@dataclass
class VoxtralStreamState:
    frames: list[torch.Tensor] = field(default_factory=list)
    emitted_frames: int = 0
    ended: bool = False


@dataclass(frozen=True)
class VoxtralStreamWindow:
    codes: torch.Tensor
    offset_frames: int
    new_frames: int


class VoxtralStreamingVocoder(
    StreamingVocoderBase[VoxtralStreamState, list[VoxtralStreamWindow]]
):
    """Decode streamed frames in left-context windows of the causal codec."""

    can_batch_stream_chunks = True
    stream_chunk_batch_max = 256

    def __init__(
        self,
        vocoder: VoxtralTTSVocoder,
        *,
        device: str | None = None,
        initial_chunk_frames: int = 5,
        chunk_frames: int = 25,
        left_context_frames: int = 25,
    ) -> None:
        self.vocoder = vocoder
        self.initial_chunk_frames = int(initial_chunk_frames)
        self.chunk_frames = int(chunk_frames)
        self.left_context_frames = int(left_context_frames)
        self.end_audio_id = AudioSpecialTokens.id(AudioSpecialTokens.end_audio)
        self.decode_stream = None
        self.codec_graphs = None
        if device is not None and str(device).startswith("cuda"):
            # Note (Jiaxin Deng): eager window decodes launch ~250 kernels and, under
            # load, sit behind the engine's replays and the GIL (~90 ms per window).
            from sglang_omni.models.voxtral_tts.audio_tokenizer import (
                VoxtralCodecGraphs,
            )

            self.decode_stream = torch.cuda.Stream(
                device=device, priority=vocoder_decode_stream_priority(torch.cuda)
            )
            self.codec_graphs = VoxtralCodecGraphs(
                vocoder.audio_tokenizer,
                self.decode_stream,
                max_frames=left_context_frames + 2 * chunk_frames,
            )
        else:
            pass
        super().__init__(
            self.vocode_payload,
            sample_rate=vocoder.audio_tokenizer.sampling_rate,
            stream_source_hint="Voxtral TTS",
        )

    async def vocode_payload(self, payload: StagePayload) -> StagePayload:
        if self.codec_graphs is None:
            return (await self.vocoder.decode_payloads([payload]))[0]
        else:
            pass
        state, _ = self.vocoder.prepare_item(payload)
        audio = self.decode_utterance(self.payload_frames(state))
        return self.vocoder.build_result(payload, state, audio, self.sample_rate)

    def payload_frames(self, state: VoxtralTTSState) -> list[torch.Tensor]:
        codes = torch.as_tensor(state.audio_codes).to(device="cpu", dtype=torch.long)
        ended = (codes[:, 0] == self.end_audio_id).nonzero()
        if ended.numel() > 0:
            codes = codes[: int(ended[0])]
        else:
            pass
        return list(codes.unbind(0))

    def decode_utterance(self, frames: list[torch.Tensor]) -> torch.Tensor:
        """Whole-utterance decode through the same left-context windows."""
        return self.decode_pending(VoxtralStreamState(frames=frames))

    def decode_pending(self, state: VoxtralStreamState) -> torch.Tensor:
        """Decode every pending frame as one batch of bounded windows."""
        emitted_frames = state.emitted_frames
        windows = []
        while state.emitted_frames < len(state.frames):
            window = self.build_window(state)
            windows.append(window)
            state.emitted_frames += window.new_frames
        state.emitted_frames = emitted_frames
        audios = self.decode_windows(windows)
        return torch.cat(
            [
                self.commit_window(state, window, audio)
                for window, audio in zip(windows, audios, strict=True)
            ]
        )

    def create_stream_state(self, request_id: str) -> VoxtralStreamState:
        del request_id
        return VoxtralStreamState()

    def validate_chunk(
        self, request_id: str, state: VoxtralStreamState, codes: torch.Tensor
    ) -> torch.Tensor:
        del state
        num_codebooks = self.vocoder.audio_tokenizer.num_codebooks
        if codes.ndim != 2 or codes.shape[1] != num_codebooks:
            raise ValueError(
                f"Voxtral TTS stream chunk for {request_id!r} must be "
                f"[T, {num_codebooks}], got {tuple(codes.shape)}"
            )
        else:
            pass
        return codes.to(dtype=torch.long)

    def ingest(
        self, request_id: str, state: VoxtralStreamState, codes: torch.Tensor
    ) -> None:
        del request_id
        for row in codes.unbind(0):
            if state.ended or int(row[0]) == self.end_audio_id:
                state.ended = True
                return
            else:
                pass
            state.frames.append(row)

    def is_ready(self, state: VoxtralStreamState, *, is_final: bool) -> bool:
        new_frames = len(state.frames) - state.emitted_frames
        threshold = (
            self.initial_chunk_frames
            if state.emitted_frames == 0
            else self.chunk_frames
        )
        return new_frames > 0 and (is_final or new_frames >= threshold)

    def build_window(self, state: VoxtralStreamState) -> VoxtralStreamWindow:
        # Note (Jiaxin Deng): the codec is causal, so enough left context reproduces
        # the full decode within its bf16 noise floor (measured: 12+ frames).
        n_warmup = self.vocoder.N_WARMUP
        total = min(len(state.frames), state.emitted_frames + 2 * self.chunk_frames)
        sequence = [state.frames[0]] * n_warmup + state.frames[:total]
        start = max(0, state.emitted_frames + n_warmup - self.left_context_frames)
        return VoxtralStreamWindow(
            codes=torch.stack(sequence[start : total + n_warmup], dim=0),
            offset_frames=state.emitted_frames + n_warmup - start,
            new_frames=total - state.emitted_frames,
        )

    def commit_window(
        self,
        state: VoxtralStreamState,
        window: VoxtralStreamWindow,
        audio: torch.Tensor,
    ) -> torch.Tensor:
        samples_per_frame = self.vocoder.audio_tokenizer.downsample_factor
        offset = window.offset_frames * samples_per_frame
        delta = audio[offset : offset + window.new_frames * samples_per_frame]
        delta = delta.contiguous()
        if state.emitted_frames == 0:
            delta = self.vocoder.apply_fade_in(delta, self.sample_rate)
        else:
            pass
        state.emitted_frames += window.new_frames
        return delta

    @torch.no_grad()
    def decode_windows(self, windows: list[VoxtralStreamWindow]) -> list[torch.Tensor]:
        codes = [window.codes for window in windows]
        if self.decode_stream is None:
            return self.vocoder.audio_tokenizer.decode_helper_batch_async(codes)
        else:
            pass
        with torch.cuda.stream(self.decode_stream):
            if self.codec_graphs.fits(codes):
                return self.codec_graphs.decode(codes)
            else:
                pass
            return self.vocoder.audio_tokenizer.decode_helper_batch_async(codes)

    def decode_delta(
        self, request_id: str, state: VoxtralStreamState, *, is_final: bool
    ) -> torch.Tensor | None:
        del request_id
        if not self.is_ready(state, is_final=is_final):
            return None
        else:
            pass
        return self.decode_pending(state)

    def select_step_participants(self) -> list[tuple[str, VoxtralStreamState]]:
        return [
            (request_id, state)
            for request_id, state in self.stream_state_items()
            if not self.is_aborted(request_id) and self.is_ready(state, is_final=False)
        ]

    def build_step_plan(
        self, participants: list[tuple[str, VoxtralStreamState]]
    ) -> list[VoxtralStreamWindow]:
        return [self.build_window(state) for _, state in participants]

    def run_step(
        self,
        participants: list[tuple[str, VoxtralStreamState]],
        plan: list[VoxtralStreamWindow],
    ) -> dict[str, torch.Tensor]:
        audios = self.decode_windows(plan)
        return {
            request_id: self.commit_window(state, window, audio)
            for (request_id, state), window, audio in zip(
                participants, plan, audios, strict=True
            )
        }

    def fallback_full_decode(
        self, request_id: str, payload: StagePayload, state: VoxtralStreamState
    ) -> torch.Tensor | None:
        del request_id, state
        final_state, _ = self.vocoder.prepare_item(payload)
        return self.decode_utterance(self.payload_frames(final_state))

    def final_result_data(
        self, request_id: str, payload: StagePayload, state: VoxtralStreamState
    ) -> Mapping[str, object]:
        del request_id, state
        final_state = load_state(payload)
        final_state.audio_codes = None
        final_state.sample_rate = self.sample_rate
        data = final_state.to_dict()
        data["modality"] = "audio"
        data["sample_rate"] = self.sample_rate
        if final_state.prompt_tokens or final_state.completion_tokens:
            data["usage"] = {
                "prompt_tokens": final_state.prompt_tokens,
                "completion_tokens": final_state.completion_tokens,
                "total_tokens": final_state.prompt_tokens
                + final_state.completion_tokens,
            }
        else:
            pass
        return data


def create_vocoder_executor(
    model_path: str, *, device: str | None = None, gpu_id: int | None = None
) -> VoxtralStreamingVocoder:
    from sglang_omni.utils.device import resolve_concrete_device

    device = str(resolve_concrete_device(device, gpu_id))
    checkpoint_dir = _resolve_checkpoint(model_path)
    logger.info("Loading Voxtral audio tokenizer for vocoding...")
    audio_tokenizer = load_audio_tokenizer(checkpoint_dir, {}, device)
    return VoxtralStreamingVocoder(VoxtralTTSVocoder(audio_tokenizer), device=device)
