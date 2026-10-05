# SPDX-License-Identifier: Apache-2.0
"""Processor for MOSS-Transcribe-Diarize checkpoints shipped without remote code."""

from __future__ import annotations

import json
from pathlib import Path

from sglang.srt.multimodal.customized_mm_processor_utils import (
    register_customized_processor,
)
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    WhisperFeatureExtractor,
)

from sglang_omni.models.moss_transcribe_diarize import stages
from sglang_omni.models.moss_transcribe_diarize.hf_config import (
    MossTranscribeDiarizeConfig,
)

AUDIO_PAD_TOKEN = "<|audio_pad|>"
PROCESSOR_CONFIG_FILE = "processor_config.json"
CHAT_TEMPLATE_FILE = "chat_template.jinja"


class MossTranscribeDiarizeLocalProcessor:
    """The prompt rendering and audio span layout the checkpoint processor defines.

    Model directories installed without executable checkpoint code carry only
    the processor settings, the chat template, and the tokenizer files.
    """

    def __init__(
        self,
        *,
        feature_extractor: WhisperFeatureExtractor,
        tokenizer: PreTrainedTokenizerBase,
        chat_template: str,
        audio_tokens_per_second: float,
        audio_merge_size: int,
        time_marker_every_seconds: int,
        enable_time_marker: bool,
    ) -> None:
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.audio_tokens_per_second = float(audio_tokens_per_second)
        self.audio_merge_size = int(audio_merge_size)
        self.time_marker_every_seconds = int(time_marker_every_seconds)
        self.enable_time_marker = bool(enable_time_marker)
        self.audio_token = AUDIO_PAD_TOKEN
        self.audio_token_id = int(tokenizer.convert_tokens_to_ids(AUDIO_PAD_TOKEN))
        self.digit_token_ids = {
            digit: single_token_id(tokenizer, digit) for digit in "0123456789"
        }

    @staticmethod
    def from_directory(checkpoint_dir: Path) -> "MossTranscribeDiarizeLocalProcessor":
        settings = json.loads((checkpoint_dir / PROCESSOR_CONFIG_FILE).read_text())
        return MossTranscribeDiarizeLocalProcessor(
            feature_extractor=WhisperFeatureExtractor.from_pretrained(checkpoint_dir),
            tokenizer=AutoTokenizer.from_pretrained(checkpoint_dir),
            chat_template=(checkpoint_dir / CHAT_TEMPLATE_FILE).read_text(),
            audio_tokens_per_second=settings["audio_tokens_per_second"],
            audio_merge_size=settings["audio_merge_size"],
            time_marker_every_seconds=settings["time_marker_every_seconds"],
            enable_time_marker=settings["enable_time_marker"],
        )

    def apply_chat_template(
        self,
        messages: list[dict[str, object]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            chat_template=self.chat_template,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )

    def _audio_span_ids(  # noqa: leading-underscore  # name the request builder calls
        self, audio_token_count: int
    ) -> list[int]:
        """Audio placeholders with the elapsed-second digits inserted every marker."""
        tokens_per_marker = int(
            self.audio_tokens_per_second * self.time_marker_every_seconds
        )
        if (
            not self.enable_time_marker
            or audio_token_count <= 0
            or tokens_per_marker <= 0
        ):
            return [self.audio_token_id] * max(audio_token_count, 0)
        else:
            pass
        duration_s = audio_token_count / self.audio_tokens_per_second
        span_ids: list[int] = []
        consumed = 0
        for marker_s in range(
            self.time_marker_every_seconds,
            int(duration_s) + 1,
            self.time_marker_every_seconds,
        ):
            marker_position = (
                marker_s // self.time_marker_every_seconds
            ) * tokens_per_marker
            span_ids.extend([self.audio_token_id] * (marker_position - consumed))
            consumed = max(consumed, marker_position)
            span_ids.extend(self.digit_token_ids[digit] for digit in str(marker_s))
        span_ids.extend([self.audio_token_id] * (audio_token_count - consumed))
        return span_ids


def single_token_id(tokenizer: PreTrainedTokenizerBase, text: str) -> int:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(f"MOSS-Transcribe-Diarize expects {text!r} to be one token")
    else:
        pass
    return int(token_ids[0])


def load_moss_transcribe_diarize_processor(
    checkpoint_dir: str, *, trust_remote_code: bool, revision: str | None = None
) -> ProcessorMixin | MossTranscribeDiarizeLocalProcessor:
    """Use the checkpoint's processor code when present, the local one otherwise."""
    checkpoint_path = Path(checkpoint_dir)
    processor_config_path = checkpoint_path / PROCESSOR_CONFIG_FILE
    if checkpoint_path.is_dir() and processor_config_path.is_file():
        auto_map = json.loads(processor_config_path.read_text()).get("auto_map", {})
        remote_module = str(auto_map.get("AutoProcessor", "")).split(".")[0]
        has_remote_code = (
            bool(remote_module) and (checkpoint_path / f"{remote_module}.py").is_file()
        )
    else:
        has_remote_code = True
    if has_remote_code:
        with stages.missing_additional_chat_templates_compat():
            return AutoProcessor.from_pretrained(
                checkpoint_dir, trust_remote_code=trust_remote_code, revision=revision
            )
    else:
        return MossTranscribeDiarizeLocalProcessor.from_directory(checkpoint_path)


class MossTranscribeDiarizeProcessorLoader:
    """SGLang's worker resolves the processor through this registration too."""

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *,
        trust_remote_code: bool,
        revision: str | None = None,
        **sglang_processor_options: object,
    ) -> ProcessorMixin | MossTranscribeDiarizeLocalProcessor:
        # note: SGLang forwards tokenizer options this checkpoint does not use.
        del sglang_processor_options
        return load_moss_transcribe_diarize_processor(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            revision=revision,
        )


register_customized_processor(MossTranscribeDiarizeProcessorLoader)(
    MossTranscribeDiarizeConfig
)
