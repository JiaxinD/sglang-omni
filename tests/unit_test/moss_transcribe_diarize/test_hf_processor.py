# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from sglang_omni.models.moss_transcribe_diarize.hf_processor import (
    MossTranscribeDiarizeLocalProcessor,
    load_moss_transcribe_diarize_processor,
)

AUDIO_TOKEN_ID = 99
DIGIT_TOKEN_IDS = {str(digit): 10 + digit for digit in range(10)}


class DigitTokenizer:
    def convert_tokens_to_ids(self, token: str) -> int:
        return {"<|audio_pad|>": AUDIO_TOKEN_ID}[token]

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [DIGIT_TOKEN_IDS[character] for character in text]


def local_processor(**overrides: object) -> MossTranscribeDiarizeLocalProcessor:
    settings: dict[str, object] = {
        "audio_tokens_per_second": 12.5,
        "audio_merge_size": 4,
        "time_marker_every_seconds": 5,
        "enable_time_marker": True,
    }
    settings.update(overrides)
    return MossTranscribeDiarizeLocalProcessor(
        feature_extractor=SimpleNamespace(),
        tokenizer=DigitTokenizer(),
        chat_template="",
        **settings,
    )


def test_audio_span_inserts_elapsed_second_markers() -> None:
    span = local_processor()._audio_span_ids(
        130
    )  # noqa: leading-underscore  # upstream name

    marker_five = [DIGIT_TOKEN_IDS["5"]]
    marker_ten = [DIGIT_TOKEN_IDS["1"], DIGIT_TOKEN_IDS["0"]]
    assert span == (
        [AUDIO_TOKEN_ID] * 62
        + marker_five
        + [AUDIO_TOKEN_ID] * 62
        + marker_ten
        + [AUDIO_TOKEN_ID] * 6
    )


def test_audio_span_without_markers_is_contiguous() -> None:
    processor = local_processor(enable_time_marker=False)

    assert (
        processor._audio_span_ids(130) == [AUDIO_TOKEN_ID] * 130
    )  # noqa: leading-underscore  # upstream name


def test_loader_prefers_the_local_processor_without_remote_code(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "processor_config.json").write_text(
        json.dumps({"audio_merge_size": 4, "time_marker_every_seconds": 5})
    )
    (tmp_path / "chat_template.jinja").write_text("template")
    loaded: list[str] = []

    def local_from_directory(checkpoint_dir: Path) -> str:
        loaded.append("local")
        return "local-processor"

    monkeypatch.setattr(
        MossTranscribeDiarizeLocalProcessor,
        "from_directory",
        staticmethod(local_from_directory),
    )

    assert (
        load_moss_transcribe_diarize_processor(str(tmp_path), trust_remote_code=True)
        == "local-processor"
    )
    assert loaded == ["local"]


REFERENCE_CHECKPOINT = os.environ.get("SGLANG_OMNI_MOSS_REFERENCE_CHECKPOINT")


@pytest.mark.skipif(
    REFERENCE_CHECKPOINT is None,
    reason="set SGLANG_OMNI_MOSS_REFERENCE_CHECKPOINT to a checkout with remote code",
)
def test_local_processor_matches_the_checkpoint_processor() -> None:
    from transformers import AutoProcessor

    from sglang_omni.models.moss_transcribe_diarize import stages
    from sglang_omni.models.moss_transcribe_diarize.request_builders import (
        render_prompt,
    )

    with stages.missing_additional_chat_templates_compat():
        reference = AutoProcessor.from_pretrained(
            REFERENCE_CHECKPOINT, trust_remote_code=True
        )
    local = MossTranscribeDiarizeLocalProcessor.from_directory(
        Path(REFERENCE_CHECKPOINT)
    )

    for prompt in ("请转写", "Transcribe the audio as plain text."):
        assert render_prompt(local, prompt) == render_prompt(reference, prompt)
    for token_count in (0, 1, 61, 62, 63, 130, 375, 3750, 15000):
        assert local._audio_span_ids(
            token_count
        ) == reference._audio_span_ids(  # noqa: leading-underscore  # upstream name
            token_count
        )
    assert local.audio_token_id == reference.audio_token_id
    assert local.audio_merge_size == reference.audio_merge_size
    assert local.feature_extractor.to_dict() == reference.feature_extractor.to_dict()


def test_sglang_worker_processor_lookup_uses_the_same_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang.srt.multimodal.customized_mm_processor_utils import (
        _CUSTOMIZED_MM_PROCESSOR,
    )

    from sglang_omni.models.moss_transcribe_diarize.hf_processor import (
        MossTranscribeDiarizeProcessorLoader,
    )

    calls: list[tuple[str, bool, str | None, dict[str, object]]] = []

    def record(
        checkpoint_dir: str,
        *,
        trust_remote_code: bool,
        revision: str | None,
        processor_options: dict[str, object],
    ) -> str:
        calls.append((checkpoint_dir, trust_remote_code, revision, processor_options))
        return "processor"

    monkeypatch.setattr(
        "sglang_omni.models.moss_transcribe_diarize.hf_processor."
        "load_moss_transcribe_diarize_processor",
        record,
    )

    loader = _CUSTOMIZED_MM_PROCESSOR["moss_transcribe_diarize"]
    processor = loader.from_pretrained(
        "/models/moss", trust_remote_code=False, revision="abc123", use_fast=True
    )

    assert loader is MossTranscribeDiarizeProcessorLoader
    assert processor == "processor"
    assert calls == [("/models/moss", False, "abc123", {"use_fast": True})]
