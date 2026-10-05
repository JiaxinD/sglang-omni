# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest
from transformers import WhisperConfig, WhisperFeatureExtractor
from voxt_omni_backend.model_views import WHISPER_LINKED_FILES, build_whisper_hf_view

FIXTURES = Path(__file__).parent / "fixtures"
TURBO_MLX_CONFIG = {
    "n_mels": 128,
    "n_audio_ctx": 1500,
    "n_audio_state": 1280,
    "n_audio_head": 20,
    "n_audio_layer": 32,
    "n_vocab": 51866,
    "n_text_ctx": 448,
    "n_text_state": 1280,
    "n_text_head": 20,
    "n_text_layer": 4,
    "model_type": "whisper",
}


def voxt_whisper_directory(root: Path) -> Path:
    model_directory = root / "mlx-audio" / "mlx-community_whisper-large-v3-turbo"
    model_directory.mkdir(parents=True)
    (model_directory / "config.json").write_text(json.dumps(TURBO_MLX_CONFIG))
    for name in WHISPER_LINKED_FILES:
        (model_directory / name).write_text(name)
    (model_directory / "generation_config.json").write_text(
        (FIXTURES / "openai-whisper-large-v3-generation_config.json").read_text()
    )
    return model_directory


def test_view_config_matches_the_published_turbo_config(tmp_path: Path) -> None:
    view = build_whisper_hf_view(voxt_whisper_directory(tmp_path), tmp_path / "view")

    generated = json.loads((view / "config.json").read_text())
    published = json.loads(
        (FIXTURES / "openai-whisper-large-v3-turbo-config.subset.json").read_text()
    )
    assert {key: generated[key] for key in published} == published
    assert WhisperConfig.from_pretrained(view).d_model == 1280


def test_view_preprocessor_matches_the_published_turbo_preprocessor(
    tmp_path: Path,
) -> None:
    view = build_whisper_hf_view(voxt_whisper_directory(tmp_path), tmp_path / "view")

    generated = WhisperFeatureExtractor.from_pretrained(view).to_dict()
    published = WhisperFeatureExtractor.from_pretrained(
        FIXTURES / "openai-whisper-large-v3-turbo-preprocessor_config.json"
    ).to_dict()
    assert generated == published


def test_view_links_weights_and_tokenizer_without_copying(tmp_path: Path) -> None:
    model_directory = voxt_whisper_directory(tmp_path)
    view = build_whisper_hf_view(model_directory, tmp_path / "view")

    for name in WHISPER_LINKED_FILES:
        assert (view / name).is_symlink()
        assert (view / name).resolve() == (model_directory / name).resolve()


def test_view_rebuild_replaces_stale_links(tmp_path: Path) -> None:
    model_directory = voxt_whisper_directory(tmp_path)
    view = tmp_path / "view"
    view.mkdir()
    (view / "weights.safetensors").symlink_to(tmp_path / "missing")

    build_whisper_hf_view(model_directory, view)

    assert (view / "weights.safetensors").resolve() == (
        model_directory / "weights.safetensors"
    ).resolve()


def test_view_rejects_an_incomplete_installation(tmp_path: Path) -> None:
    model_directory = voxt_whisper_directory(tmp_path)
    (model_directory / "tokenizer.json").unlink()

    with pytest.raises(FileNotFoundError, match="tokenizer.json"):
        build_whisper_hf_view(model_directory, tmp_path / "view")
