# SPDX-License-Identifier: Apache-2.0
"""Staleness guard tests for the fused MOSS-TTS audio LM heads."""

from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest
import torch

from sglang_omni.models.moss_tts.sglang_model import MossTTSDelaySGLangModel


def _make_stub(rows: int = 8, hidden: int = 4, n_audio: int = 3) -> SimpleNamespace:
    stacked = torch.randn(n_audio * rows, hidden)
    heads = [SimpleNamespace(weight=torch.randn(2, hidden))]  # text head
    for index in range(n_audio):
        heads.append(SimpleNamespace(weight=stacked[index * rows : (index + 1) * rows]))
    stub = SimpleNamespace(
        lm_heads=heads,
        _stacked_audio_head_weight=stacked,
        _audio_head_padded_vocab=rows,
        _audio_head_expected_ptrs=[
            stacked[index * rows : (index + 1) * rows].data_ptr()
            for index in range(n_audio)
        ],
        _fused_audio_heads_enabled=True,
    )
    stub._ensure_stacked_audio_heads = MethodType(
        lambda self: self._stacked_audio_head_weight is not None, stub
    )
    stub._fused_audio_heads_ready = MethodType(
        MossTTSDelaySGLangModel._fused_audio_heads_ready, stub
    )
    return stub


def test_fused_audio_heads_ready_when_aliased() -> None:
    stub = _make_stub()
    assert stub._fused_audio_heads_ready() is True


@pytest.mark.parametrize("replaced_index", [1, 2, 3])
def test_replacing_any_audio_head_disables_fused_path(replaced_index: int) -> None:
    stub = _make_stub()
    stub.lm_heads[replaced_index].weight = torch.randn_like(
        stub.lm_heads[replaced_index].weight
    )
    assert stub._fused_audio_heads_ready() is False
    assert stub._stacked_audio_head_weight is None
    assert stub._fused_audio_heads_enabled is False
