# SPDX-License-Identifier: Apache-2.0
"""Fixed-seed sync parity harness for MOSS-TTS Local v1.5 bit-identity.

Regression sentinel (lives under tests/regression/, not a stray unit test): the
reproducer for this PR's bit-identity claim, paired with gate_s0_sync_parity.py
and reused by #734's sync-parity hard gate. CPU tests exercise the comparison
fixtures and the apply-result data flow; the GPU test (skipped without CUDA)
verifies the frame-decode graph is deterministic and pins the S0 baseline. The
GPU baseline test must not be modified after initial commit.
"""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.moss_tts_local.request_builders import (
    MossTTSLocalSGLangRequestData,
    apply_sglang_moss_tts_local_result,
)
from sglang_omni.proto import OmniRequest, StagePayload

_N_VQ = 12


def _payload(request_id: str = "r0") -> StagePayload:
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs={"text": "hello"}, params={}, metadata={}),
        data={},
    )


def _data_with_rows(n_frames: int, seed: int) -> MossTTSLocalSGLangRequestData:
    """Build request data with fixed-seed deterministic output_rows."""
    torch.manual_seed(seed)
    data = MossTTSLocalSGLangRequestData(
        input_ids=torch.zeros(4, dtype=torch.long),
        max_new_tokens=64,
        temperature=0.0,
        output_ids=[],
        prompt_rows=torch.full((4, _N_VQ + 1), 1024, dtype=torch.long),
        stage_payload=_payload(),
        engine_start_s=0.0,
    )
    for _ in range(n_frames):
        row = torch.cat(
            [
                torch.tensor([1000]),  # non-stop slot
                torch.randint(0, 1024, (_N_VQ,)),
            ]
        )
        data.output_rows.append(row)
    return data


def _audio_codes(data: MossTTSLocalSGLangRequestData) -> torch.Tensor:
    result = apply_sglang_moss_tts_local_result(data.stage_payload, data)
    return torch.as_tensor(result.data["audio_codes"])


# ---------------------------------------------------------------------------
# CPU parity fixture tests
# ---------------------------------------------------------------------------


def test_parity_identical_output_rows_give_identical_audio_codes():
    """Same seed → identical output_rows → bit-identical audio_codes.

    Core harness assertion: PR-A must not break this.
    """
    codes_a = _audio_codes(_data_with_rows(5, seed=42))
    codes_b = _audio_codes(_data_with_rows(5, seed=42))
    assert torch.equal(codes_a, codes_b)


def test_parity_different_seeds_give_different_audio_codes():
    """Different seeds → different rows → different audio_codes.

    Validates the harness is sensitive enough to detect divergence.
    """
    codes_a = _audio_codes(_data_with_rows(5, seed=1))
    codes_b = _audio_codes(_data_with_rows(5, seed=2))
    assert not torch.equal(codes_a, codes_b), "harness must detect divergence"


def test_parity_empty_output_rows_produce_empty_audio_codes():
    data = MossTTSLocalSGLangRequestData(
        input_ids=torch.zeros(4, dtype=torch.long),
        max_new_tokens=64,
        temperature=0.0,
        output_ids=[],
        prompt_rows=torch.full((4, _N_VQ + 1), 1024, dtype=torch.long),
        stage_payload=_payload(),
        engine_start_s=0.0,
    )
    codes = _audio_codes(data)
    assert codes.shape == (0, _N_VQ)


# ---------------------------------------------------------------------------
# GPU gate scaffold (S0 baseline — do NOT modify after initial commit)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_s0_graph_replay_is_deterministic():
    """Two CUDA-graph replays with identical inputs must produce bit-identical outputs.

    This test pins the S0 gate baseline for PR-A. It exercises the same
    decode-frame kernel as the production v1.5 pipeline (MossTTSLocalTransformer
    + sample_seeded_branchless loop). Do NOT modify after initial commit.
    """
    from sglang_omni.models.moss_tts_local.local_transformer import (
        MossTTSLocalTransformer,
        sample_seeded_branchless,
    )

    device = torch.device("cuda")
    torch.manual_seed(0)
    module = MossTTSLocalTransformer(
        hidden_size=64,
        num_heads=4,
        inner_size=96,
        num_layers=1,
        max_positions=_N_VQ + 1,
        rope_base=1_000_000.0,
    ).to(device=device, dtype=torch.bfloat16)
    tables = [
        torch.randn(64, 64, device=device, dtype=torch.bfloat16) for _ in range(_N_VQ)
    ]

    def decode_frame(
        hidden: torch.Tensor,
        seeds: torch.Tensor,
        base: torch.Tensor,
    ) -> torch.Tensor:
        current = module.step(hidden, 0)
        codes = []
        for channel in range(_N_VQ):
            logits = (current.float() @ tables[channel].float().T)[:, :32]
            code = sample_seeded_branchless(
                logits,
                temperature=torch.full((hidden.shape[0],), 1.0, device=device),
                top_p=torch.full((hidden.shape[0],), 1.0, device=device),
                top_k=torch.full(
                    (hidden.shape[0],), 32, device=device, dtype=torch.long
                ),
                seeds=seeds,
                positions=base + channel + 1,
            )
            codes.append(code)
            if channel + 1 < _N_VQ:
                embed = torch.nn.functional.embedding(code, tables[channel][:32])
                current = module.step(embed.to(torch.bfloat16), channel + 1)
        return torch.stack(codes, dim=-1)

    batch = 4
    static_hidden = torch.zeros(batch, 64, device=device, dtype=torch.bfloat16)
    static_seeds = torch.zeros(batch, device=device, dtype=torch.long)
    static_base = torch.zeros(batch, device=device, dtype=torch.long)

    # Warmup two passes before capture (required for CUDA graph stability).
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            decode_frame(static_hidden, static_seeds, static_base)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graphed_codes = decode_frame(static_hidden, static_seeds, static_base)

    hidden = torch.randn(batch, 64, device=device, dtype=torch.bfloat16)
    seeds = torch.arange(batch, device=device, dtype=torch.long) * 1_234_567
    base = torch.full((batch,), 26, device=device, dtype=torch.long)

    static_hidden.copy_(hidden)
    static_seeds.copy_(seeds)
    static_base.copy_(base)
    graph.replay()
    run1 = graphed_codes.clone()

    static_hidden.copy_(hidden)
    static_seeds.copy_(seeds)
    static_base.copy_(base)
    graph.replay()
    run2 = graphed_codes.clone()

    assert torch.equal(run1, run2), (
        "CUDA graph replay with identical inputs must be bit-identical; "
        "any divergence indicates non-determinism in the decode kernel"
    )
