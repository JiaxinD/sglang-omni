# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for MOSS-TTS sampling kernels."""

from __future__ import annotations

import pytest
import torch
from sglang.kernels.ops.sampling.murmur_hash import murmur_hash32
from sglang.srt.layers.sampler import multinomial_with_seed

from sglang_omni.models.moss_tts.sampling_kernels import (
    multinomial_with_seed_and_token_ids,
    seeded_gumbel_argmax,
)

pytestmark = pytest.mark.accelerator

_UINT32_MAX_HASH_POSITION = 1_707_985_137


@pytest.mark.parametrize("equal_scores", [False, True], ids=["random", "tied"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_seeded_gumbel_argmax_matches_production_shape(
    equal_scores: bool,
) -> None:
    device = torch.device("cuda")
    rows, vocab = 32, 1025
    scores = (
        torch.zeros(rows, vocab, device=device, dtype=torch.float32)
        if equal_scores
        else torch.randn(rows, vocab, device=device, dtype=torch.float32)
    )
    seeds = torch.tensor([20260720], device=device, dtype=torch.long).expand(rows)
    positions = torch.arange(rows, device=device, dtype=torch.long)
    output = torch.empty(rows, device=device, dtype=torch.long)

    expected = multinomial_with_seed(scores, seeds, positions).view(-1)
    actual = seeded_gumbel_argmax(scores, seeds, positions, output)
    torch.cuda.synchronize()

    assert seeds.stride(0) == 0
    assert torch.equal(expected, actual)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_seeded_gumbel_argmax_matches_uint32_max_hash() -> None:
    device = torch.device("cuda")
    scores = torch.tensor([[-100.0, 0.0]], device=device)
    seeds = torch.tensor([0], device=device, dtype=torch.long)
    # This position makes MurmurHash(seed=0, position, token_id=0) UINT32_MAX.
    positions = torch.tensor(
        [_UINT32_MAX_HASH_POSITION], device=device, dtype=torch.long
    )
    output = torch.empty(1, device=device, dtype=torch.long)

    hashes = murmur_hash32(
        seeds.to(torch.uint64),
        positions,
        torch.arange(scores.shape[1], device=device),
    )
    expected = multinomial_with_seed(scores, seeds, positions).view(-1)
    actual = seeded_gumbel_argmax(scores, seeds, positions, output)
    torch.cuda.synchronize()

    assert hashes[0, 0].item() == torch.iinfo(torch.uint32).max
    assert expected.item() == 1
    assert torch.equal(expected, actual)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_token_id_sampler_matches_sglang_at_the_uint32_max_hash() -> None:
    device = torch.device("cuda")
    scores = torch.tensor([[-100.0, 0.0]], device=device)
    seeds = torch.tensor([0], device=device, dtype=torch.long)
    positions = torch.tensor(
        [_UINT32_MAX_HASH_POSITION], device=device, dtype=torch.long
    )
    token_ids = torch.arange(scores.shape[1], device=device)

    hashes = murmur_hash32(seeds.to(torch.uint64), positions, token_ids)
    expected = multinomial_with_seed(scores, seeds, positions).view(-1)
    actual = multinomial_with_seed_and_token_ids(scores, seeds, positions, token_ids)

    assert hashes[0, 0].item() == torch.iinfo(torch.uint32).max
    assert expected.item() == 1
    assert torch.equal(expected, actual)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_seeded_gumbel_argmax_rejects_strided_output() -> None:
    device = torch.device("cuda")
    rows, vocab = 2, 16
    scores = torch.randn(rows, vocab, device=device, dtype=torch.float32)
    seeds = torch.arange(rows, device=device, dtype=torch.long)
    positions = torch.arange(rows, device=device, dtype=torch.long)
    output = torch.empty(rows * 2, device=device, dtype=torch.long)[::2]

    assert output.stride(0) == 2
    with pytest.raises(ValueError, match="output must have stride 1"):
        seeded_gumbel_argmax(scores, seeds, positions, output)


def _fused_case(vocab: int, temp: float, top_p: float, top_k: int, tie: bool):
    return pytest.param(vocab, temp, top_p, top_k, tie, id=f"v{vocab}-t{temp}-p{top_p}-k{top_k}-{'tied' if tie else 'random'}")


@pytest.mark.parametrize(
    "vocab,temp,top_p,top_k,tie",
    [
        _fused_case(1025, 1.7, 0.8, 25, False),
        _fused_case(1025, 1.7, 0.8, 25, True),
        _fused_case(1025, 1.0, 0.9, 64, True),
        _fused_case(1025, 1.2, 0.5, 1, False),
        _fused_case(1025, 1.7, 0.001, 25, False),
        _fused_case(1025, 1.7, 0.999, 25, False),
        _fused_case(1025, 0.0, 0.8, 25, False),
        _fused_case(2, 1.0, 1.0, 50, False),
        _fused_case(2, 1.3, 0.7, 0, False),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_sample_seeded_fused_matches_branchless(
    vocab: int, temp: float, top_p: float, top_k: int, tie: bool
) -> None:
    from sglang_omni.models.moss_tts.sampling_kernels import (
        sample_seeded_branchless,
        sample_seeded_fused,
    )

    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(20260821)
    for rows in (1, 4, 16):
        for _ in range(20):
            logits = torch.randn(rows, vocab, device=device, generator=gen) * 4.0
            # Note (Jiaxin Deng): bf16 round-trip manufactures score ties so the
            # stable-sort tie ordering is exercised, not just clean floats.
            logits = logits.to(torch.bfloat16).float()
            if tie:
                logits = (logits * 4).round() / 4.0
            params = dict(
                temperature=torch.full((rows,), temp, device=device),
                top_p=torch.full((rows,), top_p, device=device),
                top_k=torch.full((rows,), top_k, device=device, dtype=torch.long),
                seeds=torch.randint(
                    0, 2**62, (rows,), device=device, dtype=torch.long, generator=gen
                ),
                positions=torch.randint(
                    0, 10000, (rows,), device=device, dtype=torch.long, generator=gen
                ),
            )
            expected = sample_seeded_branchless(logits, **params)
            actual = sample_seeded_fused(logits, **params)
            assert torch.equal(expected, actual)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_sample_seeded_fused_mixed_rows_and_greedy_fallback() -> None:
    from sglang_omni.models.moss_tts.sampling_kernels import (
        sample_seeded_branchless,
        sample_seeded_fused,
    )

    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(7)
    rows, vocab = 8, 1025
    logits = (torch.randn(rows, vocab, device=device, generator=gen) * 4.0).to(
        torch.bfloat16
    ).float()
    params = dict(
        temperature=torch.tensor(
            [1.7, 0.0, 1.0, 0.0, 1.2, 1.7, 0.5, 1.0], device=device
        ),
        top_p=torch.tensor([0.8, 0.8, 1.0, 0.0, 0.5, 0.9, 0.2, 0.7], device=device),
        top_k=torch.tensor([25, 25, 0, 50, 1, 64, 7, 2000], device=device),
        seeds=torch.randint(
            0, 2**62, (rows,), device=device, dtype=torch.long, generator=gen
        ),
        positions=torch.arange(rows, device=device, dtype=torch.long),
    )
    expected = sample_seeded_branchless(logits, **params)
    actual = sample_seeded_fused(logits, **params)
    assert torch.equal(expected, actual)


def test_sample_seeded_fused_rejects_large_vocab() -> None:
    from sglang_omni.models.moss_tts.sampling_kernels import (
        MAX_FUSED_SAMPLE_VOCAB,
        sample_seeded_fused,
    )

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda")
    rows = 2
    vocab = MAX_FUSED_SAMPLE_VOCAB + 1
    with pytest.raises(ValueError, match="vocab"):
        sample_seeded_fused(
            torch.randn(rows, vocab, device=device),
            temperature=torch.ones(rows, device=device),
            top_p=torch.ones(rows, device=device),
            top_k=torch.full((rows,), 25, device=device, dtype=torch.long),
            seeds=torch.zeros(rows, device=device, dtype=torch.long),
            positions=torch.arange(rows, device=device, dtype=torch.long),
        )
