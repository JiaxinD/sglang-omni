# SPDX-License-Identifier: Apache-2.0
"""Bit-identity gate for the MOSS-local fused seeded sampler.

The fused Triton kernel MUST be bit-for-bit identical to sglang's multinomial_with_seed across
the sampling param + edge matrix, or the MOSS frame-decode would change outputs. Any mismatch
fails the gate (the cut is dead). GPU + triton required.
"""

from __future__ import annotations

import itertools

import pytest
import torch

pytestmark = pytest.mark.gpu

_HAS_CUDA = torch.cuda.is_available()
if _HAS_CUDA:
    from sglang.srt.layers.sampler import multinomial_with_seed

    from sglang_omni.models.moss_tts_local.fused_sampler import (
        fused_multinomial_with_seed,
    )


def _logprobs(bs: int, vocab: int, kind: str, device: str) -> torch.Tensor:
    g = torch.randn(bs, vocab, device=device, dtype=torch.float32)
    if kind == "random":
        return g
    if kind == "peaked":
        return g * 20.0
    if kind == "flat":  # ties: argmax tie-break must match
        return torch.zeros(bs, vocab, device=device, dtype=torch.float32)
    if kind == "masked":  # mimic top_k/top_p masking: keep top 25, rest -inf
        k = min(25, vocab)
        thr = g.topk(k, dim=-1).values[:, -1:]
        return g.masked_fill(g < thr, float("-inf"))
    if kind == "single":  # all -inf except column 0
        m = torch.full((bs, vocab), float("-inf"), device=device, dtype=torch.float32)
        m[:, 0] = g[:, 0]
        return m
    if kind == "allinf":  # fallback territory: every candidate masked
        return torch.full(
            (bs, vocab), float("-inf"), device=device, dtype=torch.float32
        )
    raise ValueError(kind)


@pytest.mark.skipif(not _HAS_CUDA, reason="fused sampler needs CUDA + triton")
@pytest.mark.parametrize("vocab", [2, 1024])
def test_fused_seeded_sampler_is_bit_identical(vocab: int) -> None:
    device = "cuda"
    bs = 8
    seeds = [
        0,
        1,
        7847,
        12345,
        2**31 - 1,
    ]  # 7847/pos 12345 is the known 0xFFFFFFFF edge (sgl #25106)
    positions = [0, 1, 12, 12345]
    kinds = ["random", "peaked", "flat", "masked", "single", "allinf"]
    if vocab == 2:
        kinds = ["random", "flat", "single", "allinf"]
    torch.manual_seed(0)

    mismatches = []
    for s, p, kind in itertools.product(seeds, positions, kinds):
        lp = _logprobs(bs, vocab, kind, device)
        seed = torch.full((bs,), s, device=device, dtype=torch.long)
        pos = torch.full((bs,), p, device=device, dtype=torch.long)
        ref = multinomial_with_seed(lp, seed, pos)
        fused = fused_multinomial_with_seed(lp, seed, pos)
        if not torch.equal(ref, fused):
            mismatches.append((kind, s, p))

    assert not mismatches, (
        f"fused sampler not bit-identical to multinomial_with_seed (vocab={vocab}): {mismatches[:10]}"
    )
