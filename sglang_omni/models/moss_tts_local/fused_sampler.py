# SPDX-License-Identifier: Apache-2.0
"""MOSS-TTS-Local bit-identical fused seeded sampler (murmur + float64 Gumbel + argmax).

One Triton kernel that reproduces sglang's ``multinomial_with_seed`` bit-for-bit but folds its
~15 tiny float64 ops (per-element murmur hash, two float64 logs, add, argmax) into a single pass.
Used ONLY on the MOSS frame-decode path (``sample_seeded_branchless``); the shared
``sglang.srt.layers.sampler.multinomial_with_seed`` is NOT touched, so other models are unaffected.
Enabled by ``MOSS_FUSED_SAMPLER=1`` (default off).

The murmur sub-functions (``rotl32``/``fmix32``/``murmur3_mix``) are reused verbatim from sglang so
the hash is bit-identical by construction; the Gumbel sequence and first-index argmax are reproduced
exactly. Bit-identity across params/edges is gated by
``tests/unit_test/moss_tts_local/test_fused_sampler.py``.

Tracks sgl-project/sglang#25133 (open): when its hash->float fix lands
(``x = (h + 0.5) / 2**32`` instead of ``/ (2**32 - 1)``), sync the conversion below and re-run the gate.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl
from sglang.srt.layers.utils.hash import fmix32, murmur3_mix

_ENABLED = os.environ.get("MOSS_FUSED_SAMPLER", "0").lower() not in ("0", "", "false")


def enabled() -> bool:
    """True if the fused MOSS sampler is opted in (MOSS_FUSED_SAMPLER=1)."""
    return _ENABLED


@triton.jit
def _fused_seeded_sample_kernel(
    seed_ptr, pos_ptr, logprobs_ptr, out_ptr, num_cols, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    valid = cols < num_cols

    seed = tl.load(seed_ptr + row).to(tl.uint64)
    pos = tl.load(pos_ptr + row).to(tl.uint32)

    # murmur3 over (seed_low, seed_high, pos, col) + fmix -- same ops as the shared sgl kernel.
    h = tl.zeros([BLOCK], dtype=tl.uint32)
    h = murmur3_mix(h, (seed & 0xFFFFFFFF).to(tl.uint32))
    h = murmur3_mix(h, ((seed >> 32) & 0xFFFFFFFF).to(tl.uint32))
    h = murmur3_mix(h, pos)
    h = murmur3_mix(h, cols.to(tl.uint32))
    h ^= 16
    h = fmix32(h)

    # gumbel: reproduce x.log_().clamp_(min=f64min).neg_().log_().neg_() exactly, in float64.
    x = h.to(tl.float64) / 4294967295.0
    t = tl.log(x)
    t = tl.maximum(t, -1.7976931348623157e308)
    t = -t
    g = -tl.log(t)

    logp = tl.load(logprobs_ptr + row * num_cols + cols, mask=valid, other=0.0).to(
        tl.float64
    )
    val = tl.where(valid, g + logp, float("-inf"))

    best = tl.max(val, axis=0)
    idx = tl.where(
        val == best, cols, num_cols
    )  # first index on ties (matches torch.argmax)
    tl.store(out_ptr + row, tl.min(idx, axis=0))


def fused_multinomial_with_seed(
    logprobs: torch.Tensor, seed: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    """Bit-identical drop-in for sglang ``multinomial_with_seed``, MOSS-path only. Returns (n, 1)."""
    n, m = logprobs.shape
    out = torch.empty(n, device=logprobs.device, dtype=torch.long)
    _fused_seeded_sample_kernel[(n,)](
        seed.contiguous(),
        positions.contiguous(),
        logprobs.contiguous(),
        out,
        m,
        BLOCK=triton.next_power_of_2(m),
    )
    return out.view(n, 1)
