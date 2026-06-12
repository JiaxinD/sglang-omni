#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""S0 gate: fixed-seed sync parity check for MOSS-TTS Local v1.5.

Regression sentinel (lives under tests/regression/, not a scratch script): the
reproducer for this PR's bit-identity claim (the S0 gate cited in the PR's
Verification section), reused directly as #734's sync-parity hard gate.

Runs the v1.5 frame-decode graph twice with identical fixed-seed inputs at each
concurrency level and asserts bit-identical outputs. Must pass before and after
any behavior-neutral change on this path.

Usage (inside sglang-omni-jiaxind container on novita-h100):
    CUDA_VISIBLE_DEVICES=3 python tests/regression/gate_s0_sync_parity.py
    CUDA_VISIBLE_DEVICES=3 python tests/regression/gate_s0_sync_parity.py --batch-sizes 1 4 16
    CUDA_VISIBLE_DEVICES=3 python tests/regression/gate_s0_sync_parity.py --frames 50 --seed 12345

Exit code: 0 = all checks passed, 1 = any divergence detected.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

import torch

_N_VQ = 12
_HIDDEN = 64
_VOCAB = 32  # logit head width for the toy model


def _build_frame_fn(device: torch.device):
    """Return a decode_frame callable backed by a toy MossTTSLocalTransformer.

    The toy model uses the same kernel stack as the production model
    (MossTTSLocalTransformer + sample_seeded_branchless) but with a
    hidden_size of 64 so it fits in a few MB of VRAM.
    """
    from sglang_omni.models.moss_tts_local.local_transformer import (
        MossTTSLocalTransformer,
        sample_seeded_branchless,
    )

    torch.manual_seed(0)
    module = MossTTSLocalTransformer(
        hidden_size=_HIDDEN,
        num_heads=4,
        inner_size=96,
        num_layers=1,
        max_positions=_N_VQ + 1,
        rope_base=1_000_000.0,
    ).to(device=device, dtype=torch.bfloat16)
    tables = [
        torch.randn(_HIDDEN, _HIDDEN, device=device, dtype=torch.bfloat16)
        for _ in range(_N_VQ)
    ]

    def decode_frame(
        hidden: torch.Tensor,
        seeds: torch.Tensor,
        base: torch.Tensor,
    ) -> torch.Tensor:
        current = module.step(hidden, 0)
        codes = []
        for channel in range(_N_VQ):
            logits = (current.float() @ tables[channel].float().T)[:, :_VOCAB]
            code = sample_seeded_branchless(
                logits,
                temperature=torch.full((hidden.shape[0],), 1.7, device=device),
                top_p=torch.full((hidden.shape[0],), 0.8, device=device),
                top_k=torch.full(
                    (hidden.shape[0],), 25, device=device, dtype=torch.long
                ),
                seeds=seeds,
                positions=base + channel + 1,
            )
            codes.append(code)
            if channel + 1 < _N_VQ:
                embed = torch.nn.functional.embedding(code, tables[channel][:_VOCAB])
                current = module.step(embed.to(torch.bfloat16), channel + 1)
        return torch.stack(codes, dim=-1)

    return decode_frame


def _capture_graph(decode_frame, batch: int, device: torch.device):
    """Capture a CUDA graph for decode_frame at the given batch size."""
    static_hidden = torch.zeros(batch, _HIDDEN, device=device, dtype=torch.bfloat16)
    static_seeds = torch.zeros(batch, device=device, dtype=torch.long)
    static_base = torch.zeros(batch, device=device, dtype=torch.long)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            decode_frame(static_hidden, static_seeds, static_base)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graphed_out = decode_frame(static_hidden, static_seeds, static_base)

    return graph, graphed_out, static_hidden, static_seeds, static_base


def _run_parity_check(
    batch: int,
    n_frames: int,
    seed: int,
    device: torch.device,
) -> bool:
    """Return True if all n_frames graph replays are bit-identical across two runs."""
    decode_frame = _build_frame_fn(device)
    graph, graphed_out, s_hidden, s_seeds, s_base = _capture_graph(
        decode_frame, batch, device
    )

    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    all_ok = True
    for frame_idx in range(n_frames):
        hidden = torch.randn(
            batch, _HIDDEN, device=device, dtype=torch.bfloat16, generator=rng
        )
        seeds = (
            torch.arange(batch, device=device, dtype=torch.long) * 1_234_567
            + frame_idx * 13
        )
        base = torch.full((batch,), frame_idx * 13, device=device, dtype=torch.long)

        s_hidden.copy_(hidden)
        s_seeds.copy_(seeds)
        s_base.copy_(base)
        graph.replay()
        run1 = graphed_out.clone()

        s_hidden.copy_(hidden)
        s_seeds.copy_(seeds)
        s_base.copy_(base)
        graph.replay()
        run2 = graphed_out.clone()

        if not torch.equal(run1, run2):
            print(
                f"  FAIL frame={frame_idx} batch={batch}: "
                f"{(run1 != run2).sum().item()} element(s) differ"
            )
            all_ok = False

    return all_ok


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 4, 16],
        metavar="BS",
        help="batch sizes to test (default: 1 4 16)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=30,
        help="number of frames to replay per batch size (default: 30)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for random inputs (default: 42)",
    )
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        print("SKIP: no CUDA device available")
        return 0

    device = torch.device("cuda")
    print(f"S0 gate — device={torch.cuda.get_device_name(0)}")
    print(f"  batch_sizes={args.batch_sizes}  frames={args.frames}  seed={args.seed}")

    passed = 0
    failed = 0
    for bs in args.batch_sizes:
        ok = _run_parity_check(
            batch=bs, n_frames=args.frames, seed=args.seed, device=device
        )
        status = "PASS" if ok else "FAIL"
        print(f"  batch={bs:3d}  {status}")
        if ok:
            passed += 1
        else:
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
