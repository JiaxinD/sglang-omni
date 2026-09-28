# SPDX-License-Identifier: Apache-2.0
"""Hops over a cached prefix reproduce the whole-history causal solve."""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.fun_cosyvoice3.packed_dit import (
    PackedDiT,
    gather_rows,
    pack_rows,
    solve_flow_euler_packed,
)
from sglang_omni.models.fun_cosyvoice3.prefix_cache import (
    BLOCK_FRAMES,
    PrefixCacheRow,
    PrefixKVPool,
    grow_rows,
    release_rows,
    solve_flow_euler_prefix,
)

pytestmark = pytest.mark.accelerator

cosyvoice_dit = pytest.importorskip("cosyvoice.flow.DiT.dit")

CHUNK = 50
CHANNELS = 80
HEADS, HEAD_DIM, LAYERS = 4, 32, 3


def make_estimator() -> PackedDiT:
    torch.manual_seed(0)
    dit = (
        cosyvoice_dit.DiT(
            dim=HEADS * HEAD_DIM,
            depth=LAYERS,
            heads=HEADS,
            dim_head=HEAD_DIM,
            ff_mult=2,
            mel_dim=CHANNELS,
            mu_dim=CHANNELS,
            spk_dim=CHANNELS,
            out_channels=CHANNELS,
            static_chunk_size=CHUNK,
            num_decoding_left_chunks=-1,
        )
        .cuda()
        .eval()
    )
    # Note (Jiaxin Deng): a visible conv bias makes zero context differ from
    # zero padding after the first conv, which the cached path must reproduce.
    with torch.no_grad():
        for conv in (
            dit.input_embed.conv_pos_embed.conv1[0],
            dit.input_embed.conv_pos_embed.conv2[0],
        ):
            conv.bias.fill_(0.5)
    for module in dit.modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d)):
            module.to(torch.bfloat16)
        else:
            pass
    estimator = PackedDiT(dit, device="cuda")
    if not estimator.is_ragged:
        pytest.skip("requires FA3")
    else:
        pass
    return estimator


def closeness(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual - expected).abs().max() / expected.std()).item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_prefix_hops_match_whole_history_hops() -> None:
    estimator = make_estimator()
    device = torch.device("cuda")
    dtype = torch.bfloat16
    pool = PrefixKVPool(
        layers=LAYERS,
        steps=10,
        heads=HEADS,
        head_dim=HEAD_DIM,
        frames=32 * BLOCK_FRAMES,
        device=device,
        dtype=dtype,
    )
    torch.manual_seed(1)
    rows = 2
    # each row: prompt 50 frames of mel condition, hops ending at 100 / 200 / 400
    totals = [100, 200, 400]
    noise = torch.randn(rows, CHANNELS, 400, device=device, dtype=dtype)
    mu = torch.randn(rows, CHANNELS, 400, device=device, dtype=dtype)
    cond = torch.zeros_like(mu)
    cond[:, :, :50] = torch.randn(rows, CHANNELS, 50, device=device, dtype=dtype)
    spks = torch.randn(rows, CHANNELS, device=device, dtype=dtype)
    unit = torch.linspace(0, 1, 11, device=device, dtype=dtype)
    time_span = 1 - torch.cos(unit * 0.5 * torch.pi)
    caches = [(PrefixCacheRow(), PrefixCacheRow()) for _ in range(rows)]
    previous = 0
    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        for total in totals:
            packed = pack_rows([total] * rows, device)
            reference = solve_flow_euler_packed(
                estimator,
                gather_rows(noise[:, :, :total].transpose(1, 2), packed),
                time_span,
                gather_rows(mu[:, :, :total].transpose(1, 2), packed),
                spks,
                gather_rows(cond[:, :, :total].transpose(1, 2), packed),
                packed,
                cfg_rate=0.7,
                streaming=True,
            )
            for pair in caches:
                assert grow_rows(pool, list(pair), [total, total])
            new = [total - previous] * rows
            take = lambda x: torch.cat(
                [x[row, :, previous:total].transpose(0, 1) for row in range(rows)]
            ).unsqueeze(0)
            cached = solve_flow_euler_prefix(
                estimator,
                pool,
                take(noise),
                time_span,
                take(mu),
                spks,
                take(cond),
                new,
                caches,
                cfg_rate=0.7,
            )
            for row in range(rows):
                expected = reference[0, row * total + previous : (row + 1) * total]
                actual = cached[
                    0, row * (total - previous) : (row + 1) * (total - previous)
                ]
                assert closeness(actual, expected) < 0.05, (
                    total,
                    row,
                    closeness(actual, expected),
                )
            assert all(
                pair[0].frames == total and pair[1].frames == total for pair in caches
            )
            previous = total
    used = 32 - len(pool.free_blocks)
    assert used == rows * 2 * ((400 + BLOCK_FRAMES - 1) // BLOCK_FRAMES)
    for pair in caches:
        release_rows(pool, list(pair))
    assert len(pool.free_blocks) == 32


def test_grow_rows_takes_nothing_on_a_shortfall() -> None:
    pool = PrefixKVPool.__new__(PrefixKVPool)
    pool.free_blocks = [0, 1, 2]
    rows = [PrefixCacheRow(), PrefixCacheRow()]
    assert not grow_rows(pool, rows, [BLOCK_FRAMES * 2, BLOCK_FRAMES * 2])
    assert pool.free_blocks == [0, 1, 2] and rows[0].blocks == []
    assert grow_rows(pool, rows, [BLOCK_FRAMES, BLOCK_FRAMES * 2])
    assert rows[0].capacity == BLOCK_FRAMES and rows[1].capacity == BLOCK_FRAMES * 2
    assert pool.free_blocks == []
    release_rows(pool, rows)
    assert sorted(pool.free_blocks) == [0, 1, 2] and rows[0].frames == 0
