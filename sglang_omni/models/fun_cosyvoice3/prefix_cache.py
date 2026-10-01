# SPDX-License-Identifier: Apache-2.0
"""Prefix K/V cache for the chunk-causal streaming DiT.

A causal hop re-solves the whole utterance so far. Under the chunk-causal mask
a frame only attends to its own chunk and the chunks before it, the causal
positional convs only look left and every hop restarts from the same noise, so
a frame whose chunk is complete produces the same K and V at every Euler step
and layer on every later hop. This keeps those in a paged pool and runs a hop
over the frames past them, attending to the cached prefix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise
from typing import Protocol

import torch
import torch._dynamo as dynamo

from sglang_omni.models.fun_cosyvoice3.packed_dit import (
    FA3_PAGE_SIZE,
    PACKED_INDUCTOR_OPTIONS,
    PackedDiT,
    PackedRows,
    gather_rows,
    layer_norm,
    mish,
    pack_rows,
    packed_fa3,
    ragged_fa3,
    rotate_in_place,
    rotated,
)

BLOCK_FRAMES = 64
# Note (Jiaxin Deng): each positional conv has kernel 31, so it reads the 30
# frames before its input frame.
CONV_CONTEXT_FRAMES = 30


class PrefixForward(Protocol):
    def __call__(
        self,
        estimator: PackedDiT,
        keys: list[torch.Tensor],
        values: list[torch.Tensor],
        x: torch.Tensor,
        mu: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        t: torch.Tensor,
        rows: PackedRows,
        attention: PrefixRowAttention,
        rope: tuple[torch.Tensor, torch.Tensor],
        first_context: torch.Tensor,
        second_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


class PrefixKVPool:
    """K and V for every (Euler step, layer) in blocks of BLOCK_FRAMES pages;
    keys[step][layer] is a (pages, 1, heads, head_dim) tensor of its own."""

    def __init__(
        self,
        *,
        layers: int,
        steps: int,
        heads: int,
        head_dim: int,
        frames: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        blocks = max(int(frames) // BLOCK_FRAMES, 0)
        shape = (blocks * BLOCK_FRAMES, FA3_PAGE_SIZE, heads, head_dim)
        # Note (Jiaxin Deng): separate storages, not views of one slab: the
        # compiled hop mutates them in place only when its inputs don't alias.
        self.keys = [
            [torch.empty(shape, device=device, dtype=dtype) for _ in range(layers)]
            for _ in range(steps)
        ]
        self.values = [
            [torch.empty(shape, device=device, dtype=dtype) for _ in range(layers)]
            for _ in range(steps)
        ]
        self.free_blocks: list[int] = list(range(blocks))
        self.device = device
        # Note (Jiaxin Deng): the compile warmup installs the compiled contract.
        self.forward: PrefixForward = forward_prefix

    @property
    def free_frames(self) -> int:
        return len(self.free_blocks) * BLOCK_FRAMES

    def allocate(self, count: int) -> list[int] | None:
        if count > len(self.free_blocks):
            return None
        else:
            taken = self.free_blocks[-count:] if count else []
            del self.free_blocks[len(self.free_blocks) - count :]
            return taken

    def release(self, blocks: list[int]) -> None:
        self.free_blocks.extend(blocks)

    @staticmethod
    def bytes_per_frame(
        *, layers: int, steps: int, heads: int, head_dim: int, dtype: torch.dtype
    ) -> int:
        return (
            2
            * layers
            * steps
            * heads
            * head_dim
            * torch.tensor([], dtype=dtype).element_size()
        )


@dataclass
class PrefixCacheRow:
    """One row's (one CFG twin's) cached frames."""

    blocks: list[int] = field(default_factory=list)
    frames: int = 0
    # (steps, 2, CONV_CONTEXT_FRAMES, dim): each positional conv's input over
    # the last cached frames at each Euler step.
    conv_context: torch.Tensor | None = None

    @property
    def capacity(self) -> int:
        return len(self.blocks) * BLOCK_FRAMES

    def pages(self, device: torch.device) -> torch.Tensor:
        blocks = torch.tensor(self.blocks, dtype=torch.int32, device=device)
        return (
            blocks.unsqueeze(1) * BLOCK_FRAMES
            + torch.arange(BLOCK_FRAMES, dtype=torch.int32, device=device)
        ).reshape(-1)


def grow_rows(
    pool: PrefixKVPool, rows: list[PrefixCacheRow], frames: list[int]
) -> bool:
    """Give every row enough blocks for `frames`; on a shortfall nothing is
    taken and False is returned."""
    needed = [
        max((total + BLOCK_FRAMES - 1) // BLOCK_FRAMES - len(row.blocks), 0)
        for row, total in zip(rows, frames, strict=True)
    ]
    if sum(needed) > len(pool.free_blocks):
        return False
    else:
        for row, count in zip(rows, needed, strict=True):
            taken = pool.allocate(count)
            assert taken is not None
            row.blocks.extend(taken)
        return True


def release_rows(pool: PrefixKVPool, rows: list[PrefixCacheRow]) -> None:
    for row in rows:
        pool.release(row.blocks)
        row.blocks = []
        row.frames = 0
        row.conv_context = None


class PrefixRowAttention:
    """Queries are each row's new frames in chunk segments; keys are the row's
    cached prefix plus its new frames, all addressed through pool pages."""

    def __init__(
        self,
        *,
        prefix: list[int],
        new: list[int],
        pages: list[torch.Tensor],
        chunk_size: int,
        device: torch.device,
    ) -> None:
        seg_rows: list[int] = []
        seg_ends: list[int] = []
        offsets: list[int] = [0]
        for row, (start, count) in enumerate(zip(prefix, new, strict=True)):
            total = start + count
            frame = start
            while frame < total:
                end = min((frame // chunk_size + 1) * chunk_size, total)
                seg_rows.append(row)
                seg_ends.append(end)
                offsets.append(offsets[-1] + end - frame)
                frame = end
        self.cache_seqlens = torch.tensor(seg_ends, dtype=torch.int32, device=device)
        self.cu_seqlens_q = torch.tensor(offsets, dtype=torch.int32, device=device)
        self.max_seqlen_q = max(b - a for a, b in pairwise(offsets))
        widest = max(seg_ends)
        table = torch.zeros(len(seg_ends), widest, dtype=torch.int32, device=device)
        for segment, (row, end) in enumerate(zip(seg_rows, seg_ends, strict=True)):
            assert pages[row].numel() >= end, "row holds fewer pages than frames"
            table[segment, :end] = pages[row][:end]
        self.page_table = table
        # note(ratish): a frame's K and V are final once its whole chunk exists,
        # so a row keeps whole chunks and recomputes the rest on its next hop.
        self.committed = [
            (start + count) // chunk_size * chunk_size
            for start, count in zip(prefix, new, strict=True)
        ]
        self.tail_index = torch.tensor(
            [end - start for end, start in zip(self.committed, prefix, strict=True)],
            device=device,
        ).unsqueeze(1) + torch.arange(CONV_CONTEXT_FRAMES, device=device)
        # every new frame's page, in packed order: where this hop writes K and V
        self.write_index = torch.cat(
            [
                pages[row][start : start + count]
                for row, (start, count) in enumerate(zip(prefix, new, strict=True))
            ]
        ).to(torch.int64)
        # Note (Jiaxin Deng): the row count and width reach the compiled graph
        # only as this tensor's shape, so batches of any shape share one graph.
        slots = torch.full((len(new), max(new)), -1, dtype=torch.int64)
        offset = 0
        for row, count in enumerate(new):
            slots[row, :count] = torch.arange(offset, offset + count)
            offset += count
        self.slots = slots.to(device)

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_pool: torch.Tensor,
        value_pool: torch.Tensor,
        heads: int,
        head_dim: int,
    ) -> torch.Tensor:
        # Note (Jiaxin Deng): heads and head_dim come from the module so the
        # dynamic graph keeps them constant and the reshapes vectorize.
        page_shape = (-1, FA3_PAGE_SIZE, heads, head_dim)
        key_pool.index_copy_(0, self.write_index, key[0].reshape(page_shape))
        value_pool.index_copy_(0, self.write_index, value[0].reshape(page_shape))
        if torch.compiler.is_compiling():
            fa3 = packed_fa3
        else:
            fa3 = ragged_fa3
        out = fa3(
            query[0].reshape(-1, heads, head_dim),
            key_pool,
            value_pool,
            self.cache_seqlens,
            self.page_table,
            self.cu_seqlens_q,
            self.max_seqlen_q,
        )
        return out.reshape(1, -1, heads * head_dim)

    def mark_dynamic(self, rows: PackedRows, absolute: torch.Tensor) -> None:
        dynamo.mark_dynamic(self.page_table, (0, 1))
        dynamo.mark_dynamic(self.cu_seqlens_q, 0)
        dynamo.mark_dynamic(self.cache_seqlens, 0)
        dynamo.mark_dynamic(self.write_index, 0)
        dynamo.mark_dynamic(self.slots, (0, 1))
        dynamo.mark_dynamic(rows.starts_host, 0)
        dynamo.mark_dynamic(rows.row_ids, 0)
        dynamo.mark_dynamic(rows.positions, 0)
        dynamo.mark_dynamic(absolute, 0)


def conv_pos_embed_prefix(
    estimator: PackedDiT,
    h: torch.Tensor,
    rows: PackedRows,
    first_context: torch.Tensor,
    second_context: torch.Tensor,
    attention: PrefixRowAttention,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The two causal positional convs over each row's new frames, each fed
    the last CONV_CONTEXT_FRAMES of its own input from the prefix (zeros for
    an empty prefix, the padding the whole-sequence call uses); each context
    is (rows, CONV_CONTEXT_FRAMES, dim). Returns the new frames' embedding
    and the two next contexts."""
    # Note (Jiaxin Deng): the whole-sequence call zero-pads conv2's input,
    # not conv1's output, so the second conv needs its own cached tail.
    module = estimator.dit.input_embed.conv_pos_embed
    slots = attention.slots
    padded = torch.where(
        (slots >= 0).unsqueeze(-1), h[0][slots.clamp(min=0)], 0.0
    )  # (rows, width, dim)
    tail_index = attention.tail_index
    first_in = torch.cat((first_context.to(padded.dtype), padded), dim=1)
    first_out = mish(module.conv1[0](first_in.permute(0, 2, 1))).permute(0, 2, 1)
    second_in = torch.cat((second_context.to(first_out.dtype), first_out), dim=1)
    second_out = mish(module.conv2[0](second_in.permute(0, 2, 1))).permute(0, 2, 1)
    tail = tail_index.unsqueeze(-1).expand(-1, -1, first_in.shape[2])
    return (
        gather_rows(second_out, rows),
        first_in.gather(1, tail),
        second_in.gather(1, tail),
    )


def forward_prefix(
    estimator: PackedDiT,
    keys: list[torch.Tensor],
    values: list[torch.Tensor],
    x: torch.Tensor,
    mu: torch.Tensor,
    spks: torch.Tensor,
    cond: torch.Tensor,
    t: torch.Tensor,
    rows: PackedRows,
    attention: PrefixRowAttention,
    rope: tuple[torch.Tensor, torch.Tensor],
    first_context: torch.Tensor,
    second_context: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """DiT.forward over the new frames only. keys, values: each layer's
    (pages, 1, heads, head_dim) pool for this Euler step; x, mu, cond, spks:
    (1, total_new, channels); rope covers the rows' absolute positions.
    Returns the vector field and the two next positional-conv contexts."""
    dit = estimator.dit
    t = dit.time_embed(t)
    h = dit.input_embed.proj(torch.cat((x, cond, mu, spks), dim=-1))
    embedded, first_tail, second_tail = conv_pos_embed_prefix(
        estimator, h, rows, first_context, second_context, attention
    )
    h = embedded + h
    residual = h
    for layer, block in enumerate(dit.transformer_blocks):
        attn_norm = block.attn_norm
        modulation = attn_norm.linear(attn_norm.silu(t))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(
            modulation, 6, dim=1
        )
        norm = layer_norm(attn_norm.norm, h) * (1 + scale_msa[:, None])
        norm = norm + shift_msa[:, None]
        attn = block.attn
        normed = norm.to(attn.to_q.weight.dtype)
        query = attn.to_q(normed)
        key = attn.to_k(normed)
        value = attn.to_v(normed)
        if torch.compiler.is_compiling():
            query = rotated(query, *rope)
            key = rotated(key, *rope)
        else:
            rotate_in_place(query, *rope)
            rotate_in_place(key, *rope)
        heads = attn.heads
        out = attention(
            query,
            key,
            value,
            keys[layer],
            values[layer],
            heads,
            attn.inner_dim // heads,
        ).to(query.dtype)
        h = h + gate_msa.unsqueeze(1) * attn.to_out[1](attn.to_out[0](out))
        ff_norm = layer_norm(block.ff_norm, h) * (1 + scale_mlp[:, None])
        ff_norm = ff_norm + shift_mlp[:, None]
        h = h + gate_mlp.unsqueeze(1) * block.ff(ff_norm)
    if dit.long_skip_connection is not None:
        h = dit.long_skip_connection(torch.cat((h, residual), dim=-1))
    else:
        pass
    norm_out = dit.norm_out
    scale, shift = torch.chunk(norm_out.linear(norm_out.silu(t)), 2, dim=1)
    h = layer_norm(norm_out.norm, h) * (1 + scale)[:, None, :] + shift[:, None, :]
    return dit.proj_out(h), first_tail, second_tail


def compile_forward_prefix() -> PrefixForward:
    """The exact dynamic Inductor contract of forward_prefix, the same recipe
    as the packed causal / full contracts."""
    return torch.compile(
        forward_prefix,
        backend="inductor",
        dynamic=True,
        fullgraph=True,
        options=dict(PACKED_INDUCTOR_OPTIONS),
    )


def solve_flow_euler_prefix(
    estimator: PackedDiT,
    pool: PrefixKVPool,
    noise: torch.Tensor,
    time_span: torch.Tensor,
    mu: torch.Tensor,
    spks: torch.Tensor,
    cond: torch.Tensor,
    new: list[int],
    caches: list[tuple[PrefixCacheRow, PrefixCacheRow]],
    *,
    cfg_rate: float,
) -> torch.Tensor:
    """Euler steps over the new frames of each row with classifier free
    guidance; the conditional rows and their unconditional twins each keep
    their own cached prefix. noise, mu, cond: (1, total_new, channels) in row
    order; spks: (rows, channels). Extends every cache by the new frames."""
    device = noise.device
    dtype = spks.dtype
    total = noise.shape[1]
    twins = [pair[0] for pair in caches] + [pair[1] for pair in caches]
    prefix = [row.frames for row in twins]
    counts = list(new) * 2
    twin_rows = pack_rows(counts, device)
    # Note (Jiaxin Deng): the rows stay local for scatter/gather; RoPE alone
    # sees each frame's absolute position in its row.
    absolute = (
        twin_rows.positions + torch.tensor(prefix, device=device)[twin_rows.row_ids]
    )
    attention = PrefixRowAttention(
        prefix=prefix,
        new=counts,
        pages=[row.pages(device) for row in twins],
        chunk_size=estimator.chunk_size,
        device=device,
    )
    freqs, scale = estimator.dit.rotary_embed.forward_from_seq_len(
        max(p + n for p, n in zip(prefix, counts, strict=True))
    )
    assert not isinstance(scale, torch.Tensor), "the DiT's RoPE has no xpos scale"
    freqs = freqs[:, absolute]
    rope = (freqs.cos(), freqs.sin())
    steps = len(time_span) - 1
    dim = int(estimator.dit.input_embed.proj.out_features)
    contexts: list[torch.Tensor] = []
    for row in twins:
        if row.conv_context is None:
            contexts.append(
                torch.zeros(
                    steps, 2, CONV_CONTEXT_FRAMES, dim, device=device, dtype=dtype
                )
            )
        else:
            contexts.append(row.conv_context)
    context = torch.stack(contexts, dim=1)  # (steps, twin rows, 2, ctx, dim)
    next_context = torch.empty_like(context)
    mu_cfg = torch.cat((mu, torch.zeros_like(mu)), dim=1)
    cond_cfg = torch.cat((cond, torch.zeros_like(cond)), dim=1)
    spks_cfg = torch.cat((spks, torch.zeros_like(spks)), dim=0)
    spks_cfg = spks_cfg[twin_rows.row_ids].unsqueeze(0)
    flow_time = torch.zeros(1, device=device, dtype=dtype)
    forward = pool.forward
    if forward is not forward_prefix:
        attention.mark_dynamic(twin_rows, absolute)
    else:
        pass
    x = noise
    t, dt = time_span[0], time_span[1] - time_span[0]
    for step in range(steps):
        flow_time[:] = t
        vector_field, next_context[step, :, 0], next_context[step, :, 1] = forward(
            estimator,
            pool.keys[step],
            pool.values[step],
            torch.cat((x, x), dim=1),
            mu_cfg,
            spks_cfg,
            cond_cfg,
            flow_time,
            twin_rows,
            attention,
            rope,
            context[step, :, 0],
            context[step, :, 1],
        )
        conditional = vector_field[:, :total]
        unconditional = vector_field[:, total:]
        x = x + dt * ((1.0 + cfg_rate) * conditional - cfg_rate * unconditional)
        t = t + dt
        if step < steps - 1:
            dt = time_span[step + 2] - t
        else:
            pass
    for index, row in enumerate(twins):
        row.frames = attention.committed[index]
        row.conv_context = next_context[:, index].clone()
    return x.float()
