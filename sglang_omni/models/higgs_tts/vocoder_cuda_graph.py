# SPDX-License-Identifier: Apache-2.0
"""CUDA-graph runner for the Higgs TTS vocoder (codec) decode.

One decode of a streaming window is ~440 kernel launches over ~37 distinct
kernels (residual-VQ gather, the fc2 GEMM, then the DAC transposed-conv stack),
and at the batch-1 window sizes serving uses it is launch-bound. A graph replays
the whole chain as one launch. Shapes are keyed exactly: the decoder's receptive
field reaches the end of the window, and batch padding changes which
cuBLAS/cuDNN kernels run, so neither dimension can be padded and still land on
the same bits.

The chain is stateless -- ``HiggsAudioV2TokenizerModel.decode`` is a pure
function of the codes -- so the persistent-state registry stays empty here.

Capture happens once at startup and the runner then seals, so serving only ever
replays or falls back to eager.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from types import MethodType
from typing import NamedTuple

import torch

from sglang_omni.cuda_graph import KeyedGraphCache, PersistentStateRegistry

logger = logging.getLogger(__name__)

VOCODER_CUDA_GRAPH_ENV = "SGLANG_OMNI_HIGGS_VOCODER_CUDA_GRAPH"

_ORIGINAL_RVQ_DECODE_ATTR = "_sglang_omni_original_rvq_decode"


class CapturedVocoderGraph(NamedTuple):
    graph: torch.cuda.CUDAGraph
    static_codes: torch.Tensor
    static_audio: torch.Tensor


def capture_safe_rvq_decode(self, codes: torch.Tensor) -> torch.Tensor:
    """Upstream residual-VQ decode without its ``torch.tensor(0.0)`` seed.

    Note: (Jiaxin Deng) that seed is an unpinned host-to-device copy, which
    capture rejects outright. Accumulating from the first quantizer instead is
    bitwise identical because the 0-dim fp32 zero is a wrapped scalar and so
    never promoted the accumulation out of the codec's dtype.
    """
    quantized_out = None
    for i, indices in enumerate(codes):
        quantized = self.quantizers[i].decode(indices).to(codes.device)
        quantized_out = (
            quantized if quantized_out is None else quantized_out + quantized
        )
    return quantized_out


def patch_rvq_decode_for_cuda_graph(model) -> None:
    quantizer = getattr(model, "quantizer", None)
    if quantizer is None or hasattr(quantizer, _ORIGINAL_RVQ_DECODE_ATTR):
        return
    setattr(quantizer, _ORIGINAL_RVQ_DECODE_ATTR, quantizer.decode)
    quantizer.decode = MethodType(capture_safe_rvq_decode, quantizer)


class HiggsVocoderCudaGraphRunner:
    """Warmup-captured, sealed replay of exact-shape codec decode graphs."""

    def __init__(
        self,
        model,
        *,
        batch_sizes: Iterable[int] = (1,),
        max_keys: int = 256,
        warmup_iters: int = 3,
        min_free_gb: float = 3.0,
    ) -> None:
        self._model = model
        self._device = next(model.parameters()).device
        self._num_codebooks = int(model.config.num_quantizers)
        self._warmup_iters = int(warmup_iters)
        self._min_free_bytes = int(float(min_free_gb) * (1024**3))
        self._sealed = False
        self._persistent_state = PersistentStateRegistry()
        self._cache = KeyedGraphCache(
            name="Higgs vocoder",
            batch_sizes=batch_sizes,
            env_var=VOCODER_CUDA_GRAPH_ENV,
            max_keys=max_keys,
        )

    @property
    def persistent_state(self) -> PersistentStateRegistry:
        """Empty: the codec decode carries nothing across calls."""
        return self._persistent_state

    def captured_shapes(self) -> list[tuple[int, int]]:
        return sorted(self._cache.graphs)

    def captured_frames(self) -> list[int]:
        return sorted({frames for _, frames in self._cache.graphs})

    def captured_entry(self, batch: int, frames: int) -> CapturedVocoderGraph | None:
        return self._cache.graphs.get((batch, frames))

    @torch.no_grad()
    def _capture(self, batch: int, frames: int) -> CapturedVocoderGraph:
        static_codes = torch.zeros(
            (batch, self._num_codebooks, frames), dtype=torch.long, device=self._device
        )
        # Note: (Jiaxin Deng) warm on a side stream first so lazy conv algo picks and
        # workspace allocations settle outside the capture.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(self._warmup_iters):
                self._model.decode(static_codes)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self._persistent_state.reset()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
            graph, pool=self._cache.memory_pool(), capture_error_mode="thread_local"
        ):
            static_audio = self._model.decode(static_codes).audio_values
        self._persistent_state.snapshot_addresses()
        return CapturedVocoderGraph(graph, static_codes, static_audio)

    def _has_vram_headroom(self) -> bool:
        free, _ = torch.cuda.mem_get_info(self._device)
        if free >= self._min_free_bytes:
            return True
        logger.warning(
            "Higgs vocoder graphs: %.1f GiB free is below the %.1f GiB headroom; "
            "leaving the remaining shapes on eager",
            free / 1024**3,
            self._min_free_bytes / 1024**3,
        )
        return False

    @torch.no_grad()
    def warmup(self, shapes: Iterable[tuple[int, int]]) -> None:
        """Capture one graph per ``(batch, frames)``, then seal."""
        if self._sealed:
            logger.warning("Higgs vocoder graph warmup called after seal; ignoring")
            return
        self._sealed = True
        if not self._cache.enabled:
            return
        patch_rvq_decode_for_cuda_graph(self._model)
        keys = [
            (int(batch), int(frames))
            for batch, frames in shapes
            if self._cache.bucket_for(int(batch)) == int(batch)
        ]
        with torch.cuda.device(self._device):
            captured = self._cache.warmup(
                keys,
                lambda key: self._capture(*key),
                precheck=self._has_vram_headroom,
            )
        logger.info(
            "Higgs vocoder CUDA graphs sealed: %d of %d shapes captured",
            captured,
            len(set(keys)),
        )

    @torch.no_grad()
    def decode(self, codes_BNT: torch.Tensor) -> torch.Tensor | None:
        """Replay the graph for this exact shape, or ``None`` to run eagerly.

        Note: (Jiaxin Deng) the returned audio is cloned out of the graph's
        pool-allocated static buffer, so callers may hold it across other
        replays; returning the view itself would not survive that.
        """
        if not codes_BNT.is_cuda:
            return None
        batch, num_codebooks, frames = codes_BNT.shape
        if num_codebooks != self._num_codebooks:
            return None
        # Note: (Jiaxin Deng) an exact bucket only. Padding the batch into a wider
        # graph changes the selected kernels and so the output bits.
        if self._cache.bucket_for(batch) != batch:
            return None
        entry = self._cache.graphs.get((batch, frames))
        if entry is None:
            return None
        entry.static_codes.copy_(codes_BNT)
        entry.graph.replay()
        return entry.static_audio.clone()


__all__ = [
    "VOCODER_CUDA_GRAPH_ENV",
    "CapturedVocoderGraph",
    "HiggsVocoderCudaGraphRunner",
    "capture_safe_rvq_decode",
    "patch_rvq_decode_for_cuda_graph",
]
