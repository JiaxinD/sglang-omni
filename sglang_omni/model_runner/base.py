# SPDX-License-Identifier: Apache-2.0
"""Base model runner — shared execute() pipeline for all AR models.

Handles: ForwardBatch construction, phase-aware pre/post hooks, forward
pass, sampling, logit post-processing, and output extraction.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch

from sglang_omni.scheduling.types import ModelRunnerOutput, RequestOutput

logger = logging.getLogger(__name__)


@dataclass
class _PendingStep:
    """One decode step launched on the GPU but not yet consumed on the host.

    Async-decode (one-step lookahead) bookkeeping: a launched step has its
    forward + on-GPU sample enqueued, its collect-staging buffer async-copied
    (D2H) into ``host_buf``, and ``event`` recorded right after that copy.
    ``execute_resolve`` later waits on ``event`` and reads ``host_buf``.

    Invariant: at most one ``_PendingStep`` is live at a time (see
    ``ModelRunner._pending``). ``host_buf`` is pinned and ping-ponged between
    two buffers so resolve(N) can read one while launch(N+1)'s D2H writes the
    other (a CPU-read vs GPU-write race not covered by stream ordering —
    design.md §1.4).
    """

    event: Any  # torch.cuda.Event, recorded right after the async D2H copy
    host_buf: Any  # pinned host tensor holding this step's staging snapshot
    requests: list  # this step's sched_output.requests (resolve routing)
    schedule_batch: Any  # to set .output_ids during resolve
    batch_result: Any  # carries logits_output (device of next_token_ids)
    n_real: int  # number of real (non-padding) rows this step


class ModelRunner:
    """Base AR model runner.

    Subclasses provide phase-specific behavior:
      - prefill hooks for extend/prompt processing
      - decode hooks for single-step autoregressive decode processing
    """

    def __init__(self, tp_worker: Any, output_processor: Any):
        self.tp_worker = tp_worker
        self.output_processor = output_processor
        self.device = torch.device(f"cuda:{tp_worker.gpu_id}")
        self.model = tp_worker.model_runner.model

        # Async decode (one-step lookahead). Inert unless ``_async_enabled``
        # is set (commit 5 wires it from server_args.enable_async_decode).
        self._async_enabled: bool = False
        self._pending: _PendingStep | None = None
        self._staging_slot: int = 0
        self._host_staging_buffers: list[torch.Tensor] = []

    def _next_host_staging(self, device_staging: torch.Tensor) -> torch.Tensor:
        """Return a pinned host buffer mirroring ``device_staging``'s full
        shape, ping-ponging between two buffers on each call.

        Two buffers are required: resolve(N) reads one on the host while
        launch(N+1)'s async D2H writes the other. That CPU-read vs GPU-write
        overlap is not protected by single-stream ordering (design.md §1.4).
        Buffers are allocated lazily on first use (the base runner does not
        know the model-specific staging shape at construction time).
        """
        if not self._host_staging_buffers:
            self._host_staging_buffers = [
                torch.empty(
                    device_staging.shape,
                    dtype=device_staging.dtype,
                    device="cpu",
                    pin_memory=True,
                )
                for _ in range(2)
            ]
        buf = self._host_staging_buffers[self._staging_slot]
        self._staging_slot ^= 1
        return buf

    def execute(self, scheduler_output: Any) -> ModelRunnerOutput:
        """Full pipeline: build batch → prepare → forward → post → sample → output."""
        from sglang.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            ForwardBatch,
        )

        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)

        schedule_batch = scheduler_output.batch_data
        if schedule_batch is None:
            return ModelRunnerOutput(outputs={}, req_ids=[], req_id_to_index={})

        model_worker_batch = schedule_batch.get_model_worker_batch()
        is_prefill = bool(schedule_batch.forward_mode.is_extend())

        capture_hidden_mode = (
            self.requested_capture_hidden_mode_prefill(
                schedule_batch, scheduler_output.requests
            )
            if is_prefill
            else self.requested_capture_hidden_mode_decode(
                schedule_batch, scheduler_output.requests
            )
        )
        if capture_hidden_mode is not None:
            model_worker_batch.capture_hidden_mode = capture_hidden_mode
        elif self.output_processor._capture_hidden:
            model_worker_batch.capture_hidden_mode = CaptureHiddenMode.LAST

        forward_batch = ForwardBatch.init_new(
            model_worker_batch, self.tp_worker.model_runner
        )

        # Hook: model-specific preparation. Returns batch_result if it ran
        # a custom forward path, or None for standard forward.
        batch_result = (
            self.prepare_prefill(
                forward_batch, schedule_batch, scheduler_output.requests
            )
            if is_prefill
            else self.prepare_decode(
                forward_batch, schedule_batch, scheduler_output.requests
            )
        )

        if batch_result is None:
            # Standard forward path
            batch_result = self.tp_worker.forward_batch_generation(forward_batch)

        if (
            not schedule_batch.is_prefill_only
            and batch_result.next_token_ids is None
            and (
                self.sample_before_post_prefill(
                    forward_batch, schedule_batch, scheduler_output.requests
                )
                if is_prefill
                else self.sample_before_post_decode(
                    forward_batch, schedule_batch, scheduler_output.requests
                )
            )
        ):
            batch_result.next_token_ids = self._sample_next_token_ids(
                batch_result.logits_output,
                forward_batch,
                schedule_batch,
                scheduler_output.requests,
            )
            schedule_batch.output_ids = batch_result.next_token_ids

        # Hook: model-specific post-processing
        if is_prefill:
            self.post_prefill(
                batch_result,
                forward_batch,
                schedule_batch,
                scheduler_output.requests,
            )
        else:
            self.post_decode(
                batch_result,
                forward_batch,
                schedule_batch,
                scheduler_output.requests,
            )

        # Sampling + logit processing
        if schedule_batch.is_prefill_only:
            if batch_result.next_token_ids is None:
                batch_result.next_token_ids = torch.zeros(
                    len(model_worker_batch.seq_lens),
                    dtype=torch.long,
                    device=model_worker_batch.input_ids.device,
                )
        elif batch_result.next_token_ids is None:
            batch_result.next_token_ids = self._sample_next_token_ids(
                batch_result.logits_output,
                forward_batch,
                schedule_batch,
                scheduler_output.requests,
            )
        schedule_batch.output_ids = batch_result.next_token_ids

        # Output extraction
        outputs = self.output_processor.process(batch_result, scheduler_output)
        self.post_process_outputs(batch_result, scheduler_output, outputs)
        for sched_req in scheduler_output.requests:
            data = sched_req.data
            data.generation_steps = int(data.generation_steps) + 1
            req_output = outputs[sched_req.request_id]
            extra = req_output.extra
            if isinstance(extra, dict) and extra:
                data.extra_model_outputs.update(extra)
        req_ids = [req.request_id for req in scheduler_output.requests]
        req_id_to_index = {req_id: idx for idx, req_id in enumerate(req_ids)}

        return ModelRunnerOutput(
            outputs=outputs,
            req_ids=req_ids,
            req_id_to_index=req_id_to_index,
            can_run_cuda_graph=bool(batch_result.can_run_cuda_graph),
        )

    # ------------------------------------------------------------------
    # Hooks — override in subclasses
    # ------------------------------------------------------------------

    def prepare_prefill(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> Any | None:
        """Called before prefill forward.

        Return a batch result if the subclass handled the forward itself,
        or None to use the standard tp_worker forward path.
        """
        return None

    def prepare_decode(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> Any | None:
        """Called before decode forward."""
        return None

    def post_prefill(
        self, result: Any, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> None:
        """Called after prefill forward."""

    def post_decode(
        self, result: Any, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> None:
        """Called after decode forward."""

    def post_process_outputs(
        self,
        result: Any,
        scheduler_output: Any,
        outputs: dict[str, RequestOutput],
    ) -> None:
        """Called after output tokens are materialized into RequestOutput."""

    def sample_before_post_prefill(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> bool:
        return False

    def sample_before_post_decode(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> bool:
        return False

    def requested_capture_hidden_mode_prefill(
        self, schedule_batch: Any, requests: list
    ) -> Any | None:
        return None

    def requested_capture_hidden_mode_decode(
        self, schedule_batch: Any, requests: list
    ) -> Any | None:
        return None

    # ------------------------------------------------------------------
    # Shared logit processing
    # ------------------------------------------------------------------

    def _sample_next_token_ids(
        self,
        logits_output: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> Any:
        self._apply_repetition_penalty(logits_output, requests)
        self._apply_codec_suppress_tokens(logits_output, requests)
        return self.tp_worker.model_runner.sample(logits_output, forward_batch)

    def _apply_repetition_penalty(self, logits_output: Any, requests: list) -> None:
        logits = logits_output.next_token_logits
        if logits is None or logits.ndim != 2:
            return
        vocab = logits.shape[1]
        device = logits.device
        rep_rows: list[int] = []
        rep_toks: list[int] = []
        rep_penalties: list[float] = []
        for row_idx, sched_req in enumerate(requests):
            data = sched_req.data
            req = data.req
            penalty = req.sampling_params.repetition_penalty
            if penalty == 1.0:
                continue
            output_ids = req.output_ids
            if not output_ids:
                continue
            unique = {int(t) for t in output_ids if 0 <= int(t) < vocab}
            if not unique:
                continue
            rep_rows.extend([row_idx] * len(unique))
            rep_toks.extend(unique)
            rep_penalties.extend([float(penalty)] * len(unique))
        if rep_rows:
            orig_dtype = logits.dtype
            rows_t = torch.tensor(rep_rows, dtype=torch.long, device=device)
            toks_t = torch.tensor(rep_toks, dtype=torch.long, device=device)
            pens_t = torch.tensor(rep_penalties, dtype=torch.float32, device=device)
            scores = logits[rows_t, toks_t].to(torch.float32)
            scores = torch.where(scores > 0, scores / pens_t, scores * pens_t)
            logits[rows_t, toks_t] = scores.to(orig_dtype)

    def _apply_codec_suppress_tokens(self, logits_output: Any, requests: list) -> None:
        logits = logits_output.next_token_logits
        if logits is None or logits.ndim != 2:
            return
        vocab = logits.shape[1]
        device = logits.device
        sup_rows: list[int] = []
        sup_toks: list[int] = []
        for row_idx, sched_req in enumerate(requests):
            data = sched_req.data
            suppress_tokens = data.suppress_tokens
            if not suppress_tokens:
                req = data.req
                suppress_tokens = getattr(req, "_codec_suppress_tokens", None)
            if not suppress_tokens:
                continue
            for token_id in suppress_tokens:
                tok = int(token_id)
                if 0 <= tok < vocab:
                    sup_rows.append(row_idx)
                    sup_toks.append(tok)
        if sup_rows:
            logits[
                torch.tensor(sup_rows, dtype=torch.long, device=device),
                torch.tensor(sup_toks, dtype=torch.long, device=device),
            ] = float("-inf")
