# SPDX-License-Identifier: Apache-2.0
"""Owned inputs for replaying the native Qwen3-ASR audio encoder.

Capture belongs to a separate calibration run: copies and serialization are
not part of a performance measurement. Call on the encoder's producer stream
after encoding, before request items are converted to cached embeddings.
Enable with Qwen3-ASR factory fields ``encoder_capture_directory`` and
``encoder_capture_max_batches`` (default 16 per encoder service). Each service
uses its own subdirectory. Only executed, successful encoding calls are saved;
cache hits and failed calls are absent. Saved batches are the first batches,
not a representative sampling guarantee. The harness must bind the capture
directory to its checkpoint, source, hardware, workload, and execution settings.
This artifact describes an encoder batch, not encoder-plus-LM group capacity.
"""

import contextlib
import copy
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang_omni.profiler.work_units import current_work_unit

logger = logging.getLogger(__name__)


def _cpu_copy(tensor):
    return None if tensor is None else tensor.detach().to("cpu", copy=True)


@dataclass
class QwenEncoderSnapshot:
    inputs: list[dict]
    output: torch.Tensor
    metadata: dict

    @classmethod
    def capture(cls, items, output: torch.Tensor, *, metadata: dict):
        """Take blocking, independent copies in actual batch order.

        Feature masks and token counts affect packing and graph selection.
        Preserve missing token counts instead of inferring them from shape.
        """
        inputs = []
        for item in items:
            data = item.model_specific_data or {}
            inputs.append(
                {
                    "feature": _cpu_copy(item.feature),
                    "feature_attention_mask": _cpu_copy(
                        getattr(item, "feature_attention_mask", None)
                    ),
                    "model_specific_data": (
                        {"num_audio_tokens": data["num_audio_tokens"]}
                        if "num_audio_tokens" in data
                        else {}
                    ),
                    "audio_fingerprint": (
                        item.audio_fingerprint
                        if isinstance(getattr(item, "audio_fingerprint", None), str)
                        else None
                    ),
                    "source_device": str(item.feature.device),
                    "source_stride": list(item.feature.stride()),
                }
            )
        return cls(inputs, _cpu_copy(output), copy.deepcopy(metadata))

    def restore_items(self):
        """Return fresh CPU inputs so one replay cannot mutate another."""
        return [
            SimpleNamespace(
                feature=row["feature"].clone(),
                feature_attention_mask=_cpu_copy(row["feature_attention_mask"]),
                model_specific_data=copy.deepcopy(row["model_specific_data"]),
                audio_fingerprint=row["audio_fingerprint"],
            )
            for row in self.inputs
        ]

    def save(self, path: Path):
        """Store tensors and primitive metadata without serializing live items."""
        torch.save(
            {"inputs": self.inputs, "output": self.output, "metadata": self.metadata},
            path,
        )

    @classmethod
    def load(cls, path: Path):
        return cls(**torch.load(path, map_location="cpu", weights_only=True))


class QwenEncoderCapture:
    """Bounded synchronous capture for an explicit calibration session."""

    def __init__(self, directory: Path, *, max_batches: int, metadata: dict):
        if max_batches < 1:
            raise ValueError("max_batches must be positive")
        directory.mkdir(parents=True, exist_ok=False)
        self.directory = directory
        self.max_batches = max_batches
        self.metadata = copy.deepcopy(metadata)
        self.saved = 0
        self.failed = False
        self.closed = False
        (directory / "session.json").write_text(
            json.dumps({"metadata": metadata, "max_batches": max_batches}),
            encoding="utf-8",
        )
        self._write_status("collecting")

    def _write_status(self, state):
        try:
            temporary = self.directory / "status.tmp"
            temporary.write_text(
                json.dumps(
                    {
                        "state": state,
                        "saved_batches": self.saved,
                        "max_batches": self.max_batches,
                    }
                ),
                encoding="utf-8",
            )
            temporary.replace(self.directory / "status.json")
        except OSError:
            self.failed = True
            logger.exception("Could not persist Qwen encoder capture status")

    def close(self):
        """Called after the encoder worker has stopped, never during encoding."""
        self.closed = True
        self._write_status("failed" if self.failed else "closed")

    def record(self, items, output, *, executions=None):
        if self.closed or self.failed or self.saved >= self.max_batches:
            return
        try:
            snapshot = QwenEncoderSnapshot.capture(
                items,
                output,
                metadata={
                    **self.metadata,
                    "batch_index": self.saved,
                    "executions": executions or [],
                    "work_unit": copy.deepcopy(current_work_unit()),
                },
            )
            snapshot.save(self.directory / f"batch-{self.saved:05d}.pt")
            self.saved += 1
            self._write_status(
                "limit_reached" if self.saved == self.max_batches else "collecting"
            )
        except Exception:
            # Note (Jiaxin Deng): calibration I/O must not cause a successful
            # encoder batch to retry and change the serving workload.
            self.failed = True
            logger.exception("Qwen encoder capture failed; disabling this session")
            self._write_status("failed")


def replay_qwen_encoder(model, snapshot, *, warmup=2, repeats=5, rtol, atol):
    """Replay CPU-restored inputs through the native audio feature method.

    The caller owns an otherwise idle model and must establish GPU isolation.
    This measures an explicit CPU-input replay protocol, including native input
    packing and transfer; original accelerator placement/strides are not
    reproduced. It excludes split/cache/LM processing and output verification.
    Warmup results are checked but their timings are discarded.
    """
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be nonnegative and repeats positive")
    device = next(model.audio_tower.parameters()).device
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("encoder replay currently supports CPU and CUDA")
    stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
    if stream is not None:
        torch.cuda.synchronize(device)
    measurements = []
    for index in range(warmup + repeats):
        items = snapshot.restore_items()
        context = (
            torch.cuda.stream(stream)
            if stream is not None
            else contextlib.nullcontext()
        )
        start_event = (
            torch.cuda.Event(enable_timing=True) if stream is not None else None
        )
        end_event = torch.cuda.Event(enable_timing=True) if stream is not None else None
        with torch.inference_mode(), context:
            if start_event is not None:
                start_event.record(stream)
            started = time.perf_counter()
            output = model.get_audio_feature(items)
            submitted = time.perf_counter()
            if end_event is not None:
                end_event.record(stream)
        if stream is not None:
            stream.synchronize()
        ended = time.perf_counter()
        torch.testing.assert_close(
            _cpu_copy(output), snapshot.output, rtol=rtol, atol=atol
        )
        if index >= warmup:
            measurements.append(
                {
                    "host_submit_s": submitted - started,
                    "host_total_s": ended - started,
                    "cuda_stream_ms": (
                        start_event.elapsed_time(end_event)
                        if start_event is not None
                        else None
                    ),
                }
            )
    return {
        "measurements": measurements,
        "warmup": warmup,
        "output_verified": True,
        "rtol": rtol,
        "atol": atol,
        "device": str(device),
        "input_protocol": "fresh CPU snapshot tensors each repetition",
        "scope": "native encoder method only; CUDA stream interval includes launch gaps; no GPU-group capacity",
    }
