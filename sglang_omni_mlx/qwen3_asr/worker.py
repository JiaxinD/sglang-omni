# SPDX-License-Identifier: Apache-2.0
"""Runs every transcription on one thread, the thread that loaded the model."""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from sglang_omni_mlx.qwen3_asr.transcriber import (
    Qwen3ASRTranscriber,
    TranscriptionOptions,
    TranscriptionResult,
)


class TranscriptionWorker:
    """Serializes requests: one model, one MLX stream, one request at a time."""

    def __init__(self, model_directory: Path) -> None:
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="qwen3-asr"
        )
        self.transcriber: Qwen3ASRTranscriber = self.executor.submit(
            Qwen3ASRTranscriber, model_directory
        ).result()

    async def transcribe(
        self,
        samples: np.ndarray,
        options: TranscriptionOptions,
        cancel: threading.Event,
    ) -> TranscriptionResult:
        return await asyncio.get_running_loop().run_in_executor(
            self.executor, self.transcriber.transcribe, samples, options, cancel
        )
