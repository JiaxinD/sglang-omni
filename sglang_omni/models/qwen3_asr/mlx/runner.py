# SPDX-License-Identifier: Apache-2.0
"""SGLang MLX runner extension for Qwen3-ASR audio prefill."""

from __future__ import annotations

import logging
import os
import time

from sglang_omni.model_runner.audio_mlx import AudioMlxModelRunner

logger = logging.getLogger(__name__)

# Opt-in for single-user servers that idle between requests (Voxt): small
# per-request KV caches that are freed on release instead of pooled.
LEAN_KV_CACHE_ENV = "SGLANG_OMNI_MLX_LEAN_KV_CACHE"
LEAN_KV_CACHE_TOKENS = 1024


def lean_kv_cache() -> bool:
    return os.environ.get(LEAN_KV_CACHE_ENV, "").strip() == "1"


class Qwen3ASRMlxModelRunner(AudioMlxModelRunner):
    """Qwen3-ASR support layered on SGLang's native MLX model runner.

    The base runner continues to own cache layout, pool sizing, radix state,
    and batched decode. This mixin only supplies the unsupported Qwen3-ASR
    model class and the multimodal first-prefill operation.
    """

    model_name = "Qwen3-ASR"

    def _load_model(self) -> None:
        from mlx_lm.utils import load_model
        from sglang.srt.hardware_backend.mlx.remote_code_gate import (
            ensure_remote_code_allowed,
            resolve_model_directory,
        )

        from .config import ModelConfig
        from .model import Qwen3ASRModel

        model_path = resolve_model_directory(
            self.model_path,
            revision=self.revision,
        )
        ensure_remote_code_allowed(model_path, self.trust_remote_code)
        logger.info("Loading native MLX Qwen3-ASR model: %s", model_path)
        started = time.perf_counter()
        self.model, _config = load_model(
            model_path,
            get_model_classes=lambda config: (Qwen3ASRModel, ModelConfig),
        )
        logger.info(
            "Loaded native MLX Qwen3-ASR model in %.2fs",
            time.perf_counter() - started,
        )


    def _new_native_cache(self):  # noqa: leading-underscore  # upstream hook
        if lean_kv_cache():
            # Upstream preallocates 4096 tokens per request (~470 MB for this
            # model); one ASR request rarely needs more than 1024 and the cache
            # still doubles on overflow.
            self._max_seq_len = min(
                self._max_seq_len, LEAN_KV_CACHE_TOKENS
            )  # noqa: leading-underscore  # upstream name
        else:
            pass
        return super()._new_native_cache()

    def _release_cache(self, cache) -> None:  # noqa: leading-underscore  # upstream hook
        if lean_kv_cache():
            # Freed instead of pooled, so an idle server holds no request cache.
            return
        else:
            pass
        super()._release_cache(cache)

    def audio_layout_options(self, item) -> dict[str, object]:
        from ..swift_layout import AUDIO_LAYOUT_PARAM

        layout = (getattr(item, "model_specific_data", None) or {}).get(AUDIO_LAYOUT_PARAM)
        return {"layout": layout} if layout else {}


def make_qwen3_asr_mlx_runner_class():
    """Build the extension class after the MLX backend has been selected."""
    from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner

    class Qwen3ASRMlxRunner(Qwen3ASRMlxModelRunner, MlxModelRunner):
        pass

    return Qwen3ASRMlxRunner
