# SPDX-License-Identifier: Apache-2.0
"""SGLang MLX runner extension for Whisper encoder-decoder inference."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
from transformers import GenerationConfig, WhisperConfig

from sglang_omni.model_runner.audio_mlx import AudioMlxModelRunner

if TYPE_CHECKING:
    from sglang.srt.hardware_backend.mlx.model_runner import (
        MlxPendingDecode,
        MlxPendingPrefill,
    )
    from sglang.srt.managers.schedule_batch import Req
else:
    pass

logger = logging.getLogger(__name__)

SUPPRESSED_LOGIT = -1.0e9
WEIGHT_FILE_PATTERNS = ("weights.safetensors", "model*.safetensors")


@dataclass(kw_only=True, frozen=True)
class WhisperSuppression:
    """Additive logit masks: generation-config suppression plus timestamp tokens.

    The first sampled step also applies begin_suppress_tokens. Text-only
    decoding never emits timestamp tokens, so all of them stay masked.
    """

    first_step_mask: mx.array
    later_step_mask: mx.array

    @classmethod
    def build(
        cls,
        *,
        vocab_size: int,
        suppress_token_ids: list[int],
        begin_suppress_token_ids: list[int],
        timestamp_begin_token_id: int,
    ) -> "WhisperSuppression":
        def mask_for(token_ids: list[int]) -> mx.array:
            blocked = [token_id for token_id in token_ids if 0 <= token_id < vocab_size]
            blocked.extend(range(timestamp_begin_token_id, vocab_size))
            mask = mx.zeros((vocab_size,), dtype=mx.float32)
            mask[mx.array(sorted(set(blocked)), dtype=mx.int32)] = SUPPRESSED_LOGIT
            return mask

        return cls(
            first_step_mask=mask_for(suppress_token_ids + begin_suppress_token_ids),
            later_step_mask=mask_for(suppress_token_ids),
        )

    def first_step(self, logits: mx.array) -> mx.array:
        return logits.astype(mx.float32) + self.first_step_mask

    def later_step(self, logits: mx.array) -> mx.array:
        return logits.astype(mx.float32) + self.later_step_mask


class WhisperMlxModelRunner(AudioMlxModelRunner):
    """Encodes the 30-second window in prefill and keeps cross-attention K/V."""

    model_name = "Whisper ASR"

    def prefill_start(
        self,
        req_id: str,
        new_token_ids: list[int],
        full_token_ids: list[int],
        prefix_slot_ids: list[int],
        new_slot_ids: list[int],
        req_pool_idx: int,
        req: "Req | None" = None,
        needs_logits: bool = True,
        logit_edit_row: mx.array | None = None,
        logprob_spec: object | None = None,
    ) -> "MlxPendingPrefill":
        from sglang.srt.hardware_backend.mlx.model_runner import MlxPendingPrefill

        if req is None or prefix_slot_ids or not self.disable_radix_cache:
            raise ValueError(
                f"{self.model_name} MLX prefill requires its request, no radix "
                "prefix and disable_radix_cache=True"
            )
        else:
            pass
        if logit_edit_row is not None or logprob_spec is not None:
            raise NotImplementedError(
                f"{self.model_name} MLX prefill supports greedy decoding only"
            )
        else:
            pass
        del needs_logits, new_slot_ids
        item = self.audio_item(req)
        if item.feature is None:
            raise ValueError(f"{self.model_name} MLX prefill requires audio features")
        else:
            pass
        encoder_token_count = int(item.model_specific_data["num_audio_tokens"])
        decoder_token_ids = list(full_token_ids[encoder_token_count:])
        if not decoder_token_ids or new_token_ids[-len(decoder_token_ids) :] != (
            decoder_token_ids
        ):
            raise ValueError(
                f"{self.model_name} MLX prefill expected {encoder_token_count} "
                "encoder placeholders before the decoder prompt"
            )
        else:
            pass

        encoder_output = self.model.encoder(mx.array(self.to_numpy(item.feature)))
        cross_states = self.model.cross_attention_states(encoder_output)
        self.cross_states_by_request[req_id] = cross_states
        cache = self._acquire_cache()  # noqa: leading-underscore  # SGLang hook name
        logits = self.model.decode(
            mx.array([decoder_token_ids], dtype=mx.int32),
            cache=cache,
            cross_states=cross_states,
        )
        lazy_token = mx.argmax(self.suppression.first_step(logits[:, -1, :]), axis=-1)
        return MlxPendingPrefill(
            lazy_token=lazy_token,
            cache=cache,
            req_id=req_id,
            full_token_ids=decoder_token_ids,
            req_pool_idx=req_pool_idx,
            synced_offset=0,
            lazy_logprobs=None,
        )

    def decode_tokens(
        self, req_ids: list[str], input_ids_by_request: list[mx.array]
    ) -> mx.array:
        next_tokens = []
        for req_id, input_ids in zip(req_ids, input_ids_by_request):
            logits = self.model.decode(
                input_ids,
                cache=self._req_caches[
                    req_id
                ],  # noqa: leading-underscore  # SGLang state
                cross_states=self.cross_states_by_request[req_id],
            )
            next_tokens.append(
                mx.argmax(self.suppression.later_step(logits[:, -1, :]), axis=-1)
            )
        return mx.concatenate(next_tokens, axis=0)

    def decode_batch_start(
        self,
        req_ids: list[str],
        edit_rows: mx.array | None = None,
        logprob_spec: object | None = None,
        logits_hook: object | None = None,
    ) -> "MlxPendingDecode":
        from sglang.srt.hardware_backend.mlx.model_runner import MlxPendingDecode

        if edit_rows is not None or logprob_spec is not None or logits_hook is not None:
            raise NotImplementedError(
                f"{self.model_name} MLX decode supports greedy decoding only"
            )
        else:
            pass
        token_ids = self._req_token_ids  # noqa: leading-underscore  # SGLang state
        lazy_tokens = self.decode_tokens(
            req_ids,
            [mx.array([[token_ids[req_id][-1]]], dtype=mx.int32) for req_id in req_ids],
        )
        return MlxPendingDecode(
            lazy_tokens=lazy_tokens,
            req_ids=list(req_ids),
            caches=[
                self._req_caches[req_id] for req_id in req_ids
            ],  # noqa: leading-underscore  # SGLang state
            lazy_logprobs=None,
            logprob_spec=None,
            edit_rows=None,
        )

    def decode_batch_start_chained(
        self, prev: "MlxPendingDecode"
    ) -> "MlxPendingDecode":
        from sglang.srt.hardware_backend.mlx.model_runner import MlxPendingDecode

        lazy_tokens = self.decode_tokens(
            prev.req_ids,
            [
                prev.lazy_tokens[index : index + 1][:, None]
                for index in range(len(prev.req_ids))
            ],
        )
        return MlxPendingDecode(
            lazy_tokens=lazy_tokens,
            req_ids=prev.req_ids,
            caches=prev.caches,
            lazy_logprobs=None,
            logprob_spec=None,
            edit_rows=None,
        )

    def remove_request(self, req_id: str) -> None:
        super().remove_request(req_id)
        self.cross_states_by_request.pop(req_id, None)

    def clear(self) -> None:
        super().clear()
        self.cross_states_by_request.clear()

    def _load_model(self) -> None:  # noqa: leading-underscore  # SGLang hook name
        from mlx.utils import tree_flatten
        from sglang.srt.hardware_backend.mlx.remote_code_gate import (
            resolve_model_directory,
        )

        from .model import WhisperMlxModel, decoder_dimensions_from_hf_config

        model_directory = Path(
            resolve_model_directory(self.model_path, revision=self.revision)
        )
        weight_files = [
            path
            for pattern in WEIGHT_FILE_PATTERNS
            for path in sorted(model_directory.glob(pattern))
        ]
        if not weight_files:
            raise FileNotFoundError(f"No Whisper safetensors in {model_directory}")
        else:
            pass
        logger.info(f"Loading native MLX Whisper model: {model_directory}")
        started = time.perf_counter()
        hf_config = WhisperConfig.from_pretrained(model_directory)
        self.model = WhisperMlxModel(decoder_dimensions_from_hf_config(hf_config))
        weights: dict[str, mx.array] = {}
        for weight_file in weight_files:
            weights.update(mx.load(str(weight_file)))
        self.model.load_weights(list(self.model.sanitize(weights).items()), strict=True)
        mx.eval(self.model.parameters())
        generation_config = GenerationConfig.from_pretrained(model_directory)
        self.suppression = WhisperSuppression.build(
            vocab_size=hf_config.vocab_size,
            suppress_token_ids=list(generation_config.suppress_tokens or []),
            begin_suppress_token_ids=list(
                generation_config.begin_suppress_tokens or []
            ),
            timestamp_begin_token_id=int(generation_config.no_timestamps_token_id) + 1,
        )
        self.cross_states_by_request = {}
        parameter_count = sum(
            array.size for _, array in tree_flatten(self.model.parameters())
        )
        logger.info(
            f"Loaded native MLX Whisper model in {time.perf_counter() - started:.2f}s "
            f"from {[path.name for path in weight_files]} parameters={parameter_count}"
        )


def make_whisper_mlx_runner_class():
    """Build the extension class after the MLX backend has been selected."""
    from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner

    class WhisperMlxRunner(WhisperMlxModelRunner, MlxModelRunner):
        pass

    return WhisperMlxRunner
