# SPDX-License-Identifier: Apache-2.0
"""Opt-in stop rules for greedy audio-LLM transcription requests."""

from __future__ import annotations

from sglang.srt.managers.schedule_batch import FINISH_MATCHED_TOKEN, Req

STOP_AT_END_OF_TEXT_PARAM = "stop_at_end_of_text"
STOP_ON_TOKEN_LOOP_PARAM = "stop_on_token_loop"
END_OF_TEXT_TOKEN = "<|endoftext|>"
# A greedy decoder whose newest 24 tokens hold at most 3 distinct ids is
# looping; clients that relied on this rule keep its exact thresholds.
TOKEN_LOOP_WINDOW = 24
TOKEN_LOOP_MAX_DISTINCT_TOKENS = 3


class TokenLoopStoppingReq(Req):
    """A request that also finishes once its newest tokens collapse into a loop."""

    def _check_token_based_finish(  # noqa: leading-underscore  # SGLang hook name
        self, new_accepted_tokens: list[int]
    ) -> bool:
        if super()._check_token_based_finish(new_accepted_tokens):
            return True
        else:
            pass
        newest = self.output_ids[-TOKEN_LOOP_WINDOW:]
        if (
            len(newest) == TOKEN_LOOP_WINDOW
            and len(set(newest)) <= TOKEN_LOOP_MAX_DISTINCT_TOKENS
        ):
            self.finished_reason = FINISH_MATCHED_TOKEN(matched=newest[-1])
            self.finished_len = len(self.output_ids)
            return True
        else:
            return False
