# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.scheduling.greedy_stop_rules import (
    TOKEN_LOOP_WINDOW,
    TokenLoopStoppingReq,
)


def decode(request_class: type[Req], token_ids: list[int]) -> Req:
    sampling_params = SamplingParams(
        max_new_tokens=200, temperature=0.0, stop_token_ids=[7]
    )
    sampling_params.normalize(tokenizer=None)
    request = request_class(
        rid="loop",
        origin_input_text="",
        origin_input_ids=[1, 2],
        sampling_params=sampling_params,
        vocab_size=64,
    )
    for token_id in token_ids:
        request.output_ids.append(token_id)
        request.update_finish_state()
        if request.finished():
            break
        else:
            pass
    return request


def test_a_short_token_loop_finishes_with_the_whole_window() -> None:
    loop = [3, 4, 5] * 20

    request = decode(TokenLoopStoppingReq, loop)

    assert request.finished()
    assert len(request.output_ids) == TOKEN_LOOP_WINDOW
    assert request.finished_len in (None, TOKEN_LOOP_WINDOW)


def test_four_distinct_tokens_in_the_window_keep_decoding() -> None:
    request = decode(TokenLoopStoppingReq, [3, 4, 5, 6] * 20)

    assert not request.finished()


def test_the_stop_token_still_finishes_first() -> None:
    request = decode(TokenLoopStoppingReq, [3, 7, 4])

    assert request.finished()
    assert list(request.output_ids) == [3, 7]


def test_the_default_request_ignores_token_loops() -> None:
    request = decode(Req, [3, 4, 5] * 20)

    assert not request.finished()
