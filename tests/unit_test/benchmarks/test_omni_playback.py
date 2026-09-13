import base64
import io
import json

import numpy as np
import pytest
import soundfile as sf

from benchmarks.dataset.seedtts import SampleInput
from benchmarks.eval import benchmark_omni_seedtts


@pytest.mark.asyncio
@pytest.mark.parametrize("trailing_newline", [True, False])
@pytest.mark.parametrize(
    "arrivals,underrun",
    [((10.0, 10.75), 0.25), ((10.0, 10.25, 10.9), 0.0), ((10.0,), None)],
)
async def test_chat_sender_measures_playback_from_wav_frames(
    tmp_path, monkeypatch, trailing_newline, arrivals, underrun
):
    wav = io.BytesIO()
    sf.write(wav, np.ones((4000, 2)) * 0.1, 8000, format="WAV", subtype="PCM_16")
    event = json.dumps(
        {
            "choices": [
                {
                    "delta": {
                        "audio": {"data": base64.b64encode(wav.getvalue()).decode()}
                    }
                }
            ]
        }
    )
    now = [10.0]
    sent_ids = []
    monkeypatch.setattr(benchmark_omni_seedtts.time, "perf_counter", lambda: now[0])

    class Response:
        status = 200

        @property
        def content(self):
            return self

        async def iter_any(self):
            for index, arrival in enumerate(arrivals):
                now[0] = arrival
                line = (
                    "data: "
                    + event
                    + ("\n" if index < len(arrivals) - 1 or trailing_newline else "")
                ).encode()
                yield line[:15]
                yield line[15:]

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    class Session:
        def post(self, url, **kwargs):
            sent_ids.append(kwargs["json"]["request_id"])
            return Response()

    send = benchmark_omni_seedtts.make_send_fn(
        "omni",
        "http://localhost/v1/chat/completions",
        lang="en",
        voice_clone=False,
        speaker="Ethan",
        max_tokens=256,
        temperature=0.7,
        stream=True,
        save_audio_dir=str(tmp_path),
    )
    result = await send(Session(), SampleInput("a", "", "", "hello"))
    assert result.is_success
    assert result.audio_duration_s == pytest.approx(len(arrivals) * 0.5)
    assert result.chunk_audio_duration_s == pytest.approx([0.5] * len(arrivals))
    if underrun is None:
        assert result.max_playback_underrun_s is None
    else:
        assert result.max_playback_underrun_s == pytest.approx(underrun)
    assert result.audio_chunk_count == len(arrivals)
    assert result.first_audio_s == 10.0
    assert result.server_request_id == sent_ids[0]
    now[0] = 10.0
    repeat = await send(Session(), SampleInput("a", "", "", "hello"))
    assert repeat.request_id == result.request_id == "a"
    assert repeat.server_request_id == sent_ids[1]
    assert repeat.server_request_id != result.server_request_id
