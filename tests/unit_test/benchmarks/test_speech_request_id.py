import io

import numpy as np
import pytest
import soundfile as sf

from benchmarks.dataset.seedtts import SampleInput
from benchmarks.tasks.tts import make_tts_send_fn


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("server_id", [None, "speech-server-123"])
async def test_speech_sender_records_server_id_without_replacing_sample_id(
    stream, server_id
):
    wav = io.BytesIO()
    sf.write(wav, np.ones(2400) * 0.1, 24000, format="WAV", subtype="PCM_16")

    class Response:
        status = 200
        headers = {
            "Content-Type": "audio/pcm" if stream else "audio/wav",
            "X-Sample-Rate": "24000",
            "X-Channels": "1",
            "X-Bit-Depth": "16",
        }

        @property
        def content(self):
            return self

        async def iter_chunks(self):
            yield np.full(2400, 1000, dtype="<i2").tobytes(), True

        async def read(self):
            return wav.getvalue()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    if server_id:
        Response.headers["X-Request-ID"] = server_id

    class Session:
        def post(self, url, **kwargs):
            return Response()

    send = make_tts_send_fn(
        "tts", "http://localhost/v1/audio/speech", stream=stream, no_ref_audio=True
    )
    result = await send(Session(), SampleInput("sample-a", "", "", "hello"))
    assert result.is_success
    assert result.request_id == "sample-a"
    assert result.server_request_id == server_id
