import json

import pytest

from benchmarks.benchmarker import restage_asr
from benchmarks.benchmarker.data import RequestResult
from benchmarks.dataset.seedtts import SampleInput
from sglang_omni.restage.evaluation import SLO


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_asr_sender_records_server_request_id(tmp_path, stream):
    import wave

    from benchmarks.tasks.asr import make_asr_send_fn

    audio = tmp_path / "sample.wav"
    with wave.open(str(audio), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\0\0" * 16000)

    class Response:
        status = 200
        headers = {"X-Request-ID": "transcription-measured"}

        @property
        def content(self):
            async def chunks():
                yield b'data: {"type":"transcript.text.done","text":"hello"}\n'
                yield b"data: [DONE]\n"

            return chunks()

        async def json(self):
            return {"text": "hello"}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    class Session:
        def post(self, *args, **kwargs):
            return Response()

    result = await make_asr_send_fn("asr", "http://localhost/asr", stream=stream)(
        Session(), SampleInput("sample", "hello", str(audio), "")
    )
    assert result.is_success
    assert result.text == "hello"
    assert result.request_id == "sample"
    assert result.server_request_id == "transcription-measured"


@pytest.mark.asyncio
async def test_asr_scores_reference_audio_transcript_and_preserves_failures(
    tmp_path, monkeypatch
):
    warmup = SampleInput("warm", "different audio", "/warm.wav", "")

    async def trial(**kwargs):
        assert kwargs["warmup_sample"] is warmup
        assert kwargs["profile"] is True
        kwargs["destination"].mkdir()
        kwargs["send_factory"]("http://localhost:18000", tmp_path)
        return await kwargs["quality"](
            [
                RequestResult(request_id="a", is_success=True, text="hello world"),
                RequestResult(
                    request_id="b",
                    is_success=False,
                    text="hello world",
                    error="HTTP 500",
                ),
                RequestResult(request_id="c", is_success=True, text="wrong words"),
            ]
        )

    send_calls = []
    monkeypatch.setattr(restage_asr, "execute_trial", trial)
    monkeypatch.setattr(
        restage_asr, "make_asr_send_fn", lambda *a, **kw: send_calls.append((a, kw))
    )
    result = await restage_asr.execute_asr_trial(
        config_path=tmp_path / "asr.yaml",
        model_path="asr-checkpoint",
        samples=[
            SampleInput(k, "hello world", "/clip.wav", "unrelated TTS target")
            for k in ("a", "b", "c")
        ],
        slo=SLO(max_latency_s=2, max_rtf=1),
        rate=1,
        destination=tmp_path / "trial",
        port=18000,
        lang="en",
        max_wer=0.2,
        profile=True,
        warmup_sample=warmup,
    )
    assert result == {"a": True, "b": False, "c": False}
    assert send_calls == [
        (
            ("asr-checkpoint", "http://localhost:18000/v1/audio/transcriptions"),
            {"lang": "en", "stream": False},
        )
    ]
    details = json.loads((tmp_path / "trial/quality-detail.json").read_text())
    assert details["requests"]["a"]["target_text"] == "hello world"
    assert details["requests"]["b"]["error"] == "HTTP 500"

    workload = json.loads((tmp_path / "trial/workload.json").read_text())
    assert workload["warmup_sample"]["sample_id"] == "warm"


@pytest.mark.asyncio
@pytest.mark.parametrize("slo", [SLO(max_ttfa_s=1), SLO(max_underrun_s=1)])
async def test_asr_rejects_audio_output_slo_before_launch(tmp_path, slo):
    with pytest.raises(ValueError, match="audio-output"):
        await restage_asr.execute_asr_trial(
            config_path=tmp_path / "asr.yaml",
            model_path="asr",
            samples=[SampleInput("a", "hello", "/clip.wav", "")],
            slo=slo,
            rate=1,
            destination=tmp_path / "trial",
            port=18000,
            lang="en",
            max_wer=0.2,
        )
    assert not (tmp_path / "trial").exists()
