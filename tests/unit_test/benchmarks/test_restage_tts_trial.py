from contextlib import contextmanager

import numpy as np
import pytest
import soundfile as sf

from benchmarks.benchmarker import restage_tts
from benchmarks.benchmarker.data import RequestResult
from benchmarks.dataset.seedtts import SampleInput
from benchmarks.eval import benchmark_omni_seedtts
from sglang_omni.restage.evaluation import SLO


@pytest.mark.asyncio
async def test_omni_sender_records_invalid_audio_as_request_failure(
    tmp_path, monkeypatch
):
    async def generate(*args, **kwargs):
        raise ValueError("No audio chunks received from streaming response")

    monkeypatch.setattr(
        benchmark_omni_seedtts.VoiceCloneOmni, "generate_speech", generate
    )
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
    result = await send(None, SampleInput("a", "", "", "hello"))
    assert not result.is_success
    assert "No audio chunks" in result.error
    assert result.latency_s >= 0


@pytest.mark.asyncio
@pytest.mark.parametrize("api", ["speech", "chat"])
async def test_tts_trial_launches_asr_only_for_quality(tmp_path, monkeypatch, api):
    events = []
    wav = tmp_path / "speech.wav"
    sf.write(wav, np.full(16000, 0.1), 16000)

    @contextmanager
    def server(**kwargs):
        assert kwargs["model_path"] == "asr-checkpoint"
        assert kwargs["server_config"] == str((tmp_path / "asr.yaml").resolve())
        assert kwargs["wait_for_gpu_release"] is False
        events.append("asr-start")
        try:
            yield
        finally:
            events.append("asr-stop")

    async def transcribe(samples, **kwargs):
        assert events[-1] == "asr-start"
        assert samples[0].ref_audio == str(wav)
        assert kwargs["model_path"] == "asr-checkpoint"
        return [RequestResult(request_id="a", is_success=True, text="hello")], 1

    async def trial(**kwargs):
        events.extend(["tts-start", "tts-stop"])
        kwargs["destination"].mkdir()
        kwargs["send_factory"]("http://localhost:18000", tmp_path)
        return await kwargs["quality"](
            [RequestResult(request_id="a", is_success=True, wav_path=str(wav))]
        )

    monkeypatch.setattr(restage_tts, "managed_omni_server", server)
    monkeypatch.setattr(restage_tts, "run_asr_transcription", transcribe)
    monkeypatch.setattr(restage_tts, "execute_trial", trial)
    send_calls = []
    if api == "speech":
        monkeypatch.setattr(
            restage_tts, "make_tts_send_fn", lambda *a, **kw: send_calls.append((a, kw))
        )
    else:
        monkeypatch.setattr(
            benchmark_omni_seedtts,
            "make_send_fn",
            lambda *a, **kw: send_calls.append((a, kw)),
        )
    result = await restage_tts.execute_tts_trial(
        config_path=tmp_path / "tts.yaml",
        model_path="tts-checkpoint",
        asr_config_path=tmp_path / "asr.yaml",
        asr_model_path="asr-checkpoint",
        samples=[SampleInput("a", "", "", "hello")],
        slo=SLO(max_latency_s=2),
        rate=1,
        destination=tmp_path / "trial",
        port=18000,
        lang="en",
        max_wer=0.2,
        api=api,
        sender_options=(
            {"stream": True}
            if api == "speech"
            else {
                "stream": True,
                "voice_clone": False,
                "speaker": "Ethan",
                "max_tokens": 256,
                "temperature": 0.7,
            }
        ),
    )
    assert result == {"a": True}
    assert events == ["tts-start", "tts-stop", "asr-start", "asr-stop"]
    assert (tmp_path / "trial/quality-detail.json").exists()
    endpoint = "audio/speech" if api == "speech" else "chat/completions"
    assert send_calls[0][0] == (
        "tts-checkpoint",
        f"http://localhost:18000/v1/{endpoint}",
    )
    assert send_calls[0][1]["stream"] is True
    if api == "chat":
        assert send_calls[0][1]["lang"] == "en"


@pytest.mark.asyncio
async def test_chat_rejects_unmeasured_playback_slo(tmp_path):
    with pytest.raises(ValueError, match="playback"):
        await restage_tts.execute_tts_trial(
            config_path=tmp_path / "tts.yaml",
            model_path="omni",
            asr_config_path=tmp_path / "asr.yaml",
            asr_model_path="asr",
            samples=[SampleInput("a", "", "", "hello")],
            slo=SLO(max_underrun_s=0),
            rate=1,
            destination=tmp_path / "trial",
            port=18000,
            lang="en",
            max_wer=0.2,
            api="chat",
        )
    assert not (tmp_path / "trial").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("api", ["speech", "chat"])
async def test_tts_requires_streaming_for_first_audio_constraint(tmp_path, api):
    with pytest.raises(ValueError, match="stream"):
        await restage_tts.execute_tts_trial(
            config_path=tmp_path / "tts.yaml",
            model_path="tts",
            asr_config_path=tmp_path / "asr.yaml",
            asr_model_path="asr",
            samples=[SampleInput("a", "", "", "hello")],
            slo=SLO(max_ttfa_s=1),
            rate=1,
            destination=tmp_path / "trial",
            port=18000,
            lang="en",
            max_wer=0.2,
            api=api,
            sender_options={"stream": False},
        )
    assert not (tmp_path / "trial").exists()
