from contextlib import contextmanager

import numpy as np
import pytest
import soundfile as sf

from benchmarks.benchmarker import restage_tts
from benchmarks.benchmarker.data import RequestResult
from benchmarks.dataset.seedtts import SampleInput
from sglang_omni.restage.evaluation import SLO


@pytest.mark.asyncio
async def test_tts_trial_launches_asr_only_for_quality(tmp_path, monkeypatch):
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
        return await kwargs["quality"](
            [RequestResult(request_id="a", is_success=True, wav_path=str(wav))]
        )

    monkeypatch.setattr(restage_tts, "managed_omni_server", server)
    monkeypatch.setattr(restage_tts, "run_asr_transcription", transcribe)
    monkeypatch.setattr(restage_tts, "execute_trial", trial)
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
    )
    assert result == {"a": True}
    assert events == ["tts-start", "tts-stop", "asr-start", "asr-stop"]
    assert (tmp_path / "trial/quality-detail.json").exists()
