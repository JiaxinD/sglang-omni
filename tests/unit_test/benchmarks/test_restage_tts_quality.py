import json

import numpy as np
import pytest
import soundfile as sf

from benchmarks.benchmarker.data import RequestResult
from benchmarks.benchmarker.restage_quality import evaluate_tts_quality


@pytest.mark.asyncio
async def test_tts_quality_combines_decoded_audio_and_transcription(tmp_path):
    wav = tmp_path / "speech.wav"
    sf.write(wav, np.sin(np.arange(16000) * 0.1).astype(np.float32) * 0.1, 16000)
    results = [
        RequestResult(request_id="ok", is_success=True, wav_path=str(wav)),
        RequestResult(request_id="wrong", is_success=True, wav_path=str(wav)),
        RequestResult(
            request_id="missing", is_success=True, wav_path=str(tmp_path / "absent.wav")
        ),
    ]

    async def transcribe(samples):
        assert {s.sample_id for s in samples} == {"ok", "wrong"}
        return [
            RequestResult(request_id="ok", is_success=True, text="hello world"),
            RequestResult(request_id="wrong", is_success=True, text="something else"),
        ]

    verdicts = await evaluate_tts_quality(
        results,
        targets={"ok": "hello world", "wrong": "hello world", "missing": "hello world"},
        transcribe=transcribe,
        output=tmp_path / "quality-detail.json",
        lang="en",
        max_wer=0.2,
    )
    assert verdicts == {"ok": True, "wrong": False, "missing": False}
    detail = json.loads((tmp_path / "quality-detail.json").read_text())
    assert detail["requests"]["wrong"]["wer"] == 1
    assert detail["requests"]["ok"]["audio_duration_s"] == 1


@pytest.mark.asyncio
async def test_silent_audio_and_missing_transcripts_do_not_pass(tmp_path):
    wav = tmp_path / "silent.wav"
    sf.write(wav, np.zeros(16000), 16000)

    async def transcribe(samples):
        assert samples == []
        return []

    verdicts = await evaluate_tts_quality(
        [RequestResult(request_id="a", is_success=True, wav_path=str(wav))],
        targets={"a": "hello"},
        transcribe=transcribe,
        output=tmp_path / "quality.json",
        lang="en",
        max_wer=0.2,
    )
    assert verdicts == {"a": False}


@pytest.mark.asyncio
async def test_valid_audio_with_missing_or_failed_asr_is_not_quality_success(tmp_path):
    wav = tmp_path / "speech.wav"
    sf.write(wav, np.full(16000, 0.1), 16000)
    results = [
        RequestResult(request_id=key, is_success=True, wav_path=str(wav))
        for key in ("missing", "failed")
    ]

    async def transcribe(samples):
        return [RequestResult(request_id="failed", error="ASR unavailable")]

    verdicts = await evaluate_tts_quality(
        results,
        targets={"missing": "hello", "failed": "hello"},
        transcribe=transcribe,
        output=tmp_path / "quality.json",
        lang="en",
        max_wer=0.2,
    )
    assert verdicts == {"missing": False, "failed": False}
    detail = json.loads((tmp_path / "quality.json").read_text())
    assert detail["requests"]["missing"]["error"] == "Missing ASR result"
    assert detail["requests"]["failed"]["error"] == "ASR unavailable"
