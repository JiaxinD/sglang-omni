import json

import pytest

from benchmarks.benchmarker import restage_asr
from benchmarks.benchmarker.data import RequestResult
from benchmarks.dataset.seedtts import SampleInput
from sglang_omni.restage.evaluation import SLO


@pytest.mark.asyncio
async def test_asr_scores_reference_audio_transcript_and_preserves_failures(
    tmp_path, monkeypatch
):
    async def trial(**kwargs):
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
