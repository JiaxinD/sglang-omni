import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from benchmarks.benchmarker import restage_campaign, restage_trial, restage_tts
from benchmarks.benchmarker.data import RequestResult
from benchmarks.dataset.seedtts import SampleInput
from benchmarks.eval import benchmark_omni_seedtts
from sglang_omni.restage.evaluation import SLO


@pytest.mark.asyncio
@pytest.mark.parametrize("api", ["speech", "chat"])
@pytest.mark.parametrize("adaptive", [False, True])
async def test_repeated_tts_outputs_keep_independent_quality(
    tmp_path, monkeypatch, api, adaptive
):
    @contextmanager
    def server(**kwargs):
        yield

    def sender(*args, save_audio_dir, **kwargs):
        async def send(session, sample):
            cycle = int(sample.sample_id.split("-")[2])
            index = int(sample.sample_id.split("-")[3])
            wav = Path(save_audio_dir) / f"{sample.sample_id}.wav"
            sf.write(wav, np.full(16000, 0.1 + cycle * 0.01 + index * 0.001), 16000)
            return RequestResult(
                request_id=sample.sample_id, is_success=True, wav_path=str(wav)
            )

        return send

    async def transcribe(samples, **kwargs):
        return [
            RequestResult(
                request_id=s.sample_id,
                is_success=True,
                text="wrong" if s.sample_id == "restage-repeat-1-0" else s.ref_text,
            )
            for s in samples
        ], 1

    monkeypatch.setattr(restage_trial, "managed_omni_server", server)
    monkeypatch.setattr(restage_tts, "managed_omni_server", server)
    monkeypatch.setattr(restage_tts, "make_tts_send_fn", sender)
    monkeypatch.setattr(benchmark_omni_seedtts, "make_send_fn", sender)
    monkeypatch.setattr(restage_tts, "run_asr_transcription", transcribe)
    config = tmp_path / "config.yaml"
    config.write_text("fixture")
    samples = [
        SampleInput("original", "reference", "", "hello"),
        SampleInput("other", "another reference", "", "world"),
    ]
    output = tmp_path / "campaign"
    result = await restage_campaign.execute_campaign(
        configs={"baseline": config},
        baseline="baseline",
        rates=[100],
        repeats=1,
        arrival_seed=42,
        destination=output,
        adaptive_search=(
            {"max_rate": 100, "target_arrival_duration_s": 0.04} if adaptive else None
        ),
        trial_options=dict(
            model_path="tts",
            asr_model_path="asr",
            asr_config_path=config,
            samples=samples,
            slo=SLO(max_latency_s=10),
            port=18000,
            lang="en",
            max_wer=0.2,
            warmup=0,
            api=api,
            corpus_repeats=1 if adaptive else 2,
        ),
    )
    assert result.recommended is None
    assert [s.sample_id for s in samples] == ["original", "other"]
    if adaptive:
        output = output / f"rate-{float(100).hex()}"
    rows = json.loads((output / "completed-trials.json").read_text())
    assert rows[0]["evaluation"]["successful_requests"] == 4
    assert rows[0]["evaluation"]["good_requests"] == 3
    cell = output / rows[0]["directory"]
    workload = json.loads((cell / "workload.json").read_text())
    assert workload["corpus_repeats"] == 2
    assert workload["request_sources"] == {
        "restage-repeat-0-0": "original",
        "restage-repeat-1-0": "original",
        "restage-repeat-0-1": "other",
        "restage-repeat-1-1": "other",
    }
    quality = json.loads((cell / "quality.json").read_text())
    assert quality == {
        "restage-repeat-0-0": True,
        "restage-repeat-1-0": False,
        "restage-repeat-0-1": True,
        "restage-repeat-1-1": True,
    }
