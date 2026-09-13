import json
import sys

import pytest
from filelock import FileLock, Timeout

from benchmarks.benchmarker import restage_campaign
from benchmarks.dataset.seedtts import SampleInput
from sglang_omni.restage.evaluation import SLO, Observation, evaluate


@pytest.mark.asyncio
@pytest.mark.parametrize("task", ["tts", "asr"])
async def test_campaign_pairs_arrivals_and_exports_measured_winner(
    tmp_path, monkeypatch, task
):
    configs = {}
    for key in ("default", "split"):
        path = tmp_path / f"{key}.yaml"
        path.write_text(key)
        configs[key] = path
    calls = []

    async def trial(**kwargs):
        key = kwargs["config_path"].read_text()
        calls.append((key, kwargs["rate"], kwargs["arrival_seed"]))
        good = kwargs["rate"] == 1 or key == "split"
        return evaluate(
            [Observation("a", 0, 0.1, True, good)],
            SLO(max_latency_s=1),
            expected_requests=1,
            elapsed_s=1,
        )

    monkeypatch.setattr(restage_campaign, f"execute_{task}_trial", trial)
    result = await restage_campaign.execute_campaign(
        task=task,
        configs=configs,
        baseline="default",
        rates=[1, 2],
        repeats=2,
        arrival_seed=42,
        destination=tmp_path / "campaign",
        trial_options={},
    )
    assert result.recommended == "split"
    assert {(rate, seed) for key, rate, seed in calls if key == "default"} == {
        (rate, seed) for key, rate, seed in calls if key == "split"
    }
    assert (tmp_path / "campaign/recommended.yaml").read_text() == "split"
    rows = (tmp_path / "campaign/trials.jsonl").read_text().splitlines()
    assert len(rows) == 8
    saved = json.loads((tmp_path / "campaign/selection.json").read_text())
    assert saved["recommended"] == "split"
    assert saved["rate_gain_over_baseline"] == 2
    assert json.loads((tmp_path / "campaign/campaign.json").read_text())["task"] == task


@pytest.mark.parametrize("task", ["tts", "asr"])
@pytest.mark.parametrize("resume", [False, True])
def test_campaign_cli_resolves_task_specific_config(
    tmp_path, monkeypatch, task, resume
):
    (tmp_path / "clip.wav").write_bytes(b"fixture")
    (tmp_path / "quality.yaml").write_text("quality")
    options = {
        "samples": [
            {
                "sample_id": "a",
                "ref_text": "hello",
                "ref_audio": "clip.wav",
                "target_text": "world",
            }
        ],
        "slo": {"max_latency_s": 2},
    }
    if task == "tts":
        options["asr_config_path"] = "quality.yaml"
    spec = tmp_path / "campaign.json"
    spec.write_text(
        json.dumps(
            {
                "task": task,
                "configs": {"default": "model.yaml"},
                "trial_options": options,
            }
        )
    )
    calls = []

    async def campaign(**kwargs):
        calls.append(kwargs)
        return SLO(max_latency_s=2)

    monkeypatch.setattr(restage_campaign, "execute_campaign", campaign)
    monkeypatch.setattr(
        sys,
        "argv",
        ["campaign", "--spec", str(spec), "--output", str(tmp_path / "output")]
        + (["--resume"] if resume else []),
    )
    restage_campaign.main()
    call = calls[0]
    assert call["resume"] is resume
    assert call["configs"]["default"] == tmp_path / "model.yaml"
    assert call["trial_options"]["samples"][0].ref_text == "hello"
    assert call["trial_options"]["slo"] == SLO(max_latency_s=2)
    if task == "tts":
        assert call["trial_options"]["asr_config_path"] == tmp_path / "quality.yaml"
    else:
        assert "asr_config_path" not in call["trial_options"]


@pytest.mark.asyncio
@pytest.mark.parametrize("task", ["tts", "asr"])
@pytest.mark.parametrize("feasible", [False, True])
async def test_resume_reuses_completed_cells_and_preserves_failed_attempt(
    tmp_path, monkeypatch, task, feasible
):
    config = tmp_path / "config.yaml"
    config.write_text("baseline")
    calls = []
    fail = True

    async def trial(**kwargs):
        calls.append((kwargs["rate"], kwargs["destination"]))
        kwargs["destination"].mkdir()
        if kwargs["rate"] == 2 and fail:
            (kwargs["destination"] / "partial.txt").write_text("saved evidence")
            raise RuntimeError(f"interrupted-{len(calls)}")
        return evaluate(
            [Observation("a", 0, 0.1, True, feasible)],
            SLO(max_latency_s=1),
            expected_requests=1,
            elapsed_s=1,
        )

    monkeypatch.setattr(restage_campaign, f"execute_{task}_trial", trial)
    options = dict(
        task=task,
        configs={"baseline": config},
        baseline="baseline",
        rates=[1, 2],
        repeats=1,
        arrival_seed=42,
        destination=tmp_path / "campaign",
        trial_options={},
        run_identity="source/image/hardware/checkpoints-v1",
    )
    with pytest.raises(RuntimeError, match="interrupted"):
        await restage_campaign.execute_campaign(**options)
    failed_dir = calls[-1][1]
    with pytest.raises(RuntimeError, match="interrupted"):
        await restage_campaign.execute_campaign(**options, resume=True)
    second_failed_dir = calls[-1][1]
    assert second_failed_dir != failed_dir
    for directory, attempt in [(failed_dir, 2), (second_failed_dir, 3)]:
        failure = json.loads((directory / "execution-failure.json").read_text())
        assert failure["error"] == f"RuntimeError: interrupted-{attempt}"
    fail = False
    result = await restage_campaign.execute_campaign(**options, resume=True)
    assert result.recommended == ("baseline" if feasible else None)
    assert [rate for rate, _ in calls] == [1, 2, 2, 2]
    assert calls[-1][1] != failed_dir
    assert (failed_dir / "partial.txt").read_text() == "saved evidence"
    assert len((options["destination"] / "trials.jsonl").read_text().splitlines()) == 2
    (options["destination"] / "trials.jsonl").write_text('{"partial":')
    again = await restage_campaign.execute_campaign(**options, resume=True)
    assert again == result
    assert len(calls) == 4
    assert len((options["destination"] / "trials.jsonl").read_text().splitlines()) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changed", ["config", "snapshot", "sample", "audio", "slo", "quality", "identity"]
)
async def test_resume_rejects_changed_evidence_identity(tmp_path, monkeypatch, changed):
    config = tmp_path / "config.yaml"
    config.write_text("baseline")
    quality_config = tmp_path / "asr.yaml"
    quality_config.write_text("quality")
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"original audio")
    calls = []

    async def trial(**kwargs):
        calls.append(kwargs)
        return evaluate(
            [Observation("a", 0, 0.1, True, True)],
            SLO(max_latency_s=1),
            expected_requests=1,
            elapsed_s=1,
        )

    monkeypatch.setattr(restage_campaign, "execute_tts_trial", trial)
    options = dict(
        configs={"baseline": config},
        baseline="baseline",
        rates=[1],
        repeats=1,
        arrival_seed=42,
        destination=tmp_path / "campaign",
        trial_options={
            "samples": [SampleInput("a", "reference", str(audio), "hello")],
            "slo": SLO(max_latency_s=1),
            "asr_config_path": quality_config,
        },
        run_identity="frozen environment v1",
    )
    await restage_campaign.execute_campaign(**options)
    if changed == "config":
        config.write_text("different layout")
    elif changed == "snapshot":
        (options["destination"] / "candidate-00000.yaml").write_text("changed snapshot")
    elif changed == "sample":
        options["trial_options"]["samples"] = [SampleInput("a", "", "", "different")]
    elif changed == "audio":
        audio.write_bytes(b"different audio")
    elif changed == "slo":
        options["trial_options"]["slo"] = SLO(max_latency_s=2)
    elif changed == "quality":
        quality_config.write_text("different evaluator config")
    else:
        options["run_identity"] = "different hardware or software"
    recorded = (options["destination"] / "trials.jsonl").read_bytes()
    with pytest.raises(ValueError, match="identity|snapshot"):
        await restage_campaign.execute_campaign(**options, resume=True)
    assert len(calls) == 1
    assert (options["destination"] / "trials.jsonl").read_bytes() == recorded


@pytest.mark.asyncio
async def test_resume_requires_explicit_run_identity(tmp_path):
    with pytest.raises(ValueError, match="run_identity"):
        await restage_campaign.execute_campaign(
            configs={},
            baseline="baseline",
            rates=[1],
            repeats=1,
            arrival_seed=42,
            destination=tmp_path / "campaign",
            trial_options={},
            resume=True,
        )


@pytest.mark.asyncio
async def test_campaign_rejects_concurrent_writer(tmp_path):
    destination = tmp_path / "campaign"
    with FileLock(str(tmp_path / ".campaign.lock")):
        with pytest.raises(Timeout):
            await restage_campaign.execute_campaign(
                configs={},
                baseline="baseline",
                rates=[1],
                repeats=1,
                arrival_seed=42,
                destination=destination,
                trial_options={},
            )
    assert not destination.exists()


@pytest.mark.asyncio
async def test_campaign_batches_quality_after_generation_and_resumes_completed(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    config = tmp_path / "model.yaml"
    config.write_text("model")
    events = []
    measurements = {}
    fail = True

    async def trial(**kwargs):
        assert kwargs["defer_quality"] is True
        kwargs["destination"].mkdir()
        (kwargs["destination"] / "measurement.json").write_text("{}")
        events.append(("measure", kwargs["rate"]))
        measurement = SimpleNamespace(
            destination=kwargs["destination"], rate=kwargs["rate"]
        )
        measurements[measurement.destination] = measurement
        return measurement

    async def quality(measurements, *, on_evaluated, **kwargs):
        events.append(("quality", [m.rate for m in measurements]))
        for m in measurements:
            if fail and m.rate == 2:
                raise RuntimeError("quality interrupted")
            on_evaluated(
                m,
                evaluate(
                    [Observation("a", 0, 0.1, True, True)],
                    SLO(),
                    expected_requests=1,
                    elapsed_s=1,
                ),
            )

    def restore(source, *, destination):
        destination.mkdir()
        (destination / "measurement.json").write_bytes(
            (source / "measurement.json").read_bytes()
        )
        return SimpleNamespace(destination=destination, rate=measurements[source].rate)

    monkeypatch.setattr(restage_campaign, "restore_measurement", restore)
    monkeypatch.setattr(restage_campaign, "execute_tts_trial", trial)
    monkeypatch.setattr(restage_campaign, "evaluate_tts_batch", quality)
    options = dict(
        configs={"default": config},
        baseline="default",
        rates=[1, 2],
        repeats=1,
        arrival_seed=42,
        destination=tmp_path / "campaign",
        trial_options={},
        run_identity="frozen",
        batch_quality=True,
    )
    with pytest.raises(RuntimeError, match="quality interrupted"):
        await restage_campaign.execute_campaign(**options)
    assert events == [("measure", 1), ("measure", 2), ("quality", [1, 2])]
    checkpoint = json.loads((tmp_path / "campaign/completed-trials.json").read_text())
    assert len(checkpoint) == 1 and checkpoint[0]["rate"] == 1
    fail = False
    events.clear()
    result = await restage_campaign.execute_campaign(**options, resume=True)
    assert events == [("quality", [2])]
    assert result.recommended == "default"
    assert (
        len(json.loads((tmp_path / "campaign/completed-trials.json").read_text())) == 2
    )


@pytest.mark.asyncio
async def test_real_batch_campaign_keeps_generation_independent(tmp_path, monkeypatch):
    from contextlib import contextmanager

    import numpy as np
    import soundfile as sf

    from benchmarks.benchmarker import restage_trial, restage_tts
    from benchmarks.benchmarker.data import RequestResult

    events = []

    @contextmanager
    def generation(**kwargs):
        events.append("generation-start")
        try:
            yield
        finally:
            events.append("generation-stop")

    @contextmanager
    def asr(**kwargs):
        events.append("asr-start")
        try:
            yield
        finally:
            events.append("asr-stop")

    def sender(*args, save_audio_dir, **kwargs):
        async def send(session, sample):
            wav = Path(save_audio_dir) / "a.wav"
            sf.write(wav, np.full(16000, 0.1), 16000)
            return RequestResult(
                request_id=sample.sample_id, is_success=True, wav_path=str(wav)
            )

        return send

    async def transcribe(samples, **kwargs):
        events.append("transcribe")
        return [
            RequestResult(request_id=s.sample_id, is_success=True, text=s.ref_text)
            for s in samples
        ], 1

    from pathlib import Path

    monkeypatch.setattr(restage_trial, "managed_omni_server", generation)
    monkeypatch.setattr(restage_tts, "managed_omni_server", asr)
    monkeypatch.setattr(restage_tts, "make_tts_send_fn", sender)
    monkeypatch.setattr(restage_tts, "run_asr_transcription", transcribe)
    config = tmp_path / "model.yaml"
    config.write_text("model")
    result = await restage_campaign.execute_campaign(
        configs={"default": config, "split": config},
        baseline="default",
        rates=[1, 2],
        repeats=1,
        arrival_seed=42,
        destination=tmp_path / "campaign",
        batch_quality=True,
        trial_options=dict(
            model_path="tts",
            asr_model_path="asr",
            asr_config_path=config,
            samples=[SampleInput("a", "", "", "hello")],
            slo=SLO(max_latency_s=1),
            port=18000,
            lang="en",
            max_wer=0.2,
            warmup=0,
        ),
    )
    assert result.recommended in ("default", "split")
    assert (
        events
        == [
            "generation-start",
            "generation-stop",
            "generation-start",
            "generation-stop",
            "asr-start",
            "transcribe",
            "transcribe",
            "asr-stop",
        ]
        * 2
    )
    rows = json.loads((tmp_path / "campaign/completed-trials.json").read_text())
    assert len(rows) == 4
    for row in rows:
        dest = tmp_path / "campaign" / row["directory"]
        assert json.loads((dest / "result.json").read_text())["status"] == "complete"
        assert (
            json.loads((dest / "workload.json").read_text())["samples"][0][
                "target_text"
            ]
            == "hello"
        )
        assert (dest / "quality-detail.json").exists()


@pytest.mark.asyncio
async def test_batch_mode_is_part_of_resume_identity(tmp_path, monkeypatch):
    config = tmp_path / "model.yaml"
    config.write_text("model")

    async def trial(**kwargs):
        return evaluate(
            [Observation("a", 0, 0.1, True, True)],
            SLO(),
            expected_requests=1,
            elapsed_s=1,
        )

    monkeypatch.setattr(restage_campaign, "execute_tts_trial", trial)
    options = dict(
        configs={"default": config},
        baseline="default",
        rates=[1],
        repeats=1,
        arrival_seed=42,
        destination=tmp_path / "campaign",
        trial_options={},
        run_identity="frozen",
    )
    await restage_campaign.execute_campaign(**options)
    with pytest.raises(ValueError, match="identity differs"):
        await restage_campaign.execute_campaign(
            **options, batch_quality=True, resume=True
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["checkpoint", "trial_log", "shutdown"])
async def test_batch_failure_is_attributed_to_actual_phase(
    tmp_path, monkeypatch, failure
):
    from types import SimpleNamespace

    config = tmp_path / "model.yaml"
    config.write_text("model")

    async def trial(**kwargs):
        kwargs["destination"].mkdir()
        (kwargs["destination"] / "measurement.json").write_text("{}")
        return SimpleNamespace(destination=kwargs["destination"], rate=kwargs["rate"])

    async def quality(measurements, *, on_evaluated, **kwargs):
        for m in measurements:
            on_evaluated(
                m,
                evaluate(
                    [Observation("a", 0, 0.1, True, True)],
                    SLO(),
                    expected_requests=1,
                    elapsed_s=1,
                ),
            )
        raise RuntimeError("shutdown failed")

    write = restage_campaign._atomic_write

    def atomic(path, text):
        if (failure == "checkpoint" and path.name == "completed-trials.json") or (
            failure == "trial_log" and path.name == "trials.jsonl"
        ):
            raise OSError(f"{failure} failed")
        write(path, text)

    monkeypatch.setattr(restage_campaign, "execute_tts_trial", trial)
    monkeypatch.setattr(restage_campaign, "evaluate_tts_batch", quality)
    monkeypatch.setattr(restage_campaign, "_atomic_write", atomic)
    dest = tmp_path / "campaign"
    with pytest.raises((RuntimeError, OSError), match=f"{failure} failed"):
        await restage_campaign.execute_campaign(
            configs={"default": config},
            baseline="default",
            rates=[1, 2],
            repeats=1,
            arrival_seed=42,
            destination=dest,
            trial_options={},
            batch_quality=True,
        )
    saved = json.loads((dest / "failure.json").read_text())
    if failure in ("checkpoint", "trial_log"):
        assert saved["rate"] == 1
        assert saved["phase"] == "checkpoint"
        if failure == "checkpoint":
            assert not (dest / "completed-trials.json").exists()
        else:
            assert len(json.loads((dest / "completed-trials.json").read_text())) == 1
        assert len(list(dest.glob("*/execution-failure.json"))) == 1
    else:
        assert saved["phase"] == "batch_shutdown"
        assert saved["directory"] is None
        assert len(json.loads((dest / "completed-trials.json").read_text())) == 2
        assert not list(dest.glob("*/execution-failure.json"))


@pytest.mark.asyncio
@pytest.mark.parametrize("tamper_receipt", [False, True])
async def test_resume_reuses_saved_generation_after_quality_failure(
    tmp_path, monkeypatch, tamper_receipt
):
    from dataclasses import asdict

    from benchmarks.benchmarker.data import RequestResult
    from benchmarks.benchmarker.restage_trial import TrialMeasurement, _save_measurement

    config = tmp_path / "model.yaml"
    config.write_text("model")
    generated = []
    fail = True

    async def trial(**kwargs):
        dest = kwargs["destination"]
        dest.mkdir()
        generated.append(dest)
        result = RequestResult(
            request_id="a", is_success=True, scheduled_s=0, completed_s=0.1
        )
        meta = dict(
            status="awaiting_quality",
            measurement_complete=True,
            slo=asdict(SLO()),
            expected_requests=1,
            elapsed_s=1,
        )
        (dest / "requests.jsonl").write_text(json.dumps(asdict(result)) + "\n")
        _save_measurement(dest, meta, [result])
        return TrialMeasurement(dest, [result], SLO(), meta)

    async def quality(measurements, *, on_evaluated, **kwargs):
        if fail:
            raise RuntimeError("ASR unavailable")
        for measurement in measurements:
            assert measurement.metadata["measurement_source"] in [
                str(p) for p in generated
            ]
            on_evaluated(
                measurement,
                evaluate(
                    [Observation("a", 0, 0.1, True, True)],
                    SLO(),
                    expected_requests=1,
                    elapsed_s=1,
                ),
            )

    monkeypatch.setattr(restage_campaign, "execute_tts_trial", trial)
    monkeypatch.setattr(restage_campaign, "evaluate_tts_batch", quality)
    options = dict(
        configs={"default": config},
        baseline="default",
        rates=[1, 2],
        repeats=1,
        arrival_seed=42,
        destination=tmp_path / "campaign",
        trial_options={},
        batch_quality=True,
        run_identity="frozen",
    )
    with pytest.raises(RuntimeError, match="ASR unavailable"):
        await restage_campaign.execute_campaign(**options)
    assert len(generated) == 2
    originals = {str(p): (p / "requests.jsonl").read_bytes() for p in generated}
    if tamper_receipt:
        receipt = generated[0] / "measurement.json"
        receipt.write_text(receipt.read_text() + " ")
        with pytest.raises(ValueError, match="receipt changed"):
            await restage_campaign.execute_campaign(**options, resume=True)
        assert len(generated) == 2
        return
    fail = False
    assert (
        await restage_campaign.execute_campaign(**options, resume=True)
    ).recommended == "default"
    assert len(generated) == 2
    for p in generated:
        assert (p / "requests.jsonl").read_bytes() == originals[str(p)]
    rows = json.loads((tmp_path / "campaign/completed-trials.json").read_text())
    assert all("attempt-00001" in row["directory"] for row in rows)
