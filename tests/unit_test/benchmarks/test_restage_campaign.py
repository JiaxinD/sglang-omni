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
    options = {
        "samples": [
            {
                "sample_id": "a",
                "ref_text": "hello",
                "ref_audio": "/clip.wav",
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
