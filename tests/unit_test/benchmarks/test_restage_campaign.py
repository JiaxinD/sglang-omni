import json

import pytest

from benchmarks.benchmarker import restage_campaign
from benchmarks.dataset.seedtts import SampleInput
from sglang_omni.restage.evaluation import SLO, Observation, evaluate


def _evaluation(good=True):
    return evaluate(
        [Observation("a", 0, 0.1, True, good)],
        SLO(max_latency_s=1),
        expected_requests=1,
        elapsed_s=1,
    )


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
        return _evaluation(good=kwargs["rate"] == 1 or key == "split")

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
    assert len((tmp_path / "campaign/trials.jsonl").read_text().splitlines()) == 8
    saved = json.loads((tmp_path / "campaign/selection.json").read_text())
    assert saved["recommended"] == "split"
    assert saved["rate_gain_over_baseline"] == 2
    assert json.loads((tmp_path / "campaign/campaign.json").read_text())["task"] == task


@pytest.mark.asyncio
async def test_quality_failure_is_not_a_performance_success(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text("baseline")

    async def trial(**kwargs):
        return _evaluation(good=False)

    monkeypatch.setattr(restage_campaign, "execute_asr_trial", trial)
    result = await restage_campaign.execute_campaign(
        task="asr",
        configs={"baseline": config},
        baseline="baseline",
        rates=[1],
        repeats=1,
        arrival_seed=1,
        destination=tmp_path / "campaign",
        trial_options={},
    )
    assert result.recommended is None
    assert not (tmp_path / "campaign/recommended.yaml").exists()


@pytest.mark.asyncio
async def test_resume_reuses_completed_cells_and_preserves_failed_attempt(
    tmp_path, monkeypatch
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
        return _evaluation()

    monkeypatch.setattr(restage_campaign, "execute_asr_trial", trial)
    options = dict(
        task="asr",
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
    fail = False
    result = await restage_campaign.execute_campaign(**options, resume=True)
    assert result.recommended == "baseline"
    # The completed rate is reused; only the interrupted cell runs again.
    assert [rate for rate, _ in calls] == [1, 2, 2]
    assert calls[-1][1] != failed_dir
    assert (failed_dir / "partial.txt").read_text() == "saved evidence"
    assert len((options["destination"] / "trials.jsonl").read_text().splitlines()) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", ["config", "sample", "slo", "identity"])
async def test_resume_rejects_changed_campaign_identity(tmp_path, monkeypatch, changed):
    config = tmp_path / "config.yaml"
    config.write_text("baseline")
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"original audio")
    calls = []

    async def trial(**kwargs):
        calls.append(kwargs)
        return _evaluation()

    monkeypatch.setattr(restage_campaign, "execute_asr_trial", trial)
    options = dict(
        task="asr",
        configs={"baseline": config},
        baseline="baseline",
        rates=[1],
        repeats=1,
        arrival_seed=42,
        destination=tmp_path / "campaign",
        trial_options={
            "samples": [SampleInput("a", "reference", str(audio), "hello")],
            "slo": SLO(max_latency_s=1),
        },
        run_identity="frozen environment v1",
    )
    await restage_campaign.execute_campaign(**options)
    if changed == "config":
        config.write_text("different layout")
    elif changed == "sample":
        options["trial_options"]["samples"] = [SampleInput("a", "", "", "different")]
    elif changed == "slo":
        options["trial_options"]["slo"] = SLO(max_latency_s=2)
    else:
        options["run_identity"] = "different hardware or software"
    recorded = (options["destination"] / "trials.jsonl").read_bytes()
    with pytest.raises(ValueError, match="identity"):
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
            arrival_seed=1,
            destination=tmp_path / "campaign",
            trial_options={},
            resume=True,
        )
