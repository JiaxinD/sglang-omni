import json

import pytest

from benchmarks.benchmarker import restage_campaign
from benchmarks.dataset.seedtts import SampleInput
from sglang_omni.restage.evaluation import SLO, Observation, evaluate


@pytest.mark.asyncio
async def test_adaptive_campaign_shared_grid_duration_and_resume(tmp_path, monkeypatch):
    configs = {}
    for name in ("single", "dual"):
        path = tmp_path / f"{name}.yaml"
        path.write_text(name)
        configs[name] = path
    calls = []
    interrupted = True

    async def trial(**kwargs):
        nonlocal interrupted
        name = kwargs["config_path"].read_text()
        rate = kwargs["rate"]
        if name == "dual" and rate == 2 and interrupted:
            interrupted = False
            raise RuntimeError("interrupted")
        calls.append((name, rate, kwargs["arrival_seed"], kwargs["corpus_repeats"]))
        return evaluate(
            [Observation("a", 0, 0.1, True, rate < (2 if name == "single" else 4))],
            SLO(max_latency_s=1),
            expected_requests=1,
            elapsed_s=1,
        )

    monkeypatch.setattr(restage_campaign, "execute_asr_trial", trial)
    options = dict(
        task="asr",
        configs=configs,
        baseline="single",
        rates=[1],
        repeats=2,
        arrival_seed=7,
        destination=tmp_path / "campaign",
        run_identity="frozen",
        trial_options={
            "samples": [SampleInput("a", "hello", "clip.wav", "")],
            "corpus_repeats": 1,
        },
        adaptive_search={
            "max_rate": 8,
            "growth_factor": 2,
            "target_arrival_duration_s": 10,
        },
    )
    with pytest.raises(RuntimeError, match="interrupted"):
        await restage_campaign.execute_campaign(**options)
    parent = tmp_path / "campaign"
    failure = json.loads((parent / "failure.json").read_text())
    assert failure["rate"] == 2
    assert failure["directory"] == "rate-0x1.0000000000000p+1"
    progress = json.loads((parent / "adaptive-search.json").read_text())
    assert progress["stop_reason"] == "trial_failure"
    assert progress["rates"] == [1]
    partial = json.loads((parent / "partial-selection.json").read_text())
    assert partial["baseline_rate"] == 1
    assert not (parent / "selection.json").exists()
    result = await restage_campaign.execute_campaign(**options, resume=True)
    assert result.recommended == "dual"
    assert result.rate_gain_over_baseline == 2
    assert len(calls) == 12  # 2 candidates, 3 common rates, 2 repeats; no reruns.
    for name in configs:
        assert {(rate, seed) for key, rate, seed, _ in calls if key == name} == {
            (rate, seed) for rate in (1, 2, 4) for seed in (7, 8)
        }
    assert all(count == rate * 10 for _, rate, _, count in calls)
    saved = json.loads((tmp_path / "campaign/adaptive-search.json").read_text())
    assert saved["rates"] == [1, 2, 4]
    assert saved["stop_reason"] == "all_candidates_failed"
    assert not saved["gpu_group_calibration"]
    with pytest.raises(ValueError, match="identity"):
        await restage_campaign.execute_campaign(
            **{**options, "adaptive_search": {"max_rate": 16}}, resume=True
        )


@pytest.mark.asyncio
async def test_adaptive_ceiling_is_not_capacity_boundary(tmp_path, monkeypatch):
    config = tmp_path / "one.yaml"
    config.write_text("one")

    async def trial(**kwargs):
        return evaluate(
            [Observation("a", 0, 0.1, True, True)],
            SLO(max_latency_s=1),
            expected_requests=1,
            elapsed_s=1,
        )

    monkeypatch.setattr(restage_campaign, "execute_asr_trial", trial)
    result = await restage_campaign.execute_campaign(
        task="asr",
        configs={"one": config},
        baseline="one",
        rates=[1],
        repeats=1,
        arrival_seed=0,
        destination=tmp_path / "campaign",
        trial_options={},
        adaptive_search={"max_rate": 3},
    )
    assert result.ranking[0].upper_limit_passed
    saved = json.loads((tmp_path / "campaign/adaptive-search.json").read_text())
    assert saved["rates"] == [1, 2, 3]
    assert saved["stop_reason"] == "max_rate_reached"


@pytest.mark.asyncio
async def test_adaptive_retains_failure_inside_initial_grid(tmp_path, monkeypatch):
    config = tmp_path / "one.yaml"
    config.write_text("one")

    async def trial(**kwargs):
        return evaluate(
            [Observation("a", 0, 0.1, True, kwargs["rate"] in (1, 4))],
            SLO(max_latency_s=1),
            expected_requests=1,
            elapsed_s=1,
        )

    monkeypatch.setattr(restage_campaign, "execute_asr_trial", trial)
    result = await restage_campaign.execute_campaign(
        task="asr",
        configs={"one": config},
        baseline="one",
        rates=[1, 2, 4],
        repeats=1,
        arrival_seed=0,
        destination=tmp_path / "campaign",
        trial_options={},
        adaptive_search={"max_rate": 8},
    )
    assert result.ranking[0].nonmonotonic
    assert result.ranking[0].best_tested_rate == 4
    assert result.ranking[0].passing_prefix_rate == 1
