import json
import sys

import pytest

from benchmarks.benchmarker import restage_campaign
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
def test_campaign_cli_resolves_task_specific_config(tmp_path, monkeypatch, task):
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
        ["campaign", "--spec", str(spec), "--output", str(tmp_path / "output")],
    )
    restage_campaign.main()
    call = calls[0]
    assert call["configs"]["default"] == tmp_path / "model.yaml"
    assert call["trial_options"]["samples"][0].ref_text == "hello"
    assert call["trial_options"]["slo"] == SLO(max_latency_s=2)
    if task == "tts":
        assert call["trial_options"]["asr_config_path"] == tmp_path / "quality.yaml"
    else:
        assert "asr_config_path" not in call["trial_options"]
