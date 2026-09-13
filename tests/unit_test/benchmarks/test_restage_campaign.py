import json

import pytest

from benchmarks.benchmarker import restage_campaign
from sglang_omni.restage.evaluation import SLO, Observation, evaluate


@pytest.mark.asyncio
async def test_campaign_pairs_arrivals_and_exports_measured_winner(
    tmp_path, monkeypatch
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

    monkeypatch.setattr(restage_campaign, "execute_tts_trial", trial)
    result = await restage_campaign.execute_tts_campaign(
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
