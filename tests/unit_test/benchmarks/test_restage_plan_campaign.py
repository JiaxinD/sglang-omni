import hashlib
import json

import pytest

from benchmarks.benchmarker.restage_campaign import load_campaign_spec


def _planned_spec(tmp_path):
    plan = tmp_path / "plan"
    plan.mkdir()
    rows = []
    for index, prediction in enumerate(
        [
            {"status": "predicted", "requests_per_s": 1.0},
            {"status": "unranked", "reason": "missing_groups"},
            {"status": "predicted", "requests_per_s": 20.0},
            {"status": "predicted", "requests_per_s": 20.0},
        ]
    ):
        filename = f"candidate-{index:05d}.yaml"
        (plan / filename).write_text(f"name: {index}\n")
        rows.append(
            {
                "status": "candidate",
                "config_file": filename,
                "prediction": prediction,
                "config_sha256": hashlib.sha256(
                    (plan / filename).read_bytes()
                ).hexdigest(),
            }
        )
    rows.append({"status": "rejected", "rejection": "invalid placement"})
    (plan / "candidates.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n"
    )
    (plan / "summary.json").write_text(
        json.dumps({"complete": True, "accepted": 4, "rejected": 1})
    )
    (tmp_path / "clip.wav").write_bytes(b"fixture")
    spec = {
        "task": "asr",
        "plan_directory": "plan",
        "baseline": "candidate-00000",
        "trial_options": {
            "samples": [
                {
                    "sample_id": "a",
                    "ref_text": "hello",
                    "ref_audio": "clip.wav",
                    "target_text": "",
                }
            ],
            "slo": {"max_latency_s": 2},
        },
    }
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    return path, plan, spec


def test_plan_orders_predictions_without_dropping_baseline_or_uncalibrated(tmp_path):
    path, plan, _ = _planned_spec(tmp_path)
    loaded = load_campaign_spec(path)
    assert list(loaded["configs"]) == [
        "candidate-00000",
        "candidate-00002",
        "candidate-00003",
        "candidate-00001",
    ]
    assert all(p.parent == plan for p in loaded["configs"].values())
    assert "plan_directory" not in loaded
    assert loaded["plan_evidence"]["summary"]["complete"] is True


def test_plan_and_explicit_configs_are_mutually_exclusive(tmp_path):
    path, _, spec = _planned_spec(tmp_path)
    spec["configs"] = {"candidate-00000": "plan/candidate-00000.yaml"}
    path.write_text(json.dumps(spec))
    with pytest.raises(
        ValueError, match="configs.*plan_directory|plan_directory.*configs"
    ):
        load_campaign_spec(path)


def test_changed_plan_bytes_change_recorded_evidence(tmp_path):
    path, plan, _ = _planned_spec(tmp_path)
    first = load_campaign_spec(path)["plan_evidence"]
    manifest = plan / "candidates.jsonl"
    manifest.write_text(manifest.read_text().replace("20.0", "21.0"))
    second = load_campaign_spec(path)["plan_evidence"]
    assert first != second


def test_modified_candidate_cannot_reuse_planned_prediction(tmp_path):
    path, plan, _ = _planned_spec(tmp_path)
    (plan / "candidate-00002.yaml").write_text("name: modified\n")
    with pytest.raises(ValueError, match="regenerate plan"):
        load_campaign_spec(path)


@pytest.mark.asyncio
async def test_campaign_records_plan_and_rejects_changed_plan_on_resume(
    tmp_path, monkeypatch
):
    from benchmarks.benchmarker import restage_campaign
    from sglang_omni.restage.evaluation import SLO, Observation, evaluate

    path, plan, _ = _planned_spec(tmp_path)
    calls = []

    async def trial(**kwargs):
        calls.append(kwargs["config_path"].read_text())
        return evaluate(
            [Observation("a", 0, 0.1, True, True)],
            SLO(max_latency_s=2),
            expected_requests=1,
            elapsed_s=1,
        )

    monkeypatch.setattr(restage_campaign, "execute_asr_trial", trial)
    options = load_campaign_spec(path)
    destination = tmp_path / "results"
    await restage_campaign.execute_campaign(
        **options,
        rates=[1],
        repeats=1,
        arrival_seed=42,
        destination=destination,
        run_identity="fixed-source-hardware-workload",
    )
    assert calls == ["name: 0\n", "name: 2\n", "name: 3\n", "name: 1\n"]
    recorded = json.loads((destination / "campaign.json").read_text())
    assert recorded["plan_evidence"] == options["plan_evidence"]
    manifest = plan / "candidates.jsonl"
    manifest.write_text(manifest.read_text().replace("20.0", "21.0"))
    with pytest.raises(ValueError, match="identity differs"):
        await restage_campaign.execute_campaign(
            **load_campaign_spec(path),
            rates=[1],
            repeats=1,
            arrival_seed=42,
            destination=destination,
            run_identity="fixed-source-hardware-workload",
            resume=True,
        )
    assert len(calls) == 4
