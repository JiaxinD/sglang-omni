import json

import pytest

from benchmarks.benchmarker.restage_campaign import load_campaign_spec


def _planned_spec(tmp_path):
    plan = tmp_path / "plan"
    plan.mkdir()
    rows = []
    for index in range(3):
        filename = f"candidate-{index:05d}.yaml"
        (plan / filename).write_text(f"name: {index}\n")
        rows.append({"status": "candidate", "config_file": filename})
    rows.append({"status": "rejected", "rejection": "invalid placement"})
    (plan / "candidates.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n"
    )
    (tmp_path / "clip.wav").write_bytes(b"fixture")
    spec = {
        "task": "asr",
        "plan_directory": "plan",
        "baseline": "candidate-00001",
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


def test_plan_measures_every_accepted_candidate_starting_at_the_baseline(tmp_path):
    path, plan, _ = _planned_spec(tmp_path)
    loaded = load_campaign_spec(path)
    assert list(loaded["configs"]) == [
        "candidate-00001",
        "candidate-00000",
        "candidate-00002",
    ]
    assert all(p.parent == plan for p in loaded["configs"].values())
    assert "plan_directory" not in loaded


def test_plan_and_explicit_configs_are_mutually_exclusive(tmp_path):
    path, _, spec = _planned_spec(tmp_path)
    spec["configs"] = {"candidate-00000": "plan/candidate-00000.yaml"}
    path.write_text(json.dumps(spec))
    with pytest.raises(
        ValueError, match="configs.*plan_directory|plan_directory.*configs"
    ):
        load_campaign_spec(path)


def test_missing_baseline_candidate_is_rejected(tmp_path):
    path, _, spec = _planned_spec(tmp_path)
    spec["baseline"] = "candidate-00009"
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="baseline"):
        load_campaign_spec(path)


@pytest.mark.parametrize("enabled", [False, True])
def test_legacy_profile_option_only_accepts_disabled_value(tmp_path, enabled):
    path, _, spec = _planned_spec(tmp_path)
    spec["trial_options"]["profile"] = enabled
    path.write_text(json.dumps(spec))
    if enabled:
        with pytest.raises(ValueError, match="profile.*removed"):
            load_campaign_spec(path)
    else:
        assert "profile" not in load_campaign_spec(path)["trial_options"]
