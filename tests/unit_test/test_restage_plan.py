"""A planning run exports usable candidates and records its search scope."""

import json
from pathlib import Path

import yaml

from sglang_omni.models.qwen3_tts.config import Qwen3TTSPipelineConfig
from sglang_omni.restage.plan import SearchSpace, write_plan
from tests.unit_test.fixtures.restage import pipeline


def test_plan_exports_configs_and_records_truncated_search(tmp_path):
    config = Qwen3TTSPipelineConfig(model_path="unused-checkpoint")
    destination = tmp_path / "plan"
    summary = write_plan(
        config,
        SearchSpace(devices=[2, 5], replica_counts=[1, 2]),
        destination,
        max_candidates=2,
    )
    assert summary["complete"] is False
    assert summary["examined"] == 2
    rows = [
        json.loads(line)
        for line in (destination / "candidates.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 2
    assert all(row["status"] == "candidate" for row in rows)
    assert all(
        yaml.safe_load((destination / row["config_file"]).read_text())["config_cls"]
        == "Qwen3TTSPipelineConfig"
        for row in rows
    )
    assert json.loads((destination / "summary.json").read_text()) == summary
    assert summary["performance_measured"] is False


def test_invalid_configuration_is_logged_and_other_choices_continue(tmp_path):
    config = Qwen3TTSPipelineConfig(model_path="unused-checkpoint")
    spec = SearchSpace(
        devices=[0],
        replica_counts=[1],
        dimensions={
            "knob": [{"tts_engine.factory_path": "invalid"}, {}],
        },
    )
    summary = write_plan(config, spec, tmp_path / "plan", max_candidates=10)
    assert summary["complete"] is True
    assert summary["rejected"] == 1
    assert summary["accepted"] == 1
    rows = [
        json.loads(line)
        for line in (tmp_path / "plan/candidates.jsonl").read_text().splitlines()
    ]
    assert "internal" in rows[0]["rejection"]


def test_per_process_replica_counts_vary_one_process_only(tmp_path):
    config = pipeline()
    config.stages[1].gpu_memory_fraction = 0.2
    summary = write_plan(
        config,
        SearchSpace(devices=[0, 1], replica_counts={"engine": [1], "tail": [1, 2, 3]}),
        tmp_path / "plan",
        max_candidates=64,
    )
    assert summary["complete"] is True
    rows = [
        json.loads(line)
        for line in (tmp_path / "plan/candidates.jsonl").read_text().splitlines()
    ]
    assert {len(row["assignments"]["tail"]) for row in rows} == {1, 2, 3}
    assert {len(row["assignments"]["engine"]) for row in rows} == {1}
    accepted = [row for row in rows if row["status"] == "candidate"]
    assert {len(row["assignments"]["tail"]) for row in accepted} == {1, 2, 3}


def test_shipped_example_search_spaces_load():
    examples = Path(__file__).parents[2] / "examples/configs"
    spaces = {
        path.name: SearchSpace.model_validate_json(path.read_text(encoding="utf-8"))
        for path in examples.glob("restage_*_search.json")
    }
    assert len(spaces) == 3
    moss = spaces["restage_moss_td_pd_search.json"]
    assert moss.devices == [0, 1]
    assert moss.replica_counts == {"asr_prefill": [1], "asr_decode": [1, 2, 3]}
