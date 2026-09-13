import json
from pathlib import Path

from sglang_omni.cli.config import _resolve_sources
from sglang_omni.config.runtime import resolve_stage_typed_kwargs
from sglang_omni.models.qwen3_tts.config import Qwen3TTSPipelineConfig
from sglang_omni.restage.plan import SearchSpace, write_plan


def test_gate_search_exports_both_settings_for_each_placement(tmp_path):
    example = (
        Path(__file__).resolve().parents[3]
        / "examples/configs/restage_qwen3_tts_gate_search.json"
    )
    space = SearchSpace.model_validate_json(example.read_text())
    result = write_plan(
        Qwen3TTSPipelineConfig(model_path="unused"), space, tmp_path / "plan"
    )
    assert result["complete"]
    assert not result["performance_measured"]
    by_placement = {}
    for line in (tmp_path / "plan/candidates.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["status"] != "candidate":
            continue
        config = _resolve_sources(
            model_path=None,
            config_file=str(tmp_path / "plan" / row["config_file"]),
            text_only=False,
            mem_fraction_static=None,
            argv=[],
        ).resolved.config
        value = resolve_stage_typed_kwargs(config.stage_named("vocoder"))[
            "criticality_slack_s"
        ]
        by_placement.setdefault(
            json.dumps(row["assignments"], sort_keys=True), set()
        ).add(value)
    assert by_placement
    assert all(values == {0.0, 0.05} for values in by_placement.values())
