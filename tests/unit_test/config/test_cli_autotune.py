"""The planning CLI exports candidates without starting inference."""

import json

import yaml
from typer.testing import CliRunner

from sglang_omni.cli import app
from sglang_omni.config.manager import ConfigManager


def test_plan_command_exports_loadable_candidate(tmp_path):
    config = tmp_path / "input.yaml"
    config.write_text(
        yaml.safe_dump(
            {"config_cls": "Qwen3TTSPipelineConfig", "model_path": "unused-checkpoint"}
        ),
        encoding="utf-8",
    )
    space = tmp_path / "space.json"
    space.write_text(
        json.dumps({"devices": [2, 5], "replica_counts": [1]}), encoding="utf-8"
    )
    output = tmp_path / "output"
    result = CliRunner().invoke(
        app,
        [
            "autotune",
            "plan",
            "--config",
            str(config),
            "--search-space",
            str(space),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    summary = json.loads((output / "summary.json").read_text())
    assert summary["accepted"] == 2
    assert summary["complete"] is True
    loaded = ConfigManager.from_file(str(output / "candidate-00000.yaml")).config
    assert loaded.processes["pipeline"].replica_devices is None
    assert loaded.stage_named("tts_engine").gpu == 2
    assert "not measured" in result.output


def test_plan_command_reports_invalid_search_file(tmp_path):
    config = tmp_path / "input.yaml"
    config.write_text(
        "config_cls: Qwen3TTSPipelineConfig\nmodel_path: unused-checkpoint\n",
        encoding="utf-8",
    )
    space = tmp_path / "space.json"
    space.write_text('{"devices": []}', encoding="utf-8")
    output = tmp_path / "output"
    result = CliRunner().invoke(
        app,
        [
            "autotune",
            "plan",
            "--config",
            str(config),
            "--search-space",
            str(space),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code != 0
    assert not output.exists()
    assert "devices" in result.output
