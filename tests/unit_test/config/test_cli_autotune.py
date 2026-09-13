"""The planning CLI exports candidates without starting inference."""

import json

import pytest
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


@pytest.mark.parametrize("task", ["tts", "asr"])
def test_run_command_loads_spec_and_resumes_completed_trials(
    tmp_path, monkeypatch, task
):
    from benchmarks.benchmarker import restage_campaign
    from sglang_omni.restage.evaluation import SLO, Observation, evaluate

    config = tmp_path / "model.yaml"
    config.write_text("model")
    audio = tmp_path / "reference.wav"
    audio.write_bytes(b"fixture")
    options = dict(
        samples=[
            dict(
                sample_id="a",
                ref_text="hello",
                ref_audio="reference.wav",
                target_text="hello",
            )
        ],
        slo=dict(max_latency_s=1),
    )
    if task == "tts":
        options["asr_config_path"] = "model.yaml"
    spec = tmp_path / "campaign.json"
    spec.write_text(
        json.dumps(
            dict(
                task=task,
                configs={"default": "model.yaml"},
                baseline="default",
                rates=[1],
                repeats=1,
                arrival_seed=42,
                run_identity="fixture",
                trial_options=options,
            )
        )
    )
    calls = []

    async def trial(**kwargs):
        calls.append(kwargs)
        assert kwargs["samples"][0].ref_audio == str(audio)
        assert kwargs["config_path"].is_file()
        assert kwargs["slo"] == SLO(max_latency_s=1)
        return evaluate(
            [Observation("a", 0, 0.1, True, True)],
            kwargs["slo"],
            expected_requests=1,
            elapsed_s=1,
        )

    monkeypatch.setattr(restage_campaign, f"execute_{task}_trial", trial)
    args = [
        "autotune",
        "run",
        "--spec",
        str(spec),
        "--output",
        str(tmp_path / "output"),
    ]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["recommended"] == "default"
    assert len(calls) == 1
    resumed = CliRunner().invoke(app, args + ["--resume"])
    assert resumed.exit_code == 0, resumed.output
    assert len(calls) == 1
    assert (tmp_path / "output/recommended.yaml").read_text() == "model"
