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


@pytest.mark.parametrize("fault", ["shape", "audio", "asr_config"])
def test_run_rejects_invalid_inputs_before_model_execution(
    tmp_path, monkeypatch, fault
):
    from benchmarks.benchmarker import restage_campaign

    (tmp_path / "model.yaml").write_text("model")
    (tmp_path / "audio.wav").write_bytes(b"fixture")
    options = dict(
        samples=[
            dict(
                sample_id="a",
                ref_audio="missing.wav" if fault == "audio" else "audio.wav",
                ref_text="hello",
                target_text="hello",
            )
        ],
        slo={},
        asr_config_path="missing.yaml" if fault == "asr_config" else "model.yaml",
    )
    payload = dict(
        configs={"default": "model.yaml"},
        baseline="default",
        rates=[1],
        repeats=1,
        arrival_seed=42,
        trial_options=options,
    )
    if fault == "shape":
        payload = {}
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(payload))
    calls = []

    async def forbidden(**kwargs):
        calls.append(kwargs)
        raise AssertionError("model execution must not start")

    monkeypatch.setattr(restage_campaign, "execute_campaign", forbidden)
    result = CliRunner().invoke(
        app, ["autotune", "run", "--spec", str(spec), "--output", str(tmp_path / "out")]
    )
    assert result.exit_code == 2, result.output
    assert "--spec" in result.output
    assert not calls
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "reference",
    [
        "https://example.invalid/audio.wav",
        "data:audio/wav;base64," + "AAAA" * 3000,
        "file:///reference.wav",
    ],
    ids=["https", "data", "file"],
)
def test_campaign_loader_preserves_media_references(tmp_path, reference):
    from benchmarks.benchmarker.restage_campaign import (
        _input_identity,
        load_campaign_spec,
    )

    (tmp_path / "asr.yaml").write_text("asr")
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            dict(
                configs={},
                trial_options=dict(
                    samples=[
                        dict(
                            sample_id="a",
                            ref_text="hello",
                            ref_audio=reference,
                            target_text="hello",
                        )
                    ],
                    slo={},
                    asr_config_path="asr.yaml",
                ),
            )
        )
    )
    options = load_campaign_spec(spec)["trial_options"]
    assert options["samples"][0].ref_audio == reference
    assert list(_input_identity(options)["local_input_sha256"]) == [
        str(tmp_path / "asr.yaml")
    ]


def test_unused_reference_does_not_require_a_local_file(tmp_path):
    from benchmarks.benchmarker.restage_campaign import load_campaign_spec

    (tmp_path / "asr.yaml").write_text("asr")
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            dict(
                configs={},
                trial_options=dict(
                    samples=[
                        dict(
                            sample_id="a",
                            ref_text="hello",
                            ref_audio="unused.wav",
                            target_text="hello",
                        )
                    ],
                    slo={},
                    sender_options={"no_ref_audio": True},
                    asr_config_path="asr.yaml",
                ),
            )
        )
    )
    assert load_campaign_spec(spec)["trial_options"]["samples"][0].ref_audio == str(
        tmp_path / "unused.wav"
    )
