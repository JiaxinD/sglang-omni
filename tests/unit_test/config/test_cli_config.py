# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``sgl-omni config resolve`` and ``sgl-omni config explain``.

These commands exist to answer two questions a launch cannot: *what
configuration would this launch actually use*, and *which source set this
value*. Both answers are only worth printing if they match what the server
would compute, so the tests check the printed result against
``ConfigManager`` -- the same entry point ``sgl-omni serve`` calls -- rather
than against the command's own internals.

The fixtures drive the commands through ``--config`` rather than
``--model-path``: resolving a model path reads the model's own config from
disk, which these tests have no weights for.
"""

from __future__ import annotations

import pytest
import yaml
from typer.testing import CliRunner

from sglang_omni.cli.config import config_app
from sglang_omni.config.manager import ConfigManager

FRACTION = "runtime.resources.total_gpu_memory_fraction"
DOTTED_FRACTION = "runtime.resources.total-gpu-memory-fraction"
STAGE_INDEX = 1
"""Position of the stage the fixtures address, for the deprecated syntax."""


def output_of(result) -> str:
    """Everything the command wrote, whichever click version is installed.

    click 8.2 separates stderr from stdout; older versions fold it in and
    raise on ``result.stderr``.
    """
    text = result.stdout
    try:
        text += result.stderr
    except (AttributeError, ValueError):  # pragma: no cover - version dependent
        pass
    return text


@pytest.fixture
def runner() -> CliRunner:
    """A runner whose ``result.stdout`` is stdout alone.

    click < 8.2 folds stderr into stdout unless asked not to, which would make
    the redirect test pass a document with a deprecation notice in the middle
    of it. click >= 8.2 always separates the two and dropped the argument.
    """
    try:
        return CliRunner(mix_stderr=False)
    except TypeError:  # pragma: no cover - version dependent
        return CliRunner()


@pytest.fixture
def base_config():
    """A shipped pipeline config, so the paths are ones users actually type."""
    module = pytest.importorskip("sglang_omni.models.moss_tts.config")
    return module.MossTTSPipelineConfig(model_path="dummy")


@pytest.fixture
def stage(base_config) -> str:
    return base_config.stages[STAGE_INDEX].name


@pytest.fixture
def config_file(tmp_path, base_config, stage):
    """A V1 config file whose ``stage_overrides`` block sets one field."""
    data = base_config.model_dump(mode="json")
    data["stage_overrides"] = {
        stage: {"runtime": {"resources": {"total_gpu_memory_fraction": 0.55}}}
    }
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


@pytest.fixture
def plain_config_file(tmp_path, base_config):
    """The same pipeline with no overrides at all."""
    path = tmp_path / "plain.yaml"
    path.write_text(yaml.safe_dump(base_config.model_dump(mode="json")))
    return path


class TestResolve:
    def test_prints_the_config_a_launch_would_use(self, runner, config_file):
        """The whole point: this is what `serve` with these arguments builds."""
        result = runner.invoke(config_app, ["resolve", "--config", str(config_file)])

        assert result.exit_code == 0, output_of(result)
        expected = ConfigManager.from_file(str(config_file)).config
        assert yaml.safe_load(result.stdout) == expected.model_dump(mode="json")

    def test_dotted_arguments_are_applied_like_serve_applies_them(
        self, runner, plain_config_file, stage
    ):
        result = runner.invoke(
            config_app,
            [
                "resolve",
                "--config",
                str(plain_config_file),
                f"--stages.{stage}.{DOTTED_FRACTION}",
                "0.35",
            ],
        )

        assert result.exit_code == 0, output_of(result)
        manager = ConfigManager.from_file(str(plain_config_file))
        expected = manager.merge_config(
            manager.parse_extra_args([f"--stages.{stage}.{DOTTED_FRACTION}", "0.35"])
        )
        assert yaml.safe_load(result.stdout) == expected.model_dump(mode="json")

    def test_stdout_stays_parseable_so_it_can_be_redirected(
        self, runner, plain_config_file
    ):
        """`config resolve > pipeline.yaml` must produce a usable file.

        Uses a deprecated positional index, which puts a notice on stderr; that
        notice must not land in the middle of the document.
        """
        result = runner.invoke(
            config_app,
            [
                "resolve",
                "--config",
                str(plain_config_file),
                f"--stages.{STAGE_INDEX}.tp_size",
                "2",
            ],
        )

        assert result.exit_code == 0, output_of(result)
        assert "deprecated" in output_of(result)
        printed = yaml.safe_load(result.stdout)
        assert printed["stages"][STAGE_INDEX]["tp_size"] == 2

    def test_diff_lists_only_what_the_sources_changed(self, runner, config_file, stage):
        result = runner.invoke(
            config_app, ["resolve", "--config", str(config_file), "--show", "diff"]
        )

        assert result.exit_code == 0, output_of(result)
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        assert lines == [f"stages.{stage}.{FRACTION}: None -> 0.55"]

    def test_diff_is_empty_when_no_source_says_anything(
        self, runner, plain_config_file
    ):
        result = runner.invoke(
            config_app,
            ["resolve", "--config", str(plain_config_file), "--show", "diff"],
        )

        assert result.exit_code == 0, output_of(result)
        assert "No configuration source changed" in result.stdout

    def test_provenance_shows_the_file_losing_to_the_command_line(
        self, runner, config_file, stage
    ):
        result = runner.invoke(
            config_app,
            [
                "resolve",
                "--config",
                str(config_file),
                "--show",
                "provenance",
                f"--stages.{stage}.{DOTTED_FRACTION}",
                "0.35",
            ],
        )

        assert result.exit_code == 0, output_of(result)
        assert f"stages.{stage}.{FRACTION} = 0.35" in result.stdout
        assert "0.55  <- yaml stage overrides" in result.stdout
        assert "[superseded]" in result.stdout
        assert "0.35  <- cli dotted (command line)  [winner]" in result.stdout

    def test_model_path_is_required_without_a_config_file(self, runner):
        result = runner.invoke(config_app, ["resolve"])

        assert result.exit_code != 0
        assert "--model-path is required" in output_of(result)


class TestExplain:
    def test_names_the_winning_source_and_what_it_overrode(
        self, runner, config_file, stage
    ):
        result = runner.invoke(
            config_app,
            [
                "explain",
                f"stages.{stage}.{FRACTION}",
                "--config",
                str(config_file),
                f"--stages.{stage}.{DOTTED_FRACTION}",
                "0.35",
            ],
        )

        assert result.exit_code == 0, output_of(result)
        assert f"stages.{stage}.{FRACTION} = 0.35" in result.stdout
        assert "None  <- model default" in result.stdout
        assert "0.55  <- yaml stage overrides" in result.stdout
        assert "0.35  <- cli dotted (command line)  [winner]" in result.stdout

    def test_positional_index_is_translated_and_reported(
        self, runner, config_file, stage
    ):
        """Explaining `stages.1.x` must answer about the stage, not refuse."""
        result = runner.invoke(
            config_app,
            [
                "explain",
                f"stages.{STAGE_INDEX}.{FRACTION}",
                "--config",
                str(config_file),
            ],
        )

        assert result.exit_code == 0, output_of(result)
        assert f"stages.{stage}.{FRACTION} = 0.55" in result.stdout
        assert "positional stage indices are deprecated" in output_of(result)

    def test_untouched_path_reports_the_model_default(self, runner, config_file, stage):
        """A path no source mentions is a normal question, not an error."""
        result = runner.invoke(
            config_app,
            ["explain", f"stages.{stage}.tp_size", "--config", str(config_file)],
        )

        assert result.exit_code == 0, output_of(result)
        assert f"stages.{stage}.tp_size = 1" in result.stdout
        assert "no source touched this path" in result.stdout

    def test_unknown_path_suggests_instead_of_raising(self, runner, config_file, stage):
        result = runner.invoke(
            config_app,
            ["explain", f"stages.{stage}.tp_sizee", "--config", str(config_file)],
        )

        assert result.exit_code != 0
        assert "did you mean" in output_of(result).lower()

    def test_without_a_path_it_lists_every_touched_path(
        self, runner, config_file, stage
    ):
        result = runner.invoke(config_app, ["explain", "--config", str(config_file)])

        assert result.exit_code == 0, output_of(result)
        assert (
            f"stages.{stage}.{FRACTION} = 0.55  <- yaml stage overrides"
            in result.stdout
        )
