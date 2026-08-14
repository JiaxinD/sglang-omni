# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the canonical set surfaces: ``set:`` and ``--set``.

Two things are being pinned down here. The first is that both spellings mean
exactly the same thing as each other and as a dotted CLI override -- same
coercion, same sentinel, same path validation -- because a second way to set a
value is only worth having if it needs no separate mental model.

The second is the refusal. ``--set stages.thinker.tp_size=8`` and
``--stages.thinker.tp_size 4`` sit at the same layer *and* the same
specificity, so the resolver has nothing to break the tie with and says so
instead of inventing a rule. The tests below assert that this stays an error
rather than quietly becoming last-one-wins.
"""

from __future__ import annotations

import pytest
import yaml

from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.patch import DuplicatePatchError, Layer, SourceKind, Specificity
from sglang_omni.config.path import ConfigPathError
from sglang_omni.config.resolver import ConfigResolver
from sglang_omni.config.schema import PipelineConfig
from sglang_omni.config.sources import (
    patches_from_dotted_cli,
    patches_from_set_block,
    patches_from_set_cli,
    split_assignment,
)

MAX_SEQ_LEN = "stages.thinker.runtime.max_seq_len"
TP_SIZE = "stages.thinker.tp_size"
ROUTE_FN = "stages.preprocessing.route_fn"


def resolve(config: PipelineConfig, *patchsets) -> PipelineConfig:
    """Apply several patch sets the way a single entry point would."""
    merged = patchsets[0]
    for extra in patchsets[1:]:
        merged = merged.merge(extra)
    return ConfigResolver(config).resolve(merged).config


class TestSplitAssignment:
    def test_splits_on_the_first_equals(self):
        assert split_assignment(f"{TP_SIZE}=4") == (TP_SIZE, "4")

    def test_the_value_keeps_every_later_equals(self):
        """JSON and query-string shaped values have to survive intact."""
        path, value = split_assignment('stages.thinker.env={"A": "x=y"}')
        assert value == '{"A": "x=y"}'

    def test_surrounding_space_around_the_path_is_ignored(self):
        assert split_assignment(f"  {TP_SIZE} =4")[0] == TP_SIZE

    @pytest.mark.parametrize("text", [TP_SIZE, "", "=4", " =4"])
    def test_a_missing_path_or_equals_is_refused(self, text):
        """A ConfigPathError, so the CLI's path-error handling catches it."""
        with pytest.raises(ConfigPathError, match="PATH=VALUE"):
            split_assignment(text)


class TestSetOnTheCommandLine:
    def test_text_is_coerced_to_the_declared_type(self, pipeline_config):
        patches = patches_from_set_cli([f"{MAX_SEQ_LEN}=8192"], pipeline_config)
        (patch,) = patches
        assert patch.value == 8192
        assert isinstance(patch.value, int)

    def test_the_none_sentinel_clears_a_field(self, pipeline_config):
        """The same sentinel a dotted override honours, not the string 'none'.

        Worth an end-to-end assertion rather than a check on the patch: what
        made this go wrong in the first place was pydantic's smart union mode
        satisfying ``str | None`` with the *word*, so the field has to be read
        back off a validated config.
        """
        seeded = resolve(
            pipeline_config,
            patches_from_set_cli([f"{ROUTE_FN}=tests.fake:route"], pipeline_config),
        )
        assert seeded.stages[0].route_fn == "tests.fake:route"

        config = resolve(seeded, patches_from_set_cli([f"{ROUTE_FN}=none"], seeded))
        assert config.stages[0].route_fn is None

    def test_it_lands_on_the_cli_layer_as_an_explicit_path(self, pipeline_config):
        (patch,) = patches_from_set_cli([f"{TP_SIZE}=2"], pipeline_config)
        assert patch.source.kind is SourceKind.CLI_SET
        assert patch.layer is Layer.CLI
        assert patch.specificity is Specificity.EXPLICIT

    def test_an_unknown_path_is_refused_with_suggestions(self, pipeline_config):
        with pytest.raises(ConfigPathError) as excinfo:
            patches_from_set_cli(["stages.thinker.tp_siz=2"], pipeline_config)
        assert "did you mean" in str(excinfo.value).lower()

    def test_positional_stage_indices_are_not_accepted(self, pipeline_config):
        """New syntax, so it is not born deprecated -- and the error says why."""
        with pytest.raises(ConfigPathError, match="by name, not by index"):
            patches_from_set_cli(["stages.1.tp_size=2"], pipeline_config)

    def test_several_paths_are_set_in_one_command(self, pipeline_config):
        config = resolve(
            pipeline_config,
            patches_from_set_cli(
                [f"{MAX_SEQ_LEN}=8192", f"{TP_SIZE}=2"], pipeline_config
            ),
        )
        assert config.stages[1].runtime.max_seq_len == 8192
        assert config.stages[1].tp_size == 2


class TestSetBlock:
    def test_flat_dotted_keys_with_yaml_typed_values(self, pipeline_config):
        config = resolve(
            pipeline_config,
            patches_from_set_block(
                {MAX_SEQ_LEN: 8192, TP_SIZE: 2, "entry_stage": "preprocessing"},
                pipeline_config,
            ),
        )
        assert config.stages[1].runtime.max_seq_len == 8192
        assert config.stages[1].tp_size == 2
        assert config.entry_stage == "preprocessing"

    def test_a_quoted_number_still_reaches_an_int_field_as_an_int(
        self, pipeline_config
    ):
        (patch,) = patches_from_set_block({MAX_SEQ_LEN: "8192"}, pipeline_config)
        assert patch.value == 8192

    def test_it_lands_on_the_user_file_layer(self, pipeline_config):
        (patch,) = patches_from_set_block({TP_SIZE: 2}, pipeline_config)
        assert patch.source.kind is SourceKind.YAML_FILE
        assert patch.layer is Layer.USER_FILE

    def test_a_block_that_is_not_a_mapping_is_refused(self, pipeline_config):
        with pytest.raises(ValueError, match="must be a mapping"):
            patches_from_set_block([f"{TP_SIZE}=2"], pipeline_config)

    def test_a_key_that_is_not_a_path_is_refused(self, pipeline_config):
        with pytest.raises(ValueError, match="written as strings"):
            patches_from_set_block({4: 2}, pipeline_config)

    def test_an_unknown_path_is_refused(self, pipeline_config):
        with pytest.raises(ConfigPathError):
            patches_from_set_block({"stages.thinker.nonesuch": 2}, pipeline_config)


class TestPrecedence:
    def test_the_command_line_beats_the_file(self, pipeline_config):
        config = resolve(
            pipeline_config,
            patches_from_set_block({TP_SIZE: 2}, pipeline_config),
            patches_from_set_cli([f"{TP_SIZE}=4"], pipeline_config),
        )
        assert config.stages[1].tp_size == 4

    def test_set_and_a_dotted_override_of_different_paths_both_apply(
        self, pipeline_config
    ):
        config = resolve(
            pipeline_config,
            patches_from_dotted_cli({TP_SIZE: "4"}, pipeline_config),
            patches_from_set_cli([f"{MAX_SEQ_LEN}=8192"], pipeline_config),
        )
        assert config.stages[1].tp_size == 4
        assert config.stages[1].runtime.max_seq_len == 8192


class TestManager:
    def test_merge_config_applies_set_values(self, pipeline_config):
        config = ConfigManager(pipeline_config).merge_config(
            {}, set_values=[f"{MAX_SEQ_LEN}=8192"]
        )
        assert config.stages[1].runtime.max_seq_len == 8192

    def test_merge_config_refuses_a_path_written_both_ways(self, pipeline_config):
        with pytest.raises(DuplicatePatchError, match="set twice"):
            ConfigManager(pipeline_config).merge_config(
                {TP_SIZE: "4"}, set_values=[f"{TP_SIZE}=8"]
            )


class TestTypedFlagPatches:
    """The typed mem-fraction flags as `sgl-omni serve` now merges them."""

    @staticmethod
    def _mem_fraction(config, stage_name):
        stage = next(s for s in config.stages if s.name == stage_name)
        return stage.runtime.sglang_server_args.mem_fraction_static

    def test_the_flag_is_applied_through_merge_config(self, shipped_config):
        """`sgl-omni serve --config f.yaml --mem-fraction-static 0.8`."""
        from sglang_omni.cli.serve import patches_from_mem_fraction_flags

        config = ConfigManager(shipped_config).merge_config(
            {},
            extra_patches=patches_from_mem_fraction_flags(
                shipped_config,
                mem_fraction_static=0.8,
                thinker_mem_fraction_static=None,
                talker_mem_fraction_static=None,
            ),
        )
        assert self._mem_fraction(config, "tts_engine") == 0.8

    def test_an_explicit_set_beats_the_typed_flag(self, shipped_config):
        """`--mem-fraction-static 0.8 --set stages...=0.7`: the explicit path
        wins by specificity, instead of the typed flag silently overwriting it
        because its helper used to run after the merge."""
        from sglang_omni.cli.serve import patches_from_mem_fraction_flags

        path = "stages.tts_engine.runtime.sglang_server_args.mem_fraction_static"
        config = ConfigManager(shipped_config).merge_config(
            {},
            set_values=[f"{path}=0.7"],
            extra_patches=patches_from_mem_fraction_flags(
                shipped_config,
                mem_fraction_static=0.8,
                thinker_mem_fraction_static=None,
                talker_mem_fraction_static=None,
            ),
        )
        assert self._mem_fraction(config, "tts_engine") == 0.7


@pytest.fixture
def shipped_config():
    """A shipped pipeline config, so the paths are ones users actually type."""
    module = pytest.importorskip("sglang_omni.models.moss_tts.config")
    return module.MossTTSPipelineConfig(model_path="dummy")


def write_config(tmp_path, config, **blocks):
    data = config.model_dump(mode="json")
    data.update(blocks)
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


class TestConfigFile:
    def test_from_file_applies_the_set_block(self, tmp_path, shipped_config):
        stage = shipped_config.stages[1].name
        path = write_config(
            tmp_path, shipped_config, set={f"stages.{stage}.runtime.max_seq_len": 8192}
        )
        config = ConfigManager.from_file(str(path)).config
        assert config.stages[1].runtime.max_seq_len == 8192

    def test_set_and_stage_overrides_coexist_on_different_paths(
        self, tmp_path, shipped_config
    ):
        stage = shipped_config.stages[1].name
        path = write_config(
            tmp_path,
            shipped_config,
            set={f"stages.{stage}.runtime.max_seq_len": 8192},
            stage_overrides={
                stage: {"runtime": {"resources": {"total_gpu_memory_fraction": 0.55}}}
            },
        )
        config = ConfigManager.from_file(str(path)).config
        assert config.stages[1].runtime.max_seq_len == 8192
        assert config.stages[1].runtime.resources.total_gpu_memory_fraction == 0.55

    def test_set_and_stage_overrides_disagreeing_on_one_path_is_refused(
        self, tmp_path, shipped_config
    ):
        """Both blocks are the user's file, so neither can outrank the other."""
        stage = shipped_config.stages[1].name
        path = write_config(
            tmp_path,
            shipped_config,
            set={f"stages.{stage}.runtime.max_seq_len": 8192},
            stage_overrides={stage: {"runtime": {"max_seq_len": 4096}}},
        )
        with pytest.raises(DuplicatePatchError) as excinfo:
            ConfigManager.from_file(str(path))
        message = str(excinfo.value)
        assert "user_file layer" in message
        assert "yaml file" in message
        assert "yaml stage overrides" in message

    def test_the_command_line_beats_the_set_block(self, tmp_path, shipped_config):
        stage = shipped_config.stages[1].name
        path = write_config(
            tmp_path, shipped_config, set={f"stages.{stage}.runtime.max_seq_len": 8192}
        )
        manager = ConfigManager.from_file(str(path))
        config = manager.merge_config(
            {}, set_values=[f"stages.{stage}.runtime.max_seq_len=2048"]
        )
        assert config.stages[1].runtime.max_seq_len == 2048
