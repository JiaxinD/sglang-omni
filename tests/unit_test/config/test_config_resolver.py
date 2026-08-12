# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the resolver, including V1/V2 parity.

The parity tests are the load-bearing ones: while the resolver runs in shadow
mode it must produce byte-identical configurations to the V1 merge chain for
every syntax the V1 chain supports.
"""

from __future__ import annotations

import pytest

from sglang_omni.config.compat import (
    canonicalize_dotted_key,
    patches_from_dotted_cli,
    patches_from_stage_overrides,
)
from sglang_omni.config.manager import ConfigManager, _apply_stage_overrides
from sglang_omni.config.patch import (
    ConfigPatch,
    ConfigPatchSet,
    ConfigSource,
    SourceKind,
)
from sglang_omni.config.resolver import ConfigResolver, diff_configs
from sglang_omni.config.schema import PipelineConfig

MEM_FRACTION = "stages.thinker.runtime.sglang_server_args.mem_fraction_static"

YAML = ConfigSource(SourceKind.YAML_FILE, "configs/omni.yaml")
CLI = ConfigSource(SourceKind.CLI_SET, "--set")
MODEL = ConfigSource(SourceKind.MODEL_DEFAULT, "FakeOmniConfig")


def patchset(*patches: ConfigPatch) -> ConfigPatchSet:
    return ConfigPatchSet(list(patches))


def make(path, value, source=CLI, **kwargs):
    return ConfigPatch.create(path, value, source, **kwargs)


class TestResolve:
    def test_applies_a_leaf_patch(self, pipeline_config: PipelineConfig):
        resolved = ConfigResolver(pipeline_config).resolve(
            patchset(make(MEM_FRACTION, "0.8"))
        )
        assert resolved.value(MEM_FRACTION) == 0.8

    def test_baseline_is_untouched(self, pipeline_config: PipelineConfig):
        ConfigResolver(pipeline_config).resolve(patchset(make(MEM_FRACTION, "0.8")))
        assert (
            pipeline_config.stages[1].runtime.sglang_server_args.mem_fraction_static
            is None
        )

    def test_result_is_validated(self, pipeline_config: PipelineConfig):
        """mem_fraction_static must stay in (0, 1); the resolver rebuilds the
        model rather than assigning onto it, so validators still fire."""
        with pytest.raises(ValueError, match="mem_fraction_static"):
            ConfigResolver(pipeline_config).resolve(patchset(make(MEM_FRACTION, "1.5")))

    def test_container_patch_deep_merges(self, pipeline_config: PipelineConfig):
        resolved = ConfigResolver(pipeline_config).resolve(
            patchset(
                make("stages.thinker.runtime.max_seq_len", 4096),
                make(
                    "stages.thinker.runtime",
                    {"sglang_server_args": {"mem_fraction_static": 0.7}},
                    YAML,
                ),
            )
        )
        # The container came from a weaker layer, so the leaf patch survives it.
        assert resolved.value("stages.thinker.runtime.max_seq_len") == 4096
        assert resolved.value(MEM_FRACTION) == 0.7

    def test_conflicting_patches_are_refused(self, pipeline_config: PipelineConfig):
        with pytest.raises(ValueError, match="same precedence"):
            ConfigResolver(pipeline_config).resolve(
                patchset(make(MEM_FRACTION, 0.5, YAML), make(MEM_FRACTION, 0.9, YAML))
            )

    def test_tp_size_alias_is_kept_coherent(self, pipeline_config: PipelineConfig):
        resolved = ConfigResolver(pipeline_config).resolve(
            patchset(make("stages.thinker.tp_size", "4"))
        )
        assert resolved.config.stages[1].tp_size == 4
        assert resolved.config.stages[1].parallelism.tp == 4

    def test_parallelism_alias_syncs_the_other_way(
        self, pipeline_config: PipelineConfig
    ):
        resolved = ConfigResolver(pipeline_config).resolve(
            patchset(make("stages.thinker.parallelism.tp", "2"))
        )
        assert resolved.config.stages[1].tp_size == 2

    def test_both_sides_written_consistently_is_fine(
        self, pipeline_config: PipelineConfig
    ):
        resolved = ConfigResolver(pipeline_config).resolve(
            patchset(
                make("stages.thinker.tp_size", 2),
                make("stages.thinker.parallelism.tp", 2),
            )
        )
        assert resolved.config.stages[1].tp_size == 2


class TestProvenance:
    def test_explains_the_full_chain(self, pipeline_config: PipelineConfig):
        resolved = ConfigResolver(pipeline_config).resolve(
            patchset(
                make(MEM_FRACTION, 0.5, MODEL),
                make(MEM_FRACTION, 0.7, YAML),
                make(MEM_FRACTION, 0.9, CLI),
            )
        )
        text = resolved.provenance.explain(MEM_FRACTION)
        assert "0.9" in text and "[winner]" in text
        assert text.count("[superseded]") == 2
        assert "configs/omni.yaml" in text

    def test_records_the_pre_patch_value(self, pipeline_config: PipelineConfig):
        resolved = ConfigResolver(pipeline_config).resolve(
            patchset(make("stages.thinker.tp_size", 4))
        )
        assert resolved.provenance.baseline["stages.thinker.tp_size"] == 1

    def test_untouched_paths_raise_with_a_hint(self, pipeline_config: PipelineConfig):
        resolved = ConfigResolver(pipeline_config).resolve(
            patchset(make("stages.thinker.tp_size", 4))
        )
        with pytest.raises(KeyError, match="No configuration source touched"):
            resolved.provenance.history("stages.thinker.process")

    def test_serializable(self, pipeline_config: PipelineConfig):
        resolved = ConfigResolver(pipeline_config).resolve(
            patchset(make(MEM_FRACTION, 0.9))
        )
        payload = resolved.provenance.to_dict()
        assert payload[MEM_FRACTION]["history"][0]["winning"] is True
        assert payload[MEM_FRACTION]["history"][0]["layer_name"] == "CLI"


class TestDiff:
    def test_identical_configs_have_no_differences(
        self, pipeline_config: PipelineConfig
    ):
        assert diff_configs(pipeline_config, pipeline_config) == []

    def test_difference_is_reported_by_stage_name(
        self, pipeline_config: PipelineConfig
    ):
        changed = ConfigResolver(pipeline_config).resolve(
            patchset(make("stages.thinker.runtime.max_seq_len", 99))
        )
        differences = diff_configs(pipeline_config, changed.config)
        assert [d.path for d in differences] == ["stages.thinker.runtime.max_seq_len"]
        assert differences[0].expected is None
        assert differences[0].actual == 99


class TestCompatTranslation:
    def test_positional_index_becomes_a_name(self, pipeline_config: PipelineConfig):
        canonical, deprecation = canonicalize_dotted_key(
            "stages.1.tp_size", pipeline_config
        )
        assert canonical == "stages.thinker.tp_size"
        assert "deprecated" in deprecation

    def test_named_key_is_left_alone(self, pipeline_config: PipelineConfig):
        canonical, deprecation = canonicalize_dotted_key(
            "stages.thinker.tp_size", pipeline_config
        )
        assert canonical == "stages.thinker.tp_size"
        assert deprecation == ""

    def test_out_of_range_index_is_reported(self, pipeline_config: PipelineConfig):
        with pytest.raises(ValueError, match="stage index 9"):
            canonicalize_dotted_key("stages.9.tp_size", pipeline_config)

    def test_stage_overrides_reject_non_runtime_keys(
        self, pipeline_config: PipelineConfig
    ):
        with pytest.raises(ValueError, match="supports only runtime overrides"):
            patches_from_stage_overrides({"thinker": {"tp_size": 2}}, pipeline_config)

    def test_stage_overrides_reject_unknown_stage(
        self, pipeline_config: PipelineConfig
    ):
        with pytest.raises(ValueError, match="unknown stage 'talker'"):
            patches_from_stage_overrides({"talker": {"runtime": {}}}, pipeline_config)

    def test_stage_overrides_flatten_to_leaves(self, pipeline_config: PipelineConfig):
        patches = patches_from_stage_overrides(
            {
                "thinker": {
                    "runtime": {"sglang_server_args": {"mem_fraction_static": 0.8}}
                }
            },
            pipeline_config,
        )
        assert [p.key for p in patches] == [MEM_FRACTION]


class TestParityWithV1:
    """The V1 chain stays authoritative; the resolver must match it exactly."""

    def _v1(self, config: PipelineConfig, extra_args: dict) -> PipelineConfig:
        return ConfigManager(config).merge_config(extra_args)

    def _v2(self, config: PipelineConfig, extra_args: dict) -> PipelineConfig:
        patches = patches_from_dotted_cli(extra_args, config)
        return ConfigResolver(config).resolve(patches).config

    @pytest.mark.parametrize(
        "extra_args",
        [
            {"stages.thinker.tp_size": "4"},
            {"stages.1.tp_size": "4"},
            {"stages.thinker.parallelism.tp": "2"},
            {"stages.thinker.runtime.max_seq_len": "8192"},
            {MEM_FRACTION: "0.75"},
            {"stages.thinker.runtime.resources.total_gpu_memory_fraction": "0.35"},
            {
                "stages.thinker.runtime.max_seq_len": "8192",
                "stages.preprocessing.runtime.video_fps": "2.0",
            },
            {"placement.max_total_gpu_memory_fraction_per_gpu": "0.9"},
            {"entry_stage": "preprocessing"},
        ],
    )
    def test_dotted_cli_parity(self, pipeline_config: PipelineConfig, extra_args):
        v1 = self._v1(pipeline_config, extra_args)
        v2 = self._v2(pipeline_config, extra_args)
        assert diff_configs(v1, v2) == []

    def test_stage_overrides_parity(self, pipeline_config: PipelineConfig):
        overrides = {
            "thinker": {
                "runtime": {
                    "max_seq_len": 4096,
                    "sglang_server_args": {"mem_fraction_static": 0.8},
                    "resources": {"total_gpu_memory_fraction": 0.35},
                }
            },
            "preprocessing": {"runtime": {"video_fps": 2.0}},
        }
        v1 = _apply_stage_overrides(pipeline_config, overrides)
        v2 = (
            ConfigResolver(pipeline_config)
            .resolve(patches_from_stage_overrides(overrides, pipeline_config))
            .config
        )
        assert diff_configs(v1, v2) == []


@pytest.mark.parametrize(
    "config_name",
    ["Qwen3OmniPipelineConfig", "Qwen3OmniSpeechColocatedPipelineConfig"],
)
class TestParityOnRealModelConfigs:
    """Same parity check against the shipped 8-stage Qwen3-Omni pipelines."""

    def _config(self, config_name: str) -> PipelineConfig:
        module = pytest.importorskip("sglang_omni.models.qwen3_omni.config")
        return getattr(module, config_name)(model_path="dummy")

    def test_dotted_cli_parity_including_positional_indices(self, config_name):
        config = self._config(config_name)
        manager = ConfigManager(config)
        extra_args = manager.parse_extra_args(
            [
                "--stages.1.runtime.resources.total-gpu-memory-fraction",
                "0.05",
                "--stages.4.runtime.resources.total-gpu-memory-fraction",
                "0.35",
                "--stages.4.runtime.sglang-server-args.mem-fraction-static",
                "0.35",
                "--stages.4.tp_size",
                "2",
            ]
        )
        v1 = manager.merge_config(extra_args)
        v2 = (
            ConfigResolver(config)
            .resolve(patches_from_dotted_cli(extra_args, config))
            .config
        )
        assert diff_configs(v1, v2) == []

    def test_every_public_leaf_path_parses(self, config_name):
        """The compiler must cover the real schema, not just the fixture."""
        from sglang_omni.config.path import ConfigPath, iter_schema_paths

        config = self._config(config_name)
        stage_name = config.stages[0].name
        for candidate in iter_schema_paths(type(config), include_non_public=True):
            ConfigPath.parse(candidate.replace("*", stage_name), type(config))
