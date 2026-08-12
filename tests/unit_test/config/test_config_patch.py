# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the normalized patch representation and its precedence."""

from __future__ import annotations

import pytest

from sglang_omni.config.patch import (
    ConfigPatch,
    ConfigPatchSet,
    ConfigSource,
    DuplicatePatchError,
    Layer,
    SourceKind,
    Specificity,
)
from sglang_omni.config.path import ConfigPathError

MEM_FRACTION = "stages.thinker.runtime.sglang_server_args.mem_fraction_static"

YAML = ConfigSource(SourceKind.YAML_FILE, "configs/omni.yaml")
CLI_SET = ConfigSource(SourceKind.CLI_SET, "--set")
CLI_FLAG = ConfigSource(SourceKind.CLI_FLAG, "--mem-fraction-static")
MODEL = ConfigSource(SourceKind.MODEL_DEFAULT, "FakeOmniConfig")
ROUTER = ConfigSource(SourceKind.ROUTER, "worker-0")


def patch(path, value, source, **kwargs):
    return ConfigPatch.create(path, value, source, **kwargs)


class TestConstruction:
    def test_layer_defaults_to_the_source_kind(self):
        assert patch(MEM_FRACTION, "0.8", YAML).layer is Layer.USER_FILE
        assert patch(MEM_FRACTION, "0.8", CLI_SET).layer is Layer.CLI
        assert patch(MEM_FRACTION, "0.8", MODEL).layer is Layer.MODEL_DEFAULT
        assert patch(MEM_FRACTION, "0.8", ROUTER).layer is Layer.ROUTER

    def test_value_is_coerced_to_the_declared_type(self):
        assert patch("stages.thinker.tp_size", "4", CLI_SET).value == 4
        assert patch(MEM_FRACTION, "0.8", CLI_SET).value == 0.8

    def test_coercion_can_be_disabled_for_already_typed_sources(self):
        assert patch("stages.thinker.tp_size", "4", YAML, coerce=False).value == "4"

    def test_unknown_path_fails_at_construction(self):
        with pytest.raises(ConfigPathError):
            patch("stages.thinker.runtime.nope", 1, CLI_SET)

    def test_internal_path_is_refused(self):
        with pytest.raises(ConfigPathError, match="internal"):
            patch("stages.thinker.factory_args.lookahead", 8, CLI_SET)

    def test_explicit_layer_override_wins_over_the_source_default(self):
        assert patch(MEM_FRACTION, 0.8, YAML, layer=Layer.CLI).layer is Layer.CLI


class TestPrecedence:
    def test_cli_beats_yaml_regardless_of_insertion_order(self):
        patchset = ConfigPatchSet()
        patchset.add(patch(MEM_FRACTION, 0.9, CLI_SET))
        patchset.add(patch(MEM_FRACTION, 0.5, YAML))
        assert [p.value for p in patchset.ordered()] == [0.5, 0.9]
        assert patchset.winners()[MEM_FRACTION].value == 0.9

    def test_role_alias_beats_broadcast_alias_inside_one_layer(self):
        """``--mem-fraction-static`` and ``--talker-mem-fraction-static``
        together is legal; the per-role value must win."""
        patchset = ConfigPatchSet()
        patchset.add(patch(MEM_FRACTION, 0.6, CLI_FLAG, specificity=Specificity.ROLE))
        patchset.add(
            patch(MEM_FRACTION, 0.3, CLI_FLAG, specificity=Specificity.BROADCAST)
        )
        assert patchset.winners()[MEM_FRACTION].value == 0.6

    def test_explicit_set_beats_role_alias(self):
        patchset = ConfigPatchSet()
        patchset.add(patch(MEM_FRACTION, 0.6, CLI_FLAG, specificity=Specificity.ROLE))
        patchset.add(patch(MEM_FRACTION, 0.7, CLI_SET))
        assert patchset.winners()[MEM_FRACTION].value == 0.7

    def test_container_patch_applies_before_a_nested_one(self):
        patchset = ConfigPatchSet()
        patchset.add(patch(MEM_FRACTION, 0.9, YAML))
        patchset.add(patch("stages.thinker.runtime", {"max_seq_len": 8}, YAML))
        assert [p.key for p in patchset.ordered()] == [
            "stages.thinker.runtime",
            MEM_FRACTION,
        ]

    def test_insertion_order_is_the_last_tie_break(self):
        patchset = ConfigPatchSet()
        patchset.add(patch("stages.thinker.tp_size", 2, YAML))
        patchset.add(patch("stages.preprocessing.tp_size", 4, YAML))
        assert [p.key for p in patchset.ordered()] == [
            "stages.thinker.tp_size",
            "stages.preprocessing.tp_size",
        ]

    def test_history_keeps_every_contributor(self):
        patchset = ConfigPatchSet()
        patchset.add(patch(MEM_FRACTION, 0.5, MODEL))
        patchset.add(patch(MEM_FRACTION, 0.7, YAML))
        patchset.add(patch(MEM_FRACTION, 0.9, CLI_SET))
        history = patchset.history()[MEM_FRACTION]
        assert [p.value for p in history] == [0.5, 0.7, 0.9]
        assert [p.source.kind for p in history] == [
            SourceKind.MODEL_DEFAULT,
            SourceKind.YAML_FILE,
            SourceKind.CLI_SET,
        ]


class TestConflicts:
    def test_same_rank_disagreement_is_a_conflict(self):
        patchset = ConfigPatchSet()
        patchset.add(patch(MEM_FRACTION, 0.5, YAML))
        patchset.add(
            patch(
                MEM_FRACTION,
                0.9,
                ConfigSource(SourceKind.YAML_STAGE_OVERRIDES, "configs/omni.yaml"),
            )
        )
        assert len(patchset.conflicts()) == 1
        with pytest.raises(DuplicatePatchError) as excinfo:
            patchset.require_no_conflicts()
        message = str(excinfo.value)
        assert "yaml file" in message
        assert "yaml stage overrides" in message

    def test_agreeing_sources_are_not_a_conflict(self):
        patchset = ConfigPatchSet()
        patchset.add(patch(MEM_FRACTION, 0.5, YAML))
        patchset.add(patch(MEM_FRACTION, 0.5, YAML))
        assert patchset.conflicts() == []
        patchset.require_no_conflicts()

    def test_different_layers_are_not_a_conflict(self):
        patchset = ConfigPatchSet()
        patchset.add(patch(MEM_FRACTION, 0.5, YAML))
        patchset.add(patch(MEM_FRACTION, 0.9, CLI_SET))
        patchset.require_no_conflicts()

    def test_different_specificity_is_not_a_conflict(self):
        patchset = ConfigPatchSet()
        patchset.add(patch(MEM_FRACTION, 0.5, CLI_FLAG, specificity=Specificity.ROLE))
        patchset.add(
            patch(MEM_FRACTION, 0.9, CLI_FLAG, specificity=Specificity.BROADCAST)
        )
        patchset.require_no_conflicts()

    def test_different_paths_are_not_a_conflict(self):
        patchset = ConfigPatchSet()
        patchset.add(patch("stages.thinker.tp_size", 2, YAML))
        patchset.add(patch("stages.preprocessing.tp_size", 4, YAML))
        patchset.require_no_conflicts()


class TestBookkeeping:
    def test_merge_preserves_both_sides(self):
        first = ConfigPatchSet().add(patch("stages.thinker.tp_size", 2, YAML))
        second = ConfigPatchSet().add(patch("stages.thinker.tp_size", 4, CLI_SET))
        merged = first.merge(second)
        assert len(merged) == 2
        assert len(first) == 1
        assert merged.winners()["stages.thinker.tp_size"].value == 4

    def test_deprecations_are_reported(self):
        patchset = ConfigPatchSet()
        patchset.add(patch("stages.thinker.tp_size", 2, YAML))
        patchset.add(
            patch(
                "stages.thinker.runtime.max_seq_len",
                8,
                YAML,
                deprecated="stage_overrides is superseded by stages.<name>",
            )
        )
        assert [p.key for p in patchset.deprecations()] == [
            "stages.thinker.runtime.max_seq_len"
        ]

    def test_describe_mentions_path_value_and_source(self):
        text = ConfigPatchSet().add(patch(MEM_FRACTION, 0.8, YAML)).describe()
        assert MEM_FRACTION in text
        assert "0.8" in text
        assert "configs/omni.yaml" in text
