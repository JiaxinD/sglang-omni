# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the canonical configuration path compiler."""

from __future__ import annotations

import pytest

from sglang_omni.config.path import (
    _VISIBILITY_RULES,
    ConfigPath,
    ConfigPathError,
    PathVisibility,
    SegmentKind,
    iter_schema_paths,
    suggest_nearby,
)
from sglang_omni.config.schema import PipelineConfig

MEM_FRACTION = "stages.thinker.runtime.sglang_server_args.mem_fraction_static"


class TestParsing:
    def test_typed_leaf_carries_its_declared_type(self):
        path = ConfigPath.parse("stages.thinker.runtime.max_seq_len")
        assert path.parts == ("stages", "thinker", "runtime", "max_seq_len")
        assert path.value_type == (int | None)
        assert path.is_leaf
        assert path.stage_name == "thinker"

    def test_stage_segment_is_name_keyed(self):
        path = ConfigPath.parse("stages.thinker.tp_size")
        assert path.segments[1].kind is SegmentKind.NAMED_ITEM
        assert path.segments[2].kind is SegmentKind.FIELD

    def test_container_paths_are_not_leaves(self):
        assert not ConfigPath.parse("stages.thinker.runtime").is_leaf
        assert not ConfigPath.parse("stages").is_leaf
        assert not ConfigPath.parse("stages.thinker.env").is_leaf

    def test_scalar_list_is_a_leaf_replaced_as_a_whole(self):
        assert ConfigPath.parse("stages.thinker.stream_to").is_leaf
        with pytest.raises(ConfigPathError, match="replaced as a whole"):
            ConfigPath.parse("stages.thinker.stream_to.0")

    def test_free_mapping_accepts_any_key(self):
        path = ConfigPath.parse("stages.preprocessing.env.OMP_NUM_THREADS")
        assert path.segments[-1].kind is SegmentKind.MAPPING_KEY
        assert path.value_type is str

    def test_parent(self):
        path = ConfigPath.parse(MEM_FRACTION)
        assert path.parent.raw == "stages.thinker.runtime.sglang_server_args"
        assert ConfigPath.parse("model_path").parent is None


class TestErrors:
    def test_unknown_field_suggests_neighbours(self):
        with pytest.raises(ConfigPathError) as excinfo:
            ConfigPath.parse("stages.thinker.runtime.sglang_server_args.mem_fraction")
        message = str(excinfo.value)
        assert "mem_fraction_static" in message
        assert "did you mean" in message

    def test_positional_stage_index_is_refused_with_a_hint(self):
        with pytest.raises(ConfigPathError) as excinfo:
            ConfigPath.parse("stages.1.tp_size")
        assert "addressed by name" in str(excinfo.value)

    def test_descending_below_a_scalar_is_refused(self):
        with pytest.raises(ConfigPathError, match="leaf of type"):
            ConfigPath.parse("stages.thinker.tp_size.value")

    def test_empty_and_malformed_paths(self):
        with pytest.raises(ConfigPathError):
            ConfigPath.parse("")
        with pytest.raises(ConfigPathError, match="empty segment"):
            ConfigPath.parse("stages..tp_size")

    def test_unknown_top_level_field(self):
        with pytest.raises(ConfigPathError) as excinfo:
            ConfigPath.parse("stagez")
        assert "stages" in str(excinfo.value)


class TestVisibility:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            (MEM_FRACTION, PathVisibility.PUBLIC),
            ("stages.thinker.factory_args", PathVisibility.DEPRECATED),
            ("stages.thinker.factory_args.lookahead", PathVisibility.DEPRECATED),
            ("stages.thinker.runtime_arg_map.max_seq_len", PathVisibility.DEPRECATED),
            ("stages.thinker.name", PathVisibility.DEPRECATED),
            ("config_cls", PathVisibility.DEPRECATED),
            ("runtime_overrides.thinker", PathVisibility.DEPRECATED),
        ],
    )
    def test_classification(self, raw, expected):
        assert ConfigPath.parse(raw).visibility is expected

    def test_nothing_the_v1_chain_allowed_is_refused(self):
        """The rule table warns; it does not take away a working spelling.

        Every path the table names was writable through the V1 dotted CLI.
        Classifying one as INTERNAL/DERIVED/IDENTITY would refuse a
        configuration that launches today, which a deprecation period exists to
        avoid — so a new rule of that kind needs a version boundary, not just
        an entry here.
        """
        refused = [
            (pattern, visibility.value)
            for pattern, visibility, _ in _VISIBILITY_RULES
            if visibility not in (PathVisibility.PUBLIC, PathVisibility.DEPRECATED)
        ]
        assert not refused, f"these paths would stop working: {refused}"

    def test_require_writable_explains_why(self, monkeypatch):
        """The refusal path still works, for classifications we do apply later.

        No shipped rule is INTERNAL today — the ones that were became
        deprecations instead — so the mechanism is exercised against an
        injected rule rather than by finding a path that happens to be refused.
        """
        rules = (
            (
                "stages.*.runtime_arg_map.**",
                PathVisibility.INTERNAL,
                "runtime_arg_map is wiring owned by the model config author",
            ),
        )
        monkeypatch.setattr("sglang_omni.config.path._VISIBILITY_RULES", rules)
        path = ConfigPath.parse("stages.thinker.runtime_arg_map.max_seq_len")
        assert not path.is_writable()
        with pytest.raises(ConfigPathError, match="owned by the model config author"):
            path.require_writable()

    def test_public_paths_pass(self):
        ConfigPath.parse(MEM_FRACTION).require_writable()

    @pytest.mark.parametrize(
        "raw",
        [
            "stages.thinker.factory_args.lookahead",
            "stages.thinker.runtime_arg_map.max_seq_len",
            "stages.thinker.name",
            "config_cls",
            "runtime_overrides.thinker",
        ],
    )
    def test_deprecated_paths_are_writable_but_not_public(self, raw):
        """Deprecation warns; it does not break configurations that work today."""
        path = ConfigPath.parse(raw)
        assert path.is_writable()
        assert not path.is_public()
        path.require_writable()
        assert path.visibility_reason

    def test_rules_are_model_agnostic(self):
        """A differently-named stage hits the same rule."""
        assert (
            ConfigPath.parse("stages.code2wav.factory_args.foo").visibility
            is PathVisibility.DEPRECATED
        )


class TestCoercion:
    @pytest.mark.parametrize(
        "raw,text,expected",
        [
            ("stages.thinker.tp_size", "4", 4),
            (MEM_FRACTION, "0.75", 0.75),
            ("stages.thinker.terminal", "true", True),
            ("stages.thinker.process", "gen", "gen"),
            ("stages.thinker.gpu", "0", 0),
            ("stages.thinker.gpu", "[0, 1]", [0, 1]),
            ("stages.thinker.stream_to", '["a", "b"]', ["a", "b"]),
            ("stages.thinker.runtime.max_seq_len", "32768", 32768),
        ],
    )
    def test_typed_coercion(self, raw, text, expected):
        assert ConfigPath.parse(raw).coerce(text) == expected

    def test_string_fields_are_not_guessed_into_numbers(self):
        """The V1 walker turned every numeric-looking value into an int."""
        path = ConfigPath.parse("stages.preprocessing.env.OMP_NUM_THREADS")
        assert path.coerce("4") == "4"

    def test_non_string_values_pass_through(self):
        assert ConfigPath.parse("stages.thinker.tp_size").coerce(4) == 4

    def test_untyped_positions_fall_back_to_scalar_parsing(self):
        path = ConfigPath.parse("runtime_overrides.thinker.foo")
        assert path.coerce("true") is True
        assert path.coerce("7") == 7
        assert path.coerce("bar") == "bar"


class TestReadWrite:
    def test_read_from_model_and_from_dump(self, pipeline_config: PipelineConfig):
        path = ConfigPath.parse("stages.thinker.tp_size")
        assert path.read(pipeline_config) == 1
        assert path.read(pipeline_config.model_dump()) == 1

    def test_read_optional_none(self, pipeline_config: PipelineConfig):
        assert ConfigPath.parse(MEM_FRACTION).read(pipeline_config) is None

    def test_write_round_trips_through_validation(
        self, pipeline_config: PipelineConfig
    ):
        data = pipeline_config.model_dump()
        ConfigPath.parse(MEM_FRACTION).write(data, 0.8)
        ConfigPath.parse("stages.thinker.runtime.max_seq_len").write(data, 4096)
        rebuilt = type(pipeline_config)(**data)
        assert rebuilt.stages[1].runtime.sglang_server_args.mem_fraction_static == 0.8
        assert rebuilt.stages[1].runtime.max_seq_len == 4096

    def test_write_creates_missing_optional_container(
        self, pipeline_config: PipelineConfig
    ):
        data = pipeline_config.model_dump()
        assert data["stages"][1]["comm"] is None
        ConfigPath.parse("stages.thinker.comm.credits").write(data, 8)
        rebuilt = type(pipeline_config)(**data)
        assert rebuilt.stages[1].comm.credits == 8

    def test_write_new_mapping_key(self, pipeline_config: PipelineConfig):
        data = pipeline_config.model_dump()
        ConfigPath.parse("stages.thinker.env.CUDA_LAUNCH_BLOCKING").write(data, "1")
        rebuilt = type(pipeline_config)(**data)
        assert rebuilt.stages[1].env["CUDA_LAUNCH_BLOCKING"] == "1"

    def test_unknown_stage_lists_the_real_names(self, pipeline_config: PipelineConfig):
        data = pipeline_config.model_dump()
        with pytest.raises(ConfigPathError) as excinfo:
            ConfigPath.parse("stages.talker.tp_size").write(data, 2)
        message = str(excinfo.value)
        assert "no entry named 'talker'" in message
        assert "thinker" in message

    def test_write_does_not_touch_siblings(self, pipeline_config: PipelineConfig):
        data = pipeline_config.model_dump()
        ConfigPath.parse("stages.thinker.runtime.max_seq_len").write(data, 128)
        assert data["stages"][0]["runtime"]["max_seq_len"] is None
        assert data["stages"][1]["factory_args"] == {"lookahead": 4}


class TestSchemaEnumeration:
    def test_public_enumeration_hides_internal_paths(self):
        public = iter_schema_paths()
        assert MEM_FRACTION.replace("thinker", "*") in public
        assert "stages.*.factory_args" not in public
        assert "config_cls" not in public

    def test_full_enumeration_includes_them(self):
        every = iter_schema_paths(include_non_public=True)
        assert "stages.*.factory_args" in every
        assert "config_cls" in every

    def test_every_enumerated_path_parses(self):
        for candidate in iter_schema_paths(include_non_public=True):
            ConfigPath.parse(candidate.replace("*", "somename"))

    def test_suggest_nearby_keeps_the_stage_name(self):
        suggestions = suggest_nearby("stages.thinker.runtime.max_seq")
        assert "stages.thinker.runtime.max_seq_len" in suggestions
