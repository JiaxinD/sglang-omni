# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the deprecation notices a launch now emits.

The V1 merge chain accepted positional stage indices, ``config_cls``,
``runtime_arg_map`` and the rest without a word. The resolver knows all of
those are on their way out -- the text has been sitting on ``ConfigPatch``
since the first commit of the new core -- but until the resolver took over the
launch path there was nothing to print it. These tests are about the printing,
not about the wording of any one notice.
"""

from __future__ import annotations

import logging

import pytest

from sglang_omni.config.deprecation import reset_deprecation_state, warn_deprecations
from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.schema import PipelineConfig
from sglang_omni.config.sources import patches_from_dotted_cli


@pytest.fixture(autouse=True)
def _forget_previous_warnings():
    """The dedup set is process-wide; tests must not inherit each other's."""
    reset_deprecation_state()
    yield
    reset_deprecation_state()


def patches_for(config: PipelineConfig, **dotted: str):
    return patches_from_dotted_cli(dict(dotted), config)


class TestWarnDeprecations:
    def test_a_legacy_spelling_is_reported(self, pipeline_config):
        emitted = warn_deprecations(
            patches_for(pipeline_config, **{"stages.1.tp_size": "2"})
        )

        assert len(emitted) == 1
        assert "stages.thinker.tp_size" in emitted[0]
        assert "positional stage indices" in emitted[0]

    def test_a_current_spelling_says_nothing(self, pipeline_config):
        assert (
            warn_deprecations(
                patches_for(pipeline_config, **{"stages.thinker.tp_size": "2"})
            )
            == []
        )

    def test_the_same_notice_is_not_repeated(self, pipeline_config):
        """A Router launching eight workers in one process must not print it
        eight times; users learn to skim past a warning that always scrolls."""
        patches = patches_for(pipeline_config, **{"stages.1.tp_size": "2"})
        assert warn_deprecations(patches)
        assert warn_deprecations(patches) == []

    def test_the_same_path_from_a_different_context_is_reported_again(
        self, pipeline_config
    ):
        """Two files saying it are two things for the user to go and fix."""
        patches = patches_for(pipeline_config, **{"stages.1.tp_size": "2"})
        assert warn_deprecations(patches, context="command line")
        assert warn_deprecations(patches, context="omni.yaml")

    def test_the_context_reaches_the_message(self, pipeline_config):
        emitted = warn_deprecations(
            patches_for(pipeline_config, **{"stages.1.tp_size": "2"}),
            context="the stage_overrides block",
        )
        assert "the stage_overrides block" in emitted[0]

    def test_it_goes_to_the_log_not_only_to_the_return_value(
        self, pipeline_config, caplog
    ):
        with caplog.at_level(logging.WARNING, logger="sglang_omni.config.deprecation"):
            warn_deprecations(patches_for(pipeline_config, **{"config_cls": "Other"}))

        assert [record.levelno for record in caplog.records] == [logging.WARNING]
        assert "config_cls" in caplog.text


class TestTheLaunchPathEmitsThem:
    """The point of the module: ``serve`` used to be silent about all of this."""

    def test_merge_config_warns_about_a_positional_index(self, pipeline_config, caplog):
        manager = ConfigManager(pipeline_config)
        with caplog.at_level(logging.WARNING, logger="sglang_omni.config.deprecation"):
            merged = manager.merge_config(
                manager.parse_extra_args(["--stages.1.tp_size", "4"])
            )

        assert merged.stages[1].tp_size == 4  # the value still takes effect
        assert "positional stage indices" in caplog.text

    def test_merge_config_is_quiet_about_current_syntax(self, pipeline_config, caplog):
        manager = ConfigManager(pipeline_config)
        with caplog.at_level(logging.WARNING, logger="sglang_omni.config.deprecation"):
            manager.merge_config(
                manager.parse_extra_args(["--stages.thinker.tp_size", "4"])
            )

        assert caplog.records == []
