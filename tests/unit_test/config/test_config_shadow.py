# SPDX-License-Identifier: Apache-2.0
"""Unit tests for shadow-mode comparison.

Two properties matter here. Outside strict mode, nothing that happens in the
shadow path may break the caller. Inside strict mode, every divergence and
every shadow failure must be fatal, so CI can use it as a gate.
"""

from __future__ import annotations

import logging

import pytest

from sglang_omni.config.compat import patches_from_dotted_cli
from sglang_omni.config.manager import ConfigManager, _apply_stage_overrides
from sglang_omni.config.patch import ConfigPatchSet
from sglang_omni.config.resolver import ConfigResolver
from sglang_omni.config.schema import PipelineConfig
from sglang_omni.config.shadow import (
    SHADOW_ENV,
    SHADOW_STRICT_ENV,
    ShadowDivergenceError,
    compare_with_resolver,
    shadow_enabled,
    strict_enabled,
)

MEM_FRACTION = "stages.thinker.runtime.sglang_server_args.mem_fraction_static"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch):
    """Neither variable is set unless a test sets it."""
    monkeypatch.delenv(SHADOW_ENV, raising=False)
    monkeypatch.delenv(SHADOW_STRICT_ENV, raising=False)


def diverging(config: PipelineConfig) -> PipelineConfig:
    """A config the resolver will not reproduce from an empty patch set."""
    data = config.model_dump()
    data["stages"][1]["runtime"]["max_seq_len"] = 4096
    return type(config)(**data)


class TestEnvParsing:
    def test_shadow_is_on_by_default(self):
        assert shadow_enabled() is True

    def test_strict_is_off_by_default(self):
        assert strict_enabled() is False

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", "", "OFF"])
    def test_falsey_values_disable(self, monkeypatch, value):
        monkeypatch.setenv(SHADOW_ENV, value)
        monkeypatch.setenv(SHADOW_STRICT_ENV, value)
        assert shadow_enabled() is False
        assert strict_enabled() is False

    @pytest.mark.parametrize("value", ["1", "true", "yes", "on"])
    def test_truthy_values_enable(self, monkeypatch, value):
        monkeypatch.setenv(SHADOW_STRICT_ENV, value)
        assert strict_enabled() is True


class TestDivergenceReporting:
    def test_agreement_is_silent(self, pipeline_config, caplog):
        with caplog.at_level(logging.WARNING):
            compare_with_resolver(
                baseline=pipeline_config,
                expected=pipeline_config,
                build_patches=ConfigPatchSet,
                context="a test",
            )
        assert caplog.records == []

    def test_divergence_warns_by_default(self, pipeline_config, caplog):
        with caplog.at_level(logging.WARNING):
            compare_with_resolver(
                baseline=pipeline_config,
                expected=diverging(pipeline_config),
                build_patches=ConfigPatchSet,
                context="a test",
            )
        assert len(caplog.records) == 1
        message = caplog.records[0].message
        assert "disagreed with the V1 chain on a test" in message
        assert "stages.thinker.runtime.max_seq_len" in message
        assert "The V1 result was used" in message

    def test_divergence_raises_under_strict(self, pipeline_config, monkeypatch):
        monkeypatch.setenv(SHADOW_STRICT_ENV, "1")
        with pytest.raises(ShadowDivergenceError, match="1 difference"):
            compare_with_resolver(
                baseline=pipeline_config,
                expected=diverging(pipeline_config),
                build_patches=ConfigPatchSet,
                context="a test",
            )

    def test_disabled_skips_the_comparison(self, pipeline_config, monkeypatch, caplog):
        monkeypatch.setenv(SHADOW_ENV, "0")
        monkeypatch.setenv(SHADOW_STRICT_ENV, "1")
        with caplog.at_level(logging.WARNING):
            compare_with_resolver(
                baseline=pipeline_config,
                expected=diverging(pipeline_config),
                build_patches=ConfigPatchSet,
                context="a test",
            )
        assert caplog.records == []

    def test_long_diffs_are_truncated(self, pipeline_config, caplog):
        data = pipeline_config.model_dump()
        data["env_defaults"] = {f"K{index}": str(index) for index in range(30)}
        with caplog.at_level(logging.WARNING):
            compare_with_resolver(
                baseline=pipeline_config,
                expected=type(pipeline_config)(**data),
                build_patches=ConfigPatchSet,
                context="a test",
            )
        assert "and 10 more" in caplog.records[0].message


class TestShadowFailures:
    """A broken shadow path must never take the caller down with it."""

    def boom(self) -> ConfigPatchSet:
        raise RuntimeError("translation exploded")

    def test_failure_warns_by_default(self, pipeline_config, caplog):
        with caplog.at_level(logging.WARNING):
            compare_with_resolver(
                baseline=pipeline_config,
                expected=pipeline_config,
                build_patches=self.boom,
                context="a test",
            )
        message = caplog.records[0].message
        assert "the resolver failed on a test" in message
        assert "RuntimeError: translation exploded" in message

    def test_failure_raises_under_strict(self, pipeline_config, monkeypatch):
        monkeypatch.setenv(SHADOW_STRICT_ENV, "1")
        with pytest.raises(ShadowDivergenceError, match="translation exploded"):
            compare_with_resolver(
                baseline=pipeline_config,
                expected=pipeline_config,
                build_patches=self.boom,
                context="a test",
            )

    def test_invalid_patch_does_not_escape(self, pipeline_config, caplog):
        """factory_args is INTERNAL, so patch creation refuses it."""
        with caplog.at_level(logging.WARNING):
            compare_with_resolver(
                baseline=pipeline_config,
                expected=pipeline_config,
                build_patches=lambda: patches_from_dotted_cli(
                    {"stages.thinker.factory_args.lookahead": "8"}, pipeline_config
                ),
                context="a test",
            )
        assert "the resolver failed" in caplog.records[0].message


class TestWiredIntoV1:
    """The V1 entry points must stay authoritative and stay quiet."""

    def test_merge_config_is_clean_under_strict(
        self, pipeline_config, monkeypatch, caplog
    ):
        monkeypatch.setenv(SHADOW_STRICT_ENV, "1")
        with caplog.at_level(logging.WARNING):
            merged = ConfigManager(pipeline_config).merge_config(
                {"stages.thinker.tp_size": "4", MEM_FRACTION: "0.75"}
            )
        assert merged.stages[1].tp_size == 4
        assert caplog.records == []

    def test_stage_overrides_are_clean_under_strict(
        self, pipeline_config, monkeypatch, caplog
    ):
        monkeypatch.setenv(SHADOW_STRICT_ENV, "1")
        overrides = {"thinker": {"runtime": {"max_seq_len": 4096}}}
        with caplog.at_level(logging.WARNING):
            overridden = _apply_stage_overrides(pipeline_config, overrides)
        assert overridden.stages[1].runtime.max_seq_len == 4096
        assert caplog.records == []

    def test_v1_result_survives_a_broken_shadow(
        self, pipeline_config, monkeypatch, caplog
    ):
        """The launch must proceed even if the resolver cannot run at all."""

        def explode(self, patchset):
            raise RuntimeError("resolver is broken")

        monkeypatch.setattr(ConfigResolver, "resolve", explode)
        with caplog.at_level(logging.WARNING):
            merged = ConfigManager(pipeline_config).merge_config(
                {"stages.thinker.tp_size": "4"}
            )
        assert merged.stages[1].tp_size == 4
        assert "resolver is broken" in caplog.records[0].message


class TestCoercionDifference:
    """A case the resolver handles and V1 cannot, kept explicit.

    V1's ``_convert_scalar`` coerces by string shape, so a numeric-looking
    value destined for a ``dict[str, str]`` becomes an int and pydantic
    rejects the rebuilt model. The resolver coerces against the declared
    annotation instead. This runs before the shadow comparison, so strict
    mode does not flag it.
    """

    KEY = "stages.preprocessing.env.OMP_NUM_THREADS"

    def test_v1_cannot_set_a_numeric_looking_env_value(self, pipeline_config):
        with pytest.raises(ValueError, match="OMP_NUM_THREADS"):
            ConfigManager(pipeline_config).merge_config({self.KEY: "8"})

    def test_the_resolver_keeps_it_a_string(self, pipeline_config):
        resolved = ConfigResolver(pipeline_config).resolve(
            patches_from_dotted_cli({self.KEY: "8"}, pipeline_config)
        )
        assert resolved.config.stages[0].env == {"OMP_NUM_THREADS": "8"}
