# SPDX-License-Identifier: Apache-2.0
"""Config surface for the pipeline-level mps switch."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sglang_omni.config import PipelineConfig, StageConfig
from sglang_omni.config.manager import ConfigManager

_FACTORY = "tests.unit_test.fixtures.pipeline_fakes.dummy_factory"


def _config(**kwargs) -> PipelineConfig:
    return PipelineConfig(
        model_path="dummy",
        stages=[
            StageConfig(
                name="thinker",
                process="pipeline",
                factory_path=_FACTORY,
                gpu=0,
                terminal=True,
            )
        ],
        **kwargs,
    )


def test_mps_defaults_off():
    assert _config().mps == "off"


@pytest.mark.parametrize("mode", ["off", "on", "auto"])
def test_mps_accepts_valid_modes(mode):
    manager = ConfigManager(_config())
    cli_args = manager.parse_extra_args(["--mps", mode])

    assert manager.merge_config(cli_args).mps == mode


def test_mps_rejects_unknown_mode():
    with pytest.raises(ValidationError):
        ConfigManager(_config()).merge_config([("mps", "always")])
