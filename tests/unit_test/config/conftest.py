# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for the configuration-core unit tests."""

from __future__ import annotations

import pytest

from sglang_omni.config.schema import PipelineConfig, StageConfig


def build_pipeline_config() -> PipelineConfig:
    """A minimal two-stage pipeline that needs no model weights."""
    return PipelineConfig(
        model_path="/models/fake-omni",
        stages=[
            StageConfig(
                name="preprocessing",
                factory="tests.fake:create_preprocessing",
                process="front",
                next="thinker",
                env={"OMP_NUM_THREADS": "4"},
            ),
            StageConfig(
                name="thinker",
                factory="tests.fake:create_thinker",
                process="gen",
                terminal=True,
                factory_args={"lookahead": 4},
                runtime_arg_map={"max_seq_len": "thinker_max_seq_len"},
            ),
        ],
    )


@pytest.fixture
def pipeline_config() -> PipelineConfig:
    return build_pipeline_config()
