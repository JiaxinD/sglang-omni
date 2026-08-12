# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import yaml

from tests.test_model.tts_ci_config import TTS_CI_PRESETS, select_tts_ci_preset

REPO_ROOT = Path(__file__).resolve().parents[3]
OMNI_WORKFLOW = REPO_ROOT / ".github/workflows/omni-ci.yaml"
TTS_WORKFLOW = REPO_ROOT / ".github/workflows/test-tts-ci.yaml"


def _workflow(path: Path) -> dict:
    return yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def _select_step_script() -> str:
    jobs = _workflow(OMNI_WORKFLOW)["jobs"]
    preflight = jobs["preflight"]
    step = next(s for s in preflight["steps"] if s.get("id") == "tts")
    return step["run"]


def test_rotation_offers_every_registered_preset() -> None:
    """The workflow must be able to select every model the presets define."""
    script = _select_step_script()
    models_line = next(
        line for line in script.splitlines() if line.strip().startswith("models=(")
    )
    rotation = set(models_line.split("(", 1)[1].split(")", 1)[0].split())
    assert rotation == set(TTS_CI_PRESETS), (
        f"workflow rotation {sorted(rotation)} does not match registered presets "
        f"{sorted(TTS_CI_PRESETS)}"
    )
    for model in rotation:
        assert f"run-{model}" in script, f"missing opt-in label for {model}"


def test_single_instance_models_skip_the_mps_stage() -> None:
    """A model without an MPS config must not drag the MPS stage along."""
    script = _select_step_script()
    assert 'resolved_config=""' in script, "no single-instance branch in preflight"

    mps = _workflow(TTS_WORKFLOW)["jobs"]["stage-5-mps"]
    assert (
        "inputs.tts_mps_config != ''" in mps["if"]
    ), "stage 5 must be conditional on an MPS config being resolved"


def test_qwen3tts_enters_observing_not_gating() -> None:
    """Thresholds are only meaningful once calibrated on the CI runner."""
    _, preset = select_tts_ci_preset("qwen3tts")
    assert preset.model.model_path == "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    assert preset.model.gate_thresholds is False
    assert not preset.thresholds.non_stream_speed
    assert not preset.thresholds.stream_speed
    # The tuned operating point, not the shipped defaults, is what CI measures.
    assert "--max-running-requests 64" in preset.model.worker_extra_args
    assert "--isolate-stage vocoder" in preset.model.worker_extra_args
