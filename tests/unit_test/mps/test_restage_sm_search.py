import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from sglang_omni.config.manager import ConfigManager
from sglang_omni.pipeline import runtime_config, stage_workers
from sglang_omni.pipeline.mp_runner import _build_stage_groups
from sglang_omni.pipeline.runtime_config import prepare_pipeline_runtime
from sglang_omni.pipeline.stage_workers import _patched_spawn_env
from sglang_omni.restage.plan import SearchSpace, write_plan

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="Linux MPS spawn environment"
)
CAP = "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"


def test_mps_search_exports_distinct_process_caps_reaching_child_environment(
    tmp_path, monkeypatch
):
    root = Path(__file__).resolve().parents[3]
    space = SearchSpace.model_validate_json(
        (root / "examples/configs/restage_qwen3_tts_mps_search.json").read_text()
    )
    original = ConfigManager.from_file(
        str(root / "examples/configs/qwen3_tts_1_7b_customvoice.yaml")
    ).config
    monkeypatch.delenv(CAP, raising=False)
    monkeypatch.setattr(runtime_config, "_visible_device_count", lambda: None)
    monkeypatch.setattr(stage_workers, "get_gpu_compat_env_defaults", lambda env: {})
    destination = tmp_path / "plan"
    summary = write_plan(original, space, destination)
    assert summary["complete"] and summary["accepted"] == 20
    rows = [
        json.loads(line)
        for line in (destination / "candidates.jsonl").read_text().splitlines()
    ]
    checked = set()
    for row in rows:
        config = ConfigManager.from_file(str(destination / row["config_file"])).config
        if config.mps != "on":
            continue
        expected = {s.process: s.env[CAP] for s in config.stages if CAP in s.env}
        assert len(expected) == 2
        combination = tuple(sorted(expected.items()))
        if combination in checked:
            continue
        checked.add(combination)
        with tempfile.TemporaryDirectory(prefix="rs-sm-") as base:
            config.endpoints.base_path = base
            prep = prepare_pipeline_runtime(config)
            try:
                groups = _build_stage_groups(
                    config,
                    stages_cfg=prep.stages_cfg,
                    endpoints=prep.endpoints,
                    placement_plan=prep.placement_plan,
                    process_plan=prep.process_plan,
                    replica_topology=prep.replica_topology,
                )
                for group in groups:
                    for spec in group.process_specs:
                        with _patched_spawn_env(spec):
                            actual = subprocess.check_output(
                                [
                                    sys.executable,
                                    "-c",
                                    f"import os; print(os.environ.get('{CAP}', 'unset'))",
                                ],
                                text=True,
                            ).strip()
                        assert actual == expected.get(spec.process_name, "unset")
                        assert CAP not in os.environ
            finally:
                prep.runtime_dir.close()
    assert len(checked) == 4
    assert (("tts_engine", "100"), ("vocoder", "100")) in checked
