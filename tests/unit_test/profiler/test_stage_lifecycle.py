# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from sglang_omni.pipeline.stage.runtime import Stage
from sglang_omni.profiler.event_recorder import get_recorder
from sglang_omni.proto.messages import ProfilerStartMessage, ProfilerStopMessage
from tests.unit_test.fixtures.pipeline_fakes import (
    FakeRelay,
    FakeScheduler,
    RecordingStageControlPlane,
)


@pytest.mark.parametrize(("role", "rank"), [("leader", 0), ("follower", 1)])
def test_stage_profile_handler_preserves_rank_and_device_namespaces(
    tmp_path, role, rank
):
    stage = Stage(
        name="engine",
        role=role,
        get_next=lambda *_: None,
        gpu_id=0,
        placement_gpu_id=3,
        tp_rank=rank,
        tp_size=2,
        endpoints={},
        control_plane=RecordingStageControlPlane(),
        relay=FakeRelay(),
        scheduler=FakeScheduler(),
    )
    get_recorder().stop()
    try:
        stage._on_profiler_start(
            ProfilerStartMessage(
                run_id="run",
                trace_path_template="",
                event_dir=str(tmp_path),
                enable_torch=False,
            )
        )
        stage._on_profiler_stop(ProfilerStopMessage(run_id="run"))
    finally:
        get_recorder().stop()
    records = [
        json.loads(line)
        for line in next(tmp_path.glob("lifecycle_*.jsonl")).read_text().splitlines()
    ]
    join = next(r for r in records if r["kind"] == "join")
    assert join["worker"] == dict(
        role=role, tp_rank=rank, tp_size=2, gpu_id=0, placement_gpu_id=3
    )
    assert records[-1]["close_ok"]
