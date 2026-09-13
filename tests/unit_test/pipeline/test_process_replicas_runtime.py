# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio

import pytest

from sglang_omni.config import (
    EndpointsConfig,
    FactoryArgs,
    PipelineConfig,
    ProcessConfig,
    StageConfig,
)
from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner
from sglang_omni.profiler.lifecycle import recorder_coverage, write_inventory
from sglang_omni.profiler.profiler_control import ProfilerControlClient
from tests.unit_test.fixtures.pipeline_fakes import fake_factory_path


@pytest.mark.asyncio
async def test_replicated_multistage_process_dispatch_stream_and_shutdown(
    tmp_path,
    tmp_path_factory,
) -> None:
    config = PipelineConfig(
        model_path="model",
        entry_stage="producer",
        stages=[
            StageConfig(
                name="producer",
                process="pair",
                factory_path=fake_factory_path("make_replica_process_probe_scheduler"),
                factory=FactoryArgs(marker="producer", emit_stream=True),
                next="consumer",
                stream_to=["consumer"],
            ),
            StageConfig(
                name="consumer",
                process="pair",
                factory_path=fake_factory_path("make_replica_process_probe_scheduler"),
                factory=FactoryArgs(marker="consumer"),
                terminal=True,
                can_accept_stream_before_payload=True,
            ),
        ],
        processes={"pair": ProcessConfig(num_replicas=2)},
        endpoints=EndpointsConfig(base_path=str(tmp_path)),
    )
    runner = MultiProcessPipelineRunner(config)

    await runner.start(timeout=30.0)
    processes = [process for group in runner._groups for process in group.processes]
    profile_dir = tmp_path_factory.mktemp("request-profile")
    control = ProfilerControlClient(runner.stage_control_endpoints)
    try:
        inventory = runner.request_profile_inventory()
        write_inventory(profile_dir, "run", inventory)
        await control.broadcast_start(
            "run", "", event_dir=str(profile_dir), enable_torch=False
        )

        async def wait_for_recorders():
            while True:
                observed = recorder_coverage(profile_dir, "run")["workers"]
                if len(observed) == 4 and all(
                    row["status"] == "recorder_observed" for row in observed
                ):
                    return
                await asyncio.sleep(0.01)

        await asyncio.wait_for(wait_for_recorders(), timeout=5)
        assert len(inventory) == 4
        assert {row["pid"] for row in inventory} == {p.pid for p in processes}
        for pid in {p.pid for p in processes}:
            rows = [row for row in inventory if row["pid"] == pid]
            assert len(rows) == 2
            assert {row["stage"].split("@r")[0] for row in rows} == {
                "producer",
                "consumer",
            }
            assert all(row["tp_rank"] == 0 and row["role"] == "single" for row in rows)
        results = [
            await asyncio.wait_for(
                runner.coordinator.submit(f"req-{replica_id}", "hello"),
                timeout=10.0,
            )
            for replica_id in range(2)
        ]
    finally:
        await control.close()
        await runner.stop()

    coverage = recorder_coverage(profile_dir, "run")
    assert len(coverage["sessions"]) == 2
    assert all(len(w["sessions"]) == 1 for w in coverage["workers"])
    assert all(s["close"]["reason"] == "process_exit" for s in coverage["sessions"])
    assert all(
        s["close_status"] == "clean" and s["events_parsed"] > 0
        for s in coverage["sessions"]
    )

    for replica_id, result in enumerate(results):
        expected_process = f"process-pair@r{replica_id}"
        assert result["producer_process"] == expected_process
        assert result["consumer_process"] == expected_process
        assert result["stream_process"] == expected_process
        assert result["producer_pid"] == result["consumer_pid"]
        assert result["stream_pid"] == result["consumer_pid"]
        assert result["same_payload_object"] is True

    assert results[0]["producer_pid"] != results[1]["producer_pid"]
    assert all(not process.is_alive() for process in processes)
    assert [process.exitcode for process in processes] == [0, 0]
    assert list(tmp_path.iterdir()) == []
