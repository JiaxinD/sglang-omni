# SPDX-License-Identifier: Apache-2.0
"""Thin runner-to-MPS integration contracts."""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from sglang_omni.config import EndpointsConfig, PipelineConfig, StageConfig
from sglang_omni.config.patch import (
    ConfigPatch,
    ConfigPatchSet,
    ConfigSource,
    SourceKind,
)
from sglang_omni.config.resolver import ConfigResolver
from sglang_omni.mps.devices import MpsPhysicalDevice
from sglang_omni.mps.manager import MpsDirtyStateError, MpsError
from sglang_omni.mps.runtime import MpsPipelineRuntime
from sglang_omni.mps.state import MpsGpuPaths
from sglang_omni.pipeline import mp_runner
from sglang_omni.pipeline.stage_workers import (
    StageGroup,
    StageLaunchConfig,
    StageWorkerProcessSpec,
)
from tests.unit_test.mps.test_mps_manager import (
    GPU_UUID,
    FakeControlClient,
    make_manager,
    seed_shared_dir,
)

FAKE_GPU_UUID = "GPU-aaaaaaaa-bbbb-cccc-dddd-000000000000"
_MPS_FLAG = ConfigSource(SourceKind.CLI_FLAG, "--mps")


@pytest.fixture
def short_base():
    path = Path(tempfile.mkdtemp(prefix="runner-", dir="/tmp"))
    yield path
    shutil.rmtree(path, ignore_errors=True)


def noop_factory():  # pragma: no cover - never constructed in these tests
    raise AssertionError("factory must not run")


def _make_config(base_path: Path, *, mps: str = "auto") -> PipelineConfig:
    base = PipelineConfig(
        model_path="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        entry_stage="preprocessing",
        stages=[
            StageConfig(
                name="preprocessing",
                process="pipeline",
                factory_path=f"{__name__}.noop_factory",
                terminal=True,
            )
        ],
        endpoints=EndpointsConfig(base_path=str(base_path)),
    )
    patch = ConfigPatch.create("mps", mps, _MPS_FLAG)
    return ConfigResolver(base).resolve(ConfigPatchSet([patch])).config


class _FakeCoordinator:
    def __init__(self, events: list[str], *args, **kwargs) -> None:
        del args, kwargs
        self.events = events
        self.registered: dict[str, str] = {}

    async def start(self) -> None:
        return None

    async def run_completion_loop(self) -> None:
        await asyncio.Event().wait()

    def register_stage(self, name: str, endpoint: str) -> None:
        self.registered[name] = endpoint

    async def shutdown_stages(self) -> None:
        self.events.append("graceful shutdown")

    async def fail_pending_requests(self, error: BaseException) -> None:
        del error

    async def stop(self) -> None:
        return None


class _FakeProcess:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self._alive = True

    def is_alive(self) -> bool:
        return self._alive

    def terminate(self) -> None:
        self.events.append("stage terminate")
        self._alive = False

    def kill(self) -> None:
        self.events.append("stage kill")
        self._alive = False

    def join(self, timeout=None) -> None:
        del timeout
        self.events.append("stage join")


class _FakeGroup:
    stage_control_endpoints = {"preprocessing": "ipc://preprocessing"}
    process_count = 1

    def __init__(
        self,
        events: list[str],
        *,
        ready_error: BaseException | None = None,
        shutdown_gate: tuple[asyncio.Event, asyncio.Event] | None = None,
        direct_process: bool = False,
    ) -> None:
        self.events = events
        self.ready_error = ready_error
        self.shutdown_gate = shutdown_gate
        self.processes = [_FakeProcess(events)] if direct_process else []
        self.spawn_env = object()
        self.dead = False
        self.process_specs = [
            StageWorkerProcessSpec(
                process_name="pipeline",
                stage_specs=[
                    StageLaunchConfig(
                        stage_name="preprocessing",
                        factory=f"{__name__}.noop_factory",
                        placement_gpu_id=0,
                        gpu_id=0,
                        recv_endpoint="ipc://preprocessing",
                    )
                ],
            )
        ]

    def spawn(self, ctx, process_env_overrides=None) -> None:
        del ctx
        self.spawn_env = process_env_overrides
        self.events.append("spawn")

    async def wait_ready(self, timeout: float) -> None:
        del timeout
        self.events.append("ready")
        if self.ready_error is not None:
            raise self.ready_error

    def any_dead(self) -> bool:
        return self.dead

    def dead_summary(self) -> str:
        return "preprocessing exited" if self.dead else "(none)"

    def process_start_attempts(self) -> set[str]:
        return {"pipeline"} if "spawn" in self.events else set()

    async def shutdown(self) -> None:
        self.events.append("process shutdown")
        if self.shutdown_gate is not None:
            entered, release = self.shutdown_gate
            entered.set()
            await release.wait()

    def close_control_channels(self) -> None:
        self.events.append("channels closed")


class _FakeMps:
    def __init__(
        self,
        events: list[str],
        *,
        close_error: BaseException | None = None,
        spawn_env: dict[str, str] | None = None,
        probe_result: dict[str, str] | None = None,
        probe_gate: asyncio.Event | None = None,
    ) -> None:
        self.events = events
        self.close_error = close_error
        self.spawn_env = spawn_env or {
            "CUDA_MPS_PIPE_DIRECTORY": "/tmp/mps-pipe"
        }
        self.probe_result = probe_result or {}
        self.probe_gate = probe_gate
        self.started = False
        self.verified = False
        self.close_process_start_attempts: set[str] | None = None

    @property
    def has_leases(self) -> bool:
        return self.started

    async def start(self) -> None:
        self.events.append("MPS acquire")
        self.started = True

    def env_for_process(self, process_name: str) -> dict[str, str]:
        if process_name != "pipeline":
            return {}
        return dict(self.spawn_env)

    async def verify(self) -> None:
        self.events.append("MPS verify")
        self.verified = True

    async def probe_failures(self) -> dict[str, str]:
        if self.probe_gate is not None:
            await self.probe_gate.wait()
        return dict(self.probe_result)

    async def close(
        self,
        *,
        process_start_attempts: set[str] | None = None,
    ) -> None:
        self.close_process_start_attempts = (
            None
            if process_start_attempts is None
            else set(process_start_attempts)
        )
        self.events.append("MPS close")
        self.started = False
        if self.close_error is not None:
            raise self.close_error


def _patch_runner(
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
    group: _FakeGroup,
    fake_mps: _FakeMps | None,
) -> _FakeCoordinator:
    coordinator = _FakeCoordinator(events)
    monkeypatch.setattr(
        mp_runner,
        "Coordinator",
        lambda *args, **kwargs: coordinator,
    )
    monkeypatch.setattr(
        mp_runner,
        "_build_stage_groups",
        lambda *args, **kwargs: [group],
    )
    if fake_mps is not None:
        monkeypatch.setattr(
            mp_runner,
            "create_for_pipeline",
            lambda mode, specs: fake_mps,
        )
    return coordinator


class _OneGpuDeviceInfo:
    def inspect(self, gpu_ids):
        return {
            gpu_id: MpsPhysicalDevice(GPU_UUID, None)
            for gpu_id in gpu_ids
        }


class _SpawnQueue:
    def close(self) -> None:
        return None

    def join_thread(self) -> None:
        return None


class _PreStartFailureContext:
    def Event(self):
        raise OSError("process synchronization resource exhausted")


class _ProcessStartFailure:
    def start(self) -> None:
        raise OSError("Process.start failed")


class _ProcessStartFailureContext:
    def Event(self):
        return object()

    def Queue(self):
        return _SpawnQueue()

    def Process(self, **kwargs):
        del kwargs
        return _ProcessStartFailure()


def _real_mps_group() -> StageGroup:
    return StageGroup(
        "pipeline",
        [
            StageWorkerProcessSpec(
                process_name="pipeline",
                stage_specs=[
                    StageLaunchConfig(
                        stage_name="preprocessing",
                        factory=f"{__name__}.noop_factory",
                        placement_gpu_id=0,
                        gpu_id=0,
                        recv_endpoint="ipc://preprocessing",
                    )
                ],
            )
        ],
    )


def _shared_mps_runtime(
    root: Path,
    group: StageGroup,
) -> tuple[MpsPipelineRuntime, FakeControlClient, MpsGpuPaths]:
    client = FakeControlClient()
    paths = seed_shared_dir(
        root,
        client,
        daemon_pid=999,
        owners={888: True},
        clients={8000: [202]},
    )
    runtime = MpsPipelineRuntime.create(
        mode="on",
        process_specs=group.process_specs,
        device_info=_OneGpuDeviceInfo(),
        client=client,
        state_root=root,
    )
    assert runtime is not None
    client.client_tokens[202] = "healthy-coowner"
    manager = runtime.managers[GPU_UUID]
    manager.poll_interval = 0.0
    manager.drain_timeout = 0.02
    manager.stop_timeout = 0.02
    return runtime, client, paths


@pytest.mark.asyncio
async def test_mps_hooks_follow_resolved_spawn_lifecycle(short_base, monkeypatch):
    events: list[str] = []
    group = _FakeGroup(events)
    fake_mps = _FakeMps(events)
    coordinator = _patch_runner(monkeypatch, events, group, fake_mps)
    runner = mp_runner.MultiProcessPipelineRunner(_make_config(short_base))

    await runner.start()

    assert events.index("MPS acquire") < events.index("spawn")
    assert events.index("spawn") < events.index("ready")
    assert events.index("ready") < events.index("MPS verify")
    assert group.spawn_env == {
        "pipeline": {"CUDA_MPS_PIPE_DIRECTORY": "/tmp/mps-pipe"}
    }
    assert fake_mps.verified
    assert coordinator.registered == {"preprocessing": "ipc://preprocessing"}

    await runner.stop()

    assert events.index("graceful shutdown") < events.index("process shutdown")
    assert events.index("process shutdown") < events.index("MPS close")


@pytest.mark.asyncio
async def test_mps_startup_error_cleans_children_before_close(short_base, monkeypatch):
    events: list[str] = []
    group = _FakeGroup(
        events,
        ready_error=RuntimeError("ready failed"),
        direct_process=True,
    )
    fake_mps = _FakeMps(events)
    _patch_runner(monkeypatch, events, group, fake_mps)
    runner = mp_runner.MultiProcessPipelineRunner(_make_config(short_base))

    with pytest.raises(RuntimeError, match="ready failed"):
        await runner.start()

    assert events.index("stage terminate") < events.index("MPS close")
    assert not fake_mps.has_leases
    assert fake_mps.close_process_start_attempts == {"pipeline"}


@pytest.mark.asyncio
async def test_mps_startup_cancellation_cleans_children_before_close(
    short_base,
    monkeypatch,
):
    events: list[str] = []
    group = _FakeGroup(
        events,
        ready_error=asyncio.CancelledError(),
        direct_process=True,
    )
    fake_mps = _FakeMps(events)
    _patch_runner(monkeypatch, events, group, fake_mps)
    runner = mp_runner.MultiProcessPipelineRunner(_make_config(short_base))

    with pytest.raises(asyncio.CancelledError):
        await runner.start()

    assert events.index("stage terminate") < events.index("MPS close")
    assert not fake_mps.has_leases
    assert fake_mps.close_process_start_attempts == {"pipeline"}


@pytest.mark.asyncio
async def test_startup_cancellation_remains_primary_when_mps_close_is_dirty(
    short_base,
    monkeypatch,
):
    events: list[str] = []
    group = _FakeGroup(
        events,
        ready_error=asyncio.CancelledError(),
        direct_process=True,
    )
    dirty = MpsDirtyStateError("dirty state persisted")
    fake_mps = _FakeMps(events, close_error=dirty)
    _patch_runner(monkeypatch, events, group, fake_mps)
    runner = mp_runner.MultiProcessPipelineRunner(_make_config(short_base))

    with pytest.raises(asyncio.CancelledError) as exc_info:
        await runner.start()

    assert exc_info.value.__cause__ is dirty
    assert events.index("stage terminate") < events.index("MPS close")


@pytest.mark.asyncio
async def test_failure_before_first_mps_process_start_rolls_back_cleanly(
    short_base,
    monkeypatch,
):
    events: list[str] = []
    group = _real_mps_group()
    runtime, client, paths = _shared_mps_runtime(short_base, group)
    foreign_clients = client.snapshot(paths.pipe_dir)
    _patch_runner(monkeypatch, events, group, runtime)
    monkeypatch.setattr(
        mp_runner.multiprocessing,
        "get_context",
        lambda _method: _PreStartFailureContext(),
    )
    runner = mp_runner.MultiProcessPipelineRunner(
        _make_config(short_base, mps="on")
    )

    with pytest.raises(OSError, match="synchronization resource exhausted"):
        await runner.start()

    manager = make_manager(short_base, client)
    assert not (paths.owners_dir / str(os.getpid())).exists()
    assert (paths.owners_dir / "888").read_text() == "active\n"
    assert client.snapshot(paths.pipe_dir) == foreign_clients
    assert client.daemon_process_alive(999)

    later_lease = manager.acquire({"later": "later-owner"})
    manager.release(later_lease, clients_could_have_attached=False)
    assert (paths.owners_dir / "888").read_text() == "active\n"
    assert client.daemon_process_alive(999)


@pytest.mark.asyncio
async def test_attempted_mps_process_start_keeps_fail_closed_cleanup(
    short_base,
    monkeypatch,
):
    events: list[str] = []
    group = _real_mps_group()
    runtime, client, paths = _shared_mps_runtime(short_base, group)
    foreign_clients = client.snapshot(paths.pipe_dir)
    _patch_runner(monkeypatch, events, group, runtime)
    monkeypatch.setattr(
        mp_runner.multiprocessing,
        "get_context",
        lambda _method: _ProcessStartFailureContext(),
    )
    runner = mp_runner.MultiProcessPipelineRunner(
        _make_config(short_base, mps="on")
    )

    with pytest.raises(OSError, match="Process.start failed") as exc_info:
        await runner.start()

    assert isinstance(exc_info.value.__cause__, MpsDirtyStateError)
    assert "ownership is incomplete" in str(exc_info.value.__cause__)
    assert (paths.owners_dir / str(os.getpid())).read_text() == "retained\n"
    assert (paths.owners_dir / "888").read_text() == "active\n"
    assert client.snapshot(paths.pipe_dir) == foreign_clients
    assert client.daemon_process_alive(999)
    with pytest.raises(MpsError, match="retained"):
        make_manager(short_base, client).acquire({"later": "later-owner"})


@pytest.mark.asyncio
async def test_mps_watchdog_waits_for_terminal_cleanup_and_reports_close_error(
    short_base,
    monkeypatch,
    caplog,
):
    events: list[str] = []
    entered = asyncio.Event()
    release = asyncio.Event()
    group = _FakeGroup(events, shutdown_gate=(entered, release))
    dirty = MpsDirtyStateError("dirty state persisted")
    probe_gate = asyncio.Event()
    fake_mps = _FakeMps(
        events,
        close_error=dirty,
        probe_result={FAKE_GPU_UUID: "daemon identity changed"},
        probe_gate=probe_gate,
    )
    _patch_runner(monkeypatch, events, group, fake_mps)
    runner = mp_runner.MultiProcessPipelineRunner(_make_config(short_base))
    await runner.start()

    probe_gate.set()
    await entered.wait()
    waiter = asyncio.create_task(runner.wait_failed())
    await asyncio.sleep(0)
    assert not waiter.done()

    release.set()
    with pytest.raises(RuntimeError, match="MPS health check failed") as exc_info:
        await waiter

    assert "MPS cleanup after runtime failure failed: dirty state persisted" in caplog.text
    assert FAKE_GPU_UUID in str(exc_info.value)
    assert "daemon identity changed" in str(exc_info.value)
    assert exc_info.value.__cause__ is dirty


@pytest.mark.asyncio
async def test_mps_off_keeps_merge_base_spawn_and_failure_order(
    short_base,
    monkeypatch,
):
    events: list[str] = []
    entered = asyncio.Event()
    release = asyncio.Event()
    group = _FakeGroup(events, shutdown_gate=(entered, release))
    _patch_runner(monkeypatch, events, group, fake_mps=None)

    original_sleep = asyncio.sleep

    async def checkpoint(_delay: float) -> None:
        await original_sleep(0)

    monkeypatch.setattr(mp_runner.asyncio, "sleep", checkpoint)

    def unexpected_mps(*args, **kwargs):
        del args, kwargs
        raise AssertionError("mps=off must not create an MPS runtime")

    monkeypatch.setattr(mp_runner, "create_for_pipeline", unexpected_mps)
    runner = mp_runner.MultiProcessPipelineRunner(
        _make_config(short_base, mps="off")
    )
    await runner.start()
    assert group.spawn_env is None

    group.dead = True
    waiter = asyncio.create_task(runner.wait_failed())
    await entered.wait()
    assert not waiter.done()
    release.set()
    with pytest.raises(RuntimeError, match="Dead stage process"):
        await waiter
    assert "MPS close" not in events
