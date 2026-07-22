from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from sglang_omni_router.config import RouterConfig, WorkerConfig
from sglang_omni_router.supervisor import (
    RouterSupervisor,
    SupervisorContext,
    SupervisorFailure,
)


class FakeProcess:
    _next_pid = 1000

    def __init__(self) -> None:
        FakeProcess._next_pid += 1
        self.pid = FakeProcess._next_pid
        self.returncode: int | None = None
        self.wait_called = False
        self.stop_order: list[str] | None = None
        self.label = ""

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.wait_called = True
        if self.returncode is None:
            raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout or 0)
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15
        if self.stop_order is not None:
            self.stop_order.append(self.label)

    def kill(self) -> None:
        self.returncode = -9

    def die(self, code: int = 1) -> None:
        self.returncode = code


class Harness:
    """Captures spawn calls and hands out fakes; injectable fake clock."""

    def __init__(
        self, config: RouterConfig, n: int, tmp_path: Path, prefer_uds: bool = False
    ) -> None:
        self.now = 0.0
        self.dp_spawns: list[tuple[int, int, FakeProcess]] = []
        self.cp_spawns: list[tuple[SupervisorContext, FakeProcess]] = []
        self.supervisor = RouterSupervisor(
            config,
            router_processes=n,
            workdir=str(tmp_path),
            prefer_uds=prefer_uds,
            spawn_dp=self._spawn_dp,
            spawn_cp=self._spawn_cp,
            clock=lambda: self.now,
        )

    def _spawn_dp(
        self, ctx: SupervisorContext, index: int, generation: int
    ) -> FakeProcess:
        process = FakeProcess()
        process.label = f"dp{index}"
        self.dp_spawns.append((index, generation, process))
        return process

    def _spawn_cp(self, ctx: SupervisorContext) -> FakeProcess:
        process = FakeProcess()
        process.label = "cp"
        self.cp_spawns.append((ctx, process))
        return process


def _free_port() -> int:
    import socket

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


def _config() -> RouterConfig:
    return RouterConfig(
        host="127.0.0.1",
        port=_free_port(),
        workers=[WorkerConfig(url="http://127.0.0.1:8101")],
    )


def test_supervisor_binds_an_ipv6_host(tmp_path: Path) -> None:
    # --host ::1 works through the N=1 uvicorn path; the supervisor's shared
    # socket must select the family the same way uvicorn does instead of
    # hardcoding AF_INET
    import socket

    probe = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    try:
        probe.bind(("::1", 0))
    except OSError:
        pytest.skip("IPv6 loopback unavailable in this environment")
    finally:
        probe.close()

    config = RouterConfig(
        host="::1",
        port=_free_port(),
        workers=[WorkerConfig(url="http://127.0.0.1:8101")],
    )
    harness = Harness(config, n=1, tmp_path=tmp_path)
    harness.supervisor.start()
    try:
        assert harness.supervisor._socket.family == socket.AF_INET6
    finally:
        harness.supervisor.shutdown()


def test_supervisor_binds_hostnames_as_ipv4_like_uvicorn(tmp_path: Path) -> None:
    # a hostname host (no colon) must bind AF_INET exactly like uvicorn's
    # family selection, not whichever family getaddrinfo lists first
    import socket

    config = RouterConfig(
        host="localhost",
        port=_free_port(),
        workers=[WorkerConfig(url="http://127.0.0.1:8101")],
    )
    harness = Harness(config, n=1, tmp_path=tmp_path)
    harness.supervisor.start()
    try:
        assert harness.supervisor._socket.family == socket.AF_INET
    finally:
        harness.supervisor.shutdown()


def test_shutdown_closes_the_listener_before_draining_children(
    tmp_path: Path,
) -> None:
    # if the parent held the listener until after the CP drained, connections
    # would keep landing in the backlog after every acceptor was gone
    harness = Harness(_config(), n=2, tmp_path=tmp_path)
    harness.supervisor.start()
    listener_closed_at_signal: list[bool] = []
    for _, _, process in harness.dp_spawns:
        original = process.terminate

        def _terminate(orig=original):
            listener_closed_at_signal.append(harness.supervisor._socket is None)
            orig()

        process.terminate = _terminate  # type: ignore[method-assign]
    harness.supervisor.shutdown()
    assert listener_closed_at_signal and all(listener_closed_at_signal)


def test_start_spawns_cp_and_n_dps_with_the_env_contract(tmp_path: Path) -> None:
    harness = Harness(_config(), n=3, tmp_path=tmp_path)
    harness.supervisor.start()

    assert len(harness.cp_spawns) == 1
    assert [(i, g) for i, g, _ in harness.dp_spawns] == [(0, 1), (1, 1), (2, 1)]
    context = harness.supervisor.context
    written = json.loads(Path(context.config_path).read_text(encoding="utf-8"))
    assert written["workers"][0]["url"] == "http://127.0.0.1:8101"
    assert context.internal_token
    assert context.cp_epoch
    assert context.internal_tcp_url is not None  # prefer_uds=False fallback
    assert context.socket_fd > 0
    harness.supervisor.shutdown()


def test_dead_dp_is_reaped_before_reclaim_and_respawned_with_next_generation(
    tmp_path: Path,
) -> None:
    harness = Harness(_config(), n=2, tmp_path=tmp_path)
    harness.supervisor.start()
    exits: list[tuple[int, int, int, bool]] = []
    harness.supervisor.on_dp_exit = lambda index, generation, code: exits.append(
        (index, generation, code, harness.dp_spawns[index][2].wait_called)
    )

    harness.now = 100.0  # well past the rapid window
    _, _, first = harness.dp_spawns[1]
    first.die(code=137)
    harness.supervisor.poll_once()

    # reclaimed exactly once, only after the reap, with the old generation
    assert exits == [(1, 1, 137, True)]
    assert harness.supervisor.dp_slots[1].generation == 2
    assert [(i, g) for i, g, _ in harness.dp_spawns] == [(0, 1), (1, 1), (1, 2)]
    harness.supervisor.shutdown()


def test_rapidly_dying_slot_fails_closed(tmp_path: Path) -> None:
    harness = Harness(_config(), n=1, tmp_path=tmp_path)
    harness.supervisor.start()

    with pytest.raises(SupervisorFailure, match="failing"):
        for _ in range(3):
            harness.now += 1.0  # each death 1s after spawn, inside the window
            _, _, process = harness.dp_spawns[-1]
            process.die()
            harness.supervisor.poll_once()
    harness.supervisor.shutdown()


def test_slow_deaths_do_not_accumulate_toward_the_cap(tmp_path: Path) -> None:
    harness = Harness(_config(), n=1, tmp_path=tmp_path)
    harness.supervisor.start()

    for _ in range(5):
        harness.now += 60.0  # every death is long after its spawn
        _, _, process = harness.dp_spawns[-1]
        process.die()
        harness.supervisor.poll_once()

    assert harness.supervisor.dp_slots[0].generation == 6
    assert harness.supervisor.dp_slots[0].rapid_deaths == 0
    harness.supervisor.shutdown()


def test_shutdown_stops_dps_before_the_cp(tmp_path: Path) -> None:
    harness = Harness(_config(), n=2, tmp_path=tmp_path)
    harness.supervisor.start()
    order: list[str] = []
    for _, _, process in harness.dp_spawns:
        process.stop_order = order
    harness.cp_spawns[0][1].stop_order = order

    harness.supervisor.shutdown()

    assert order == ["dp0", "dp1", "cp"]


def test_cp_restart_gets_a_fresh_epoch(tmp_path: Path) -> None:
    harness = Harness(_config(), n=1, tmp_path=tmp_path)
    harness.supervisor.start()
    first_epoch = harness.cp_spawns[0][0].cp_epoch

    harness.cp_spawns[0][1].die()
    harness.supervisor.poll_once()

    assert len(harness.cp_spawns) == 2
    assert harness.cp_spawns[1][0].cp_epoch != first_epoch
    harness.supervisor.shutdown()


def test_router_processes_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=">= 1"):
        RouterSupervisor(_config(), router_processes=0, workdir=str(tmp_path))


def test_cp_restart_unlinks_a_stale_uds_socket(tmp_path: Path) -> None:
    harness = Harness(_config(), n=1, tmp_path=tmp_path, prefer_uds=True)
    harness.supervisor.start()
    uds_path = Path(harness.supervisor.context.internal_uds)
    uds_path.write_text("", encoding="utf-8")  # simulate the leftover socket

    harness.cp_spawns[0][1].die()
    harness.supervisor.poll_once()

    # the replacement CP must be able to bind: the stale file is gone
    assert not uds_path.exists()
    assert len(harness.cp_spawns) == 2
    harness.supervisor.shutdown()


def test_start_rolls_back_when_a_dp_spawn_fails(tmp_path: Path) -> None:
    config = _config()
    spawned: list[FakeProcess] = []
    listener_closed_at_signal: list[bool] = []

    def _track(process: FakeProcess) -> FakeProcess:
        original = process.terminate

        def _terminate(orig=original):
            # supervisor is defined below; the closure resolves it at call time
            listener_closed_at_signal.append(supervisor._socket is None)
            orig()

        process.terminate = _terminate  # type: ignore[method-assign]
        spawned.append(process)
        return process

    def _spawn_dp(ctx: SupervisorContext, index: int, generation: int) -> FakeProcess:
        if index == 1:
            raise RuntimeError("spawn exploded")
        return _track(FakeProcess())

    def _spawn_cp(ctx: SupervisorContext) -> FakeProcess:
        return _track(FakeProcess())

    supervisor = RouterSupervisor(
        config,
        router_processes=2,
        workdir=str(tmp_path),
        prefer_uds=False,
        spawn_dp=_spawn_dp,
        spawn_cp=_spawn_cp,
    )
    with pytest.raises(RuntimeError, match="spawn exploded"):
        supervisor.start()

    # everything already spawned was stopped, artifacts are gone
    assert all(process.returncode is not None for process in spawned)
    assert not (tmp_path / "router_config.json").exists()
    with pytest.raises(RuntimeError, match="not started"):
        _ = supervisor.context
    # rollback mirrors shutdown(): the listener was dropped before any child
    # was signaled, and the real port refuses connections afterwards
    assert listener_closed_at_signal and all(listener_closed_at_signal)
    import socket as socket_module

    probe = socket_module.socket(socket_module.AF_INET, socket_module.SOCK_STREAM)
    probe.settimeout(1.0)
    try:
        with pytest.raises(OSError):
            probe.connect(("127.0.0.1", config.port))
    finally:
        probe.close()


def test_run_forever_stops_cleanly_on_request_stop(tmp_path: Path) -> None:
    harness = Harness(_config(), n=1, tmp_path=tmp_path)
    polls: list[int] = []
    original = harness.supervisor.poll_once

    def _poll_then_stop() -> None:
        polls.append(1)
        original()
        if len(polls) >= 2:
            harness.supervisor.request_stop()

    harness.supervisor.start()
    harness.supervisor.poll_once = _poll_then_stop  # type: ignore[method-assign]
    harness.supervisor.run_forever(poll_interval_secs=0.01)

    assert len(polls) == 2
    # shutdown ran: all children stopped
    assert all(p.returncode is not None for _, _, p in harness.dp_spawns)
    assert harness.cp_spawns[0][1].returncode is not None


def test_run_forever_installs_and_restores_signal_handlers(tmp_path: Path) -> None:
    import signal as signal_module

    harness = Harness(_config(), n=1, tmp_path=tmp_path)
    harness.supervisor.start()
    before = signal_module.getsignal(signal_module.SIGTERM)
    seen: list[object] = []
    original = harness.supervisor.poll_once

    def _poll_capture() -> None:
        original()
        handler = signal_module.getsignal(signal_module.SIGTERM)
        seen.append(handler)
        handler(signal_module.SIGTERM, None)  # what a service stop delivers

    harness.supervisor.poll_once = _poll_capture  # type: ignore[method-assign]
    harness.supervisor.run_forever(poll_interval_secs=0.01)

    assert len(seen) == 1  # the handler stopped the loop on first delivery
    assert signal_module.getsignal(signal_module.SIGTERM) == before


def test_run_forever_shutdown_survives_an_unrestorable_previous_handler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # a C-level previous handler reads back as None and cannot be
    # reinstalled; that must not skip shutdown() and leak workdir/shm/UDS
    import sglang_omni_router.supervisor as sup

    def fake_signal(signum, handler):
        if handler is None:
            raise TypeError("signal handler must be signal.SIG_IGN, SIG_DFL, ...")
        return None  # pretend the previous handler was installed in C

    monkeypatch.setattr(sup.signal, "signal", fake_signal)

    harness = Harness(_config(), n=1, tmp_path=tmp_path)
    harness.supervisor.start()
    original = harness.supervisor.poll_once

    def _poll_then_stop() -> None:
        original()
        harness.supervisor.request_stop()

    harness.supervisor.poll_once = _poll_then_stop  # type: ignore[method-assign]
    harness.supervisor.run_forever(poll_interval_secs=0.01)

    # shutdown still ran and stopped every child despite the un-restorable handler
    assert all(p.returncode is not None for _, _, p in harness.dp_spawns)
    assert harness.cp_spawns[0][1].returncode is not None


def test_rapidly_dying_cp_fails_closed(tmp_path: Path) -> None:
    harness = Harness(_config(), n=1, tmp_path=tmp_path)
    harness.supervisor.start()

    with pytest.raises(SupervisorFailure, match="control plane"):
        for _ in range(3):
            harness.now += 1.0
            harness.cp_spawns[-1][1].die()
            harness.supervisor.poll_once()
    harness.supervisor.shutdown()


def test_child_env_sets_exactly_one_internal_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sglang_omni_router.supervisor import INTERNAL_TCP_URL_ENV, INTERNAL_UDS_ENV

    # a stale UDS variable inherited from the parent must not leak through
    monkeypatch.setenv(INTERNAL_UDS_ENV, "/stale/internal.sock")
    harness = Harness(_config(), n=1, tmp_path=tmp_path)  # TCP fallback mode
    harness.supervisor.start()
    env = harness.supervisor.context.child_env()
    assert INTERNAL_UDS_ENV not in env
    assert env[INTERNAL_TCP_URL_ENV].startswith("http://127.0.0.1:")
    harness.supervisor.shutdown()


def test_watch_supervisor_liveness_sigterms_on_pipe_eof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os
    import time as time_module

    from sglang_omni_router.supervisor import (
        DEATH_PIPE_FD_ENV,
        watch_supervisor_liveness,
    )

    read_fd, write_fd = os.pipe()
    killed: list[tuple[int, int]] = []
    monkeypatch.setenv(DEATH_PIPE_FD_ENV, str(read_fd))
    monkeypatch.setattr(
        "sglang_omni_router.supervisor.os.kill",
        lambda pid, sig: killed.append((pid, sig)),
    )
    watch_supervisor_liveness()
    time_module.sleep(0.05)
    assert killed == []  # supervisor alive: watcher blocks quietly

    os.close(write_fd)  # supervisor died: EOF
    deadline = time_module.monotonic() + 3.0
    while not killed and time_module.monotonic() < deadline:
        time_module.sleep(0.02)
    assert len(killed) == 1
    os.close(read_fd)


def test_context_carries_the_death_pipe_and_shutdown_closes_it(
    tmp_path: Path,
) -> None:
    import os

    from sglang_omni_router.supervisor import DEATH_PIPE_FD_ENV

    harness = Harness(_config(), n=1, tmp_path=tmp_path)
    harness.supervisor.start()
    fd = harness.supervisor.context.death_pipe_fd
    assert fd >= 0
    assert harness.supervisor.context.child_env()[DEATH_PIPE_FD_ENV] == str(fd)

    harness.supervisor.shutdown()
    with pytest.raises(OSError):
        os.fstat(fd)  # both ends closed with the supervisor


def test_supervisor_rejects_a_bound_below_the_process_count(
    tmp_path: Path,
) -> None:
    config = RouterConfig(
        host="127.0.0.1",
        port=_free_port(),
        workers=[WorkerConfig(url="http://127.0.0.1:8101")],
        max_connections=2,
    )
    with pytest.raises(ValueError, match="below"):
        RouterSupervisor(config, router_processes=4, workdir=str(tmp_path))


def test_supervisor_rejects_least_request_with_multiple_processes(
    tmp_path: Path,
) -> None:
    config = RouterConfig(
        host="127.0.0.1",
        port=_free_port(),
        workers=[WorkerConfig(url="http://127.0.0.1:8101")],
        policy="least_request",
    )
    with pytest.raises(ValueError, match="round_robin"):
        RouterSupervisor(config, router_processes=2, workdir=str(tmp_path))
    # single process keeps supporting least_request
    RouterSupervisor(config, router_processes=1, workdir=str(tmp_path))


def test_supervisor_creates_and_reclaims_admission_slots(tmp_path: Path) -> None:
    from sglang_omni_router.admission_shm import SlotCodec, admission_file_size
    from sglang_omni_router.supervisor import ADMISSION_SHM_ENV

    harness = Harness(_config(), n=2, tmp_path=tmp_path)
    harness.supervisor.start()
    shm_path = harness.supervisor.context.admission_shm_path
    assert (tmp_path / "admission.shm").exists()
    assert (tmp_path / "admission.shm").stat().st_size == admission_file_size(2)
    assert harness.supervisor.context.child_env()[ADMISSION_SHM_ENV] == shm_path

    # a DP claims its slot and dies holding in-flight budget
    shm = harness.supervisor._admission_mmap
    SlotCodec(shm, 1).write(
        inflight=5,
        peak_sum=5,
        rejected_total=2,
        generation=1,
        pid=harness.dp_spawns[1][2].pid,
        heartbeat_ts=1.0,
    )
    harness.now = 100.0
    harness.dp_spawns[1][2].die()
    harness.supervisor.poll_once()

    # reclaimed after the reap, before the replacement runs
    view = SlotCodec(shm, 1).read()
    assert view.inflight == 0
    assert view.generation == 0
    assert harness.supervisor.dp_slots[1].generation == 2
    harness.supervisor.shutdown()
    assert not (tmp_path / "admission.shm").exists()
