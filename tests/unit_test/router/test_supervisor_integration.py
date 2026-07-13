"""Real-process supervisor integration (Linux only: fd passing + UDS).

Spawns the actual CP and DP children, so it exercises the shared listen
socket, the UDS internal channel, registration/heartbeat, crash respawn with
a bumped generation, and clean shutdown.
"""

from __future__ import annotations

import os
import signal
import sys
import time

import httpx
import pytest

from sglang_omni_router.config import RouterConfig, WorkerConfig
from sglang_omni_router.internal_channel import INTERNAL_TOKEN_HEADER
from sglang_omni_router.supervisor import RouterSupervisor

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="fd passing + UDS are Linux-only"
)


def _free_port() -> int:
    import socket

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


def _wait_until(predicate, timeout: float = 20.0, interval: float = 0.25):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(interval)
    raise AssertionError("condition not met within timeout")


@pytest.fixture()
def supervisor():
    config = RouterConfig(
        host="127.0.0.1",
        port=_free_port(),
        workers=[WorkerConfig(url="http://127.0.0.1:1")],  # never healthy; fine
        health_check_interval_secs=1,
    )
    instance = RouterSupervisor(config, router_processes=2)
    instance.start()
    try:
        yield instance, config
    finally:
        instance.shutdown()


def _internal_client(instance: RouterSupervisor) -> httpx.Client:
    context = instance.context
    return httpx.Client(
        transport=httpx.HTTPTransport(uds=context.internal_uds),
        base_url="http://internal",
        headers={INTERNAL_TOKEN_HEADER: context.internal_token},
        timeout=5.0,
    )


def _data_planes(instance: RouterSupervisor) -> list[dict]:
    try:
        with _internal_client(instance) as client:
            return client.get("/internal/data_planes").json()["data_planes"]
    except httpx.HTTPError:
        # CP still booting (UDS not bound yet) or restarting
        return []


def test_two_dps_share_the_data_socket_and_register(supervisor) -> None:
    instance, config = supervisor
    url = f"http://127.0.0.1:{config.port}/live"

    def _live():
        try:
            return httpx.get(url, timeout=2.0).status_code == 200
        except httpx.HTTPError:
            return False

    _wait_until(_live)
    # both DPs register and heartbeat over the UDS channel
    records = _wait_until(
        lambda: (lambda planes: planes if len(planes) == 2 else None)(
            _data_planes(instance)
        )
    )
    assert sorted(record["dp_index"] for record in records) == [0, 1]
    assert all(record["generation"] == 1 for record in records)

    # the data port keeps answering while both children accept from the
    # shared queue
    for _ in range(20):
        assert httpx.get(url, timeout=2.0).status_code == 200


def test_killed_dp_is_respawned_with_a_bumped_generation(supervisor) -> None:
    instance, config = supervisor
    _wait_until(lambda: len(_data_planes(instance)) == 2)

    victim = instance.dp_slots[0].process
    os.kill(victim.pid, signal.SIGKILL)
    _wait_until(lambda: victim.poll() is not None)
    instance.poll_once()

    assert instance.dp_slots[0].generation == 2
    records = _wait_until(
        lambda: (
            lambda planes: (
                planes
                if any(r["dp_index"] == 0 and r["generation"] == 2 for r in planes)
                else None
            )
        )(_data_planes(instance))
    )
    assert any(r["generation"] == 2 for r in records)
    # data port still serves after the respawn
    url = f"http://127.0.0.1:{config.port}/live"
    _wait_until(lambda: httpx.get(url, timeout=2.0).status_code == 200)


def test_shutdown_leaves_no_children(supervisor) -> None:
    instance, _ = supervisor
    pids = [slot.process.pid for slot in instance.dp_slots.values()]
    assert instance.cp_process is not None
    pids.append(instance.cp_process.pid)

    instance.shutdown()

    for pid in pids:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def test_admin_crud_via_the_data_port_reaches_the_cp(supervisor) -> None:
    instance, config = supervisor
    _wait_until(lambda: len(_data_planes(instance)) == 2)
    base = f"http://127.0.0.1:{config.port}"

    created = httpx.post(
        f"{base}/workers", json={"url": "http://127.0.0.1:2"}, timeout=15.0
    )
    assert created.status_code == 200

    listed = httpx.get(f"{base}/workers", timeout=5.0)
    assert listed.status_code == 200
    urls = {worker["url"] for worker in listed.json()["workers"]}
    assert urls == {"http://127.0.0.1:1", "http://127.0.0.1:2"}

    # the aggregate /health also rides the forwarding path to the CP
    health = httpx.get(f"{base}/health", timeout=5.0)
    assert health.status_code in (200, 503)
    assert "data_planes" in health.json()


def test_admission_shm_is_wired_end_to_end(supervisor) -> None:
    instance, config = supervisor
    _wait_until(lambda: len(_data_planes(instance)) == 2)
    base = f"http://127.0.0.1:{config.port}"

    def _health():
        try:
            return httpx.get(f"{base}/health", timeout=5.0).json()
        except httpx.HTTPError:
            return {}

    payload = _wait_until(
        lambda: (lambda p: p if p.get("admission_slots") else None)(_health())
    )
    assert [slot["generation"] for slot in payload["admission_slots"]] == [1, 1]
    assert payload["admission"]["inflight"] == 0
    assert payload["live_data_planes"] == 2

    # SIGKILL one DP: the supervisor reclaims its slot after the reap and the
    # respawned generation-2 process claims it again
    victim = instance.dp_slots[0].process
    os.kill(victim.pid, signal.SIGKILL)
    _wait_until(lambda: victim.poll() is not None)
    instance.poll_once()

    _wait_until(
        lambda: _health().get("admission_slots", [{}])[0].get("generation") == 2
    )
