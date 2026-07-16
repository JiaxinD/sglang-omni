from __future__ import annotations

import asyncio
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from sglang_omni_router.config import RouterConfig, WorkerConfig
from sglang_omni_router.control_plane import create_control_plane_app
from sglang_omni_router.health import HealthChecker
from sglang_omni_router.internal_channel import INTERNAL_TOKEN_HEADER
from sglang_omni_router.snapshot import SnapshotReader
from sglang_omni_router.worker import build_workers


def _config(failure_threshold: int = 1) -> RouterConfig:
    return RouterConfig(
        workers=[WorkerConfig(url="http://worker-a:8101")],
        health_failure_threshold=failure_threshold,
        health_success_threshold=1,
    )


class _Upstream:
    """Mock worker upstream tracking admin calls."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request.url.path)
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"})
        return httpx.Response(
            200, json={"success": True, "message": "ok", "results": []}
        )


def _cp_app(tmp_path: Path, upstream: _Upstream | None = None, **kwargs):
    upstream = upstream or _Upstream()
    client = httpx.AsyncClient(transport=httpx.MockTransport(upstream.handler))
    snapshot_path = str(tmp_path / "workers.json")
    app = create_control_plane_app(
        _config(**{k: v for k, v in kwargs.items() if k == "failure_threshold"}),
        snapshot_path=snapshot_path,
        cp_epoch="epoch-test",
        client=client,
        **{k: v for k, v in kwargs.items() if k != "failure_threshold"},
    )
    return app, snapshot_path, upstream


def _read_snapshot(path: str):
    reader = SnapshotReader(path)
    assert reader.maybe_reload() is True
    return reader.snapshot


@pytest.mark.asyncio
async def test_health_checker_on_tick_fires_after_each_sweep() -> None:
    config = _config()
    workers = build_workers(config.workers)
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"status": "healthy"})
        )
    )
    ticks: list[int] = []
    checker = HealthChecker(
        workers=workers,
        config=config,
        client=client,
        on_tick=lambda: ticks.append(1),
    )
    await checker.check_all_workers_health()
    await checker.check_all_workers_health()
    assert len(ticks) == 2
    await client.aclose()


def test_cp_publishes_snapshot_on_startup_and_crud(tmp_path: Path) -> None:
    app, snapshot_path, _ = _cp_app(tmp_path)
    with TestClient(app) as client:
        first = _read_snapshot(snapshot_path)
        assert first.cp_epoch == "epoch-test"
        assert [w.url for w in first.workers] == ["http://worker-a:8101"]

        assert (
            client.post("/workers", json={"url": "http://worker-b:8102"}).status_code
            == 200
        )
        after_add = _read_snapshot(snapshot_path)
        assert after_add.seq > first.seq
        assert [w.url for w in after_add.workers] == [
            "http://worker-a:8101",
            "http://worker-b:8102",
        ]

        worker_id = after_add.workers[1].worker_id
        assert (
            client.put(f"/workers/{worker_id}", json={"disabled": True}).status_code
            == 200
        )
        after_disable = _read_snapshot(snapshot_path)
        assert after_disable.workers[1].disabled is True
        assert after_disable.workers[1].routable is False

        assert client.delete(f"/workers/{worker_id}").status_code == 200
        after_delete = _read_snapshot(snapshot_path)
        assert [w.url for w in after_delete.workers] == ["http://worker-a:8101"]
    # The supervisor owns cleanup; the CP leaves this state for crash recovery.
    assert Path(snapshot_path).exists()


def test_cp_restart_recovers_worker_crud(tmp_path: Path) -> None:
    app, snapshot_path, upstream = _cp_app(tmp_path)
    with TestClient(app) as client:
        assert (
            client.post(
                "/workers",
                json={"url": "http://worker-b:8102", "capabilities": ["speech"]},
            ).status_code
            == 200
        )
        snapshot = _read_snapshot(snapshot_path)
        added = snapshot.workers[1]
        assert (
            client.put(
                f"/workers/{added.worker_id}", json={"disabled": True}
            ).status_code
            == 200
        )
        assert (
            client.delete(f"/workers/{snapshot.workers[0].worker_id}").status_code
            == 200
        )

    restarted_client = httpx.AsyncClient(
        transport=httpx.MockTransport(upstream.handler)
    )
    restarted = create_control_plane_app(
        _config(),
        snapshot_path=snapshot_path,
        cp_epoch="epoch-restarted",
        client=restarted_client,
        journal_path=str(tmp_path / "update_journal.json"),
    )
    with TestClient(restarted) as client:
        workers = client.get("/workers").json()["workers"]
        assert [worker["url"] for worker in workers] == ["http://worker-b:8102"]
        assert workers[0]["capabilities"] == ["speech"]
        assert workers[0]["disabled"] is True
        assert _read_snapshot(snapshot_path).cp_epoch == "epoch-restarted"


def test_worker_failure_report_evicts_at_threshold_and_republishes(
    tmp_path: Path,
) -> None:
    app, snapshot_path, _ = _cp_app(tmp_path, failure_threshold=1)
    with TestClient(app) as client:
        snapshot = _read_snapshot(snapshot_path)
        worker_id = snapshot.workers[0].worker_id

        response = client.post(
            "/internal/worker_failure",
            json=_failure(worker_id, seq=1),
        )
        assert response.status_code == 200
        assert response.json()["worker_state"] == "unhealthy"

        after = _read_snapshot(snapshot_path)
        assert after.workers[0].routable is False

        assert (
            client.post(
                "/internal/worker_failure", json=_failure("missing", seq=2)
            ).status_code
            == 404
        )


def test_internal_routes_on_cp_require_the_token_when_configured(
    tmp_path: Path,
) -> None:
    app, snapshot_path, _ = _cp_app(tmp_path, internal_token="cp-secret")
    with TestClient(app) as client:
        snapshot = _read_snapshot(snapshot_path)
        worker_id = snapshot.workers[0].worker_id
        body = _failure(worker_id, seq=1)
        assert client.post("/internal/worker_failure", json=body).status_code == 403
        assert (
            client.post(
                "/internal/worker_failure",
                json=body,
                headers={INTERNAL_TOKEN_HEADER: "cp-secret"},
            ).status_code
            == 200
        )


def test_weight_update_waits_for_dp_ack_then_broadcasts(tmp_path: Path) -> None:
    upstream = _Upstream()
    app, snapshot_path, _ = _cp_app(tmp_path, upstream=upstream)
    with TestClient(app) as client:
        client.post(
            "/internal/register", json={"dp_index": 0, "generation": 1, "pid": 1}
        )
        # ack everything the CP will ever publish in this test
        client.post(
            "/internal/heartbeat",
            json={
                "dp_index": 0,
                "generation": 1,
                "pid": 1,
                "last_applied_seq": 10_000,
                "last_applied_epoch": "epoch-test",
            },
        )
        response = client.post("/update_weights_from_disk", json={"path": "/m"})
        assert response.status_code == 200
        assert "/update_weights_from_disk" in upstream.calls


def test_weight_update_fails_closed_when_a_live_dp_never_acks(
    tmp_path: Path,
) -> None:
    upstream = _Upstream()
    app, snapshot_path, _ = _cp_app(
        tmp_path, upstream=upstream, dp_ack_timeout_secs=0.2
    )
    with TestClient(app) as client:
        client.post(
            "/internal/register", json={"dp_index": 0, "generation": 1, "pid": 1}
        )
        client.post(
            "/internal/heartbeat",
            json={
                "dp_index": 0,
                "generation": 1,
                "pid": 1,
                "last_applied_seq": 0,
                "last_applied_epoch": "epoch-test",
            },
        )
        response = client.post("/update_weights_from_disk", json={"path": "/m"})
        assert response.status_code == 503
        assert "aborted" in response.json()["error"]["message"]
        # no broadcast was sent and the disable flip was fully restored
        assert "/update_weights_from_disk" not in upstream.calls
        workers = client.get("/workers").json()["workers"]
        assert all(worker["disabled"] is False for worker in workers)
        assert _read_snapshot(snapshot_path).workers[0].disabled is False


def test_a_registered_dp_gone_silent_holds_the_barrier_without_an_expected_count(
    tmp_path: Path,
) -> None:
    # no supervisor-provided fleet size: a DP that registered and then went
    # silent may still be serving from a stale snapshot, so it must hold the
    # barrier closed (matching the expected-count path) rather than be dropped
    # into a zero-ACK broadcast.
    upstream = _Upstream()
    app, snapshot_path, _ = _cp_app(
        tmp_path, upstream=upstream, dp_ack_timeout_secs=0.2, dp_liveness_secs=5.0
    )
    with TestClient(app) as client:
        client.post(
            "/internal/register", json={"dp_index": 0, "generation": 1, "pid": 1}
        )
        record = app.state.internal_channel.data_planes[0]
        record.last_seen_at = time.time() - 60.0  # registered but not live

        response = client.post("/update_weights_from_disk", json={"path": "/m"})
        assert response.status_code == 503
        assert "/update_weights_from_disk" not in upstream.calls


def test_cp_admin_surface_enforces_the_admin_key(tmp_path: Path) -> None:
    app, _, _ = _cp_app(tmp_path, admin_api_key="cp-admin-key")
    with TestClient(app) as client:
        assert client.get("/model_info").status_code == 401
        assert (
            client.get(
                "/model_info", headers={"Authorization": "Bearer cp-admin-key"}
            ).status_code
            == 200
        )


def test_cp_aggregates_dp_counters_into_worker_listings(tmp_path: Path) -> None:
    app, snapshot_path, _ = _cp_app(tmp_path)
    with TestClient(app) as client:
        snapshot = _read_snapshot(snapshot_path)
        worker_id = snapshot.workers[0].worker_id

        def _report(seq: int, routed: int, ok: int, bad: int, active: int) -> dict:
            return {
                "dp_index": 0,
                "generation": 1,
                "counter_seq": seq,
                "workers": [
                    {
                        "worker_id": worker_id,
                        "routed_total": routed,
                        "successful_total": ok,
                        "failed_total": bad,
                        "current_active": active,
                    }
                ],
            }

        # first contact = baseline; growth after it is what gets displayed
        assert (
            client.post("/internal/counters", json=_report(1, 0, 0, 0, 0)).status_code
            == 200
        )
        assert (
            client.post("/internal/counters", json=_report(2, 7, 6, 1, 2)).status_code
            == 200
        )

        listed = client.get("/workers").json()["workers"][0]
        assert listed["routed_requests"] == 7
        assert listed["successful_requests"] == 6
        assert listed["failed_requests"] == 1
        assert listed["active_requests"] == 2

        # stale generation is fenced with 409 (pydantic bound: use gen >= 1)
        stale = {**_report(9, 9, 9, 0, 0), "generation": 1}
        client.post(
            "/internal/counters", json={**_report(1, 1, 1, 0, 0), "generation": 2}
        )
        assert client.post("/internal/counters", json=stale).status_code == 409


def _ack_everything(client, dp_index: int = 0) -> None:
    client.post(
        "/internal/register",
        json={"dp_index": dp_index, "generation": 1, "pid": dp_index + 1},
    )
    client.post(
        "/internal/heartbeat",
        json={
            "dp_index": dp_index,
            "generation": 1,
            "pid": dp_index + 1,
            "last_applied_seq": 10_000,
            "last_applied_epoch": "epoch-test",
        },
    )


def test_a_stale_epoch_seq_cannot_acknowledge_the_barrier(tmp_path: Path) -> None:
    # cross-epoch aliasing: seq 10_000 from a PREVIOUS CP's numbering must
    # not satisfy this CP's barrier
    upstream = _Upstream()
    app, _, _ = _cp_app(tmp_path, upstream=upstream, dp_ack_timeout_secs=0.2)
    with TestClient(app) as client:
        client.post(
            "/internal/register", json={"dp_index": 0, "generation": 1, "pid": 1}
        )
        client.post(
            "/internal/heartbeat",
            json={
                "dp_index": 0,
                "generation": 1,
                "pid": 1,
                "last_applied_seq": 10_000,
                "last_applied_epoch": "some-older-epoch",
            },
        )
        response = client.post("/update_weights_from_disk", json={"path": "/m"})
        assert response.status_code == 503
        assert "/update_weights_from_disk" not in upstream.calls


def test_barrier_fails_closed_while_an_expected_dp_is_absent(
    tmp_path: Path,
) -> None:
    # a serving DP the CP has not heard from (e.g. right after a CP restart)
    # is not an implicit ACK
    upstream = _Upstream()
    app, _, _ = _cp_app(
        tmp_path,
        upstream=upstream,
        dp_ack_timeout_secs=0.2,
        expected_data_planes=2,
    )
    with TestClient(app) as client:
        _ack_everything(client, dp_index=0)  # dp 1 never registers
        response = client.post("/update_weights_from_disk", json={"path": "/m"})
        assert response.status_code == 503
        assert "1" in response.json()["error"]["message"]
        assert "/update_weights_from_disk" not in upstream.calls


def test_worker_crud_is_rejected_while_an_update_holds_the_lock(
    tmp_path: Path,
) -> None:
    app, _, _ = _cp_app(tmp_path)
    with TestClient(app) as client:
        lock = app.state.admin_update_lock
        asyncio.run(_hold(lock))  # leaves the lock held
        try:
            response = client.post("/workers", json={"url": "http://w-x:1"})
            assert response.status_code == 409
            assert "in progress" in response.json()["error"]["message"]
            workers = client.get("/workers").json()["workers"]
            assert all(w["url"] != "http://w-x:1" for w in workers)
        finally:
            lock.release()


async def _hold(lock: asyncio.Lock) -> None:
    await lock.acquire()


def _ready_heartbeat(client, dp_index: int, pid: int) -> None:
    client.post(
        "/internal/register",
        json={"dp_index": dp_index, "generation": 1, "pid": pid},
    )
    client.post(
        "/internal/heartbeat",
        json={
            "dp_index": dp_index,
            "generation": 1,
            "pid": pid,
            "last_applied_seq": 1,
            "last_applied_epoch": "epoch-test",
        },
    )


def test_cp_health_reports_degraded_and_unhealthy_dp_states(
    tmp_path: Path,
) -> None:
    app, _, _ = _cp_app(tmp_path, expected_data_planes=2)
    with TestClient(app) as client:
        # no DP has registered: nothing can serve -> 503
        response = client.get("/health")
        assert response.status_code == 503
        assert response.json()["status"] == "unhealthy"

        # registered but never applied THIS epoch's snapshot: still not
        # serving-ready (it would shed as snapshot_stale)
        client.post(
            "/internal/register", json={"dp_index": 0, "generation": 1, "pid": 1}
        )
        response = client.get("/health")
        assert response.status_code == 503
        assert response.json()["status"] == "unhealthy"
        assert response.json()["missing_data_planes"] == [0, 1]

        # one of two DPs serving-ready: degraded -> 200
        _ready_heartbeat(client, 0, 1)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "degraded"
        assert response.json()["serving_ready_data_planes"] == 1
        assert response.json()["missing_data_planes"] == [1]

        # full roster -> healthy
        _ready_heartbeat(client, 1, 2)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

        # identity, not cardinality: an unexpected index does not fill a hole
        _ready_heartbeat(client, 99, 3)
        payload = client.get("/health").json()
        assert payload["status"] == "healthy"
        assert payload["unexpected_data_planes"] == [99]


def test_cp_health_aggregates_the_admission_shm(tmp_path: Path) -> None:
    from sglang_omni_router.admission_shm import SharedAdmission, create_admission_file

    shm_path = str(tmp_path / "admission.shm")
    create_admission_file(shm_path, 2)
    with open(shm_path, "r+b") as f:
        import mmap as mmap_module

        buf = mmap_module.mmap(f.fileno(), 0)
        dp0 = SharedAdmission(
            buf, slots=2, own_index=0, max_inflight=8, generation=1, pid=11
        )
        dp0.try_acquire()

        app, _, _ = _cp_app(
            tmp_path, expected_data_planes=2, admission_shm_path=shm_path
        )
        with TestClient(app) as client:
            client.post(
                "/internal/register",
                json={"dp_index": 0, "generation": 1, "pid": 11},
            )
            payload = client.get("/health").json()
            assert payload["admission"]["inflight"] == 1
            # the CP reports the bound from ITS config (same source as the
            # DPs in production); the slot array only carries the counts
            assert payload["admission"]["max_inflight"] == 128
            assert payload["admission_slots"][0]["generation"] == 1
            assert payload["admission_slots"][1]["generation"] == 0
        buf.close()


def _failure(worker_id: str, *, seq: int, dp: int = 0, generation: int = 1) -> dict:
    return {
        "worker_id": worker_id,
        "status_code": 502,
        "dp_index": dp,
        "generation": generation,
        "failure_seq": seq,
    }


def test_redelivered_failure_events_are_counted_once(tmp_path: Path) -> None:
    # a retry of one failure (same failure_seq) must not push a healthy
    # worker over the eviction threshold
    app, snapshot_path, _ = _cp_app(tmp_path, failure_threshold=2)
    with TestClient(app) as client:
        worker_id = _read_snapshot(snapshot_path).workers[0].worker_id

        first = client.post("/internal/worker_failure", json=_failure(worker_id, seq=1))
        assert first.status_code == 200
        redelivered = client.post(
            "/internal/worker_failure", json=_failure(worker_id, seq=1)
        )
        assert redelivered.json().get("deduplicated") is True
        workers = client.get("/workers").json()["workers"]
        assert workers[0]["health_state"] != "unhealthy"

        # a genuinely distinct second event does evict at threshold 2
        client.post("/internal/worker_failure", json=_failure(worker_id, seq=2))
        workers = client.get("/workers").json()["workers"]
        assert workers[0]["health_state"] == "unhealthy"


def test_cp_restart_recovers_an_unresolved_update_fail_closed(
    tmp_path: Path,
) -> None:
    from sglang_omni_router.update_journal import UpdateJournal

    # a previous CP died mid-broadcast: its journal survives in the workdir
    snapshot_path = str(tmp_path / "workers.json")
    journal_path = str(tmp_path / "update_journal.json")
    journal = UpdateJournal(journal_path)
    worker_id = "http%3A%2F%2Fworker-a%3A8101"
    journal.begin("/update_weights_from_disk", [worker_id])

    upstream = _Upstream()
    client = httpx.AsyncClient(transport=httpx.MockTransport(upstream.handler))
    app = create_control_plane_app(
        _config(),
        snapshot_path=snapshot_path,
        cp_epoch="epoch-test",
        client=client,
        journal_path=journal_path,
    )
    with TestClient(app) as tc:
        # the journaled target is kept disabled (fail closed), even though a
        # fresh registry plus healthy probes would have re-enabled it
        snapshot = _read_snapshot(snapshot_path)
        assert snapshot.workers[0].disabled is True
        assert snapshot.workers[0].routable is False

        # a retry of the update is rejected while the journal is unresolved
        response = tc.post("/update_weights_from_disk", json={"path": "/m"})
        assert response.status_code == 409
        assert "did not complete" in response.json()["error"]["message"]
        assert "/update_weights_from_disk" not in upstream.calls

        # operator verified the weights: re-enable resolves the journal
        assert (
            tc.put(f"/workers/{worker_id}", json={"disabled": False}).status_code == 200
        )
        assert journal.pending() == []
        response = tc.post("/update_weights_from_disk", json={"path": "/m"})
        assert response.status_code == 200
        assert "/update_weights_from_disk" in upstream.calls


def test_a_completed_update_leaves_no_journal(tmp_path: Path) -> None:
    upstream = _Upstream()
    app, snapshot_path, _ = _cp_app(tmp_path, upstream=upstream)
    with TestClient(app) as client:
        assert (
            client.post("/update_weights_from_disk", json={"path": "/m"}).status_code
            == 200
        )
        assert not (tmp_path / "update_journal.json").exists()
        workers = client.get("/workers").json()["workers"]
        assert all(worker["disabled"] is False for worker in workers)


def test_stale_incarnation_failures_cannot_evict_a_readded_worker(
    tmp_path: Path,
) -> None:
    # ABA: a request routed through incarnation A fails after the URL was
    # deleted and re-added as incarnation B; the old report must not act on B
    app, snapshot_path, _ = _cp_app(tmp_path, failure_threshold=1)
    with TestClient(app) as client:
        worker_id = _read_snapshot(snapshot_path).workers[0].worker_id
        old_incarnation = _read_snapshot(snapshot_path).workers[0].incarnation
        assert old_incarnation

        client.delete(f"/workers/{worker_id}")
        client.post("/workers", json={"url": "http://worker-a:8101"})
        new_incarnation = _read_snapshot(snapshot_path).workers[-1].incarnation
        assert new_incarnation != old_incarnation

        stale = {**_failure(worker_id, seq=1), "incarnation": old_incarnation}
        response = client.post("/internal/worker_failure", json=stale)
        assert response.json().get("stale_incarnation") is True
        workers = client.get("/workers").json()["workers"]
        assert workers[0]["health_state"] != "unhealthy"

        current = {**_failure(worker_id, seq=2), "incarnation": new_incarnation}
        client.post("/internal/worker_failure", json=current)
        workers = client.get("/workers").json()["workers"]
        assert workers[0]["health_state"] == "unhealthy"


def test_health_excludes_non_serving_and_slot_mismatched_dps(
    tmp_path: Path,
) -> None:
    from sglang_omni_router.admission_shm import SharedAdmission, create_admission_file

    shm_path = str(tmp_path / "admission.shm")
    create_admission_file(shm_path, 2)
    import mmap as mmap_module

    with open(shm_path, "r+b") as f:
        buf = mmap_module.mmap(f.fileno(), 0)
        # slot 0 claimed by generation 1; slot 1 never claimed (crashed
        # before its replacement registered)
        SharedAdmission(buf, slots=2, own_index=0, max_inflight=8, generation=1, pid=11)
        app, _, _ = _cp_app(
            tmp_path, expected_data_planes=2, admission_shm_path=shm_path
        )
        with TestClient(app) as client:
            _ready_heartbeat(client, 0, 11)
            # dp1's record claims generation 1 but its slot is unclaimed:
            # it must not count as serving (the process is gone)
            _ready_heartbeat(client, 1, 12)
            payload = client.get("/health").json()
            assert payload["serving_ready_data_planes"] == 1
            assert payload["missing_data_planes"] == [1]
            assert payload["status"] == "degraded"

            # a DP that self-reports not-serving (snapshot_stale) drops out
            client.post(
                "/internal/heartbeat",
                json={
                    "dp_index": 0,
                    "generation": 1,
                    "pid": 11,
                    "last_applied_seq": 1,
                    "last_applied_epoch": "epoch-test",
                    "serving": False,
                },
            )
            payload = client.get("/health").json()
            assert payload["serving_ready_data_planes"] == 0
            assert payload["status"] == "unhealthy"
        buf.close()


def test_broadcast_crash_keeps_targets_disabled_and_journaled(tmp_path: Path) -> None:
    # results=None (broadcast started, outcome unknown) must keep ALL targets
    # disabled and journaled - not only for init_weights_update_group
    from sglang_omni_router.app import _restore_admin_disabled_state
    from sglang_omni_router.worker import build_workers

    workers = build_workers(
        [
            WorkerConfig(url="http://worker-a:8101"),
            WorkerConfig(url="http://worker-b:8102"),
        ]
    )
    previous = {w.worker_id: False for w in workers}
    _restore_admin_disabled_state(
        "/update_weights_from_disk", workers, previous, None, outcome_known=False
    )
    assert all(w.disabled for w in workers)


def test_completed_update_restores_a_previously_disabled_worker_without_journal(
    tmp_path: Path,
) -> None:
    # a worker disabled BEFORE the update must not be left journaled after a
    # clean completion (else every future update wedges on the 409 gate)
    upstream = _Upstream()
    journal_path = str(tmp_path / "j.json")
    app, snapshot_path, _ = _cp_app(
        tmp_path, upstream=upstream, journal_path=journal_path
    )
    with TestClient(app) as client:
        created = client.post("/workers", json={"url": "http://worker-b:8102"})
        wid = created.json()["worker"]["worker_id"]
        client.put(f"/workers/{wid}", json={"disabled": True})  # pre-disabled

        assert (
            client.post("/update_weights_from_disk", json={"path": "/m"}).status_code
            == 200
        )
        # journal cleared, pre-disabled worker still disabled, updates not wedged
        assert not Path(journal_path).exists()
        assert (
            client.post("/update_weights_from_disk", json={"path": "/m"}).status_code
            == 200
        )


def test_recovery_drops_journaled_workers_that_no_longer_exist(
    tmp_path: Path,
) -> None:
    from sglang_omni_router.update_journal import UpdateJournal

    journal_path = str(tmp_path / "j.json")
    UpdateJournal(journal_path).begin("/x", ["gone-worker-id"])
    upstream = _Upstream()
    app, snapshot_path, _ = _cp_app(
        tmp_path, upstream=upstream, journal_path=journal_path
    )
    with TestClient(app) as client:
        # the journaled id is not in the registry: it cannot serve mixed
        # weights, so recovery drops it and updates are not wedged
        assert not Path(journal_path).exists()
        assert (
            client.post("/update_weights_from_disk", json={"path": "/m"}).status_code
            == 200
        )


def test_unreadable_journal_disables_the_whole_pool_on_recovery(
    tmp_path: Path,
) -> None:
    journal_path = str(tmp_path / "j.json")
    Path(journal_path).write_text("corrupt", encoding="utf-8")
    upstream = _Upstream()
    app, snapshot_path, _ = _cp_app(
        tmp_path, upstream=upstream, journal_path=journal_path
    )
    with TestClient(app) as client:
        snapshot = _read_snapshot(snapshot_path)
        assert snapshot.workers[0].disabled is True


def test_re_enabling_a_journaled_worker_requires_admin_auth(tmp_path: Path) -> None:
    from sglang_omni_router.update_journal import UpdateJournal

    journal_path = str(tmp_path / "j.json")
    worker_id = "http%3A%2F%2Fworker-a%3A8101"
    UpdateJournal(journal_path).begin("/x", [worker_id])
    upstream = _Upstream()
    app, snapshot_path, _ = _cp_app(
        tmp_path,
        upstream=upstream,
        journal_path=journal_path,
        admin_api_key="secret-key",
    )
    with TestClient(app) as client:
        # discarding a journal entry via re-enable is admin-sensitive even
        # though ordinary worker CRUD is not
        assert (
            client.put(f"/workers/{worker_id}", json={"disabled": False}).status_code
            == 401
        )
        assert UpdateJournal(journal_path).pending() == [worker_id]
        ok = client.put(
            f"/workers/{worker_id}",
            json={"disabled": False},
            headers={"Authorization": "Bearer secret-key"},
        )
        assert ok.status_code == 200
        assert UpdateJournal(journal_path).pending() == []


def test_health_reports_admission_error_instead_of_500_when_shm_unstable(
    tmp_path: Path,
) -> None:
    from sglang_omni_router.admission_shm import (
        SharedAdmission,
        create_admission_file,
        retired_slot_index,
    )

    shm_path = str(tmp_path / "admission.shm")
    create_admission_file(shm_path, 2)
    import mmap as mmap_module

    with open(shm_path, "r+b") as f:
        buf = mmap_module.mmap(f.fileno(), 0)
        SharedAdmission(buf, slots=2, own_index=0, max_inflight=8, generation=1, pid=11)
        # wedge the retired slot mid-write (a fold that never completes)
        from sglang_omni_router.admission_shm import SlotCodec

        SlotCodec(buf, retired_slot_index(2)).begin_write()

        app, _, _ = _cp_app(
            tmp_path, expected_data_planes=2, admission_shm_path=shm_path
        )
        with TestClient(app) as client:
            _ready_heartbeat(client, 0, 11)
            response = client.get("/health")
            assert response.status_code in (200, 503)  # never 500
            assert "admission_error" in response.json()
        buf.close()
