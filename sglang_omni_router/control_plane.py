# SPDX-License-Identifier: Apache-2.0
"""Control-plane app: the single owner of registry, health, and admin state.

In the multi-process router exactly one CP process runs this app. It owns the
mutable worker registry, the HealthChecker, the admin update lock, and the
privileged admin surface; it publishes the routable-worker snapshot after
every state change (health tick, worker CRUD, disable flips around weight
updates) for the stateless data planes to consume; and it hosts the internal
DP channel, including hot-path worker-failure reports and the weight-update
ACK barrier (all live DPs must acknowledge the disabled-worker snapshot
before a weight update is broadcast; on timeout the update fails closed).
"""

from __future__ import annotations

import asyncio
import logging
import mmap
import os
import time
from contextlib import asynccontextmanager
from typing import cast

import httpx
from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pydantic import Field as PydanticField

from sglang_omni.http.admin_auth import resolve_admin_api_key
from sglang_omni_router.admission_shm import (
    AdmissionAggregateView,
    SeqlockUnstableError,
    admission_file_size,
)
from sglang_omni_router.app import (
    _error_response,
    _find_worker,
    _pool_summary,
    _worker_pool_status_response,
    register_admin_routes,
    register_public_metadata_routes,
)
from sglang_omni_router.config import RouterConfig, WorkerConfig
from sglang_omni_router.health import HealthChecker
from sglang_omni_router.internal_channel import (
    InternalChannelState,
    make_internal_token_dependency,
    register_internal_routes,
)
from sglang_omni_router.observability import (
    CounterReport,
    DataPlaneCounterLedger,
    StaleCounterGenerationError,
)
from sglang_omni_router.snapshot import SnapshotReader, SnapshotWorker, SnapshotWriter
from sglang_omni_router.update_journal import (
    JournalUnreadableError,
    UpdateJournal,
    default_journal_path,
)
from sglang_omni_router.worker import Worker, WorkerState, build_workers

logger = logging.getLogger("sglang_omni_router.control_plane")

DEFAULT_DP_ACK_TIMEOUT_SECS = 10.0
DEFAULT_DP_LIVENESS_SECS = 6.0
# Note: (Jiaxin Deng) the snapshot doubles as the CP liveness signal for the
# DP stale-timeout, so it is republished on a fixed cadence, not only on
# state changes (the health tick alone can be slower than the DP max age).
DEFAULT_SNAPSHOT_KEEPALIVE_SECS = 2.0
_ACK_POLL_INTERVAL_SECS = 0.05


class WorkerFailureReport(BaseModel):
    worker_id: str
    status_code: int | None = None
    error: str | None = None
    # the incarnation pins the report to the worker object it was observed
    # on: a failure from a deleted-and-re-added URL must not evict the new one
    incarnation: str = ""
    # event identity: retries of ONE failure reuse one failure_seq, so
    # re-delivery cannot count a single failure toward eviction twice
    dp_index: int = PydanticField(ge=0)
    generation: int = PydanticField(ge=1)
    failure_seq: int = PydanticField(ge=1)


def snapshot_workers(workers: list[Worker]) -> list[SnapshotWorker]:
    return [
        SnapshotWorker(
            url=worker.url,
            worker_id=worker.worker_id,
            incarnation=worker.incarnation,
            model=worker.model,
            capabilities=sorted(worker.capabilities),
            routable=worker.is_routable,
            state=worker.state,
            disabled=worker.disabled,
        )
        for worker in workers
    ]


def restore_workers(snapshot_path: str, config: RouterConfig) -> list[Worker]:
    if not os.path.exists(snapshot_path):
        return build_workers(config.workers)

    reader = SnapshotReader(snapshot_path)
    if not reader.maybe_reload() or reader.snapshot is None:
        raise RuntimeError(f"cannot recover worker registry from {snapshot_path}")

    workers = []
    for entry in reader.snapshot.workers:
        worker = Worker(
            config=WorkerConfig(
                url=entry.url,
                model=entry.model,
                capabilities=set(entry.capabilities),
            )
        )
        if entry.incarnation:
            worker.incarnation = entry.incarnation
        worker.state = cast(WorkerState, entry.state)
        worker.disabled = entry.disabled
        workers.append(worker)
    return workers


def create_control_plane_app(
    config: RouterConfig,
    *,
    snapshot_path: str,
    cp_epoch: str,
    internal_token: str | None = None,
    client: httpx.AsyncClient | None = None,
    health_client: httpx.AsyncClient | None = None,
    admin_api_key: str | None = None,
    dp_ack_timeout_secs: float = DEFAULT_DP_ACK_TIMEOUT_SECS,
    dp_liveness_secs: float = DEFAULT_DP_LIVENESS_SECS,
    snapshot_keepalive_secs: float = DEFAULT_SNAPSHOT_KEEPALIVE_SECS,
    expected_data_planes: int | None = None,
    admission_shm_path: str | None = None,
    journal_path: str | None = None,
) -> FastAPI:
    if expected_data_planes is not None and expected_data_planes < 1:
        raise ValueError("expected_data_planes must be >= 1 when set")
    workers = restore_workers(snapshot_path, config)
    writer = SnapshotWriter(snapshot_path, cp_epoch)
    internal_state = InternalChannelState()
    ledger = DataPlaneCounterLedger()
    # a STABLE path (default keyed by host:port): the journal must survive a
    # full supervisor restart, not just a CP respawn inside one supervisor
    journal = UpdateJournal(
        journal_path or default_journal_path(config.host, config.port)
    )

    admission_view: AdmissionAggregateView | None = None
    admission_shm_file = None
    if admission_shm_path and expected_data_planes:
        admission_shm_file = open(admission_shm_path, "rb")
        admission_view = AdmissionAggregateView(
            mmap.mmap(
                admission_shm_file.fileno(),
                admission_file_size(expected_data_planes),
                access=mmap.ACCESS_READ,
            ),
            expected_data_planes,
        )

    # The CP only fans out health checks, admin broadcasts, and /v1/models
    # merges; it never relays data traffic, so its pool is sized to the
    # worker count, not to the admission bound.
    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(config.request_timeout_secs),
            limits=httpx.Limits(max_connections=max(16, 2 * len(workers))),
        )
    owns_health_client = health_client is None and owns_client
    if health_client is None:
        health_client = client
    health_checker = HealthChecker(
        workers=workers,
        config=config,
        client=health_client,
        on_tick=lambda: publish(),
    )

    def publish() -> int:
        writer.publish(snapshot_workers(workers))
        return writer.seq

    async def dp_snapshot_ack_barrier() -> tuple[bool, list[int]]:
        target_seq = writer.seq
        loop = asyncio.get_running_loop()
        deadline = loop.time() + dp_ack_timeout_secs
        while True:
            now = time.time()
            live = {
                record.dp_index: record
                for record in internal_state.data_planes.values()
                if now - record.last_seen_at <= dp_liveness_secs
            }
            # an ACK must come from THIS epoch: a bare seq carried over from
            # a previous CP's numbering must never satisfy the barrier
            pending = {
                index
                for index, record in live.items()
                if record.last_applied_epoch != cp_epoch
                or record.last_applied_seq < target_seq
            }
            if expected_data_planes is not None:
                # fail closed on absent DPs: a serving DP the (possibly just
                # restarted) CP has not heard from is not an implicit ACK
                pending.update(
                    index for index in range(expected_data_planes) if index not in live
                )
            if not pending:
                return True, []
            if loop.time() >= deadline:
                return False, sorted(pending)
            await asyncio.sleep(_ACK_POLL_INTERVAL_SECS)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.router_config = config
        app.state.workers = workers
        app.state.http_client = client
        app.state.health_http_client = health_client
        app.state.health_checker = health_checker
        app.state.admin_update_lock = asyncio.Lock()
        app.state.internal_channel = internal_state
        app.state.snapshot_writer = writer
        app.state.counter_ledger = ledger
        app.state.worker_stats_overlay = ledger.overlay
        app.state.on_registry_change = publish
        app.state.dp_snapshot_ack_barrier = dp_snapshot_ack_barrier
        app.state.update_journal = journal
        try:
            unresolved = journal.pending()
        except JournalUnreadableError:
            # a transaction may be in progress but its targets are unreadable:
            # fail closed by disabling the whole pool until an operator clears
            # the journal
            unresolved = None
        if unresolved is None:
            for worker in workers:
                worker.set_disabled(True)
            logger.critical(
                "weight-update journal is present but unreadable; disabling "
                "the entire worker pool until it is inspected and cleared"
            )
        elif unresolved:
            # crash recovery: an update died mid-transaction under a previous
            # CP; keep its targets disabled (fail closed) until an operator
            # verifies weight versions and re-enables them. A journaled worker
            # that no longer exists in the registry cannot serve mixed weights,
            # so drop it (otherwise a deleted target wedges every future
            # update behind the 409 gate).
            live = {worker.worker_id for worker in workers}
            still_present = [wid for wid in unresolved if wid in live]
            for worker_id in still_present:
                _find_worker(workers, worker_id).set_disabled(True)
            if len(still_present) != len(unresolved):
                journal.keep(still_present)
            if still_present:
                logger.critical(
                    f"unresolved weight update in the journal; keeping "
                    f"{len(still_present)} target worker(s) disabled until "
                    "verified and re-enabled"
                )
        publish()
        await health_checker.start()

        async def _keepalive() -> None:
            while True:
                await asyncio.sleep(snapshot_keepalive_secs)
                try:
                    publish()
                except Exception:
                    # a transient write failure must not end the cadence:
                    # the snapshot doubles as the CP liveness signal
                    logger.exception("snapshot keepalive publish failed")

        keepalive_task = asyncio.create_task(_keepalive())
        try:
            yield
        finally:
            keepalive_task.cancel()
            try:
                await keepalive_task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("snapshot keepalive task ended abnormally")
            await health_checker.stop()
            if admission_shm_file is not None:
                admission_shm_file.close()
            if owns_health_client and health_client is not client:
                await health_client.aclose()
            if owns_client:
                await client.aclose()

    app = FastAPI(title="sglang-omni-router-cp", version="0.1.0", lifespan=lifespan)
    resolved_key = resolve_admin_api_key(admin_api_key)

    @app.get("/live")
    async def live() -> JSONResponse:
        return JSONResponse({"status": "alive"})

    @app.get("/ready")
    async def ready() -> JSONResponse:
        return _worker_pool_status_response(
            workers,
            available_status="ready",
            unavailable_status="not_ready",
            overlay=ledger.overlay,
        )

    @app.get("/health")
    async def health() -> JSONResponse:
        now = time.time()
        live_dps = [
            record
            for record in internal_state.data_planes.values()
            if now - record.last_seen_at <= dp_liveness_secs
        ]
        # serving-ready = live AND applied THIS CP's snapshot AND
        # self-reports serving AND (when the slot array is mapped) still owns
        # its admission slot - a crashed generation drops out immediately
        # instead of lingering for the liveness window
        slot_generations: dict[int, int] = {}
        admission_error = None
        admission_slots = None
        if admission_view is not None:
            try:
                admission_slots = admission_view.per_slot(now=now)
                slot_generations = {
                    slot["index"]: slot["generation"] for slot in admission_slots
                }
            except SeqlockUnstableError as exc:
                # the shm is momentarily unstable (a fold in flight): report a
                # read error rather than a 500, and do not gate readiness on
                # an unreadable slot array
                admission_error = str(exc)
        ready_indices = {
            record.dp_index
            for record in live_dps
            if record.last_applied_epoch == cp_epoch
            and record.serving
            and (
                admission_view is None
                or admission_error is not None
                or slot_generations.get(record.dp_index) == record.generation
            )
        }
        routable = sum(1 for worker in workers if worker.is_routable)
        if expected_data_planes is not None:
            # identity, not cardinality: {0, 99} is not a full roster of 2
            expected_set = set(range(expected_data_planes))
            missing = sorted(expected_set - ready_indices)
            unexpected = sorted({record.dp_index for record in live_dps} - expected_set)
            serving = len(expected_set & ready_indices)
        else:
            missing = []
            unexpected = []
            serving = len(ready_indices)
        # degraded, not dead: some DPs missing is a warning while at least
        # one keeps serving; 503 only when nothing can serve at all
        no_service = routable == 0 or (
            expected_data_planes is not None and serving == 0
        )
        degraded = bool(missing) and serving > 0
        if no_service:
            status = "unhealthy"
        elif degraded:
            status = "degraded"
        else:
            status = "healthy"
        payload = _pool_summary(workers, status=status, overlay=ledger.overlay)
        payload["data_planes"] = [
            record.model_dump()
            for _, record in sorted(internal_state.data_planes.items())
        ]
        payload["live_data_planes"] = len(live_dps)
        payload["serving_ready_data_planes"] = serving
        payload["missing_data_planes"] = missing
        payload["unexpected_data_planes"] = unexpected
        payload["expected_data_planes"] = expected_data_planes
        if admission_view is not None:
            if admission_error is not None:
                payload["admission_error"] = admission_error
            else:
                try:
                    payload["admission"] = admission_view.to_dict(
                        config.effective_max_inflight
                    )
                    payload["admission_slots"] = admission_slots
                except SeqlockUnstableError as exc:
                    payload["admission_error"] = str(exc)
        return JSONResponse(payload, status_code=503 if no_service else 200)

    register_admin_routes(app, workers, config, admin_api_key=resolved_key)
    register_public_metadata_routes(app, workers, config)
    register_internal_routes(app, internal_state, token=internal_token)

    internal_auth = Depends(make_internal_token_dependency(internal_token))

    applied_failure_seqs: dict[tuple[int, int], dict] = {}

    def _failure_already_applied(report: WorkerFailureReport) -> bool:
        # exact-id dedup (not high-water: distinct events may arrive out of
        # order), with a pruning floor to bound memory
        state = applied_failure_seqs.setdefault(
            (report.dp_index, report.generation), {"seqs": set(), "floor": 0}
        )
        if report.failure_seq <= state["floor"]:
            return True
        if report.failure_seq in state["seqs"]:
            return True
        state["seqs"].add(report.failure_seq)
        if len(state["seqs"]) > 4096:
            state["floor"] = max(state["seqs"]) - 1024
            state["seqs"] = {seq for seq in state["seqs"] if seq > state["floor"]}
        return False

    @app.post("/internal/worker_failure", dependencies=[internal_auth])
    async def worker_failure(report: WorkerFailureReport) -> JSONResponse:
        worker = _find_worker(workers, report.worker_id)
        if worker is None:
            return _error_response(404, "worker not found")
        if report.incarnation and worker.incarnation != report.incarnation:
            return JSONResponse({"status": "ok", "stale_incarnation": True})
        if _failure_already_applied(report):
            return JSONResponse({"status": "ok", "deduplicated": True})
        worker.record_request_failure(
            failure_threshold=config.health_failure_threshold,
            status_code=report.status_code,
            error=report.error,
        )
        publish()
        return JSONResponse(
            {"status": "ok", "worker_state": worker.state, "disabled": worker.disabled}
        )

    @app.post("/internal/counters", dependencies=[internal_auth])
    async def counters(report: CounterReport) -> JSONResponse:
        try:
            applied = ledger.apply(report)
        except StaleCounterGenerationError as exc:
            return _error_response(409, str(exc))
        return JSONResponse({"status": "ok", "applied": applied})

    return app
