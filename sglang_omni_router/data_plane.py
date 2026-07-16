# SPDX-License-Identifier: Apache-2.0
"""Data-plane app: a stateless relay fed by the control plane's snapshot.

The DP never mutates the registry. It polls the worker snapshot into a local
view (persistent Worker objects, so per-process counters survive refreshes),
relays the four model routes through the standard ProxyHandler, sheds new
requests when the snapshot outlives the stale timeout (the CP is presumed
gone), reports eviction-relevant upstream failures to the CP, and heartbeats
its identity plus last_applied_seq so the CP watchdog and the weight-update
ACK barrier can see it. A 409 heartbeat means this process was fenced out by
a newer generation and must stop serving.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import os
import signal
from contextlib import asynccontextmanager
from typing import Callable

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from sglang_omni_router.app import (
    _merge_models,
    _worker_pool_status_response,
    register_data_routes,
)
from sglang_omni_router.config import RouterConfig, WorkerConfig
from sglang_omni_router.internal_channel import INTERNAL_TOKEN_HEADER
from sglang_omni_router.proxy import HOP_BY_HOP_HEADERS, ProxyHandler
from sglang_omni_router.selector import WorkerSelector
from sglang_omni_router.snapshot import SnapshotReader, WorkerSnapshot
from sglang_omni_router.worker import Worker

logger = logging.getLogger("sglang_omni_router.data_plane")

DEFAULT_DP_REFRESH_INTERVAL_SECS = 0.2
DEFAULT_SNAPSHOT_MAX_AGE_SECS = 10.0
DEFAULT_HEARTBEAT_INTERVAL_SECS = 2.0
DEFAULT_COUNTER_FLUSH_INTERVAL_SECS = 1.0
# Failure events are bounded-retry best-effort delivery, NOT at-least-once:
# after the attempts below the event is dropped and the CP health probe is
# the backstop.
_FAILURE_REPORT_MAX_ATTEMPTS = 3
_FAILURE_REPORT_BACKOFF_SECS = 0.2
# in-flight report tasks are bounded so a slow CP cannot accumulate
# unbounded retry tasks (and their sockets) on the DP
_FAILURE_REPORT_MAX_PENDING = 64
# Consecutive definitive (non-428) registration rejections before the DP
# fails closed; a DP the CP refuses to register cannot be covered by the
# weight-update ACK barrier.
_REGISTER_REJECT_LIMIT = 10


class DataPlaneWorkerView:
    """Snapshot-fed worker view with persistent Worker objects.

    Applying a snapshot updates health/disabled/config in place and only
    creates or drops workers on membership changes, so per-process state
    (active_requests, routed counters) survives refreshes.
    """

    def __init__(self) -> None:
        self._workers: dict[str, Worker] = {}
        self.last_applied_seq = 0
        self.last_applied_epoch = ""

    def workers(self) -> list[Worker]:
        return list(self._workers.values())

    def apply(self, snapshot: WorkerSnapshot) -> None:
        seen: set[str] = set()
        for entry in snapshot.workers:
            seen.add(entry.url)
            config_kwargs: dict = {"url": entry.url, "model": entry.model}
            if entry.capabilities:
                config_kwargs["capabilities"] = set(entry.capabilities)
            config = WorkerConfig(**config_kwargs)
            worker = self._workers.get(entry.url)
            incarnation_changed = (
                worker is not None
                and entry.incarnation
                and worker.incarnation != entry.incarnation
            )
            if worker is None or incarnation_changed:
                # a new URL, or the same URL deleted and re-added (new
                # incarnation): a FRESH Worker object, so an in-flight request
                # holding the old object keeps the old incarnation and a late
                # failure cannot be misattributed to the new worker (ABA)
                worker = Worker(config=config)
                if entry.incarnation:
                    worker.incarnation = entry.incarnation
                self._workers[entry.url] = worker
            else:
                if (
                    worker.model != config.model
                    or worker.capabilities != config.capabilities
                ):
                    worker.replace_config(config)
                if entry.incarnation:
                    worker.incarnation = entry.incarnation
            worker.state = entry.state
            worker.disabled = entry.disabled
        for url in list(self._workers):
            if url not in seen:
                del self._workers[url]
        self.last_applied_seq = snapshot.seq
        self.last_applied_epoch = getattr(snapshot, "cp_epoch", "")


def _default_fence_reaction() -> None:
    # Fenced out by a newer generation: stop serving. SIGTERM lets uvicorn
    # finish in-flight responses instead of dropping them.
    os.kill(os.getpid(), signal.SIGTERM)


def dp_client_limits(config: RouterConfig, total_data_planes: int) -> httpx.Limits:
    """Per-DP upstream pool: full-size connections (a skewed client mix can
    pin the whole global bound onto one DP, which must still relay it), but
    keep-alives split across DPs so the cluster's idle-connection total stays
    around one pool's worth."""
    keepalive = -(-config.upstream_pool_size // max(1, total_data_planes))
    return httpx.Limits(
        max_connections=config.upstream_pool_size,
        max_keepalive_connections=keepalive,
    )


def create_data_plane_app(
    config: RouterConfig,
    *,
    snapshot_path: str,
    dp_index: int,
    generation: int,
    client: httpx.AsyncClient | None = None,
    internal_client: httpx.AsyncClient | None = None,
    internal_token: str | None = None,
    metadata_client: httpx.AsyncClient | None = None,
    admission=None,
    total_data_planes: int = 1,
    dp_refresh_interval_secs: float = DEFAULT_DP_REFRESH_INTERVAL_SECS,
    snapshot_max_age_secs: float = DEFAULT_SNAPSHOT_MAX_AGE_SECS,
    heartbeat_interval_secs: float = DEFAULT_HEARTBEAT_INTERVAL_SECS,
    counter_flush_interval_secs: float = DEFAULT_COUNTER_FLUSH_INTERVAL_SECS,
    on_fenced: Callable[[], None] = _default_fence_reaction,
) -> FastAPI:
    view = DataPlaneWorkerView()
    reader = SnapshotReader(snapshot_path)

    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(config.request_timeout_secs),
            limits=dp_client_limits(config, total_data_planes),
        )
    # /v1/models fans out on its own small pool: metadata traffic is not
    # admission-controlled and must not occupy the sized relay pool (an
    # admitted model request must never queue behind it)
    owns_metadata_client = metadata_client is None
    if metadata_client is None:
        metadata_client = httpx.AsyncClient(
            timeout=httpx.Timeout(config.health_check_timeout_secs),
            limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
        )

    failure_tasks: set[asyncio.Task] = set()
    failure_seq_counter = itertools.count(1)

    def _report_failure(
        worker: Worker, status_code: int | None, error: str | None
    ) -> None:
        # every distinct failure is reported with its own failure_seq (the
        # CP's eviction threshold counts events, and dedups re-delivery by
        # this id); retries of one event reuse the same id. The storm is
        # self-limiting: at the threshold the CP unpublishes the worker and
        # this DP stops routing to it within one refresh.
        if internal_client is None:
            return
        if len(failure_tasks) >= _FAILURE_REPORT_MAX_PENDING:
            # bounded, best-effort by contract; health probes are the backstop
            logger.debug("worker_failure report dropped: too many pending")
            return
        loop = asyncio.get_running_loop()
        payload = {
            "worker_id": worker.worker_id,
            "incarnation": worker.incarnation,
            "status_code": status_code,
            "error": error,
            "dp_index": dp_index,
            "generation": generation,
            "failure_seq": next(failure_seq_counter),
        }

        async def _send() -> None:
            last_error: httpx.HTTPError | None = None
            for attempt in range(_FAILURE_REPORT_MAX_ATTEMPTS):
                try:
                    await internal_client.post(
                        "/internal/worker_failure",
                        json=payload,
                        headers=_internal_headers(internal_token),
                    )
                    return
                except httpx.HTTPError as exc:
                    last_error = exc
                    await asyncio.sleep(_FAILURE_REPORT_BACKOFF_SECS * (2**attempt))
            logger.debug(
                f"worker_failure report dropped after "
                f"{_FAILURE_REPORT_MAX_ATTEMPTS} attempts: {last_error}"
            )

        task = loop.create_task(_send())
        failure_tasks.add(task)
        task.add_done_callback(failure_tasks.discard)

    proxy = ProxyHandler(
        config=config,
        workers=[],
        selector=WorkerSelector(config.policy, rr_offset=dp_index),
        client=client,
        worker_provider=view.workers,
        on_worker_failure=_report_failure,
        admission=admission,
    )

    def _stale_gate() -> JSONResponse | None:
        age = reader.age_secs()
        if age is not None and age <= snapshot_max_age_secs:
            return None
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "message": (
                        "worker snapshot missing or stale; the control plane "
                        "is unreachable and new requests are shed"
                    ),
                    "type": "unavailable_error",
                    "code": 503,
                }
            },
        )

    async def _refresh_loop() -> None:
        while True:
            if reader.maybe_reload():
                view.apply(reader.snapshot)
            await asyncio.sleep(dp_refresh_interval_secs)

    async def _heartbeat_loop() -> None:
        if internal_client is None:
            return
        identity = {
            "dp_index": dp_index,
            "generation": generation,
            "pid": os.getpid(),
        }
        registered = False
        rejected_registrations = 0
        while True:
            try:
                if not registered:
                    response = await internal_client.post(
                        "/internal/register",
                        json=identity,
                        headers=_internal_headers(internal_token),
                    )
                else:
                    response = await internal_client.post(
                        "/internal/heartbeat",
                        json={
                            **identity,
                            "last_applied_seq": view.last_applied_seq,
                            "last_applied_epoch": view.last_applied_epoch,
                            # self-reported serving state: shedding as
                            # snapshot_stale must show up in CP /health
                            "serving": _stale_gate() is None,
                        },
                        headers=_internal_headers(internal_token),
                    )
                if response.status_code == 409:
                    logger.critical("fenced out by a newer generation; stopping")
                    on_fenced()
                    return
                if response.status_code == 428:
                    # the CP restarted and lost registrations: re-register,
                    # this is NOT a fence
                    registered = False
                elif response.status_code == 200:
                    registered = True
                    rejected_registrations = 0
                else:
                    # definitive rejection (e.g. token/config mismatch): an
                    # unregistered DP is invisible to the weight-update ACK
                    # barrier, so it must not keep serving indefinitely
                    registered = False
                    rejected_registrations += 1
                    logger.warning(
                        f"internal registration rejected with "
                        f"{response.status_code} "
                        f"({rejected_registrations}/{_REGISTER_REJECT_LIMIT})"
                    )
                    if rejected_registrations >= _REGISTER_REJECT_LIMIT:
                        logger.critical(
                            "registration persistently rejected; failing closed"
                        )
                        on_fenced()
                        return
            except httpx.HTTPError:
                # CP unreachable: the stale-snapshot gate governs serving;
                # keep retrying and re-register once the CP is back.
                registered = False
            await asyncio.sleep(heartbeat_interval_secs)

    async def _counter_flush_loop() -> None:
        if internal_client is None:
            return
        counter_seq = 0
        while True:
            await asyncio.sleep(counter_flush_interval_secs)
            counter_seq += 1
            payload = {
                "dp_index": dp_index,
                "generation": generation,
                "counter_seq": counter_seq,
                "workers": [
                    {
                        "worker_id": worker.worker_id,
                        "routed_total": worker.routed_requests,
                        "successful_total": worker.successful_requests,
                        "failed_total": worker.failed_requests,
                        "current_active": worker.active_requests,
                    }
                    for worker in view.workers()
                ],
            }
            if admission is not None and hasattr(admission, "touch"):
                # keep the shm slot heartbeat fresh even when idle
                admission.touch()
            try:
                response = await internal_client.post(
                    "/internal/counters",
                    json=payload,
                    headers=_internal_headers(internal_token),
                )
                if response.status_code == 409:
                    logger.critical("counter report fenced by a newer generation")
                    on_fenced()
                    return
            except httpx.HTTPError:
                # cumulative totals: the next flush carries everything, so a
                # dropped report loses nothing
                pass

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.router_config = config
        app.state.worker_view = view
        app.state.snapshot_reader = reader
        app.state.http_client = client
        app.state.proxy = proxy
        app.state.admission_controller = proxy.admission
        tasks = [asyncio.create_task(_refresh_loop())]
        if internal_client is not None:
            tasks.append(asyncio.create_task(_heartbeat_loop()))
            tasks.append(asyncio.create_task(_counter_flush_loop()))
        try:
            yield
        finally:
            for task in tasks:
                task.cancel()
            for task in tasks:
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            if owns_client:
                await client.aclose()
            if owns_metadata_client:
                await metadata_client.aclose()
            if internal_client is not None:
                await internal_client.aclose()

    app = FastAPI(title="sglang-omni-router-dp", version="0.1.0", lifespan=lifespan)
    # same external CORS policy as the single-process app: the DP is the
    # public surface, and browser preflights must keep working
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/live")
    async def live() -> JSONResponse:
        return JSONResponse({"status": "alive"})

    @app.get("/ready")
    async def ready() -> JSONResponse:
        stale = _stale_gate()
        if stale is not None:
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "reason": "snapshot_stale"},
            )
        return _worker_pool_status_response(
            view.workers(),
            available_status="ready",
            unavailable_status="not_ready",
        )

    @app.get("/v1/models")
    async def models(request: Request) -> JSONResponse:
        # public metadata stays DP-local: fan out over the snapshot view
        return await _merge_models(
            view.workers(),
            metadata_client,
            request,
            timeout_secs=config.health_check_timeout_secs,
        )

    register_data_routes(app, proxy, gate=_stale_gate)
    if internal_client is not None:
        _register_cp_forwarding(app, internal_client, config)
    return app


# The privileged admin surface plus the aggregate /health live on the CP;
# a DP relays them verbatim over the internal channel. Auth is NOT checked
# here: the Authorization header is forwarded and the CP re-checks it.
_FORWARDED_CP_ROUTES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("/health", ("GET",)),
    ("/workers", ("GET", "POST")),
    ("/workers/{worker_path:path}", ("GET", "PUT", "DELETE")),
    ("/model_info", ("GET", "POST")),
    ("/pause_generation", ("POST",)),
    ("/continue_generation", ("POST",)),
    ("/update_weights_from_disk", ("POST",)),
    ("/update_weights_from_tensor", ("POST",)),
    ("/init_weights_update_group", ("POST",)),
    ("/destroy_weights_update_group", ("POST",)),
    ("/update_weights_from_distributed", ("POST",)),
    ("/weights_checker", ("GET", "POST")),
)

# reuse the canonical hop-by-hop set: the body is re-framed as fixed-length
# bytes (httpx sets Content-Length), so a client transfer-encoding/te must not
# ride along or the CP sees a Content-Length and a Transfer-Encoding at once
_FORWARD_REQUEST_STRIP = HOP_BY_HOP_HEADERS | {
    "host",
    "content-length",
    "accept-encoding",
}
_FORWARD_RESPONSE_STRIP = {
    "content-length",
    "date",
    "server",
    "connection",
    "transfer-encoding",
    "content-encoding",
}


def _register_cp_forwarding(
    app: FastAPI, internal_client: httpx.AsyncClient, config: RouterConfig
) -> None:
    def _payload_too_large() -> JSONResponse:
        return JSONResponse(
            status_code=413,
            content={
                "error": {
                    "message": (
                        "request body exceeds max_payload_size "
                        f"({config.max_payload_size} bytes)"
                    ),
                    "type": "payload_too_large",
                    "code": 413,
                }
            },
        )

    async def _forward(request: Request) -> Response:
        # bound the body BEFORE buffering: auth happens on the CP, so this
        # pre-auth path must not let an unauthenticated client exhaust DP
        # memory with a huge or chunked body
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > config.max_payload_size:
            return _payload_too_large()
        received = bytearray()
        async for chunk in request.stream():
            received.extend(chunk)
            if len(received) > config.max_payload_size:
                return _payload_too_large()
        body = bytes(received)
        # raw_path keeps the client's percent-encoding intact (worker ids
        # contain encoded slashes that a decode/re-encode round trip loses)
        raw_path = request.scope.get("raw_path")
        url = raw_path.decode("ascii") if raw_path else request.url.path
        query = request.scope.get("query_string", b"")
        if query:
            url = f"{url}?{query.decode('ascii')}"
        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in _FORWARD_REQUEST_STRIP
        }
        try:
            upstream = await internal_client.request(
                request.method, url, content=body, headers=headers
            )
        except httpx.HTTPError as exc:
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "message": f"control plane unreachable: {exc}",
                        "type": "control_plane_error",
                        "code": 502,
                    }
                },
            )
        response_headers = {
            key: value
            for key, value in upstream.headers.items()
            if key.lower() not in _FORWARD_RESPONSE_STRIP
        }
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            headers=response_headers,
        )

    for path, methods in _FORWARDED_CP_ROUTES:
        app.api_route(path, methods=list(methods))(_forward)


def _internal_headers(token: str | None) -> dict[str, str]:
    if token:
        return {INTERNAL_TOKEN_HEADER: token}
    return {}


def create_dp_app_from_env() -> FastAPI:
    """uvicorn factory entry point for a supervisor-spawned data plane."""
    import mmap as mmap_module

    from sglang_omni_router.admission_shm import SharedAdmission, admission_file_size
    from sglang_omni_router.app_factory import load_config_from_env
    from sglang_omni_router.internal_channel import INTERNAL_TOKEN_ENV
    from sglang_omni_router.supervisor import (
        ADMISSION_SHM_ENV,
        DP_GENERATION_ENV,
        DP_INDEX_ENV,
        EXPECTED_DPS_ENV,
        INTERNAL_TCP_URL_ENV,
        INTERNAL_UDS_ENV,
        SNAPSHOT_PATH_ENV,
    )

    config = load_config_from_env()
    token = os.environ.get(INTERNAL_TOKEN_ENV)
    timeout = httpx.Timeout(config.request_timeout_secs)
    uds = os.environ.get(INTERNAL_UDS_ENV)
    internal_limits = httpx.Limits(max_connections=8, max_keepalive_connections=4)
    if uds:
        internal_client = httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(uds=uds, limits=internal_limits),
            base_url="http://internal",
            timeout=timeout,
        )
    else:
        tcp_url = os.environ.get(INTERNAL_TCP_URL_ENV)
        if not tcp_url:
            raise RuntimeError(
                "neither the internal UDS nor the internal TCP URL is set; "
                "this factory is only meant to be spawned by the supervisor"
            )
        internal_client = httpx.AsyncClient(
            base_url=tcp_url, timeout=timeout, limits=internal_limits
        )

    dp_index = int(os.environ[DP_INDEX_ENV])
    generation = int(os.environ[DP_GENERATION_ENV])
    total = int(os.environ.get(EXPECTED_DPS_ENV, "1"))

    admission = None
    shm_path = os.environ.get(ADMISSION_SHM_ENV)
    admission_file = None
    if shm_path:
        admission_file = open(shm_path, "r+b")
        admission_mmap = mmap_module.mmap(
            admission_file.fileno(), admission_file_size(total)
        )
        admission = SharedAdmission(
            admission_mmap,
            slots=total,
            own_index=dp_index,
            max_inflight=config.effective_max_inflight,
            generation=generation,
            pid=os.getpid(),
            on_fenced=_default_fence_reaction,
        )

    app = create_data_plane_app(
        config,
        snapshot_path=os.environ[SNAPSHOT_PATH_ENV],
        dp_index=dp_index,
        generation=generation,
        internal_client=internal_client,
        internal_token=token,
        admission=admission,
        total_data_planes=total,
    )
    # keep the shm file object alive for the app's lifetime
    app.state.admission_shm_file = admission_file
    return app
