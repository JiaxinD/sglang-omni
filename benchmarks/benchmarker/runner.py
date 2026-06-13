# SPDX-License-Identifier: Apache-2.0
"""BenchmarkRunner: warmup + concurrent dispatch with semaphore and rate limiting."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Coroutine

import aiohttp
import numpy as np
from tqdm.asyncio import tqdm

from benchmarks.benchmarker.data import RequestResult

logger = logging.getLogger(__name__)

SendFn = Callable[[aiohttp.ClientSession, Any], Coroutine[Any, Any, RequestResult]]


def planned_arrival_offsets(
    load_mode: str,
    request_count: int,
    request_rate: float,
    arrival_seed: int,
) -> list[float]:
    """Seconds-from-start arrival time for each request under an open-loop schedule."""
    if request_count <= 0:
        return []
    if load_mode == "openloop_fixed":
        return [index / request_rate for index in range(request_count)]
    if load_mode == "openloop_poisson":
        rng = random.Random(f"{arrival_seed}:arrival")
        offsets: list[float] = []
        elapsed = 0.0
        for _ in range(request_count):
            offsets.append(elapsed)
            elapsed += rng.expovariate(request_rate)
        return offsets
    raise ValueError(f"unsupported load_mode for arrival offsets: {load_mode!r}")


@dataclass
class RunConfig:
    max_concurrency: int = 1
    request_rate: float = float("inf")
    warmup: int = 1
    disable_tqdm: bool = False
    timeout_s: int = 300
    # Open-loop fields. Default keeps the historical closed-loop behavior.
    load_mode: str = "closed_loop"
    arrival_seed: int = 0
    max_inflight_guard: int | None = None


class BenchmarkRunner:
    """Support concurrent requests sending in a single benchmark run.

    Note (chenyang):
    max_concurrency is default to 1, thus all the requests are runs sequentially.

    TODO (chenyang):
    Current concurrency implementation of models are not fully supported.
    https://github.com/sgl-project/sglang-omni/issues/229
    https://github.com/sgl-project/sglang-omni/issues/228
    """

    def __init__(self, config: RunConfig) -> None:
        self.config = config
        self.wall_clock_s: float = 0.0
        self.dispatch_meta: dict[str, Any] = {}

    async def run(self, samples: list, send_fn: SendFn) -> list[RequestResult]:
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_s)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            if self.config.warmup > 0:
                await self._warmup(session, samples, send_fn)

            logger.info(
                "Benchmarking %d requests (max_concurrency=%s)...",
                len(samples),
                self.config.max_concurrency,
            )
            t0 = time.perf_counter()
            if self.config.load_mode == "closed_loop":
                results = await self._dispatch(session, samples, send_fn)
            else:
                results = await self._dispatch_openloop(session, samples, send_fn)
            self.wall_clock_s = time.perf_counter() - t0
        return results

    async def _warmup(
        self,
        session: aiohttp.ClientSession,
        samples: list,
        send_fn: SendFn,
    ) -> None:
        count = min(self.config.warmup, len(samples))
        logger.info("Warmup (%d requests)...", count)
        for i in range(count):
            result = await send_fn(session, samples[i])
            status = "ok" if result.is_success else result.error
            logger.info("  warmup %d/%d: %s", i + 1, count, status)

    async def _dispatch(
        self,
        session: aiohttp.ClientSession,
        samples: list,
        send_fn: SendFn,
    ) -> list[RequestResult]:
        semaphore = (
            asyncio.Semaphore(self.config.max_concurrency)
            if self.config.max_concurrency
            else None
        )
        pbar = tqdm(total=len(samples), disable=self.config.disable_tqdm)

        async def _limited(sample: Any) -> RequestResult:
            if semaphore:
                async with semaphore:
                    result = await send_fn(session, sample)
            else:
                result = await send_fn(session, sample)
            pbar.update(1)
            return result

        try:
            tasks: list[asyncio.Task] = []
            for sample in samples:
                if self.config.request_rate != float("inf"):
                    interval = np.random.exponential(1.0 / self.config.request_rate)
                    await asyncio.sleep(interval)
                tasks.append(asyncio.create_task(_limited(sample)))

            results: list[RequestResult] = list(await asyncio.gather(*tasks))
        finally:
            pbar.close()
        return results

    async def _dispatch_openloop(
        self,
        session: aiohttp.ClientSession,
        samples: list,
        send_fn: SendFn,
    ) -> list[RequestResult]:
        """Launch requests on a planned arrival schedule, NOT gated by completions.

        No semaphore: an arrival is launched at its planned time even while earlier
        requests are still in flight. This is what makes the load open-loop.
        """
        offsets = planned_arrival_offsets(
            self.config.load_mode,
            len(samples),
            self.config.request_rate,
            self.config.arrival_seed,
        )
        pbar = tqdm(total=len(samples), disable=self.config.disable_tqdm)
        guard = self.config.max_inflight_guard
        active = 0
        peak_inflight = 0
        guard_tripped = False

        started = 0.0

        async def _launch(
            sample: Any, planned_offset: float, inflight_at_launch: int
        ) -> RequestResult:
            nonlocal active
            actual_start = time.perf_counter() - started
            try:
                result = await send_fn(session, sample)
            finally:
                active -= 1
                pbar.update(1)
            # All schedule times are relative to dispatch start (one clock).
            result.planned_start_s = planned_offset
            result.actual_start_s = actual_start
            result.start_delay_s = max(0.0, actual_start - planned_offset)
            result.inflight_at_launch = inflight_at_launch
            return result

        try:
            started = time.perf_counter()
            tasks: list[asyncio.Task] = []
            for sample, offset in zip(samples, offsets, strict=True):
                delay = (started + offset) - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
                # Hard backstop: at the ceiling, abort and mark the run invalid.
                # Never wait for a slot to free up; that would make the load closed.
                if guard is not None and active >= guard:
                    guard_tripped = True
                    break
                active += 1
                peak_inflight = max(peak_inflight, active)
                tasks.append(asyncio.create_task(_launch(sample, offset, active)))
            results: list[RequestResult] = list(await asyncio.gather(*tasks))
        finally:
            pbar.close()

        self.dispatch_meta = {
            "load_mode": self.config.load_mode,
            "arrival_seed": self.config.arrival_seed,
            "request_rate": self.config.request_rate,
            "planned_count": len(samples),
            "launched_count": len(tasks),
            "peak_inflight": peak_inflight,
            "guard_tripped": guard_tripped,
        }
        return results
