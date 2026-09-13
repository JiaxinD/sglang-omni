"""Correlate measured requests with the existing request-event profiler."""

import json
import re
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

import aiohttp

from benchmarks.benchmarker.data import RequestResult
from sglang_omni.profiler.views import (
    RequestTimeline,
    compute_stage_intervals,
    iter_events,
)


@asynccontextmanager
async def request_profile(url: str, *, event_dir: Path, run_id: str):
    """Enable JSONL events on an owned service; stop only this profiling run."""
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=30)
    ) as session:
        async with session.post(
            f"{url}/start_request_profile",
            json={"run_id": run_id, "event_dir": str(event_dir.resolve())},
        ) as response:
            response.raise_for_status()
        try:
            yield
        finally:
            async with session.post(
                f"{url}/stop_request_profile", json={"run_id": run_id}
            ) as response:
                response.raise_for_status()


def write_profile_report(
    results: list[RequestResult], *, source: Path, run_id: str, output: Path
) -> None:
    """Keep process-local intervals separate; do not interpret them as service time."""
    by_server = {
        r.server_request_id: r.request_id for r in results if r.server_request_id
    }
    grouped = defaultdict(list)
    for event in iter_events(source):
        rid = event.get("request_id")
        if event.get("run_id") != run_id or not rid:
            continue
        parent = rid
        if parent not in by_server:
            child = re.fullmatch(r"(transcription-.+?)-chunk-\d+(?:-retry)?", rid)
            if child is None:
                continue
            parent = child.group(1)
        if parent in by_server:
            grouped[(parent, rid, event.get("pid"))].append(event)
    requests = {
        r.request_id: {
            "server_request_id": r.server_request_id,
            "event_count": 0,
            "intervals": [],
        }
        for r in results
    }
    for (parent, rid, pid), events in grouped.items():
        events.sort(key=lambda event: event["timestamp_ns"])
        row = requests[by_server[parent]]
        row["event_count"] += len(events)
        intervals = compute_stage_intervals({rid: RequestTimeline(rid, events)})
        row["intervals"].extend(
            {**asdict(interval), "pid": pid} for interval in intervals
        )
    output.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "requests": requests,
                "calibration_ready": False,
                "scope": "Observed event pairs only; no proof of full stage coverage, isolation or GPU service time.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
