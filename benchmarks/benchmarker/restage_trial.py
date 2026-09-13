"""Run one Restage candidate using the shared serving benchmark machinery."""

import json
import math
import uuid
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AsyncExitStack
from dataclasses import asdict
from pathlib import Path
from typing import Any

from benchmarks.benchmarker.data import RequestResult
from benchmarks.benchmarker.restage import to_observation
from benchmarks.benchmarker.restage_profile import request_profile, write_profile_report
from benchmarks.benchmarker.runner import BenchmarkRunner, RunConfig, SendFn
from benchmarks.benchmarker.utils import managed_omni_server
from sglang_omni.restage.evaluation import SLO, Evaluation, evaluate


async def execute_trial(
    *,
    config_path: Path,
    model_path: str,
    samples: list[Any],
    send_factory: Callable[[str, Path], SendFn],
    quality: Callable[[list[RequestResult]], Awaitable[Mapping[str, bool | None]]],
    slo: SLO,
    rate: float,
    destination: Path,
    port: int,
    warmup: int = 1,
    startup_timeout_s: int = 1800,
    request_timeout_s: int = 300,
    arrival_seed: int | None = None,
    profile: bool = False,
) -> Evaluation:
    """Launch, measure, stop, then evaluate quality and the joint SLO.

    The caller provides an admitted GPU allocation, model-specific sender,
    explicit warmup count and quality evaluation. The run is open-loop.
    Quality runs after serving stops so its resource use cannot contaminate
    the timed cohort. This function owns only the launched service group.
    """
    if not samples or not math.isfinite(rate) or rate <= 0:
        raise ValueError("A trial needs samples and a finite positive arrival rate")
    destination.mkdir(parents=True, exist_ok=False)
    audio_dir = destination / "audio"
    audio_dir.mkdir()
    metadata = {
        "status": "running",
        "config_path": str(config_path.resolve()),
        "model_path": model_path,
        "rate": rate,
        "expected_requests": len(samples),
        "slo": asdict(slo),
        "warmup": warmup,
        "arrival_seed": arrival_seed,
        "profile_run_id": str(uuid.uuid4()) if profile else None,
    }
    output = destination / "result.json"

    def save():
        output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    save()
    runner = BenchmarkRunner(
        RunConfig(
            max_concurrency=0,
            request_rate=rate,
            warmup=warmup,
            disable_tqdm=True,
            timeout_s=request_timeout_s,
            arrival_seed=arrival_seed,
        )
    )
    try:
        send = send_factory(f"http://127.0.0.1:{port}", audio_dir)
        with (destination / "requests.jsonl").open("w", encoding="utf-8") as handle:

            def record(result):
                handle.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
                handle.flush()

            runner.on_result = record
            with managed_omni_server(
                model_path=model_path,
                server_config=str(config_path.resolve()),
                port=port,
                host="127.0.0.1",
                log_file=destination / "server.log",
                timeout=startup_timeout_s,
                wait_for_gpu_release=False,
            ):
                async with AsyncExitStack() as stack:
                    if profile:
                        await stack.enter_async_context(
                            request_profile(
                                f"http://127.0.0.1:{port}",
                                event_dir=destination / "request-events",
                                run_id=metadata["profile_run_id"],
                            )
                        )
                    results = await runner.run(samples, send)
        if profile:
            write_profile_report(
                results,
                source=destination / "request-events",
                run_id=metadata["profile_run_id"],
                output=destination / "profile-report.json",
            )
        quality_results = await quality(results)
        (destination / "quality.json").write_text(
            json.dumps(dict(quality_results), indent=2), encoding="utf-8"
        )
        observations = [
            to_observation(result, quality_pass=quality_results.get(result.request_id))
            for result in results
        ]
        evaluation = evaluate(
            observations,
            slo,
            expected_requests=len(samples),
            elapsed_s=runner.wall_clock_s,
        )
        metadata.update(
            status="complete",
            elapsed_s=runner.wall_clock_s,
            evaluation=asdict(evaluation),
        )
        save()
        return evaluation
    except BaseException as exc:
        metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save()
        raise
