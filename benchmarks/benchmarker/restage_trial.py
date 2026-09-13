"""Run one Restage candidate using the shared serving benchmark machinery."""

import hashlib
import json
import math
import uuid
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AsyncExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from benchmarks.benchmarker.data import RequestResult
from benchmarks.benchmarker.restage import to_observation
from benchmarks.benchmarker.restage_profile import request_profile, write_profile_report
from benchmarks.benchmarker.runner import BenchmarkRunner, RunConfig, SendFn
from benchmarks.benchmarker.utils import managed_omni_server
from sglang_omni.restage.evaluation import SLO, Evaluation, evaluate


@dataclass
class TrialMeasurement:
    """Completed serving measurements awaiting a separate quality decision."""

    destination: Path
    results: list[RequestResult]
    slo: SLO
    metadata: dict[str, Any]


def _save_result(destination, metadata):
    (destination / "result.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_measurement(destination, metadata, results):
    receipt = {
        "metadata": metadata,
        "request_ids": [result.request_id for result in results],
        "requests_sha256": _sha256(destination / "requests.jsonl"),
        "audio_sha256": {
            str(Path(result.wav_path).resolve()): (
                _sha256(result.wav_path) if Path(result.wav_path).is_file() else None
            )
            for result in results
            if result.wav_path
        },
    }
    temporary = destination / "measurement.json.tmp"
    temporary.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    temporary.replace(destination / "measurement.json")


def restore_measurement(source: Path, *, destination: Path) -> TrialMeasurement:
    """Copy a finalized generation receipt into a new quality attempt.

    The immutable receipt exists only after generation and service shutdown.
    Mutable quality status is not used to infer measurement completeness.
    Original requests, audio and quality attempts remain in place.
    """
    receipt = json.loads((source / "measurement.json").read_text(encoding="utf-8"))
    raw = (source / "requests.jsonl").read_bytes()
    if hashlib.sha256(raw).hexdigest() != receipt["requests_sha256"]:
        raise ValueError(
            f"Saved requests changed since measurement: {source / 'requests.jsonl'}"
        )
    for path, expected in receipt["audio_sha256"].items():
        actual = _sha256(path) if Path(path).is_file() else None
        if actual != expected:
            raise ValueError(f"Saved audio changed since measurement: {path}")
    metadata = receipt["metadata"]
    metadata.setdefault("measurement_source", str(source))
    rows = [
        json.loads(line) for line in raw.decode("utf-8").split("\n") if line.strip()
    ]
    by_id = {row["request_id"]: RequestResult(**row) for row in rows}
    results = [by_id[request_id] for request_id in receipt["request_ids"]]
    measurement = TrialMeasurement(
        destination, results, SLO(**metadata["slo"]), metadata
    )
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "requests.jsonl").write_bytes(raw)
    (destination / "measurement.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    if (source / "workload.json").exists():
        (destination / "workload.json").write_bytes(
            (source / "workload.json").read_bytes()
        )
    _save_result(destination, metadata)
    return measurement


async def measure_trial(
    *,
    config_path: Path,
    model_path: str,
    samples: list[Any],
    send_factory: Callable[[str, Path], SendFn],
    slo: SLO,
    rate: float,
    destination: Path,
    port: int,
    warmup: int = 1,
    startup_timeout_s: int = 1800,
    request_timeout_s: int = 300,
    arrival_seed: int | None = None,
    profile: bool = False,
) -> TrialMeasurement:
    """Launch, measure and stop the candidate without assigning quality.

    The caller provides an admitted GPU allocation, model-specific sender,
    explicit warmup count. The run is open-loop. The returned measurement
    is not a completed evaluation and cannot be ranked as a passing trial.
    This function owns only the launched service group.
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
    _save_result(destination, metadata)
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
                    metadata.update(
                        measurement_complete=True, elapsed_s=runner.wall_clock_s
                    )
                    _save_result(destination, metadata)
        if profile:
            write_profile_report(
                results,
                source=destination / "request-events",
                run_id=metadata["profile_run_id"],
                output=destination / "profile-report.json",
            )
        metadata["status"] = "awaiting_quality"
        _save_measurement(destination, metadata, results)
        _save_result(destination, metadata)
        return TrialMeasurement(destination, results, slo, metadata)
    except BaseException as exc:
        metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        _save_result(destination, metadata)
        raise


async def evaluate_trial(
    measurement: TrialMeasurement,
    *,
    quality: Callable[[list[RequestResult]], Awaitable[Mapping[str, bool | None]]],
) -> Evaluation:
    """Finalize a waiting measurement after its serving process has stopped."""
    destination = measurement.destination
    metadata = measurement.metadata
    if metadata["status"] != "awaiting_quality":
        raise ValueError("Only measurements awaiting quality can be finalized")
    try:
        quality_results = await quality(measurement.results)
        (destination / "quality.json").write_text(
            json.dumps(dict(quality_results), indent=2), encoding="utf-8"
        )
        observations = [
            to_observation(result, quality_pass=quality_results.get(result.request_id))
            for result in measurement.results
        ]
        evaluation = evaluate(
            observations,
            measurement.slo,
            expected_requests=metadata["expected_requests"],
            elapsed_s=metadata["elapsed_s"],
        )
        metadata.update(
            status="complete",
            evaluation=asdict(evaluation),
        )
        _save_result(destination, metadata)
        return evaluation
    except BaseException as exc:
        metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        _save_result(destination, metadata)
        raise


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
    """Measure an owned service, stop it, then evaluate quality and the joint SLO."""
    measurement = await measure_trial(
        config_path=config_path,
        model_path=model_path,
        samples=samples,
        send_factory=send_factory,
        slo=slo,
        rate=rate,
        destination=destination,
        port=port,
        warmup=warmup,
        startup_timeout_s=startup_timeout_s,
        request_timeout_s=request_timeout_s,
        arrival_seed=arrival_seed,
        profile=profile,
    )
    return await evaluate_trial(measurement, quality=quality)
