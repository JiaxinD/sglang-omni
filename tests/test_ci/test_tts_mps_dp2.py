# SPDX-License-Identifier: Apache-2.0
"""Non-streaming TTS validation on one H100 running CUDA MPS DP2.

Author: Jiaxin Deng

The canonical TTS stages already prove the model is correct under ordinary DP2
across two H100s. This file proves a different thing: that the same quality
contract still holds when two production Router workers share one H100 through
a private MPS namespace. It is a placement test, not a fifth TTS stage, so its
outputs are evidence only and never feed a later stage.

Three claims have to hold together, and each has its own oracle here:

* the two replicas really are separate MPS clients on the intended GPU, checked
  by diffing GPU client sets around launch rather than trusting the launcher;
* they really do execute concurrently, checked by a bounded canary whose
  server-side CLOCK_MONOTONIC model-path intervals must actually overlap, since
  two serialized replicas would otherwise pass every aggregate metric;
* quality does not move, checked by reusing the canonical WER, similarity,
  UTMOS and audio-integrity assertions unchanged rather than restating them.

Performance references are separate from those assertions and are calibrated
worst-of-five, see `benchmarks/eval/tts_mps_perf.py`.

Teardown is part of the contract. A dirty process, port, state directory, or
leftover GPU client fails the test, because the next CI stage on the same
runner would otherwise inherit the mess.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import pytest

from benchmarks.dataset.prepare import DATASETS, download_dataset
from benchmarks.eval.tts_mps_perf import MPS_PERFORMANCE_REFERENCES, check_mps_performance
from benchmarks.metrics.wer import print_wer_summary
from tests.test_model.omni_router_utils import (
    _find_available_port_range,
    assert_workers_served_requests_since,
    launch_managed_router,
    router_get_json,
)
from tests.test_model.test_tts_ci import (
    _PRESET,
    _THRESHOLDS,
    SEEDTTS_DATASET_LABEL,
    SEEDTTS_EN_FULLSET_SAMPLES,
    TTS_MODEL_PATH,
    TTS_SIMILARITY_MAX_SAMPLES,
    TTS_WORKER_EXTRA_ARGS,
    _assert_full_seedtts_en_speed_results,
    _assert_full_seedtts_en_wer_results,
    _assert_similarity_results,
    _assert_tts_audio_result_integrity,
    _assert_utmos_results,
    _run_benchmark,
    _run_similarity,
    _run_utmos,
    _run_wer_transcribe,
)
from tests.utils import (
    QWEN3_ASR_ROUTER_STARTUP_TIMEOUT,
    QWEN3_ASR_WER_MODEL_PATH,
    MetricCheckCollector,
    assert_wer_results,
    wait_for_gpu_memory_release,
)
from tests.utils.tts_mps_overlap import build_overlap_verdict
from tests.utils.tts_mps_runtime import (
    MpsLaunchSpec,
    atomic_write_json,
    capture_gpu_clients,
    derive_core_blocks,
    launch_replicas,
    new_summary,
    read_model_path_activity,
    require_clean_cleanup,
    require_exact_request_counts,
    start_request_profiles,
    stop_request_profiles,
    teardown_replicas,
    update_summary,
    verify_active_gpu_visibility,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_ENV = "TTS_MPS_OUTPUT_ROOT"
STATE_ENV = "TTS_MPS_STATE_ROOT"
RUN_ID_ENV = "TTS_MPS_RUN_ID"
CONFIG_ENV = "TTS_MPS_CONFIG"
GPU_ENV = "TTS_MPS_GPU_ID"
BASE_PORT_ENV = "TTS_MPS_BASE_PORT"
EXACT_SHA_ENV = "TTS_MPS_EXACT_SHA"
CANARY_REQUESTS = 8
CANARY_CONCURRENCY = 8
CONCURRENCY = 16
VALIDATION_FILE = "benchmark_validation.json"
# Mirrors the CPU preflight resolution in omni-ci.yaml; the launcher pins N=2.
DEFAULT_CONFIGS = {
    "higgs": "examples/mps_dp/configs/higgs_h100_dp3.yaml",
    "moss": "examples/mps_dp/configs/moss_local_h100_dp2.yaml",
}
pytestmark = [pytest.mark.benchmark, pytest.mark.gpu]


def _selected_model() -> str:
    model = os.environ.get("TTS_CI_MODEL", "higgs").strip() or "higgs"
    if model not in DEFAULT_CONFIGS:
        raise ValueError(f"unsupported TTS MPS model {model!r}")
    return model


def _env_or(name: str, fallback: str) -> str:
    return os.environ.get(name, "").strip() or fallback


def _physical_gpu_id() -> int:
    """Resolve the physical GPU index the launcher should bind.

    The launcher resolves a GPU UUID through nvidia-smi, which enumerates
    physically and ignores CUDA_VISIBLE_DEVICES, so a harness that scopes this
    process with CVD would otherwise be silently overridden and the stage would
    run on physical GPU 0.
    """
    explicit = os.environ.get(GPU_ENV, "").strip()
    if explicit:
        return int(explicit)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    first = visible.split(",")[0].strip() if visible else ""
    return int(first) if first.isdigit() else 0


def _resolve_roots(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path]:
    # Note: (Jiaxin Deng) the dedicated CI job supplies these roots so artifacts
    # land on the runner. Calibration and local repro supply neither, so they
    # fall back instead of hard-failing; otherwise the stage is observable only
    # from inside CI and can never be calibrated.
    configured = os.environ.get(OUTPUT_ENV, "").strip()
    output_root = (
        Path(configured).resolve()
        if configured
        else tmp_path_factory.mktemp("tts-mps-ci")
    )
    configured_state = os.environ.get(STATE_ENV, "").strip()
    # The state root carries the MPS control socket, which must fit AF_UNIX
    # sun_path, so the fallback is a short temp dir rather than the deep
    # pytest basetemp.
    state_root = (
        Path(configured_state).resolve()
        if configured_state
        else Path(tempfile.mkdtemp(prefix="omni-mps-"))
    )
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root, state_root


def _ensure_summary(model: str, output_root: Path, run_id: str) -> Path:
    path = output_root / "tts_mps_summary.json"
    if not path.exists():
        atomic_write_json(
            path,
            new_summary(
                exact_sha=_env_or(EXACT_SHA_ENV, "local"),
                run_id=run_id,
                run_attempt=os.environ.get("GITHUB_RUN_ATTEMPT", "local"),
                selected_model=model,
            ),
        )
    return path


def _launch_spec(
    model: str,
    output_root: Path,
    state_root: Path,
    run_id: str,
) -> MpsLaunchSpec:
    config = Path(_env_or(CONFIG_ENV, DEFAULT_CONFIGS[model]))
    if not config.is_absolute():
        config = PROJECT_ROOT / config
    gpu_id = _physical_gpu_id()
    return MpsLaunchSpec(
        repository_root=PROJECT_ROOT,
        output_dir=output_root,
        state_root=state_root,
        run_id=run_id,
        config_path=config.resolve(),
        gpu_id=gpu_id,
        base_port=(
            int(os.environ[BASE_PORT_ENV])
            if os.environ.get(BASE_PORT_ENV)
            else _find_available_port_range(2)
        ),
        core_blocks=derive_core_blocks(gpu_id),
        python_bin=sys.executable,
        serve_extra_args=(
            f"{TTS_WORKER_EXTRA_ARGS} {_PRESET.worker_extra_args}".strip()
        ),
    )


def _write_validation(
    output_root: Path,
    *,
    valid: bool,
    threshold_assertion_failed: bool,
    detail: dict | None = None,
) -> None:
    atomic_write_json(
        output_root / VALIDATION_FILE,
        {
            "valid": valid,
            "threshold_assertion_failed": threshold_assertion_failed,
            "detail": detail or {},
        },
    )


def _write_activity(path: Path, events: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for event in events:
            stream.write(json.dumps(event, sort_keys=True))
            stream.write("\n")


def test_tts_mps_non_streaming(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    model = _selected_model()
    output_root, state_root = _resolve_roots(tmp_path_factory)
    # new_summary() requires a run-<suffix> path component.
    run_id = _env_or(RUN_ID_ENV, f"run-local-{os.getpid()}")
    summary_path = _ensure_summary(model, output_root, run_id)
    spec = _launch_spec(model, output_root, state_root, run_id)
    dataset_repo = DATASETS["seedtts"]
    download_dataset(dataset_repo, quiet=True)

    total_started = time.perf_counter()
    generation_started = total_started
    snapshot = None
    cleanup_recorded = False
    canonical_dir = output_root / "canonical"
    canary_dir = output_root / "overlap-canary"
    before_workers: dict | None = None
    after_workers: dict | None = None
    speed_results: dict | None = None
    overlap: dict | None = None
    baseline_gpu_clients = capture_gpu_clients()
    update_summary(
        summary_path,
        runtime={
            "status": "collecting",
            "baseline_gpu_clients": baseline_gpu_clients,
        },
    )
    try:
        snapshot = launch_replicas(spec)
        active_gpu_clients = capture_gpu_clients()
        visible_mps_clients = verify_active_gpu_visibility(
            baseline_gpu_clients,
            active_gpu_clients,
            gpu_uuid=snapshot.manifest["gpu_uuid"],
        )
        update_summary(
            summary_path,
            runtime={
                "status": "collecting",
                "baseline_gpu_clients": baseline_gpu_clients,
                "active_gpu_clients": active_gpu_clients,
                "visible_mps_clients": visible_mps_clients,
            },
        )
        with launch_managed_router(
            tmp_path_factory=tmp_path_factory,
            model_path=TTS_MODEL_PATH,
            model_name=TTS_MODEL_PATH,
            worker_extra_args="",
            external_worker_urls=list(spec.worker_urls),
            wait_timeout=_PRESET.startup_timeout,
            log_prefix="tts_mps_router_logs",
        ) as router:
            before_workers = router_get_json(router.port, "/workers")
            speed_results = _run_benchmark(
                router.port,
                dataset_repo,
                str(canonical_dir),
                concurrency=CONCURRENCY,
            )
            performance = check_mps_performance(
                model=model,
                concurrency=CONCURRENCY,
                summary=speed_results["summary"],
            )
            update_summary(summary_path, runtime={"performance": performance})
            # Note: (Jiaxin Deng) a reference violation is collected, not raised
            # here, so the run still produces a complete observation for
            # calibration instead of aborting before the quality oracles.
            canonical_checks = MetricCheckCollector("TTS MPS canonical generation")
            canonical_checks.check(
                performance["status"] == "pass",
                "MPS performance references: "
                + "; ".join(performance["failed_checks"]),
            )
            _assert_full_seedtts_en_speed_results(
                speed_results,
                label="TTS MPS non-stream c16",
                collector=canonical_checks,
            )
            _assert_tts_audio_result_integrity(
                speed_results["summary"],
                speed_results["per_request"],
                label="TTS MPS non-stream c16",
                collector=canonical_checks,
            )
            assert_workers_served_requests_since(
                port=router.port,
                before_snapshot=before_workers,
                label="TTS MPS canonical generation",
                min_total_requests=SEEDTTS_EN_FULLSET_SAMPLES,
            )
            # The canonical benchmark produced a complete result set, so the
            # observation stands even when a reference assertion below fails.
            # Calibration needs that distinction; see CONTRACT.md.
            _write_validation(
                output_root,
                valid=True,
                threshold_assertion_failed=bool(canonical_checks.failures),
                detail={"performance": performance},
            )
            canonical_checks.assert_all()

            start_request_profiles(snapshot, spec.run_id)
            canary_started = time.perf_counter()
            try:
                canary_results = _run_benchmark(
                    router.port,
                    dataset_repo,
                    str(canary_dir),
                    concurrency=CANARY_CONCURRENCY,
                    max_samples=CANARY_REQUESTS,
                    warmup=0,
                )
            finally:
                stop_request_profiles(snapshot, spec.run_id)
            canary_counts = require_exact_request_counts(
                canary_results["summary"],
                expected_requests=CANARY_REQUESTS,
            )
            events = read_model_path_activity(
                snapshot,
                min_terminal_events=CANARY_REQUESTS,
            )
            _write_activity(output_root / "replica_activity.jsonl", events)
            overlap = build_overlap_verdict(
                events,
                expected_run_id=spec.run_id,
                min_successes_per_replica=2,
                min_matched_overlap_count=2,
                measurement_uncertainty_ns=1_000_000,
            )
            overlap["canary_wall_time_s"] = time.perf_counter() - canary_started
            overlap["request_counts"] = canary_counts
            after_workers = router_get_json(router.port, "/workers")

        cleanup = teardown_replicas(
            spec,
            snapshot,
            baseline_gpu_clients=baseline_gpu_clients,
        )
        cleanup_recorded = True
        update_summary(summary_path, cleanup=cleanup)
        require_clean_cleanup(cleanup)
        wait_for_gpu_memory_release()
        update_summary(
            summary_path,
            runtime={
                "status": "pass",
                "baseline_gpu_clients": baseline_gpu_clients,
                "active_gpu_clients": active_gpu_clients,
                "visible_mps_clients": visible_mps_clients,
                "gpu_uuid": snapshot.manifest["gpu_uuid"],
                "weight_share": False,
                "replicas": [
                    {
                        "index": item.index,
                        "pid": item.pid,
                        "pgid": item.pgid,
                        "port": item.port,
                        "max_total_tokens": item.kv_tokens,
                    }
                    for item in snapshot.replicas
                ],
                "router_workers_before": before_workers,
                "router_workers_after": after_workers,
                "overlap": overlap,
                "performance": performance,
            },
            timing={
                "generation_and_cleanup_s": time.perf_counter() - generation_started,
            },
        )
    except BaseException as exc:
        if snapshot is not None and not cleanup_recorded:
            try:
                cleanup = teardown_replicas(
                    spec,
                    snapshot,
                    baseline_gpu_clients=baseline_gpu_clients,
                )
                cleanup_recorded = True
                update_summary(summary_path, cleanup=cleanup)
                require_clean_cleanup(cleanup)
            except BaseException as cleanup_exc:
                if not cleanup_recorded:
                    update_summary(
                        summary_path,
                        cleanup={"status": "dirty", "error": repr(cleanup_exc)},
                    )
        current = json.loads(summary_path.read_text(encoding="utf-8"))
        runtime = current.get("runtime")
        if not isinstance(runtime, dict) or runtime.get("status") not in {
            "fail",
            "pass",
        }:
            update_summary(
                summary_path,
                runtime={
                    "status": "fail",
                    "error": repr(exc),
                    "baseline_gpu_clients": baseline_gpu_clients,
                },
            )
        raise

    assert speed_results is not None
    evaluator_started = time.perf_counter()
    with launch_managed_router(
        tmp_path_factory=tmp_path_factory,
        model_path=QWEN3_ASR_WER_MODEL_PATH,
        model_name=QWEN3_ASR_WER_MODEL_PATH,
        worker_extra_args="",
        wait_timeout=QWEN3_ASR_ROUTER_STARTUP_TIMEOUT,
        log_prefix="tts_mps_asr_router_logs",
    ) as asr_router:
        wer_results = _run_wer_transcribe(
            dataset_repo,
            str(canonical_dir),
            asr_router_port=asr_router.port,
            concurrency=CONCURRENCY,
        )
    wait_for_gpu_memory_release()
    update_summary(
        summary_path,
        evaluator={
            "status": "pass",
            "model": QWEN3_ASR_WER_MODEL_PATH,
            "duration_s": time.perf_counter() - evaluator_started,
        },
    )

    similarity_checkpoint = os.environ.get("SEEDTTS_SIM_CHECKPOINT")
    similarity_results = _run_similarity(
        dataset_repo,
        str(canonical_dir),
        similarity_checkpoint,
        max_samples=TTS_SIMILARITY_MAX_SAMPLES,
    )
    utmos_results = _run_utmos(str(canonical_dir))
    quality = MetricCheckCollector("TTS MPS quality")
    _assert_full_seedtts_en_wer_results(
        wer_results,
        label="TTS MPS non-stream c16",
        collector=quality,
    )
    assert_wer_results(
        wer_results,
        _THRESHOLDS.wer_corpus,
        collector=quality,
    )
    _assert_similarity_results(
        similarity_results,
        _THRESHOLDS.similarity_mean_min,
        collector=quality,
    )
    _assert_utmos_results(
        utmos_results,
        _THRESHOLDS.utmos_mean_min,
        collector=quality,
    )
    correctness = {
        "status": "pass" if not quality.failures else "fail",
        "canonical_requests": len(speed_results["per_request"]),
        "wer": wer_results["summary"],
        "similarity": similarity_results["summary"],
        "utmos": utmos_results["summary"],
        "overlap": overlap,
        "failures": list(quality.failures),
    }
    update_summary(
        summary_path,
        correctness=correctness,
        timing={
            **json.loads(summary_path.read_text(encoding="utf-8"))["timing"],
            "total_s": time.perf_counter() - total_started,
        },
    )
    print_wer_summary(
        wer_results["summary"],
        TTS_MODEL_PATH,
        dataset=SEEDTTS_DATASET_LABEL,
    )
    quality.assert_all()
