# SPDX-License-Identifier: Apache-2.0
"""Standalone non-streaming TTS validation on one-H100 CUDA MPS DP2."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / ".github" / "scripts"))

from tts_mps_overlap import build_overlap_verdict  # noqa: E402
from tts_mps_runtime import (  # noqa: E402
    MpsLaunchSpec,
    atomic_write_json,
    capture_gpu_clients,
    derive_core_blocks,
    find_free_port_pair,
    launch_replicas,
    new_summary,
    read_model_path_activity,
    require_clean_cleanup,
    start_request_profiles,
    stop_request_profiles,
    teardown_replicas,
    update_summary,
    verify_active_gpu_visibility,
)

from benchmarks.dataset.prepare import DATASETS, download_dataset  # noqa: E402
from benchmarks.eval.tts_mps_perf import check_mps_performance  # noqa: E402
from benchmarks.metrics.wer import print_wer_summary  # noqa: E402
from tests.test_model.omni_router_utils import (  # noqa: E402
    assert_workers_served_requests_since,
    launch_managed_router,
    router_get_json,
)
from tests.test_model.test_tts_ci import (  # noqa: E402
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
from tests.utils import (  # noqa: E402
    QWEN3_ASR_ROUTER_STARTUP_TIMEOUT,
    QWEN3_ASR_WER_MODEL_PATH,
    MetricCheckCollector,
    assert_wer_results,
    wait_for_gpu_memory_release,
)

OUTPUT_ENV = "TTS_MPS_OUTPUT_ROOT"
STATE_ENV = "TTS_MPS_STATE_ROOT"
RUN_ID_ENV = "TTS_MPS_RUN_ID"
CONFIG_ENV = "TTS_MPS_CONFIG"
GPU_ENV = "TTS_MPS_GPU_ID"
BASE_PORT_ENV = "TTS_MPS_BASE_PORT"
EXACT_SHA_ENV = "TTS_MPS_EXACT_SHA"
CANARY_REQUESTS = 8
CANARY_CONCURRENCY = 8
pytestmark = [
    pytest.mark.benchmark,
    pytest.mark.skipif(
        not os.environ.get(OUTPUT_ENV),
        reason="standalone MPS validation is enabled only by its dedicated CI job",
    ),
]


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} is required for TTS MPS validation")
    return value


def _summary_path() -> Path:
    return Path(_required(OUTPUT_ENV)).resolve() / "tts_mps_summary.json"


def _ensure_summary(model: str) -> Path:
    path = _summary_path()
    if not path.exists():
        atomic_write_json(
            path,
            new_summary(
                exact_sha=_required(EXACT_SHA_ENV),
                run_id=_required(RUN_ID_ENV),
                run_attempt=os.environ.get("GITHUB_RUN_ATTEMPT", "local"),
                selected_model=model,
            ),
        )
    return path


def _launch_spec() -> MpsLaunchSpec:
    config = Path(_required(CONFIG_ENV))
    if not config.is_absolute():
        config = PROJECT_ROOT / config
    return MpsLaunchSpec(
        repository_root=PROJECT_ROOT,
        output_dir=Path(_required(OUTPUT_ENV)).resolve(),
        state_root=Path(_required(STATE_ENV)).resolve(),
        run_id=_required(RUN_ID_ENV),
        config_path=config.resolve(),
        gpu_id=int(os.environ.get(GPU_ENV, "0")),
        base_port=(
            int(os.environ[BASE_PORT_ENV])
            if os.environ.get(BASE_PORT_ENV)
            else find_free_port_pair()
        ),
        core_blocks=derive_core_blocks(int(os.environ.get(GPU_ENV, "0"))),
        python_bin=sys.executable,
        serve_extra_args=(
            f"{TTS_WORKER_EXTRA_ARGS} {_PRESET.worker_extra_args}".strip()
        ),
    )


def _write_activity(path: Path, events: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for event in events:
            stream.write(json.dumps(event, sort_keys=True))
            stream.write("\n")


def test_tts_mps_non_streaming(
    tmp_path_factory: pytest.TempPathFactory,
    selected_tts_concurrencies: tuple[int, ...],
) -> None:
    assert selected_tts_concurrencies == (16,)
    model = os.environ.get("TTS_CI_MODEL", "higgs")
    output_root = Path(_required(OUTPUT_ENV)).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = _ensure_summary(model)
    spec = _launch_spec()
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
                concurrency=16,
            )
            performance = check_mps_performance(
                model=model,
                concurrency=16,
                summary=speed_results["summary"],
            )
            if performance["status"] != "pass":
                update_summary(
                    summary_path,
                    runtime={"status": "fail", "performance": performance},
                )
                raise AssertionError(
                    "MPS normal-load safety floor failed: "
                    + ", ".join(performance["failed_checks"])
                )
            canonical_checks = MetricCheckCollector("TTS MPS canonical generation")
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
                )
            finally:
                stop_request_profiles(snapshot, spec.run_id)
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
            overlap["request_count"] = CANARY_REQUESTS
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
            concurrency=16,
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
