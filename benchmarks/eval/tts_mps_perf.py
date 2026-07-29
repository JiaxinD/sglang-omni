# SPDX-License-Identifier: Apache-2.0
"""Performance policy for the standalone TTS MPS validation stage."""

from __future__ import annotations

from typing import Any

# Note (Jiaxin Deng): these floors sit well below the accepted H100 samples so
# the new stage catches gross regressions without turning one early sample into
# a capacity benchmark.
MPS_PERFORMANCE_FLOORS: dict[str, dict[str, Any]] = {
    "higgs": {
        "concurrency": 16,
        "minimum": {
            "throughput_qps": 10.0,
            "audio_throughput_s_per_s": 43.0,
            "output_throughput": 1100.0,
            "output_tok_per_req_s": 80.0,
        },
        "maximum": {"latency_mean_s": 1.8, "rtf_mean": 0.45},
        "provenance": {
            "temporary": True,
            "method": "wide_floor_below_two_exact_sha_h100_mps_samples",
            "accepted_run_ids": [30196509700, 30199556014],
        },
    },
    "moss": {
        "concurrency": 16,
        "minimum": {
            "throughput_qps": 7.0,
            "audio_throughput_s_per_s": 30.0,
            "output_throughput": 380.0,
            "output_tok_per_req_s": 40.0,
        },
        "maximum": {"latency_mean_s": 2.4, "rtf_mean": 0.60},
        "provenance": {
            "temporary": True,
            "method": "wide_floor_with_over_30_percent_slack_from_h100_sample",
            "accepted_run_ids": [30202304743],
        },
    },
}


def check_mps_performance(
    *,
    model: str,
    concurrency: int,
    summary: dict[str, Any],
) -> dict[str, Any]:
    floor = MPS_PERFORMANCE_FLOORS.get(model)
    if floor is None:
        raise ValueError(f"unsupported MPS performance model {model!r}")
    if concurrency != floor["concurrency"]:
        raise ValueError(
            f"MPS performance floor requires concurrency {floor['concurrency']}"
        )
    failed: list[str] = []
    checks: dict[str, Any] = {}
    for metric, threshold in floor["minimum"].items():
        observed = summary.get(metric)
        passed = isinstance(observed, (int, float)) and observed >= threshold
        checks[metric] = {
            "operator": ">=",
            "threshold": threshold,
            "observed": observed,
            "pass": passed,
        }
        if not passed:
            failed.append(metric)
    for metric, threshold in floor["maximum"].items():
        observed = summary.get(metric)
        passed = isinstance(observed, (int, float)) and observed <= threshold
        checks[metric] = {
            "operator": "<=",
            "threshold": threshold,
            "observed": observed,
            "pass": passed,
        }
        if not passed:
            failed.append(metric)
    total = summary.get("total_requests")
    if (
        not isinstance(total, int)
        or total <= 0
        or summary.get("completed_requests") != total
        or summary.get("failed_requests") != 0
    ):
        failed.append("request_completion")
    return {
        "status": "pass" if not failed else "fail",
        "concurrency": concurrency,
        "checks": checks,
        "failed_checks": failed,
        "provenance": floor["provenance"],
    }
