# SPDX-License-Identifier: Apache-2.0
"""Performance policy for the TTS MPS DP2 validation stage.

References are raw worst-of-five observations from a CI-comparable H100, held
pre-slack exactly as `.claude/skills/tune-ci-thresholds` writes them. CI
assertion slack is derived here, never baked into the stored reference, so a
recalibration only ever rewrites the `_REF` literals.

`None` means the metric has not been calibrated yet. The stage still collects
and reports the observation, then fails the assertion, so an uncalibrated gate
can never pass silently.
"""

from __future__ import annotations

from typing import Any

# Same uniform slack the TTS generation stages use, kept here so this policy
# does not import from the test tree.
MPS_SLACK_HIGHER = 0.75
MPS_SLACK_LOWER = 1.25

MPS_CONCURRENCY = 16

MPS_HIGGS_THROUGHPUT_QPS_REF: float | None = None
MPS_HIGGS_OUTPUT_TOK_PER_REQ_S_REF: float | None = None
MPS_HIGGS_LATENCY_MEAN_S_REF: float | None = None
MPS_HIGGS_RTF_MEAN_REF: float | None = None

MPS_MOSS_THROUGHPUT_QPS_REF: float | None = None
MPS_MOSS_OUTPUT_TOK_PER_REQ_S_REF: float | None = None
MPS_MOSS_LATENCY_MEAN_S_REF: float | None = None
MPS_MOSS_RTF_MEAN_REF: float | None = None

_MINIMUM = "minimum"
_MAXIMUM = "maximum"

MPS_PERFORMANCE_REFERENCES: dict[str, dict[str, tuple[float | None, str, str]]] = {
    "higgs": {
        "throughput_qps": (MPS_HIGGS_THROUGHPUT_QPS_REF, _MINIMUM, "req/s"),
        "output_tok_per_req_s": (
            MPS_HIGGS_OUTPUT_TOK_PER_REQ_S_REF,
            _MINIMUM,
            "tok/req-s",
        ),
        "latency_mean_s": (MPS_HIGGS_LATENCY_MEAN_S_REF, _MAXIMUM, "s"),
        "rtf_mean": (MPS_HIGGS_RTF_MEAN_REF, _MAXIMUM, ""),
    },
    "moss": {
        "throughput_qps": (MPS_MOSS_THROUGHPUT_QPS_REF, _MINIMUM, "req/s"),
        "output_tok_per_req_s": (
            MPS_MOSS_OUTPUT_TOK_PER_REQ_S_REF,
            _MINIMUM,
            "tok/req-s",
        ),
        "latency_mean_s": (MPS_MOSS_LATENCY_MEAN_S_REF, _MAXIMUM, "s"),
        "rtf_mean": (MPS_MOSS_RTF_MEAN_REF, _MAXIMUM, ""),
    },
}


def derive_threshold(reference: float, direction: str) -> float:
    """Apply CI slack to a stored pre-slack reference."""
    if direction == _MINIMUM:
        return reference * MPS_SLACK_HIGHER
    return reference * MPS_SLACK_LOWER


def check_mps_performance(
    *,
    model: str,
    concurrency: int,
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Compare an observed benchmark summary against the calibrated references."""
    references = MPS_PERFORMANCE_REFERENCES.get(model)
    if references is None:
        raise ValueError(f"unsupported MPS performance model {model!r}")
    if concurrency != MPS_CONCURRENCY:
        raise ValueError(
            f"MPS performance references require concurrency {MPS_CONCURRENCY}"
        )

    checks: dict[str, Any] = {}
    failed: list[str] = []
    uncalibrated: list[str] = []
    for metric, (reference, direction, unit) in references.items():
        observed = summary.get(metric)
        if not isinstance(observed, (int, float)):
            failed.append(f"{metric} missing from benchmark summary")
            continue
        if reference is None:
            uncalibrated.append(metric)
            checks[metric] = {
                "observed": observed,
                "reference": None,
                "threshold": None,
                "direction": direction,
                "unit": unit,
                "pass": False,
            }
            continue
        threshold = derive_threshold(reference, direction)
        passed = (
            observed >= threshold if direction == _MINIMUM else observed <= threshold
        )
        checks[metric] = {
            "observed": observed,
            "reference": reference,
            "threshold": threshold,
            "direction": direction,
            "unit": unit,
            "pass": passed,
        }
        if not passed:
            comparator = ">=" if direction == _MINIMUM else "<="
            failed.append(
                f"{metric} {observed:.4f}{unit} violates {comparator} "
                f"{threshold:.4f}{unit} (reference {reference:.4f}{unit})"
            )

    total = summary.get("total_requests")
    if (
        not isinstance(total, int)
        or total <= 0
        or summary.get("completed_requests") != total
        or summary.get("failed_requests") != 0
    ):
        failed.append("request_completion")

    if uncalibrated:
        failed.append(
            "uncalibrated performance references: "
            + ", ".join(sorted(uncalibrated))
            + "; run .claude/skills/tune-ci-thresholds with 5 repeats"
        )
    return {
        "status": "pass" if not failed else "fail",
        "model": model,
        "concurrency": concurrency,
        "checks": checks,
        "failed_checks": failed,
        "uncalibrated": sorted(uncalibrated),
    }
