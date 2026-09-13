"""Fit the inherited service-time law without reusing historical coefficients."""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import nnls

from sglang_omni.restage.workload import ServiceLaw


@dataclass(frozen=True)
class ServiceSample:
    request_id: str
    context_tokens: int
    audio_seconds: float
    service_s: float


@dataclass(frozen=True)
class ServiceCalibration:
    law: ServiceLaw
    training_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    training_mae_s: float
    validation_mae_s: float | None
    validation_predictions: tuple[dict, ...]
    context_range: tuple[int, int]
    audio_range_s: tuple[float, float]


def fit_service_law(
    training: Sequence[ServiceSample],
    *,
    validation: Sequence[ServiceSample],
    source: str,
    delta_floor_s: float,
    prefill_rate: float,
) -> ServiceCalibration:
    """Fit s(L,D)=a+bL+cL²+dD with nonnegative least squares.

    Supply quality-accepted isolated service measurements from one model,
    configuration, hardware and stack. Queueing latency under load is not
    an isolated service measurement. The caller supplies separately measured
    stall parameters; this fit does not estimate them or saturated capacity.
    Independent validation requests are scored without entering the fit.
    """
    all_samples = [*training, *validation]
    ids = [sample.request_id for sample in all_samples]
    if len(set(ids)) != len(ids):
        raise ValueError("Training and validation requests must be unique and disjoint")
    if not source:
        raise ValueError("Calibration source is required")
    if (
        not math.isfinite(delta_floor_s)
        or delta_floor_s < 0
        or not math.isfinite(prefill_rate)
        or prefill_rate <= 0
    ):
        raise ValueError("Stall floor must be nonnegative and prefill rate positive")
    for sample in all_samples:
        if (
            not sample.request_id
            or not math.isfinite(sample.context_tokens)
            or sample.context_tokens < 0
            or not math.isfinite(sample.audio_seconds)
            or sample.audio_seconds < 0
            or not math.isfinite(sample.service_s)
            or sample.service_s <= 0
        ):
            raise ValueError(
                "Samples need IDs, finite nonnegative workloads and positive service time"
            )
    if len(training) < 4:
        raise ValueError(
            "At least four independent workload points must identify the law"
        )
    matrix = np.array(
        [
            [1, sample.context_tokens, sample.context_tokens**2, sample.audio_seconds]
            for sample in training
        ],
        dtype=float,
    )
    scales = np.linalg.norm(matrix, axis=0)
    if np.any(scales == 0) or np.linalg.matrix_rank(matrix / scales) < 4:
        raise ValueError(
            "Workload variation cannot identify all four service-law terms"
        )
    observed = np.array([sample.service_s for sample in training])
    coefficients, _ = nnls(matrix / scales, observed)
    a, b, c, audio_slope = (float(value) for value in coefficients / scales)
    law = ServiceLaw(
        a=a,
        b=b,
        c=c,
        audio_slope=audio_slope,
        delta_floor_s=delta_floor_s,
        prefill_rate=prefill_rate,
        source=source,
    )
    predictions = tuple(
        {
            "request_id": sample.request_id,
            "context_tokens": sample.context_tokens,
            "audio_seconds": sample.audio_seconds,
            "observed_s": sample.service_s,
            "predicted_s": law.s(sample.context_tokens, sample.audio_seconds),
        }
        for sample in validation
    )
    return ServiceCalibration(
        law=law,
        training_ids=tuple(sample.request_id for sample in training),
        validation_ids=tuple(sample.request_id for sample in validation),
        training_mae_s=float(
            np.mean(np.abs(matrix @ (coefficients / scales) - observed))
        ),
        validation_mae_s=(
            float(
                np.mean(
                    [abs(row["predicted_s"] - row["observed_s"]) for row in predictions]
                )
            )
            if predictions
            else None
        ),
        validation_predictions=predictions,
        context_range=(
            min(s.context_tokens for s in training),
            max(s.context_tokens for s in training),
        ),
        audio_range_s=(
            min(s.audio_seconds for s in training),
            max(s.audio_seconds for s in training),
        ),
    )
