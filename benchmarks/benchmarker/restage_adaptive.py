"""Shared-grid adaptive deployment measurements, separate from GPU calibration."""

import hashlib
import json
import math
import shutil
from dataclasses import asdict

from pydantic import BaseModel, ConfigDict, Field

from sglang_omni.restage.evaluation import Evaluation
from sglang_omni.restage.search import search_rates
from sglang_omni.restage.selection import select_candidate


class AdaptiveSearch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_rate: float = Field(gt=0, allow_inf_nan=False)
    growth_factor: float = Field(default=2, gt=1, allow_inf_nan=False)
    target_arrival_duration_s: float | None = Field(
        default=None, gt=0, allow_inf_nan=False
    )


async def execute_adaptive_campaign(
    *,
    configs,
    baseline,
    rates,
    repeats,
    arrival_seed,
    destination,
    trial_options,
    task,
    run_identity,
    resume,
    batch_quality,
    plan_evidence,
    adaptive_search,
):
    # Use the existing campaign for every common rate, including its checkpoints,
    # quality gates and failure handling. The parent lock is held by the caller.
    from benchmarks.benchmarker.restage_campaign import (
        _atomic_write,
        _file_hash,
        _input_identity,
        execute_campaign,
    )

    settings = AdaptiveSearch.model_validate(adaptive_search)
    if not rates or any(not math.isfinite(rate) or rate <= 0 for rate in rates):
        raise ValueError("Rates must be nonempty, finite and positive")
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if baseline not in configs:
        raise ValueError("Include the baseline configuration")
    grid = sorted(set(float(rate) for rate in rates))
    if settings.max_rate < grid[-1]:
        raise ValueError("Adaptive max_rate must cover the initial rates")
    duration = settings.target_arrival_duration_s
    if duration is not None:
        if task != "asr":
            raise ValueError("Automatic corpus duration currently requires ASR")
        if not trial_options.get("samples"):
            raise ValueError("Automatic corpus duration requires samples")
        corpus_repeats = trial_options.get("corpus_repeats", 1)
        if type(corpus_repeats) is not int or corpus_repeats < 1:
            raise ValueError("corpus_repeats must be a positive integer")

    config_bytes = {name: path.read_bytes() for name, path in configs.items()}
    snapshots = {
        name: destination / f"candidate-{index:05d}.yaml"
        for index, name in enumerate(configs)
    }
    identity = dict(
        schema_version=1,
        run_identity=run_identity,
        task=task,
        baseline=baseline,
        initial_rates=grid.copy(),
        repeats=repeats,
        arrival_seed=arrival_seed,
        adaptive_search=settings.model_dump(),
        trial_inputs=_input_identity(trial_options),
        batch_quality=batch_quality,
        plan_evidence=plan_evidence,
        configs={
            name: dict(
                file=snapshots[name].name, sha256=hashlib.sha256(data).hexdigest()
            )
            for name, data in config_bytes.items()
        },
    )
    manifest = destination / "adaptive-campaign.json"
    if resume:
        if json.loads(manifest.read_text(encoding="utf-8")) != identity:
            raise ValueError("Adaptive campaign identity differs from the recorded run")
        for name, path in snapshots.items():
            if _file_hash(path) != identity["configs"][name]["sha256"]:
                raise ValueError(f"Candidate snapshot changed: {name}")
    else:
        destination.mkdir(parents=True, exist_ok=False)
        for name, path in snapshots.items():
            path.write_bytes(config_bytes[name])
        _atomic_write(manifest, json.dumps(identity, indent=2))

    observed = {name: {} for name in configs}
    results = {}
    index = 0
    while index < len(grid):
        rate = grid[index]
        child = destination / f"rate-{rate.hex()}"
        options = dict(trial_options)
        if duration is not None:
            options["corpus_repeats"] = max(
                corpus_repeats, math.ceil(rate * duration / len(options["samples"]))
            )
        try:
            await execute_campaign(
                configs=snapshots,
                baseline=baseline,
                rates=[rate],
                repeats=repeats,
                arrival_seed=arrival_seed,
                destination=child,
                trial_options=options,
                task=task,
                run_identity=run_identity,
                resume=child.exists(),
                batch_quality=batch_quality,
                plan_evidence=plan_evidence,
            )
        except BaseException as exc:
            _atomic_write(
                destination / "failure.json",
                json.dumps(
                    dict(
                        rate=rate,
                        directory=child.name,
                        error=f"{type(exc).__name__}: {exc}",
                    ),
                    indent=2,
                ),
            )
            _atomic_write(
                destination / "adaptive-search.json",
                json.dumps(
                    dict(
                        rates=grid[:index],
                        stop_reason="trial_failure",
                        failed_rate=rate,
                        gpu_group_calibration=False,
                    ),
                    indent=2,
                ),
            )
            if results:
                partial = asdict(select_candidate(results, baseline=baseline))
                partial["scope"] = (
                    "Partial comparison at completed common rates; adaptive search interrupted"
                )
                _atomic_write(
                    destination / "partial-selection.json",
                    json.dumps(partial, indent=2),
                )
            raise
        for row in json.loads(
            (child / "completed-trials.json").read_text(encoding="utf-8")
        ):
            observed[row["candidate"]][rate, row["repeat"]] = Evaluation(
                **row["evaluation"]
            )
        measured_rates = grid[: index + 1]
        for name in configs:

            async def recorded(rate, repeat):
                return observed[name][rate, repeat]

            results[name] = await search_rates(
                recorded, measured_rates, repeats=repeats
            )
        _atomic_write(
            destination / "adaptive-search.json",
            json.dumps(
                dict(
                    rates=measured_rates,
                    stop_reason="running",
                    gpu_group_calibration=False,
                ),
                indent=2,
            ),
        )
        if index == len(grid) - 1:
            if not any(result.points[-1].feasible for result in results.values()):
                stop_reason = "all_candidates_failed"
                break
            if rate >= settings.max_rate:
                stop_reason = "max_rate_reached"
                break
            next_rate = min(settings.max_rate, rate * settings.growth_factor)
            if next_rate <= rate:
                raise ValueError(
                    "Adaptive growth does not advance the floating-point rate"
                )
            grid.append(next_rate)
        index += 1

    selection = select_candidate(results, baseline=baseline)
    for index, name in enumerate(configs):
        _atomic_write(
            destination / f"candidate-{index:05d}-search.json",
            json.dumps(asdict(results[name]), indent=2),
        )
    _atomic_write(
        destination / "selection.json", json.dumps(asdict(selection), indent=2)
    )
    _atomic_write(
        destination / "adaptive-search.json",
        json.dumps(
            dict(
                rates=measured_rates,
                stop_reason=stop_reason,
                gpu_group_calibration=False,
                scope="Finite shared-grid deployment observations; failures are not mathematical capacity upper bounds",
            ),
            indent=2,
        ),
    )
    if selection.recommended is not None:
        shutil.copyfile(
            snapshots[selection.recommended], destination / "recommended.yaml"
        )
    return selection
