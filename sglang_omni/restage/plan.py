"""Record a finite Restage search and export its unmeasured candidates."""

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from sglang_omni.config.schema import PipelineConfig
from sglang_omni.config.sources import dump_user_config
from sglang_omni.config.topology import compile_logical_processes
from sglang_omni.restage.candidates import enumerate_candidates
from sglang_omni.restage.capacity import (
    CapacityCatalog,
    CapacityContext,
    capacity_requirements,
    predict_group_capacity,
)
from sglang_omni.restage.configurations import enumerate_configurations


class SearchSpace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    devices: list[int] = Field(min_length=1)
    replica_counts: list[int] = Field(default_factory=lambda: [1], min_length=1)
    dimensions: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)


def _records(config, space):
    for variant in enumerate_configurations(config, space.dimensions):
        context = {"selections": variant.selections}
        if variant.config is None:
            yield {
                **context,
                "status": "rejected",
                "rejection": variant.rejection,
            }, None
            continue
        try:
            logical, stages = compile_logical_processes(variant.config)
            gpu_stages = {stage.name for stage in stages if stage.gpu is not None}
            counts = {
                process.name: space.replica_counts
                for process in logical.processes
                if gpu_stages.intersection(process.stage_names)
            }
            for candidate in enumerate_candidates(
                variant.config, space.devices, counts
            ):
                yield {
                    **context,
                    "key": candidate.key,
                    "assignments": candidate.assignments,
                    "status": (
                        "candidate" if candidate.config is not None else "rejected"
                    ),
                    "rejection": candidate.rejection,
                }, candidate.config
        except ValueError as exc:
            yield {**context, "status": "rejected", "rejection": str(exc)}, None


def write_plan(
    config: PipelineConfig,
    space: SearchSpace,
    destination: Path,
    *,
    max_candidates: int = 256,
    capacity_catalog: CapacityCatalog | None = None,
    capacity_context: CapacityContext | None = None,
) -> dict[str, Any]:
    """Write candidate YAML and rejection records without launching a server.

    The limit counts all examined records, including rejected configurations.
    The summary states whether the declared search space was exhausted.
    """
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    if (capacity_catalog is None) != (capacity_context is None):
        raise ValueError("Capacity catalog and context must be supplied together")
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "search-space.json").write_text(
        space.model_dump_json(indent=2), encoding="utf-8"
    )
    summary = {
        "complete": False,
        "examined": 0,
        "accepted": 0,
        "rejected": 0,
        "performance_measured": False,
        "max_candidates": max_candidates,
    }
    records = iter(_records(config, space))
    missing_requirements = {}
    if capacity_catalog is not None:
        summary.update(predicted=0, unranked=0)
    with (destination / "candidates.jsonl").open("w", encoding="utf-8") as log:
        for index in range(max_candidates):
            item = next(records, None)
            if item is None:
                summary["complete"] = True
                break
            row, candidate_config = item
            if candidate_config is not None:
                filename = f"candidate-{index:05d}.yaml"
                (destination / filename).write_text(
                    yaml.safe_dump(dump_user_config(candidate_config), sort_keys=False),
                    encoding="utf-8",
                )
                row["config_file"] = filename
                if capacity_catalog is not None:
                    requirements = capacity_requirements(
                        candidate_config, row["assignments"], capacity_context
                    )
                    prediction = predict_group_capacity(requirements, capacity_catalog)
                    row["prediction"] = prediction
                    summary[prediction["status"]] += 1
                    missing = set(prediction.get("missing_groups", []))
                    for group in requirements:
                        if group["key"] in missing:
                            missing_requirements.setdefault(
                                group["key"],
                                {**group, "representative_candidate_config": filename},
                            )
                summary["accepted"] += 1
            else:
                summary["rejected"] += 1
            summary["examined"] += 1
            log.write(json.dumps(row, ensure_ascii=False) + "\n")
            log.flush()
        else:
            summary["complete"] = next(records, None) is None
    if capacity_catalog is not None:
        (destination / "calibration-requirements.json").write_text(
            json.dumps(list(missing_requirements.values()), indent=2), encoding="utf-8"
        )
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
