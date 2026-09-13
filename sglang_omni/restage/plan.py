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
from sglang_omni.restage.configurations import enumerate_configurations


class SearchSpace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    devices: list[int] = Field(min_length=1)
    # Note (Jiaxin Deng): a mapping keeps one process fixed while another varies,
    # such as Moss PD decode replicas 1/2/3 against a single prefill process.
    replica_counts: list[int] | dict[str, list[int]] = Field(
        default_factory=lambda: [1]
    )
    dimensions: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)


def _replica_counts(space, process_names):
    if isinstance(space.replica_counts, dict):
        missing = process_names - set(space.replica_counts)
        if missing:
            raise ValueError(f"Replica counts missing GPU processes: {sorted(missing)}")
        return {name: space.replica_counts[name] for name in process_names}
    return {name: space.replica_counts for name in process_names}


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
            counts = _replica_counts(
                space,
                {
                    process.name
                    for process in logical.processes
                    if gpu_stages.intersection(process.stage_names)
                },
            )
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
) -> dict[str, Any]:
    """Write candidate YAML and rejection records without launching a server.

    The limit counts all examined records, including rejected configurations.
    The summary states whether the declared search space was exhausted.
    """
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
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
                summary["accepted"] += 1
            else:
                summary["rejected"] += 1
            summary["examined"] += 1
            log.write(json.dumps(row, ensure_ascii=False) + "\n")
            log.flush()
        else:
            summary["complete"] = next(records, None) is None
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def load_plan(directory: Path, baseline: str) -> dict[str, Path]:
    """Return every accepted candidate of a plan, baseline first."""
    directory = directory.resolve()
    configs = {}
    for line in (
        (directory / "candidates.jsonl").read_text(encoding="utf-8").splitlines()
    ):
        row = json.loads(line)
        if row["status"] != "candidate":
            continue
        config = (directory / row["config_file"]).resolve()
        if not config.is_relative_to(directory) or not config.is_file():
            raise ValueError(
                f"Plan configuration must exist inside its directory: {config}"
            )
        configs[config.stem] = config
    if baseline not in configs:
        raise ValueError("Include the baseline among accepted plan candidates")
    return {baseline: configs.pop(baseline), **configs}
