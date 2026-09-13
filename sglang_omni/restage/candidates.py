"""Materialize Restage process placements through the serving config schema."""

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations_with_replacement, permutations

from sglang_omni.config.placement import build_stage_placement_plan
from sglang_omni.config.schema import PipelineConfig, ProcessConfig
from sglang_omni.config.topology import (
    build_process_topology_plan,
    compile_logical_processes,
)
from sglang_omni.pipeline.replicas import expand_replica_stages


@dataclass(frozen=True)
class Candidate:
    key: str
    assignments: dict[str, tuple[tuple[int, ...], ...]]
    config: PipelineConfig | None
    rejection: str | None = None


def enumerate_candidates(
    config: PipelineConfig,
    devices: Sequence[int],
    replica_counts: Mapping[str, Sequence[int]],
) -> Iterator[Candidate]:
    """Enumerate placements for explicit per-process replica count choices.

    Keeps the input's TP degrees, memory budgets and process membership.
    Both dedicated and shared devices are considered, including idle GPUs.
    All candidates in this finite space are yielded; invalid runtime layouts
    carry their rejection reason. This is not a performance ranking.
    """
    if not devices or len(set(devices)) != len(devices) or min(devices) < 0:
        raise ValueError("Device budget must contain distinct nonnegative GPU IDs")
    devices = tuple(sorted(devices))
    logical, stages = compile_logical_processes(config)
    gpu_stages = {stage.name for stage in stages if stage.gpu is not None}
    processes = [p for p in logical.processes if gpu_stages.intersection(p.stage_names)]
    for process in processes:
        if process.tp_size > len(devices):
            raise ValueError(
                f"Process {process.name!r} tp_size={process.tp_size} exceeds "
                f"the {len(devices)}-GPU budget"
            )
    if set(replica_counts) != {p.name for p in processes}:
        raise ValueError("Replica choices must cover exactly the GPU processes")
    for name, counts in replica_counts.items():
        if not counts or any(type(count) is not int or count < 1 for count in counts):
            raise ValueError(f"Process {name!r} needs positive integer replica choices")

    def assignments_at(index, partial):
        if index == len(processes):
            yield dict(partial)
            return
        process = processes[index]
        rank_groups = tuple(permutations(devices, process.tp_size))
        for count in sorted(set(replica_counts[process.name])):
            # Note (Jiaxin Deng): replicas are interchangeable, but TP rank order
            # can affect communication; remove only replica permutations.
            for replicas in combinations_with_replacement(rank_groups, count):
                partial[process.name] = replicas
                yield from assignments_at(index + 1, partial)

    config_data = config.model_dump(mode="json")
    for assignments in assignments_at(0, {}):
        encoded = json.dumps(
            {"config": config_data, "assignments": assignments},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        key = hashlib.sha256(encoded).hexdigest()
        try:
            result = materialize_candidate(config, assignments, devices)
        except ValueError as exc:
            yield Candidate(key, assignments, None, str(exc))
        else:
            yield Candidate(key, assignments, result)


def materialize_candidate(
    config: PipelineConfig,
    assignments: Mapping[str, Sequence[Sequence[int]]],
    devices: Sequence[int],
) -> PipelineConfig:
    """Copy a config with one TP-rank device tuple per process replica.

    Device IDs are in the server's visible-device namespace. Assign every
    GPU process; CPU-only processes retain their existing policy. This checks
    declared runtime constraints, not measured memory use or performance.
    """
    logical, stages = compile_logical_processes(config)
    gpu_stages = {stage.name for stage in stages if stage.gpu is not None}
    gpu_processes = {
        process.name: process
        for process in logical.processes
        if gpu_stages.intersection(process.stage_names)
    }
    if set(assignments) != set(gpu_processes):
        raise ValueError(
            f"Assignments must cover exactly the GPU processes: {sorted(gpu_processes)}"
        )
    budget = set(devices)
    result = config.model_copy(deep=True)
    for name, replicas in assignments.items():
        process = gpu_processes[name]
        if not replicas:
            raise ValueError(f"Process {name!r} needs at least one replica")
        for ranks in replicas:
            if len(ranks) != process.tp_size:
                raise ValueError(
                    f"Process {name!r} needs tp_size={process.tp_size} ranks"
                )
            if not set(ranks) <= budget:
                raise ValueError(
                    f"Process {name!r} uses devices outside the GPU budget"
                )
        result.processes[name] = ProcessConfig(
            num_replicas=len(replicas),
            replica_devices=(
                [device for ranks in replicas for device in ranks]
                if len(replicas) > 1
                else None
            ),
        )
        for stage in result.stages:
            if stage.name in process.stage_names and stage.gpu is not None:
                stage.gpu = list(replicas[0]) if stage.tp_size > 1 else replicas[0][0]
    result = type(config).model_validate(result.model_dump())
    # Note (Jiaxin Deng): expand before placement checks so every replica contributes
    # to the same memory and process constraints used by serving startup.
    plan, stage_copies = compile_logical_processes(result)
    expanded, topology = expand_replica_stages(stage_copies, plan)
    placement = build_stage_placement_plan(
        result, stages_cfg=expanded, replica_instances=topology.replicas
    )
    build_process_topology_plan(result, placement, stages_cfg=expanded)
    return result
