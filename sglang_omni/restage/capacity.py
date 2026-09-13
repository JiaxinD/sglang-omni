# SPDX-License-Identifier: Apache-2.0
"""Resource groups for calibrated Restage capacity predictions."""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CapacityContext(BaseModel):
    """Caller-verified identities of the calibration target."""

    model_config = ConfigDict(extra="forbid")
    model_revision: str = Field(min_length=1)
    hardware: str = Field(min_length=1)
    stack: str = Field(min_length=1)
    workload: str = Field(min_length=1)


class CapacityPoint(BaseModel):
    """Measured aggregate group capacity at the target per-request demand.

    A source must measure the whole replica pool, with sufficient supplied work.
    This record does not turn a host duration or a load-limited run into capacity.
    """

    model_config = ConfigDict(extra="forbid")
    group_key: str = Field(min_length=1)
    requests_per_s: float = Field(gt=0, allow_inf_nan=False)
    run_id: str = Field(min_length=1)
    evidence: str = Field(min_length=1)


class CapacityCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid")
    points: list[CapacityPoint] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_groups(self):
        keys = [point.group_key for point in self.points]
        if len(set(keys)) != len(keys):
            raise ValueError("Capacity catalog must have one point per group key")
        return self


def _digest(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def capacity_requirements(config, assignments, context: CapacityContext) -> list[dict]:
    """Identify GPU capacity groups without assuming device equivalence.

    Keep the full pipeline's non-placement parameters: an upstream change can
    change downstream demand. Only GPU assignment and replica placement/count
    move from this global identity into each group's members. Stack identity
    must cover the implementation and its model-specific factory defaults.
    """
    data = config.model_dump(mode="json")
    for stage in data["stages"]:
        stage["gpu"] = "assigned" if stage["gpu"] is not None else None
    for process in data["processes"].values():
        process.pop("num_replicas", None)
        process.pop("replica_devices", None)
    configuration_hash = _digest(
        {
            "class": f"{type(config).__module__}.{type(config).__qualname__}",
            "config": data,
        }
    )
    requirements = []
    for group in capacity_resource_groups(assignments):
        signature = {
            "context": context.model_dump(),
            "configuration_hash": configuration_hash,
            "members": [asdict(replica) for replica in group],
        }
        requirements.append({"key": _digest(signature), **signature})
    return requirements


def predict_group_capacity(requirements: list[dict], catalog: CapacityCatalog) -> dict:
    """Combine measured GPU groups as a bottleneck proxy, not an SLO verdict."""
    points = {point.group_key: point for point in catalog.points}
    missing = [group["key"] for group in requirements if group["key"] not in points]
    if missing or not requirements:
        return {
            "status": "unranked",
            "reason": "missing_groups" if missing else "no_gpu_groups",
            "missing_groups": missing,
        }
    matched = [points[group["key"]] for group in requirements]
    minimum = min(point.requests_per_s for point in matched)
    return {
        "status": "predicted",
        "basis": "gpu_group_min",
        "requests_per_s": minimum,
        "bottleneck_groups": [
            point.group_key for point in matched if point.requests_per_s == minimum
        ],
        "groups": [point.model_dump() for point in matched],
        "scope": "GPU-group bottleneck proxy; CPU, transport and cross-group coupling unmodeled; not measured candidate capacity or SLO feasibility",
    }


@dataclass(frozen=True)
class ProcessReplica:
    process: str
    replica: int
    ranks: tuple[int, ...]


def capacity_resource_groups(
    assignments: Mapping[str, Sequence[Sequence[int]]],
) -> tuple[tuple[ProcessReplica, ...], ...]:
    """Group validated candidates by GPU overlap and process replica pools.

    Inputs come from Candidate.assignments in the visible-device namespace.
    Preserve TP rank ordering and replica multiplicity. These groups describe
    GPU contention and replicas serving the same logical process. A replica
    pool must be calibrated together instead of taking the minimum of its
    single-replica capacities. CPU and cross-group transport coupling remain
    outside this partition. No capacity or device equivalence is inferred here.
    """
    replicas = [
        ProcessReplica(name, index, tuple(ranks))
        for name in sorted(assignments)
        for index, ranks in enumerate(assignments[name])
    ]
    remaining = set(range(len(replicas)))
    groups = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        members = [first]
        pending = [first]
        while pending:
            current = pending.pop()
            devices = set(replicas[current].ranks)
            # Note (Jiaxin Deng): overlapping TP tuples share a resource even
            # when the tuples differ; merging only equal tuples double-counts it.
            neighbors = sorted(
                index
                for index in remaining
                if replicas[index].process == replicas[current].process
                or devices.intersection(replicas[index].ranks)
            )
            remaining.difference_update(neighbors)
            members.extend(neighbors)
            pending.extend(neighbors)
        groups.append(tuple(replicas[index] for index in sorted(members)))
    return tuple(groups)
