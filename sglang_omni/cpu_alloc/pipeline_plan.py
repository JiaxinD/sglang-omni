# SPDX-License-Identifier: Apache-2.0
"""Build a CPU allocation plan for a resolved pipeline topology."""

from __future__ import annotations

import json
import logging
import os
from collections import Counter

from sglang_omni.config.placement import StagePlacementPlan
from sglang_omni.config.topology import ProcessTopologyPlan
from sglang_omni.cpu_alloc.allocator import (
    CpuAllocationPlan,
    ProcessCpuDemand,
    allocate,
)
from sglang_omni.cpu_alloc.cost import resolve_stage_cpu_costs
from sglang_omni.cpu_alloc.topology import (
    CpuTopology,
    discover_topology,
    gpu_numa_nodes,
)
from sglang_omni.utils.cpu import cgroup_cpu_quota_count

logger = logging.getLogger(__name__)


def _logical_to_physical_gpus(logical_ids: set[int]) -> dict[int, int] | None:
    """Map placement GPU ids to physical indices via CUDA_VISIBLE_DEVICES."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or not visible.strip():
        return {gpu: gpu for gpu in logical_ids}
    entries = [part.strip() for part in visible.split(",") if part.strip()]
    if not all(entry.isdigit() for entry in entries):
        # UUID-based masks cannot be mapped to nvidia-smi indices here.
        return None
    physical = [int(entry) for entry in entries]
    mapping: dict[int, int] = {}
    for gpu in logical_ids:
        if gpu >= len(physical):
            return None
        mapping[gpu] = physical[gpu]
    return mapping


def _replicated_process_names(process_plan) -> set[str]:
    """Names of process groups that expand into multiple replicas.

    Duck-typed against the process-replica work (whole-process
    ``num_replicas``): an exclusive grant computed once per logical process
    would be inherited by every replica and stop being exclusive, so those
    groups stay in the shared pool until per-replica planning lands.
    """
    processes = getattr(process_plan, "processes", None)
    names = getattr(processes, "replicated_process_names", None)
    if callable(names):
        try:
            names = names()
        except TypeError:
            return set()
    if names is None:
        return set()
    return {str(name) for name in names}


def _iter_process_entries(
    process_plan: ProcessTopologyPlan,
    placement_plan: StagePlacementPlan,
):
    """Yield (process_name, stage_names, anchor_gpu_ids) per OS process.

    TP ranks each own a process and anchor to their own rank's GPU, not to
    the stage's whole GPU list.
    """
    for group in process_plan.groups:
        if group.gpu_id is not None:
            gpus = [group.gpu_id]
        else:
            gpus = [
                gpu_id
                for stage_name in group.stage_names
                for gpu_id in (
                    placement_plan.stages[stage_name].gpu_ids
                    if stage_name in placement_plan.stages
                    else ()
                )
            ]
        yield group.name, list(group.stage_names), gpus
    for stage_name, process_names in process_plan.tp_stage_to_processes.items():
        placement = placement_plan.stages.get(stage_name)
        for rank, process_name in enumerate(process_names):
            rank_gpu = (
                placement.gpu_ids[rank]
                if placement is not None and rank < len(placement.gpu_ids)
                else None
            )
            yield process_name, [stage_name], (
                [rank_gpu] if rank_gpu is not None else []
            )


def _majority_numa_node(
    gpu_ids: list[int],
    gpu_numa: dict[int, int | None],
) -> int | None:
    nodes: Counter[int] = Counter()
    for gpu_id in gpu_ids:
        node = gpu_numa.get(gpu_id)
        if node is not None:
            nodes[node] += 1
    if not nodes:
        return None
    return nodes.most_common(1)[0][0]


def build_pipeline_cpu_plan(
    config,
    *,
    placement_plan: StagePlacementPlan,
    process_plan: ProcessTopologyPlan,
    topology: CpuTopology | None = None,
    gpu_numa: dict[int, int | None] | None = None,
) -> CpuAllocationPlan | None:
    """Build the per-process CPU plan, or None when planning cannot help.

    Returns None (with a log line) when the model declares no stage costs or
    when the host topology cannot be discovered, so enabling the allocator on
    an unsupported setup never changes behavior.
    """
    costs = resolve_stage_cpu_costs(config)
    if not costs:
        logger.info(
            "cpu_alloc: %s declares no stage_cpu_costs(); allocator is a no-op",
            type(config).__name__,
        )
        return None

    if topology is None:
        try:
            topology = discover_topology()
        except (OSError, RuntimeError, ValueError, AttributeError) as exc:
            logger.warning("cpu_alloc: topology discovery failed, disabled: %s", exc)
            return None

    if gpu_numa is None:
        logical_gpus = set(placement_plan.gpus)
        mapping = _logical_to_physical_gpus(logical_gpus)
        if mapping is None:
            logger.warning(
                "cpu_alloc: cannot map CUDA_VISIBLE_DEVICES to physical GPU "
                "indices; planning without NUMA anchoring"
            )
            gpu_numa = {gpu: None for gpu in logical_gpus}
        else:
            physical_numa = gpu_numa_nodes(mapping.values())
            gpu_numa = {
                logical: physical_numa.get(physical)
                for logical, physical in mapping.items()
            }

    quota = cgroup_cpu_quota_count()
    if quota is not None and quota < len(topology.universe):
        logger.warning(
            "cpu_alloc: cgroup CPU quota (%s CPUs) is below the affinity "
            "universe (%d CPUs); exclusive grants cannot guarantee cycles",
            quota,
            len(topology.universe),
        )

    replicated = _replicated_process_names(process_plan)
    demands = []
    entries = sorted(
        _iter_process_entries(process_plan, placement_plan), key=lambda e: e[0]
    )
    for process_name, stage_names, anchor_gpus in entries:
        serial = sum(
            costs[s].exclusive_cores
            for s in stage_names
            if s in costs and costs[s].host_class == "serial-loop"
        )
        pool = sum(
            costs[s].pool_width or 0
            for s in stage_names
            if s in costs and costs[s].host_class == "parallel-pool"
        )
        if process_name in replicated and (serial or pool):
            logger.warning(
                "cpu_alloc: process %s is replicated; per-replica planning is "
                "not supported yet, keeping it in the shared pool",
                process_name,
            )
            serial = pool = 0
        demands.append(
            ProcessCpuDemand(
                process_name=process_name,
                numa_node=_majority_numa_node(anchor_gpus, gpu_numa),
                serial_cores=serial,
                pool_cores=pool,
            )
        )

    plan = allocate(topology, demands)
    logger.info("cpu_alloc plan: %s", json.dumps(plan.to_dict(), sort_keys=True))
    return plan
