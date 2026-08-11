# SPDX-License-Identifier: Apache-2.0
"""Build a CPU allocation plan for a resolved pipeline topology."""

from __future__ import annotations

import json
import logging

from sglang_omni.config.placement import StagePlacementPlan
from sglang_omni.config.topology import ProcessTopologyPlan
from sglang_omni.cpu_alloc.allocator import (
    CpuAllocationPlan,
    ProcessCpuDemand,
    allocate,
)
from sglang_omni.cpu_alloc.cost import resolve_stage_cpu_costs
from sglang_omni.cpu_alloc.topology import CpuTopology, discover_topology
from sglang_omni.utils.cpu import cgroup_cpu_quota_count

logger = logging.getLogger(__name__)


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


def _iter_process_entries(process_plan: ProcessTopologyPlan):
    """Yield (process_name, stage_names) per OS process, TP ranks included."""
    for group in process_plan.groups:
        yield group.name, list(group.stage_names)
    for stage_name, process_names in process_plan.tp_stage_to_processes.items():
        for process_name in process_names:
            yield process_name, [stage_name]


def build_pipeline_cpu_plan(
    config,
    *,
    placement_plan: StagePlacementPlan,
    process_plan: ProcessTopologyPlan,
    topology: CpuTopology | None = None,
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
    entries = sorted(_iter_process_entries(process_plan), key=lambda e: e[0])
    for process_name, stage_names in entries:
        exclusive = sum(
            costs[s].exclusive_cores
            for s in stage_names
            if s in costs and costs[s].host_class == "serial-loop"
        )
        if process_name in replicated and exclusive:
            logger.warning(
                "cpu_alloc: process %s is replicated; per-replica planning is "
                "not supported yet, keeping it in the shared pool",
                process_name,
            )
            exclusive = 0
        demands.append(
            ProcessCpuDemand(
                process_name=process_name,
                numa_node=None,
                exclusive_cores=exclusive,
            )
        )

    plan = allocate(topology, demands)
    logger.info("cpu_alloc plan: %s", json.dumps(plan.to_dict(), sort_keys=True))
    return plan
