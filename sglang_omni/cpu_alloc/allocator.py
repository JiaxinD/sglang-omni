# SPDX-License-Identifier: Apache-2.0
"""Two-pool CPU allocation over physical cores.

Exclusive demands are granted whole physical cores (all SMT siblings move
together, so no foreign thread lands on a sibling). Everything else shares
the remaining CPUs of its NUMA node. When a node cannot satisfy every
exclusive demand, the processes that do not fit move to the shared pool,
which is today's behavior; it never silently oversubscribes an exclusive
grant, and every degradation is recorded in ``CpuAllocationPlan.events``.
"""

from __future__ import annotations

from dataclasses import dataclass

from sglang_omni.cpu_alloc.topology import CpuTopology, PhysicalCore, format_cpulist


@dataclass(frozen=True)
class ProcessCpuDemand:
    """Aggregated exclusive-core demand of one OS process."""

    process_name: str
    numa_node: int | None
    exclusive_cores: int = 0

    def __post_init__(self) -> None:
        if self.exclusive_cores < 0:
            raise ValueError(f"Process {self.process_name!r}: core demand must be >= 0")


@dataclass(frozen=True)
class ProcessCpuAssignment:
    process_name: str
    cpu_ids: tuple[int, ...]
    exclusive: bool
    numa_node: int | None = None


@dataclass(frozen=True)
class CpuAllocationPlan:
    assignments: dict[str, ProcessCpuAssignment]
    shared_pools: dict[int | None, tuple[int, ...]]
    events: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "assignments": {
                name: {
                    "cpus": format_cpulist(assignment.cpu_ids),
                    "exclusive": assignment.exclusive,
                }
                for name, assignment in sorted(self.assignments.items())
            },
            "shared_pools": {
                str(node): format_cpulist(cpus)
                for node, cpus in sorted(
                    self.shared_pools.items(), key=lambda item: str(item[0])
                )
            },
            "events": list(self.events),
        }


@dataclass
class _NodeState:
    free_cores: list[PhysicalCore]
    reserved_shared: int


def _anchor_node(
    demand: ProcessCpuDemand,
    node_states: dict[int, _NodeState],
    projected: dict[int, int],
    events: list[str],
) -> int | None:
    """Node to grant exclusive cores on, or None to stay in the shared pool.

    An explicit node is honored; without one the demand goes to the node with
    the most remaining capacity, so exclusive grants spread across sockets
    instead of piling onto the first one. An explicit node that has no usable
    cores stays in the shared pool: a guessed grant could pin the process to
    the wrong socket, which is worse than not pinning.
    """
    if demand.numa_node is not None:
        if demand.numa_node in node_states:
            return demand.numa_node
        events.append(
            f"process {demand.process_name}: NUMA node {demand.numa_node} has "
            f"no usable cores in the universe; keeping it in the shared pool"
        )
        return None
    return max(
        node_states,
        key=lambda n: (
            len(node_states[n].free_cores)
            - node_states[n].reserved_shared
            - projected[n],
            -n,
        ),
    )


def allocate(
    topology: CpuTopology,
    demands: list[ProcessCpuDemand],
    *,
    min_shared_physical_cores: int = 1,
) -> CpuAllocationPlan:
    """Allocate exclusive physical cores per process and build shared pools."""
    names = [d.process_name for d in demands]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate process names in demands: {names}")
    if min_shared_physical_cores < 1:
        raise ValueError("min_shared_physical_cores must be >= 1")

    events: list[str] = []
    node_states = {
        node: _NodeState(
            free_cores=list(topology.cores_on_node(node)),
            reserved_shared=min_shared_physical_cores,
        )
        for node in topology.numa_nodes
    }

    exclusive_demands = sorted(
        (d for d in demands if d.exclusive_cores),
        key=lambda d: d.process_name,
    )
    anchored: dict[str, int] = {}
    projected = dict.fromkeys(node_states, 0)
    for demand in exclusive_demands:
        node = _anchor_node(demand, node_states, projected, events)
        if node is not None:
            anchored[demand.process_name] = node
            projected[node] += demand.exclusive_cores
    exclusive_demands = [d for d in exclusive_demands if d.process_name in anchored]

    granted: dict[str, int] = {
        d.process_name: d.exclusive_cores for d in exclusive_demands
    }
    for node, state in node_states.items():
        local = sorted(
            (d for d in exclusive_demands if anchored[d.process_name] == node),
            key=lambda d: (d.exclusive_cores, d.process_name),
        )
        budget = len(state.free_cores) - state.reserved_shared
        for demand in local:
            if demand.exclusive_cores <= budget:
                budget -= demand.exclusive_cores
                continue
            granted[demand.process_name] = 0
            events.append(
                f"node {node}: {demand.process_name} wants "
                f"{demand.exclusive_cores} core(s) but only {max(budget, 0)} "
                f"remain; moved to the shared pool"
            )

    assignments: dict[str, ProcessCpuAssignment] = {}
    for demand in exclusive_demands:
        node = anchored[demand.process_name]
        state = node_states[node]
        count = granted[demand.process_name]
        if count == 0:
            continue
        cores, state.free_cores = (
            state.free_cores[:count],
            state.free_cores[count:],
        )
        cpu_ids = tuple(sorted(c for core in cores for c in core.cpu_ids))
        assignments[demand.process_name] = ProcessCpuAssignment(
            process_name=demand.process_name,
            cpu_ids=cpu_ids,
            exclusive=True,
            numa_node=node,
        )

    shared_pools: dict[int | None, tuple[int, ...]] = {
        node: tuple(sorted(c for core in state.free_cores for c in core.cpu_ids))
        for node, state in node_states.items()
    }
    all_shared = tuple(sorted(cpu for cpus in shared_pools.values() for cpu in cpus))
    shared_pools[None] = all_shared

    for demand in demands:
        if demand.process_name in assignments:
            continue
        node = demand.numa_node if demand.numa_node in node_states else None
        cpu_ids = shared_pools[node]
        if not cpu_ids:
            cpu_ids = all_shared
        assignments[demand.process_name] = ProcessCpuAssignment(
            process_name=demand.process_name,
            cpu_ids=cpu_ids,
            exclusive=False,
            numa_node=node,
        )

    return CpuAllocationPlan(
        assignments=assignments,
        shared_pools=shared_pools,
        events=tuple(events),
    )
