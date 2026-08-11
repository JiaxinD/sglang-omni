# SPDX-License-Identifier: Apache-2.0
import pytest

from sglang_omni.cpu_alloc.allocator import ProcessCpuDemand, allocate
from sglang_omni.cpu_alloc.topology import discover_topology


@pytest.fixture
def topology(dual_node_sysfs):
    return discover_topology(range(16), sysfs_root=dual_node_sysfs)


def demand(name, node=0, serial=0, pool=0):
    return ProcessCpuDemand(
        process_name=name, numa_node=node, serial_cores=serial, pool_cores=pool
    )


class TestAllocate:
    def test_exclusive_gets_whole_physical_cores(self, topology):
        plan = allocate(topology, [demand("ar", serial=1), demand("shared")])
        ar = plan.assignments["ar"]
        assert ar.exclusive
        assert ar.cpu_ids == (0, 8)  # both SMT siblings of one core
        assert plan.events == ()

    def test_shared_process_gets_node_remainder(self, topology):
        plan = allocate(topology, [demand("ar", serial=1), demand("shared")])
        shared = plan.assignments["shared"]
        assert not shared.exclusive
        assert shared.cpu_ids == (1, 2, 3, 9, 10, 11)
        assert 0 not in shared.cpu_ids and 8 not in shared.cpu_ids

    def test_numa_anchoring(self, topology):
        plan = allocate(
            topology, [demand("a", node=0, serial=1), demand("b", node=1, serial=1)]
        )
        assert plan.assignments["a"].cpu_ids == (0, 8)
        assert plan.assignments["b"].cpu_ids == (4, 12)
        assert plan.assignments["a"].numa_node == 0
        assert plan.assignments["b"].numa_node == 1

    def test_unknown_numa_stays_shared_with_event(self, topology):
        # A guessed anchor could pin a GPU process to the wrong socket, so an
        # unresolvable node keeps today's shared behavior instead.
        plan = allocate(topology, [demand("a", node=7, serial=1)])
        assert not plan.assignments["a"].exclusive
        assert any("node 7" in event for event in plan.events)

    def test_unanchored_exclusive_demand_stays_shared(self, topology):
        plan = allocate(topology, [demand("a", node=None, serial=1)])
        assert not plan.assignments["a"].exclusive
        assert plan.assignments["a"].cpu_ids == tuple(range(16))
        assert any("no resolvable NUMA anchor" in event for event in plan.events)

    def test_pool_width_shrinks_before_serial(self, topology):
        # Node 0 has 4 physical cores; 1 stays shared, budget = 3.
        plan = allocate(
            topology,
            [demand("loop", serial=1), demand("pool", pool=4)],
        )
        loop = plan.assignments["loop"]
        pool = plan.assignments["pool"]
        assert loop.exclusive and len(loop.cpu_ids) == 2
        assert pool.exclusive and len(pool.cpu_ids) == 4  # shrunk 4 -> 2 cores
        assert any("shrank pool width" in event for event in plan.events)

    def test_pool_moves_to_shared_before_serial(self, topology):
        # Budget 3: three serial demands + one pool cannot all fit.
        plan = allocate(
            topology,
            [
                demand("s1", serial=1),
                demand("s2", serial=1),
                demand("s3", serial=1),
                demand("p1", pool=2),
            ],
        )
        assert all(plan.assignments[n].exclusive for n in ("s1", "s2", "s3"))
        assert not plan.assignments["p1"].exclusive
        assert any("moved to the shared pool" in event for event in plan.events)

    def test_serial_degrades_last_and_is_logged(self, topology):
        plan = allocate(
            topology,
            [demand(f"s{i}", serial=1) for i in range(5)],
        )
        exclusive = [n for n, a in plan.assignments.items() if a.exclusive]
        shared = [n for n, a in plan.assignments.items() if not a.exclusive]
        assert len(exclusive) == 3 and len(shared) == 2
        assert any("exclusivity lost" in event for event in plan.events)

    def test_exclusive_grants_are_disjoint(self, topology):
        plan = allocate(
            topology,
            [demand("a", serial=1), demand("b", serial=1), demand("c", pool=1)],
        )
        seen: set[int] = set()
        for assignment in plan.assignments.values():
            if assignment.exclusive:
                assert not (seen & set(assignment.cpu_ids))
                seen.update(assignment.cpu_ids)

    def test_shared_pool_never_empty(self, topology):
        plan = allocate(topology, [demand(f"s{i}", serial=1) for i in range(8)])
        assert plan.shared_pools[0]  # min_shared_physical_cores reserved

    def test_deterministic(self, topology):
        demands = [demand("b", serial=1), demand("a", serial=1), demand("z")]
        first = allocate(topology, demands)
        second = allocate(topology, list(reversed(demands)))
        assert first.to_dict() == second.to_dict()

    def test_duplicate_names_raise(self, topology):
        with pytest.raises(ValueError, match="Duplicate"):
            allocate(topology, [demand("a"), demand("a")])

    def test_negative_demand_raises(self, topology):
        with pytest.raises(ValueError, match="must be >= 0"):
            ProcessCpuDemand(process_name="a", numa_node=0, serial_cores=-1)

    def test_no_anchor_shared_gets_union(self, topology):
        plan = allocate(topology, [demand("cpuonly", node=None)])
        assert plan.assignments["cpuonly"].cpu_ids == tuple(range(16))

    def test_to_dict_shape(self, topology):
        plan = allocate(topology, [demand("ar", serial=1)])
        data = plan.to_dict()
        assert data["assignments"]["ar"] == {"cpus": "0,8", "exclusive": True}
        assert "0" in data["shared_pools"] and "None" in data["shared_pools"]
