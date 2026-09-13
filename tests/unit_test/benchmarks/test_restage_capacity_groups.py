# SPDX-License-Identifier: Apache-2.0
import copy

import pytest

from sglang_omni.restage.capacity import (
    CapacityCatalog,
    CapacityContext,
    capacity_requirements,
    capacity_resource_groups,
    predict_group_capacity,
)


class Config:
    def __init__(self):
        self.data = {
            "stages": [{"name": "a", "gpu": 0, "tp_size": 1, "batch": 8}],
            "processes": {"a": {"num_replicas": 1, "replica_devices": None}},
            "chunk_size": 16,
        }

    def model_dump(self, **kwargs):
        return copy.deepcopy(self.data)


def context():
    return CapacityContext(
        model_revision="rev", hardware="host", stack="tree", workload="inputs"
    )


def test_signature_separates_group_layout_from_global_demand():
    config = Config()
    first = capacity_requirements(config, {"a": ((0,),)}, context())[0]
    config.data["stages"][0]["gpu"] = 2
    config.data["processes"]["a"]["num_replicas"] = 2
    moved = capacity_requirements(config, {"a": ((2,), (3,))}, context())[0]
    assert first["configuration_hash"] == moved["configuration_hash"]
    assert first["key"] != moved["key"]
    config.data["chunk_size"] = 32
    changed = capacity_requirements(config, {"a": ((0,),)}, context())[0]
    assert changed["configuration_hash"] != first["configuration_hash"]


def test_all_context_fields_invalidate_matching_and_config_is_unchanged():
    config = Config()
    before = config.model_dump()
    original = capacity_requirements(config, {"a": ((0,),)}, context())[0]
    for field in CapacityContext.model_fields:
        changed_context = context().model_copy(update={field: "different"})
        changed = capacity_requirements(config, {"a": ((0,),)}, changed_context)[0]
        assert changed["key"] != original["key"]
    assert config.data == before


def test_prediction_requires_all_groups_and_reports_the_bottleneck():
    requirements = [{"key": "first"}, {"key": "second"}]
    catalog = CapacityCatalog(
        points=[
            {
                "group_key": "first",
                "requests_per_s": 10,
                "run_id": "a",
                "evidence": "a.json",
            },
            {
                "group_key": "second",
                "requests_per_s": 4,
                "run_id": "b",
                "evidence": "b.json",
            },
        ]
    )
    prediction = predict_group_capacity(requirements, catalog)
    assert prediction["requests_per_s"] == 4
    assert prediction["bottleneck_groups"] == ["second"]
    assert prediction["basis"] == "gpu_group_min"
    assert predict_group_capacity(requirements, CapacityCatalog()) == {
        "status": "unranked",
        "reason": "missing_groups",
        "missing_groups": ["first", "second"],
    }
    assert predict_group_capacity([], catalog)["reason"] == "no_gpu_groups"


def test_catalog_rejects_ambiguous_or_nonfinite_measurements():
    point = {
        "group_key": "group",
        "requests_per_s": 1,
        "run_id": "run",
        "evidence": "result",
    }
    with pytest.raises(ValueError, match="one point"):
        CapacityCatalog(points=[point, point])
    for value in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            CapacityCatalog(points=[{**point, "requests_per_s": value}])


def test_dedicated_replicas_form_one_capacity_pool():
    groups = capacity_resource_groups({"asr": ((0,), (1,)), "tail": ((2,),)})
    assert [[(r.process, r.replica, r.ranks) for r in g] for g in groups] == [
        [("asr", 0, (0,)), ("asr", 1, (1,))],
        [("tail", 0, (2,))],
    ]


def test_partial_tp_overlap_connects_the_entire_resource_group():
    assignments = {"thinker": ((1, 0),), "bridge": ((1, 2),), "tail": ((2,),)}
    groups = capacity_resource_groups(assignments)
    assert len(groups) == 1
    assert {r.process for r in groups[0]} == {"thinker", "bridge", "tail"}
    assert next(r for r in groups[0] if r.process == "thinker").ranks == (1, 0)
    assert assignments["thinker"] == ((1, 0),)


def test_colocated_replicas_preserve_multiplicity():
    groups = capacity_resource_groups({"asr": ((0,), (0,))})
    assert len(groups) == 1
    assert [(r.process, r.replica) for r in groups[0]] == [("asr", 0), ("asr", 1)]


def test_group_order_is_independent_of_mapping_insertion_order():
    assert capacity_resource_groups(
        {"z": ((2,),), "a": ((0,),)}
    ) == capacity_resource_groups({"a": ((0,),), "z": ((2,),)})
    assert capacity_resource_groups({}) == ()


def test_replica_pool_and_gpu_overlap_merge_transitively():
    groups = capacity_resource_groups(
        {"a": ((0,), (1,)), "b": ((1,), (2,)), "c": ((3,),)}
    )
    assert [[r.process for r in group] for group in groups] == [
        ["a", "a", "b", "b"],
        ["c"],
    ]
