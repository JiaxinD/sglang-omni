# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from sglang_omni.models.qwen3_asr.config import Qwen3ASRPipelineConfig
from sglang_omni.restage.candidates import enumerate_candidates
from sglang_omni.restage.capacity import (
    CapacityCatalog,
    CapacityContext,
    capacity_requirements,
)
from sglang_omni.restage.plan import SearchSpace, write_plan


def test_real_candidate_layout_preserves_demand_identity():
    config = Qwen3ASRPipelineConfig(model_path="test-checkpoint")
    context = CapacityContext(
        model_revision="test", hardware="test", stack="test", workload="test"
    )
    candidates = list(enumerate_candidates(config, [0, 1], {"asr": [1]}))
    assert len(candidates) == 2
    groups = []
    for candidate in candidates:
        assert candidate.config is not None
        assert set(candidate.assignments) == {"asr"}
        assert all(ranks for ranks in candidate.assignments["asr"])
        requirements = capacity_requirements(
            candidate.config, candidate.assignments, context
        )
        assert len(requirements) == 1
        groups.append(requirements[0])
    assert groups[0]["configuration_hash"] == groups[1]["configuration_hash"]
    assert groups[0]["key"] != groups[1]["key"]


def test_catalog_preserves_candidates_and_emits_missing_calibration(tmp_path):
    config = Qwen3ASRPipelineConfig(model_path="test-checkpoint")
    space = SearchSpace(devices=[0, 1], replica_counts=[1, 2])
    context = CapacityContext(
        model_revision="test", hardware="test", stack="test", workload="test"
    )
    plain = tmp_path / "plain"
    pending = tmp_path / "pending"
    plain_summary = write_plan(config, space, plain)
    pending_summary = write_plan(
        config,
        space,
        pending,
        capacity_catalog=CapacityCatalog(),
        capacity_context=context,
    )
    assert pending_summary["accepted"] == plain_summary["accepted"]
    assert pending_summary["unranked"] == plain_summary["accepted"]
    assert pending_summary["predicted"] == 0
    plain_rows = [
        json.loads(s) for s in (plain / "candidates.jsonl").read_text().splitlines()
    ]
    rows = [
        json.loads(s) for s in (pending / "candidates.jsonl").read_text().splitlines()
    ]
    assert [
        {k: v for k, v in row.items() if k != "prediction"} for row in rows
    ] == plain_rows
    for row in plain_rows:
        if row["status"] == "candidate":
            assert (plain / row["config_file"]).read_bytes() == (
                pending / row["config_file"]
            ).read_bytes()
    requirements = json.loads((pending / "calibration-requirements.json").read_text())
    assert {r["key"] for r in requirements} == {
        key
        for row in rows
        if row["status"] == "candidate"
        for key in row["prediction"]["missing_groups"]
    }
    point = requirements[0]
    catalog = CapacityCatalog(
        points=[
            {
                "group_key": point["key"],
                "requests_per_s": 10,
                "run_id": "synthetic-test",
                "evidence": "synthetic-test-only",
            }
        ]
    )
    scored = write_plan(
        config,
        space,
        tmp_path / "scored",
        capacity_catalog=catalog,
        capacity_context=context,
    )
    assert scored["predicted"] >= 1
    assert scored["performance_measured"] is False
    assert scored["predicted"] + scored["unranked"] == scored["accepted"]


def test_catalog_requires_context_before_creating_output(tmp_path):
    target = tmp_path / "out"
    with pytest.raises(ValueError, match="together"):
        write_plan(
            Qwen3ASRPipelineConfig(model_path="test"),
            SearchSpace(devices=[0]),
            target,
            capacity_catalog=CapacityCatalog(),
        )
    assert not target.exists()
