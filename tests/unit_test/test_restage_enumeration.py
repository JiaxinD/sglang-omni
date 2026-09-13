"""Finite budget searches retain co-location and replica alternatives."""

import pytest

from sglang_omni.restage.candidates import enumerate_candidates
from tests.unit_test.fixtures.restage import pipeline


def test_two_card_search_includes_split_colocated_and_tail_replicas():
    config = pipeline()
    config.stages[1].gpu_memory_fraction = 0.2
    results = list(
        enumerate_candidates(config, (2, 5), {"engine": (1,), "tail": (1, 2, 3)})
    )
    accepted = [item for item in results if item.config is not None]
    assert any(
        item.assignments == {"engine": ((2,),), "tail": ((5,), (5,))}
        for item in accepted
    )
    assert any(
        item.assignments == {"engine": ((2,),), "tail": ((2,), (5,), (5,))}
        for item in accepted
    )
    assert any(
        item.assignments == {"engine": ((2,),), "tail": ((2,),)} for item in accepted
    )
    assert all(
        set(
            d
            for replicas in item.assignments.values()
            for ranks in replicas
            for d in ranks
        )
        <= {2, 5}
        for item in results
    )
    assert len({item.key for item in results}) == len(results)


def test_rejected_candidate_keeps_runtime_reason():
    config = pipeline()
    config.stages[1].gpu_memory_fraction = 0.8
    results = list(enumerate_candidates(config, (0,), {"engine": (1,), "tail": (1,)}))
    assert len(results) == 1
    assert results[0].config is None
    assert "exceeds placement limit" in results[0].rejection


def test_insufficient_tp_budget_explains_empty_search():
    with pytest.raises(ValueError, match="tp_size=2 exceeds"):
        list(enumerate_candidates(pipeline(2), (0,), {"engine": (1,), "tail": (1,)}))
