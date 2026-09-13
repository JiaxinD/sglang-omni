"""Search configuration changes use the same typed patches as serving."""

from sglang_omni.config.topology import compile_logical_processes
from sglang_omni.restage.candidates import enumerate_candidates
from sglang_omni.restage.configurations import enumerate_configurations
from tests.unit_test.fixtures.restage import pipeline


def test_tp_and_memory_choices_form_independent_dimensions():
    source = pipeline()
    before = source.model_dump()
    variants = list(
        enumerate_configurations(
            source,
            {
                "tp": [
                    {"engine.tp_size": 1, "engine.gpu": [0]},
                    {"engine.tp_size": 2, "engine.gpu": [0, 1]},
                ],
                "memory": [
                    {"engine.gpu_memory_fraction": 0.4},
                    {"engine.gpu_memory_fraction": 0.6},
                ],
            },
        )
    )
    assert len(variants) == 4
    assert all(v.config is not None for v in variants)
    assert {
        (
            v.config.stage_named("engine").tp_size,
            v.config.stage_named("engine").gpu_memory_fraction,
        )
        for v in variants
    } == {(1, 0.4), (1, 0.6), (2, 0.4), (2, 0.6)}
    assert source.model_dump() == before


def test_process_grouping_is_a_search_dimension():
    variants = list(
        enumerate_configurations(
            pipeline(),
            {
                "group": [{}, {"tail.process": "engine"}],
            },
        )
    )
    assert [v.config.stage_named("tail").process for v in variants] == [
        "tail",
        "engine",
    ]


def test_conflicting_choices_are_reported_not_silently_overwritten():
    variants = list(
        enumerate_configurations(
            pipeline(),
            {
                "a": [{"engine.tp_size": 1}],
                "b": [{"engine.tp_size": 2}],
            },
        )
    )
    assert variants[0].config is None
    assert "set twice" in variants[0].rejection


def test_internal_topology_edits_stay_forbidden():
    variant = next(
        enumerate_configurations(
            pipeline(),
            {
                "bad": [{"engine.factory_path": "untrusted.factory"}],
            },
        )
    )
    assert variant.config is None
    assert "internal" in variant.rejection


def test_grouping_and_tp_variants_feed_budget_enumeration():
    source = pipeline()
    source.stages[1].gpu_memory_fraction = 0.2
    variants = enumerate_configurations(
        source,
        {
            "topology": [
                {},
                {"tail.process": "engine"},
                {"engine.tp_size": 2, "engine.gpu": [0, 1]},
            ],
        },
    )
    accepted_counts = []
    for variant in variants:
        assert variant.config is not None
        logical, stages = compile_logical_processes(variant.config)
        gpu_stages = {s.name for s in stages if s.gpu is not None}
        counts = {
            p.name: (1,)
            for p in logical.processes
            if gpu_stages.intersection(p.stage_names)
        }
        accepted_counts.append(
            sum(
                c.config is not None
                for c in enumerate_candidates(variant.config, (2, 5), counts)
            )
        )
    assert accepted_counts == [4, 2, 4]
