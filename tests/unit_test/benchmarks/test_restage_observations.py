from benchmarks.benchmarker.data import RequestResult
from benchmarks.benchmarker.restage import to_observation
from sglang_omni.restage.evaluation import SLO, evaluate


def test_adapter_uses_absolute_audio_time_including_dispatch_lag():
    result = RequestResult(
        request_id="a",
        is_success=True,
        latency_s=0.2,
        audio_ttfp_s=0.05,
        audio_duration_s=2,
        scheduled_s=10,
        dispatched_s=11,
        completed_s=11.2,
        first_audio_s=11.05,
    )
    observation = to_observation(result, quality_pass=True)
    verdict = evaluate(
        [observation], SLO(max_ttfa_s=0.5), expected_requests=1, elapsed_s=1.2
    )
    assert not verdict.feasible
    assert observation.first_output_s == 11.05


def test_legacy_relative_audio_time_is_not_treated_as_absolute():
    result = RequestResult(
        request_id="a",
        is_success=True,
        audio_ttfp_s=0.05,
        scheduled_s=10,
        dispatched_s=11,
        completed_s=11.2,
    )
    observation = to_observation(result, quality_pass=None)
    assert observation.first_output_s is None
    assert observation.quality_pass is None
