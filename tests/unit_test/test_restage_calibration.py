import pytest

from sglang_omni.restage.calibration import ServiceSample, fit_service_law


def sample(key, context, audio, extra=0):
    return ServiceSample(
        key,
        context,
        audio,
        0.2 + 0.001 * context + 0.000002 * context**2 + 0.1 * audio + extra,
    )


def test_service_fit_recovers_law_and_reports_independent_prediction_error():
    training = [
        sample(str(i), context, audio)
        for i, (context, audio) in enumerate(
            [(0, 1), (100, 1), (200, 1), (0, 4), (150, 2)]
        )
    ]
    heldout = [sample("holdout", 50, 3, extra=0.1)]
    fitted = fit_service_law(
        training,
        validation=heldout,
        source="synthetic-unit-test",
        delta_floor_s=0.01,
        prefill_rate=1000,
    )
    assert fitted.law.a == pytest.approx(0.2)
    assert fitted.law.b == pytest.approx(0.001)
    assert fitted.law.c == pytest.approx(0.000002)
    assert fitted.law.audio_slope == pytest.approx(0.1)
    assert fitted.training_mae_s == pytest.approx(0, abs=1e-10)
    assert fitted.validation_mae_s == pytest.approx(0.1)
    assert fitted.validation_predictions[0]["observed_s"] == heldout[0].service_s
    assert fitted.validation_predictions[0]["predicted_s"] == pytest.approx(
        heldout[0].service_s - 0.1
    )


def test_calibration_rejects_reused_validation_requests_and_unidentifiable_design():
    rows = [sample(str(i), 100, 1) for i in range(5)]
    with pytest.raises(ValueError, match="disjoint"):
        fit_service_law(
            rows,
            validation=rows[:1],
            source="test",
            delta_floor_s=0.01,
            prefill_rate=1000,
        )
    with pytest.raises(ValueError, match="identify"):
        fit_service_law(
            rows, validation=[], source="test", delta_floor_s=0.01, prefill_rate=1000
        )
