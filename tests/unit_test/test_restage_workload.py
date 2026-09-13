"""Contracts for Restage's migrated service-time predictor."""

import pytest

from sglang_omni.restage.workload import SGLANG_OMNI_QWEN3_LAW, ServiceLaw


def test_historical_anchor_is_preserved():
    law = SGLANG_OMNI_QWEN3_LAW
    expected = 0.692 + 2.634e-9 * 6343**2
    assert law.s(6343) == pytest.approx(expected)
    assert law.T_sat(4.5, 6343) == pytest.approx(4.5 / expected)
    assert "Blackwell" in law.source


def test_output_duration_changes_service_time_off_anchor():
    law = SGLANG_OMNI_QWEN3_LAW
    short = 0.2078 + 2.634e-9 * 512**2 + 0.1076 * 2
    long = 0.2078 + 2.634e-9 * 512**2 + 0.1076 * 8
    assert law.s(512, 2) == pytest.approx(short)
    assert law.s(512, 8) == pytest.approx(long)
    assert law.T_sat(8, 512) == pytest.approx(8 / long)
    assert law.T_sat(8, 512) < 4 * law.T_sat(2, 512)


def test_context_only_law_and_stall_floor_remain_supported():
    law = ServiceLaw(1, 0.01, 0, 0.1, 1000, source="test calibration")
    assert law.s(100, 8) == pytest.approx(2)
    assert law.T_sat(8, 100) == pytest.approx(4)
    assert law.delta(10) == pytest.approx(0.1)
    assert law.delta(1000) == pytest.approx(1)
