# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.profiler.work_units import annotate_execution_device


def test_cuda_properties_only_annotate_new_attempts(monkeypatch):
    queried = []

    def properties(device):
        queried.append(device)
        return SimpleNamespace(name="test GPU", multi_processor_count=32)

    monkeypatch.setattr(torch.cuda, "get_device_properties", properties)
    monkeypatch.setenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "25")
    device = torch.device("cuda:2")
    records = [{"old": True}]
    with annotate_execution_device(device, records):
        records.extend([{"outcome": "raised"}, {"outcome": "host_returned"}])
    assert queried == [device]
    assert records[0] == {"old": True}
    assert records[1]["device"] == records[2]["device"]
    assert records[1]["device"]["index"] == 2
    assert records[1]["device"]["device_reported_sm_count"] == 32
    assert records[1]["device"]["mps_active_thread_percentage"] == "25"


def test_cpu_and_empty_executions_do_not_query_cuda(monkeypatch):
    def unexpected(*args):
        pytest.fail("CUDA query was not needed")

    monkeypatch.setattr(torch.cuda, "get_device_properties", unexpected)
    records = []
    with annotate_execution_device(torch.device("cpu"), records):
        records.append({})
    assert records == [{"device": {"type": "cpu", "index": None}}]
    with annotate_execution_device(torch.device("cuda:0"), records):
        pass


def test_query_failure_preserves_original_encoder_error(monkeypatch):
    def unavailable(*args):
        raise RuntimeError("driver query failed")

    monkeypatch.setattr(torch.cuda, "get_device_properties", unavailable)
    records = []
    with pytest.raises(ValueError, match="encoder error"):
        with annotate_execution_device(torch.device("cuda:0"), records):
            records.append({"outcome": "raised"})
            raise ValueError("encoder error")
    assert records[0]["device"]["query_error_type"] == "RuntimeError"
