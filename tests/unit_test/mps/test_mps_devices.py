# SPDX-License-Identifier: Apache-2.0
"""Parent-process CUDA identity and NVML capability tests."""

from __future__ import annotations

import sys
import uuid
from enum import IntEnum
from types import SimpleNamespace

import pytest

import sglang_omni.mps.devices as devices_module
from sglang_omni.mps.devices import NvmlDeviceInfo, _resolve_cuda_device_uuids


GPU_A = "GPU-aaaaaaaa-bbbb-cccc-dddd-000000000001"
GPU_B = "GPU-aaaaaaaa-bbbb-cccc-dddd-000000000007"


class _CudaStatus(IntEnum):
    SUCCESS = 0
    INVALID_DEVICE = 101


class _FakeDriver:
    def __init__(self, uuids: dict[int, str]):
        self.uuids = uuids
        self.calls: list[tuple[str, object]] = []

    def cuInit(self, flags):
        self.calls.append(("cuInit", flags))
        return (_CudaStatus.SUCCESS,)

    def cuDeviceGet(self, ordinal):
        self.calls.append(("cuDeviceGet", ordinal))
        if ordinal not in self.uuids:
            return _CudaStatus.INVALID_DEVICE, None
        return _CudaStatus.SUCCESS, ordinal

    def cuDeviceGetUuid(self, device):
        self.calls.append(("cuDeviceGetUuid", device))
        raw_uuid = uuid.UUID(self.uuids[device].removeprefix("GPU-")).bytes
        return _CudaStatus.SUCCESS, SimpleNamespace(bytes=raw_uuid)


def _fake_pynvml(failed_uuids: set[str] | None = None):
    class NvmlError(Exception):
        pass

    class NvmlNotSupported(NvmlError):
        pass

    handles: list[str] = []
    failures = failed_uuids or set()

    def by_uuid(raw_uuid):
        gpu_uuid = raw_uuid.decode()
        if gpu_uuid in failures:
            raise NvmlError(f"cannot inspect {gpu_uuid}")
        handles.append(gpu_uuid)
        return gpu_uuid

    return SimpleNamespace(
        handles=handles,
        NVMLError=NvmlError,
        NVMLError_NotSupported=NvmlNotSupported,
        NVML_DEVICE_MIG_ENABLE=1,
        nvmlInit=lambda: None,
        nvmlDeviceGetHandleByUUID=by_uuid,
        nvmlDeviceGetUUID=lambda handle: handle,
        nvmlDeviceGetCudaComputeCapability=lambda _handle: (9, 0),
        nvmlDeviceGetMigMode=lambda _handle: (0, 0),
    )


def test_parent_driver_resolves_unique_ordinals_with_one_init():
    driver = _FakeDriver({0: GPU_A, 1: GPU_B})

    resolved, errors = _resolve_cuda_device_uuids([1, 0, 1], driver)

    assert resolved == {
        0: GPU_A,
        1: GPU_B,
    }
    assert errors == {}
    assert driver.calls.count(("cuInit", 0)) == 1
    assert {value for name, value in driver.calls if name == "cuDeviceGet"} == {
        0,
        1,
    }


def test_parent_driver_preserves_success_when_another_ordinal_is_invalid():
    resolved, errors = _resolve_cuda_device_uuids(
        [0, 9],
        _FakeDriver({0: GPU_A}),
    )

    assert resolved == {0: GPU_A}
    assert "cuDeviceGet(9)" in errors[9]
    assert "INVALID_DEVICE" in errors[9]


def test_nvml_uses_driver_uuid_when_cuda_order_differs_from_nvml_index(monkeypatch):
    pynvml = _fake_pynvml()
    monkeypatch.setitem(sys.modules, "pynvml", pynvml)
    calls: list[tuple[int, ...]] = []

    def resolve(ordinals):
        calls.append(tuple(ordinals))
        return {0: GPU_B, 1: GPU_A}, {}

    monkeypatch.setattr(devices_module, "_resolve_cuda_device_uuids", resolve)

    devices = NvmlDeviceInfo().inspect([1, 0, 1])

    assert {ordinal: device.gpu_uuid for ordinal, device in devices.items()} == {
        0: GPU_B,
        1: GPU_A,
    }
    assert calls == [(0, 1)]
    assert pynvml.handles == [GPU_B, GPU_A]


def test_cuda_resolution_failure_never_falls_back_to_nvml_index(monkeypatch):
    pynvml = _fake_pynvml()
    monkeypatch.setitem(sys.modules, "pynvml", pynvml)

    def fail(_ordinals):
        return {}, {2: "cuDeviceGet(2) failed with INVALID_DEVICE"}

    monkeypatch.setattr(devices_module, "_resolve_cuda_device_uuids", fail)

    device = NvmlDeviceInfo().inspect([2])[2]

    assert device.gpu_uuid is None
    assert "INVALID_DEVICE" in device.unsupported_reason
    assert pynvml.handles == []


def test_nvml_failure_preserves_driver_resolved_uuid(monkeypatch):
    pynvml = _fake_pynvml({GPU_B})
    monkeypatch.setitem(sys.modules, "pynvml", pynvml)
    monkeypatch.setattr(
        devices_module,
        "_resolve_cuda_device_uuids",
        lambda _ordinals: ({0: GPU_A, 1: GPU_B}, {}),
    )

    inspected = NvmlDeviceInfo().inspect([0, 1])

    assert inspected[0] == devices_module.MpsPhysicalDevice(GPU_A)
    assert inspected[1].gpu_uuid == GPU_B
    assert "NVML query failed" in inspected[1].unsupported_reason
