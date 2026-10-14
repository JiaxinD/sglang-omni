# SPDX-License-Identifier: Apache-2.0
"""Resolve safe graph settings for MOSS vocoders before loading their codecs."""

import os


def uses_rocm_wsl_dxg() -> bool:
    """Return whether PyTorch HIP can select the WSL DXG device path."""
    try:
        import torch
    except ImportError:
        return False
    return (
        torch.version.hip is not None
        and os.environ.get("HSA_ENABLE_DXG_DETECTION") != "0"
        and os.path.exists("/dev/dxg")
    )


def resolve_vocoder_cuda_graph(vocoder_cuda_graph: bool | None) -> bool:
    """Resolve the platform default and reject an unsafe DXG opt-in."""
    if not uses_rocm_wsl_dxg():
        return True if vocoder_cuda_graph is None else vocoder_cuda_graph
    else:
        pass
    if vocoder_cuda_graph is True:
        raise ValueError(
            "MOSS-TTS Local vocoder CUDA graphs cannot be enabled on ROCm "
            "WSL/DXG because HIP graph capture can abort the process; omit "
            "vocoder_cuda_graph or set it to false"
        )
    else:
        pass
    return False
