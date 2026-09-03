"""Per-stage SM caps backed by CUDA Green Contexts.

A stage that issues short bursts of wide kernels can hold most of the device's
SMs for milliseconds at a time, stalling a latency-sensitive stage sharing the
GPU through MPS. Setting ``sm_cap`` on the bursty stage bounds its SM footprint.

Whether a cap helps, and which stage to cap, is a property of the pipeline and
has to be measured: see ``docs/basic_usage/stage_sm_cap.md``.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang_omni.config.schema import StageConfig

# Note (Jiaxin Deng): the driver splits a device into groups of at least this
# many SMs; H100 and H200 both quantize to 8. Devices that split differently
# need GREEN_CTX_SPLIT set through the stage's `env` block.
SM_GROUP_SIZE = 8

BOOTSTRAP_ENV = "SGLANG_OMNI_SM_CAP_BOOTSTRAP"


class SmCapError(ValueError):
    """Raised when a requested SM cap cannot be applied."""


def resolve_bootstrap_path(explicit: str | None = None) -> str:
    """Return the bootstrap library path, or raise if it is not configured."""
    path = explicit or os.environ.get(BOOTSTRAP_ENV, "")
    if not path:
        raise SmCapError(
            "sm_cap needs the green-context bootstrap library: build it with "
            f"`make -C tools/green_ctx` and point ${BOOTSTRAP_ENV} at the .so"
        )
    if not os.path.isfile(path):
        raise SmCapError(f"sm_cap bootstrap library not found: {path}")
    return path


def sm_cap_env(sm_cap: int, bootstrap: str) -> dict[str, str]:
    """Return the spawn env that caps a stage process to *sm_cap* SMs.

    The cap is an upper bound, not a reservation: two capped stages may cover
    overlapping SMs, and an uncapped stage still sees the whole device.
    """
    if sm_cap <= 0 or sm_cap % SM_GROUP_SIZE:
        raise SmCapError(
            f"sm_cap={sm_cap} must be a positive multiple of {SM_GROUP_SIZE}"
        )
    return {
        "GREEN_CTX_SM": str(sm_cap),
        "GREEN_CTX_SPLIT": str(SM_GROUP_SIZE),
        "GREEN_CTX_GROUP_COUNT": str(sm_cap // SM_GROUP_SIZE),
        "LD_PRELOAD": bootstrap,
        # Note (Jiaxin Deng): kept separately because the verification below
        # must name the library it expects, not whatever LD_PRELOAD ended up as.
        BOOTSTRAP_ENV: bootstrap,
    }


def verify_sm_cap(bootstrap: str, expected_sm: int) -> int:
    """Check that this process really runs capped, and return its SM count.

    Raises ``SmCapError`` when the cap did not take effect. The check runs on a
    freshly created thread: the bootstrap binds new threads through its
    ``pthread_create`` interposer, so a thread that sees the cap proves the
    library was preloaded rather than merely loadable.
    """
    import ctypes
    import threading

    library = ctypes.CDLL(bootstrap)
    library.green_ctx_actual_sm.restype = ctypes.c_uint
    library.green_ctx_current_sm.restype = ctypes.c_uint

    actual = int(library.green_ctx_actual_sm())
    if actual != expected_sm:
        raise SmCapError(
            f"sm_cap={expected_sm} requested but the green context has {actual} SMs"
        )

    observed: list[int] = []
    thread = threading.Thread(
        target=lambda: observed.append(int(library.green_ctx_current_sm())),
        name="sm-cap-verify",
    )
    thread.start()
    thread.join()
    if observed != [actual]:
        raise SmCapError(
            f"sm_cap={expected_sm} did not reach a new thread (it sees "
            f"{observed[0] if observed else 'no'} SMs); check that LD_PRELOAD "
            f"names {bootstrap}"
        )
    return actual


def stage_sm_cap_env(stage_cfg: StageConfig) -> dict[str, str]:
    """Green-context env for *stage_cfg*, empty when it declares no cap."""
    if stage_cfg.sm_cap is None:
        return {}
    # Note (Jiaxin Deng): stage env defaults never override os.environ, so an
    # LD_PRELOAD inherited from the parent would silently drop the cap.
    if "LD_PRELOAD" in os.environ:
        raise SmCapError(
            f"stage {stage_cfg.name!r} sets sm_cap but LD_PRELOAD is already "
            "set in the parent environment, which would shadow the "
            "green-context bootstrap; unset it or add the bootstrap to it"
        )
    return sm_cap_env(stage_cfg.sm_cap, resolve_bootstrap_path())
