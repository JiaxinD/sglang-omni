# SPDX-License-Identifier: Apache-2.0
"""Shadow-mode comparison between the V1 merge chain and the resolver.

While the resolver is being introduced, the V1 chain stays authoritative: it
computes the configuration that is actually used. The resolver runs alongside
it on the same inputs, and any divergence is *reported* rather than applied.

Two environment variables control this:

``SGLANG_OMNI_CONFIG_SHADOW``
    Set to ``0``/``false``/``no``/``off`` to skip the shadow run entirely.
    On by default -- the whole point is to collect evidence from real runs.

``SGLANG_OMNI_CONFIG_SHADOW_STRICT``
    When enabled, a divergence raises instead of logging a warning. Intended
    for CI, where "the resolver reproduces V1 exactly" is a gate rather than
    an observation.

Outside strict mode this module must never be able to break a launch. A bug
in the resolver, an input the compat layer cannot translate yet, anything at
all -- it is caught here and logged, and the V1 result is returned untouched.
"""

from __future__ import annotations

import logging
import os
from typing import Callable

from sglang_omni.config.patch import ConfigPatchSet
from sglang_omni.config.resolver import ConfigDifference, ConfigResolver, diff_configs
from sglang_omni.config.schema import PipelineConfig

logger = logging.getLogger(__name__)

__all__ = [
    "SHADOW_ENV",
    "SHADOW_STRICT_ENV",
    "ShadowDivergenceError",
    "shadow_enabled",
    "strict_enabled",
    "compare_with_resolver",
]

SHADOW_ENV = "SGLANG_OMNI_CONFIG_SHADOW"
SHADOW_STRICT_ENV = "SGLANG_OMNI_CONFIG_SHADOW_STRICT"

_FALSEY = {"", "0", "false", "no", "off"}


class ShadowDivergenceError(RuntimeError):
    """The resolver disagreed with the V1 chain, and strict mode is on."""


def shadow_enabled() -> bool:
    return os.environ.get(SHADOW_ENV, "1").strip().lower() not in _FALSEY


def strict_enabled() -> bool:
    return os.environ.get(SHADOW_STRICT_ENV, "").strip().lower() not in _FALSEY


def compare_with_resolver(
    *,
    baseline: PipelineConfig,
    expected: PipelineConfig,
    build_patches: Callable[[], ConfigPatchSet],
    context: str,
) -> None:
    """Resolve ``build_patches()`` against ``baseline`` and compare to ``expected``.

    ``expected`` is the V1 result and stays authoritative; this function only
    ever reports. ``build_patches`` is a callable rather than a patch set so
    that translating the inputs is itself covered by the failure handling.
    """
    if not shadow_enabled():
        return

    strict = strict_enabled()
    try:
        resolved = ConfigResolver(baseline).resolve(build_patches())
    except Exception as exc:  # noqa: BLE001 - shadow mode must not break launches
        _report_failure(context, exc, strict)
        return

    try:
        differences = diff_configs(expected, resolved.config)
    except Exception as exc:  # noqa: BLE001
        _report_failure(context, exc, strict)
        return

    if differences:
        _report_divergence(context, differences, strict)


def _report_failure(context: str, exc: Exception, strict: bool) -> None:
    message = (
        f"config shadow mode: the resolver failed on {context} "
        f"({type(exc).__name__}: {exc}). The V1 result was used."
    )
    if strict:
        raise ShadowDivergenceError(
            f"{message} {SHADOW_STRICT_ENV} is set, so this is fatal."
        ) from exc
    logger.warning(message)


def _report_divergence(
    context: str,
    differences: list[ConfigDifference],
    strict: bool,
) -> None:
    shown = differences[:20]
    body = "\n".join(f"  {difference.render()}" for difference in shown)
    if len(differences) > len(shown):
        body += f"\n  ... and {len(differences) - len(shown)} more"

    message = (
        f"config shadow mode: the resolver disagreed with the V1 chain on "
        f"{context} ({len(differences)} difference(s)). The V1 result was "
        f"used.\n{body}"
    )
    if strict:
        raise ShadowDivergenceError(
            f"{message}\n{SHADOW_STRICT_ENV} is set, so this is fatal."
        )
    logger.warning(message)
