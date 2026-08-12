# SPDX-License-Identifier: Apache-2.0
"""Translation of V1 configuration syntax into canonical patches.

This is the **only** module allowed to know about legacy spellings: positional
stage indices, the ``stage_overrides`` block, and (later) the typed CLI flags.
Everything downstream sees canonical paths and nothing else, so the legacy
surface can be deprecated by deleting entries here rather than by unpicking
merge logic spread across four files.
"""

from __future__ import annotations

from typing import Any

from sglang_omni.config.patch import (
    ConfigPatch,
    ConfigPatchSet,
    ConfigSource,
    SourceKind,
)
from sglang_omni.config.path import ConfigPath
from sglang_omni.config.schema import PipelineConfig

__all__ = [
    "canonicalize_dotted_key",
    "patches_from_dotted_cli",
    "patches_from_stage_overrides",
]

_INDEX_DEPRECATION = "positional stage indices are deprecated; address stages by name"


def canonicalize_dotted_key(key: str, config: PipelineConfig) -> tuple[str, str]:
    """Rewrite a V1 dotted key into canonical form.

    Returns ``(canonical_key, deprecation)`` where ``deprecation`` is empty when
    the key was already canonical.
    """
    parts = key.split(".")
    for index, part in enumerate(parts[:-1]):
        if part != "stages" or not parts[index + 1].isdigit():
            continue
        position = int(parts[index + 1])
        if position >= len(config.stages):
            raise ValueError(
                f"{key!r} refers to stage index {position}, but the pipeline has "
                f"{len(config.stages)} stages"
            )
        parts[index + 1] = config.stages[position].name
        return ".".join(parts), _INDEX_DEPRECATION
    return key, ""


def patches_from_dotted_cli(
    extra_args: dict[str, Any],
    config: PipelineConfig,
    *,
    origin: str = "extra CLI args",
) -> ConfigPatchSet:
    """Normalize ``--stages.thinker.tp_size 4`` style arguments."""
    patchset = ConfigPatchSet()
    for key, value in extra_args.items():
        canonical, deprecation = canonicalize_dotted_key(key, config)
        patchset.add(
            ConfigPatch.create(
                canonical,
                value,
                ConfigSource(SourceKind.CLI_DOTTED, origin),
                root=type(config),
                deprecated=deprecation,
            )
        )
    return patchset


def patches_from_stage_overrides(
    stage_overrides: dict[str, Any],
    config: PipelineConfig,
    *,
    origin: str = "",
) -> ConfigPatchSet:
    """Normalize the YAML ``stage_overrides`` block.

    The V1 validation messages are reproduced verbatim: this block is a
    documented user-facing surface and its errors are part of the contract.
    """
    if not isinstance(stage_overrides, dict):
        raise ValueError(
            "stage_overrides must be a mapping from stage name to overrides"
        )

    known = {stage.name for stage in config.stages}
    source = ConfigSource(SourceKind.YAML_STAGE_OVERRIDES, origin)
    patchset = ConfigPatchSet()

    for stage_name, override in stage_overrides.items():
        if stage_name not in known:
            raise ValueError(f"stage_overrides references unknown stage {stage_name!r}")
        if not isinstance(override, dict):
            raise ValueError(f"stage_overrides.{stage_name} must be a mapping")

        unsupported = sorted(set(override) - {"runtime"})
        if unsupported:
            raise ValueError(
                f"stage_overrides.{stage_name} supports only runtime overrides; "
                f"got unsupported keys {unsupported}"
            )

        if "runtime" not in override:
            continue
        runtime_override = override["runtime"]
        if not isinstance(runtime_override, dict):
            raise ValueError(f"stage_overrides.{stage_name}.runtime must be a mapping")

        prefix = f"stages.{stage_name}.runtime"
        for path, value in _flatten(prefix, runtime_override, type(config)):
            patchset.add(
                ConfigPatch.create(path, value, source, root=type(config), coerce=False)
            )

    return patchset


def _flatten(
    prefix: str,
    value: dict[str, Any],
    root: type[PipelineConfig],
) -> list[tuple[str, Any]]:
    """Split a nested override into one patch per leaf.

    Stopping at schema leaves reproduces the V1 deep merge while keeping
    provenance per value rather than per block.
    """
    out: list[tuple[str, Any]] = []
    for key, child in value.items():
        path = f"{prefix}.{key}"
        if isinstance(child, dict) and not ConfigPath.parse(path, root).is_leaf:
            out.extend(_flatten(path, child, root))
        else:
            out.append((path, child))
    return out
