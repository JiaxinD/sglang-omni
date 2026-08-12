# SPDX-License-Identifier: Apache-2.0
"""The single place where configuration sources are merged.

``ConfigResolver.resolve`` takes a baseline config plus a
:class:`~sglang_omni.config.patch.ConfigPatchSet` and produces exactly two
things: the validated configuration, and the provenance that explains it.

What it deliberately does *not* do:

* it does not know about YAML, CLI flags or Router workers — sources normalize
  themselves into patches before they get here;
* it does not compute placement, process topology or SGLang server args —
  those are downstream consumers of the resolved value, and must not write
  back into it.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from sglang_omni.config.patch import ConfigPatch, ConfigPatchSet
from sglang_omni.config.path import ConfigPath, ConfigPathError
from sglang_omni.config.provenance import ProvenanceMap
from sglang_omni.config.schema import PipelineConfig

__all__ = ["ConfigResolver", "ResolvedConfig", "ConfigDifference", "diff_configs"]


# Fields that are two spellings of one value. When a patch writes exactly one
# side, the other has to follow or the rebuilt model rejects the pair. This
# replaces ``ConfigManager._sync_stage_parallelism_aliases``, which recognised
# the same pair by string-matching dotted CLI keys.
_ALIAS_PAIRS: tuple[tuple[str, str], ...] = (
    ("stages.*.tp_size", "stages.*.parallelism.tp"),
)


@dataclass(frozen=True)
class ResolvedConfig:
    """A validated configuration together with the story of how it got there."""

    config: PipelineConfig
    provenance: ProvenanceMap
    patches: ConfigPatchSet

    def value(self, path: str) -> Any:
        return ConfigPath.parse(path, type(self.config)).read(self.config)


class ConfigResolver:
    """Applies a patch set to a baseline config."""

    def __init__(self, base: PipelineConfig) -> None:
        self._base = base

    @property
    def config_cls(self) -> type[PipelineConfig]:
        return type(self._base)

    def resolve(self, patchset: ConfigPatchSet) -> ResolvedConfig:
        patchset.require_no_conflicts()

        data = self._base.model_dump()
        provenance = ProvenanceMap.from_patchset(patchset)

        ordered = patchset.ordered()
        for patch in ordered:
            provenance.record_baseline(patch.key, _safe_read(patch.path, data))

        for patch in ordered:
            _apply(data, patch)

        _sync_aliases(data, _written_paths(ordered))

        config = self.config_cls(**data)
        return ResolvedConfig(config=config, provenance=provenance, patches=patchset)


# ----------------------------------------------------------------------
# application
# ----------------------------------------------------------------------


def _apply(data: dict[str, Any], patch: ConfigPatch) -> None:
    """Assign a leaf, or deep-merge a mapping written at a container path."""
    if patch.path.is_leaf or not isinstance(patch.value, dict):
        patch.path.write(data, deepcopy(patch.value))
        return

    existing = _safe_read(patch.path, data)
    if isinstance(existing, dict):
        patch.path.write(data, _deep_merge(existing, patch.value))
    else:
        patch.path.write(data, deepcopy(patch.value))


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _safe_read(path: ConfigPath, data: dict[str, Any]) -> Any:
    """Read a path that may not exist yet (a new mapping key, for instance)."""
    try:
        return path.read(data)
    except ConfigPathError:
        return None


def _written_paths(patches: list[ConfigPatch]) -> set[str]:
    """Every leaf path a patch set touches, expanding container patches."""
    out: set[str] = set()
    for patch in patches:
        out.add(patch.key)
        if isinstance(patch.value, dict):
            out.update(f"{patch.key}.{suffix}" for suffix in _dotted_keys(patch.value))
    return out


def _dotted_keys(value: dict[str, Any], prefix: str = "") -> list[str]:
    out: list[str] = []
    for key, child in value.items():
        dotted = f"{prefix}{key}"
        out.append(dotted)
        if isinstance(child, dict):
            out.extend(_dotted_keys(child, f"{dotted}."))
    return out


def _sync_aliases(data: dict[str, Any], written: set[str]) -> None:
    """Mirror a value onto its alias when only one side was written."""
    stages = data.get("stages")
    if not isinstance(stages, list):
        return

    for stage in stages:
        if not isinstance(stage, dict) or "name" not in stage:
            continue
        name = stage["name"]
        for left, right in _ALIAS_PAIRS:
            left_path = left.replace("*", name)
            right_path = right.replace("*", name)
            left_written = left_path in written
            right_written = right_path in written
            if left_written == right_written:
                continue
            source, target = (
                (left_path, right_path) if left_written else (right_path, left_path)
            )
            value = ConfigPath.parse(source).read(data)
            ConfigPath.parse(target).write(data, value)


# ----------------------------------------------------------------------
# shadow-mode comparison
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ConfigDifference:
    path: str
    expected: Any
    actual: Any

    def render(self) -> str:
        return f"{self.path}: expected {self.expected!r}, got {self.actual!r}"


def diff_configs(
    expected: PipelineConfig | dict[str, Any],
    actual: PipelineConfig | dict[str, Any],
) -> list[ConfigDifference]:
    """Compare two configs field by field, addressing stages by name.

    Used while the resolver runs in shadow mode: the V1 chain stays
    authoritative and any divergence is reported instead of taking effect.
    """
    return _diff(_as_dump(expected), _as_dump(actual), "")


def _as_dump(value: PipelineConfig | dict[str, Any]) -> dict[str, Any]:
    return value.model_dump() if isinstance(value, PipelineConfig) else value


def _diff(expected: Any, actual: Any, prefix: str) -> list[ConfigDifference]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        out: list[ConfigDifference] = []
        for key in sorted(set(expected) | set(actual)):
            child = f"{prefix}.{key}" if prefix else key
            if key not in expected or key not in actual:
                out.append(ConfigDifference(child, expected.get(key), actual.get(key)))
                continue
            out.extend(_diff(expected[key], actual[key], child))
        return out

    if _is_named_list(expected) and _is_named_list(actual):
        out = []
        expected_by_name = {item["name"]: item for item in expected}
        actual_by_name = {item["name"]: item for item in actual}
        for name in sorted(set(expected_by_name) | set(actual_by_name)):
            child = f"{prefix}.{name}" if prefix else name
            if name not in expected_by_name or name not in actual_by_name:
                out.append(
                    ConfigDifference(
                        child, expected_by_name.get(name), actual_by_name.get(name)
                    )
                )
                continue
            out.extend(_diff(expected_by_name[name], actual_by_name[name], child))
        return out

    if expected != actual:
        return [ConfigDifference(prefix, expected, actual)]
    return []


def _is_named_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, dict) and "name" in item for item in value)
    )
