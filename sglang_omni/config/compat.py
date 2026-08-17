# SPDX-License-Identifier: Apache-2.0
"""Translation of V1 legacy configuration spellings into canonical patches.

This is the **only** module allowed to know about legacy spellings: positional
stage indices in dotted keys (``stages.1.tp_size``), the YAML
``stage_overrides`` block, and the retired ``stages.*.parallelism.tp`` mirror
of ``stages.*.tp_size``. Everything downstream sees canonical paths and
nothing else, so the legacy surface can be deprecated by deleting entries here
rather than by unpicking merge logic spread across four files.

The canonical surfaces themselves -- dotted CLI flags, ``--set`` and the YAML
``set:`` block -- live in :mod:`sglang_omni.config.sources`.
"""

from __future__ import annotations

from typing import Any

import yaml

from sglang_omni.config.patch import (
    ConfigPatch,
    ConfigPatchSet,
    ConfigSource,
    SourceKind,
)
from sglang_omni.config.path import ConfigPath, ConfigPathError
from sglang_omni.config.schema import PipelineConfig
from sglang_omni.config.sources import SET_BLOCK_KEY, patches_from_set_block

__all__ = [
    "canonicalize_dotted_key",
    "patches_from_stage_overrides",
    "sources_from_config_file",
]

_INDEX_DEPRECATION = "positional stage indices are deprecated; address stages by name"
_PARALLELISM_DEPRECATION = (
    "parallelism.tp is a deprecated spelling of tp_size; write "
    "stages.<name>.tp_size instead"
)


def canonicalize_dotted_key(key: str, config: PipelineConfig) -> tuple[str, str]:
    """Rewrite a V1 dotted key into canonical form.

    Returns ``(canonical_key, deprecation)`` where ``deprecation`` is empty when
    the key was already canonical.
    """
    parts = key.split(".")
    notices: list[str] = []
    for index, part in enumerate(parts[:-1]):
        if part != "stages" or not parts[index + 1].isdigit():
            continue
        position = int(parts[index + 1])
        if position >= len(config.stages):
            # A ConfigPathError rather than a bare ValueError, so the CLI
            # entry points that already turn path errors into readable
            # BadParameter messages catch this one too.
            names = [stage.name for stage in config.stages]
            raise ConfigPathError(
                f"{key!r} refers to stage index {position}, but the pipeline has "
                f"only {len(config.stages)} stages, named: {', '.join(names)}",
                raw=key,
                resolved_prefix=".".join(parts[: index + 1]),
                suggestions=tuple(
                    ".".join(parts[: index + 1] + [name] + parts[index + 2 :])
                    for name in names[:3]
                ),
            )
        parts[index + 1] = config.stages[position].name
        notices.append(_INDEX_DEPRECATION)
        break
    if len(parts) == 4 and parts[0] == "stages" and parts[2:] == ["parallelism", "tp"]:
        parts = parts[:2] + ["tp_size"]
        notices.append(_PARALLELISM_DEPRECATION)
    return ".".join(parts), "; ".join(notices)


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


def sources_from_config_file(
    file_path: str,
) -> tuple[PipelineConfig, ConfigPatchSet]:
    """Split a config file into the config it declares and its overrides.

    ``ConfigManager.from_file`` folds the override blocks into the config it
    returns, which is what launching needs and the opposite of what explaining
    needs: by the time the caller holds the config, the block that set a value
    has already been absorbed into it. Keeping the two apart is what lets
    ``sgl-omni config explain`` name the file as a source.

    Both blocks come back in one patch set rather than one per block, so that
    ``set:`` and ``stage_overrides`` writing the same path is caught as the
    conflict it is instead of being settled by whichever block is applied
    second.
    """
    # Local import: the registry pulls in model packages, several of which
    # import this module's callers.
    from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY

    with open(file_path, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config file {file_path!r} must contain a mapping")

    data = dict(data)
    stage_overrides = data.pop("stage_overrides", {})
    set_block = data.pop(SET_BLOCK_KEY, None)
    legacy_tp = _extract_legacy_stage_parallelism(data)
    config_cls = PIPELINE_CONFIG_REGISTRY.get_config_cls_by_name(data["config_cls"])
    config = config_cls(**data)
    patches = patches_from_stage_overrides(
        stage_overrides, config, origin=str(file_path)
    )
    if set_block is not None:
        patches = patches.merge(
            patches_from_set_block(set_block, config, origin=str(file_path))
        )
    for stage_name, tp in legacy_tp:
        patches.add(
            ConfigPatch.create(
                f"stages.{stage_name}.tp_size",
                tp,
                ConfigSource(SourceKind.YAML_FILE, str(file_path)),
                root=config_cls,
                deprecated=_PARALLELISM_DEPRECATION,
            )
        )
    return config, patches


def _extract_legacy_stage_parallelism(data: dict[str, Any]) -> list[tuple[str, Any]]:
    """Pull V1 ``parallelism`` blocks out of the stage documents, in place.

    ``StageConfig`` no longer has the field, so a file that still writes it
    would be refused as an unknown key. The block held one value -- ``tp``, a
    mirror of ``tp_size`` -- and comes back as ``(stage_name, tp)`` pairs for
    the caller to turn into deprecated ``tp_size`` patches. The value is also
    written into the stage document itself, because validation runs on the
    document before any patch applies and cross-checks ``tp_size`` against the
    ``gpu`` list. A block that contradicts an explicit ``tp_size`` is refused
    here with the same message the schema used to produce.
    """
    stages = data.get("stages")
    if not isinstance(stages, list):
        return []

    out: list[tuple[str, Any]] = []
    for stage in stages:
        if not isinstance(stage, dict) or "parallelism" not in stage:
            continue
        name = stage.get("name")
        if not isinstance(name, str):
            continue  # the schema reports the missing name itself
        block = stage.pop("parallelism")
        if block is None:
            continue
        if not isinstance(block, dict):
            raise ValueError(f"Stage {name!r}: parallelism must be a mapping")
        unsupported = sorted(set(block) - {"tp"})
        if unsupported:
            raise ValueError(
                f"Stage {name!r}: parallelism supports only tp; "
                f"got unsupported keys {unsupported}"
            )
        if "tp" not in block:
            continue
        tp = block["tp"]
        if "tp_size" in stage and stage["tp_size"] != tp:
            raise ValueError(
                f"Stage {name!r}: tp_size={stage['tp_size']} conflicts with "
                f"parallelism.tp={tp}"
            )
        stage["tp_size"] = tp
        out.append((name, tp))
    return out


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
