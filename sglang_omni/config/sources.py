# SPDX-License-Identifier: Apache-2.0
"""The canonical user-facing surfaces for setting a configuration path.

Two spellings, one path language. In YAML, per-stage settings live under the
``stages:`` mapping, keyed by stage name::

    config_cls: MossTTSPipelineConfig
    model_path: OpenMOSS-Team/MOSS-TTS

    stages:
      tts_engine:
        tp_size: 2
        engine:
          mem_fraction_static: 0.7
      vocoder:
        model:
          dtype: bfloat16

On the command line the same paths appear as dotted flags, with the
``stages.`` prefix implied -- the flag starts from the stage name, exactly
as the mapping does::

    sgl-omni serve --config omni.yaml \\
        --tts_engine.tp_size 2 \\
        --tts_engine.engine.mem_fraction_static 0.7

Every stage name must be one the config class defines: stage topology --
which stages exist, how they route, where requests enter -- lives in the
model's ``config.py`` and is not user configuration. The file overrides
settings on those stages; every leaf it writes becomes one patch, so
provenance and conflict detection stay per value.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import yaml

from sglang_omni.config.patch import (
    ConfigPatch,
    ConfigPatchSet,
    ConfigSource,
    SourceKind,
    Specificity,
)
from sglang_omni.config.path import ConfigPath, ConfigPathError
from sglang_omni.config.schema import PipelineConfig

__all__ = [
    "dump_user_config",
    "patches_from_dotted_cli",
    "patches_from_model_path_flag",
    "patches_from_shared_block",
    "patches_from_stages_mapping",
    "sources_from_config_file",
]


# Top-level blocks earlier config surfaces accepted. They are refused with
# directions rather than falling through to a generic unknown-field error,
# because every one of them has an exact spelling in the current surface.
_REMOVED_TOP_LEVEL_BLOCKS: dict[str, str] = {
    "stage_overrides": (
        "the stage_overrides block was removed; write the same settings "
        "under the stages: mapping, e.g. stages.<name>.gpu_memory_fraction"
    ),
    "set": (
        "the set: block was removed; write each setting at its own place in "
        "the document, e.g. under the stages: mapping"
    ),
    "runtime_overrides": (
        "runtime_overrides was removed; write stages.<name>.engine.*, "
        "stages.<name>.scheduler.* or stages.<name>.model.* instead"
    ),
    "entry_stage": (
        "entry_stage is not user configuration; the entry stage is declared "
        "by the model's config class"
    ),
}

_STAGES_LIST_GUIDANCE = (
    "stages must be a mapping keyed by stage name; the list form is no "
    "longer accepted. Move each entry under its name and drop the name: "
    "key:\n\n    stages:\n      thinker:\n        tp_size: 2"
)


def patches_from_dotted_cli(
    extra_args: Mapping[str, Any] | Iterable[tuple[str, Any]],
    config: PipelineConfig,
    *,
    origin: str = "extra CLI args",
) -> ConfigPatchSet:
    """Normalize ``--thinker.engine.mem_fraction_static 0.6`` style arguments.

    The ``stages.`` prefix is implied on the command line -- the flag starts
    from the stage name, exactly as the YAML ``stages:`` mapping starts from
    it -- and the explicit prefix is refused so each path has one spelling.
    Accepts a mapping or ordered ``(key, value)`` pairs; the pairs keep a
    repeated flag visible, so the resolver's duplicate check rules on it
    instead of the parser keeping whichever spelling came last.
    """
    # Local import: the spelling rewrites are compat's job, and compat imports
    # this module, so importing it at module level would be a cycle.
    from sglang_omni.config.compat import canonicalize_dotted_key

    items = extra_args.items() if isinstance(extra_args, Mapping) else extra_args
    patchset = ConfigPatchSet()
    for key, value in items:
        head, _, rest = str(key).partition(".")
        if head == "stages" and rest:
            raise ConfigPathError(
                f"--{key}: the stages. prefix is implied on the command "
                f"line; write --{rest}",
                raw=str(key),
            )
        canonical = canonicalize_dotted_key(key, config)
        patchset.add(
            ConfigPatch.create(
                canonical,
                value,
                ConfigSource(SourceKind.CLI_DOTTED, origin),
                root=type(config),
            )
        )
    return patchset


def patches_from_model_path_flag(
    model_path: str,
    config: PipelineConfig,
    *,
    origin: str = "--model-path",
) -> ConfigPatchSet:
    """Translate the typed ``--model-path`` flag into a CLI-layer patch.

    Only needed alongside ``--config``: without a file the flag *selects* the
    baseline and there is nothing to override. With one, the flag is a command
    line source like any other and must outrank the file's ``model_path`` --
    as a patch, so that a launch and ``config explain`` agree on the value and
    on where it came from.
    """
    source = ConfigSource(SourceKind.CLI_FLAG, origin)
    return ConfigPatchSet().add(
        ConfigPatch.create("model_path", model_path, source, root=type(config))
    )


def patches_from_shared_block(
    shared_block: Any,
    config_cls: type[PipelineConfig],
    stage_names: Iterable[str],
    *,
    origin: str = "",
) -> ConfigPatchSet:
    """Expand the ``shared:`` selector list into per-stage patches.

    Each entry pairs a ``select:`` block with a stage-shaped body; the body is
    applied to every stage the selector matches, one leaf patch per stage, at
    ``Specificity.ROLE`` -- so a path written explicitly for one stage (under
    ``stages:`` or as a dotted flag) always outranks the broadcast without
    being a conflict. Selectors:

    * ``stages`` -- explicit stage names (each must exist);
    * ``engine`` -- ``true`` matches every SGLang engine stage;
    * ``exclude`` -- removes stages after the positive selectors matched.

    The expansion happens before duplicate checking and validation; resolved
    values are stored per stage, and provenance shows each expanded patch with
    the entry's index so ``config explain`` can name the selector that wrote
    a value. Two entries expanding onto one leaf are a conflict, exactly as
    two explicit writes at one precedence are.
    """
    if not isinstance(shared_block, list):
        raise ValueError(
            "shared must be a list of {select: ..., <settings>} "
            f"entries, got {type(shared_block).__name__}"
        )

    names = list(stage_names)
    patches = ConfigPatchSet()
    for index, entry in enumerate(shared_block):
        label = f"shared[{index}]"
        if not isinstance(entry, Mapping) or "select" not in entry:
            raise ValueError(f"{label} must be a mapping with a select: block")
        body = {key: value for key, value in entry.items() if key != "select"}
        if not body:
            raise ValueError(f"{label} selects stages but writes nothing")
        matched = _select_stages(entry["select"], config_cls, names, label=label)
        source = ConfigSource(SourceKind.YAML_FILE, origin, detail=label)
        for stage_name in matched:
            for path, value in _flatten(f"stages.{stage_name}", dict(body), config_cls):
                patches.add(
                    ConfigPatch.create(
                        path,
                        value,
                        source,
                        root=config_cls,
                        specificity=Specificity.ROLE,
                        coerce=False,
                    )
                )
    return patches


def _select_stages(
    select: Any,
    config_cls: type[PipelineConfig],
    stage_names: list[str],
    *,
    label: str,
) -> list[str]:
    """Resolve one ``select:`` block to the stage names it matches."""
    if not isinstance(select, Mapping) or not select:
        raise ValueError(f"{label}.select must be a non-empty mapping")
    unknown_keys = set(select) - {"stages", "engine", "exclude"}
    if unknown_keys:
        raise ValueError(
            f"{label}.select has unknown selector(s) {sorted(unknown_keys)}; "
            "supported: stages, engine, exclude"
        )

    matched = list(stage_names)
    if "stages" in select:
        wanted = select["stages"]
        if not isinstance(wanted, list) or not all(
            isinstance(name, str) for name in wanted
        ):
            raise ValueError(f"{label}.select.stages must be a list of names")
        missing = [name for name in wanted if name not in stage_names]
        if missing:
            raise ValueError(
                f"{label}.select.stages names unknown stage(s) {missing}; "
                f"this pipeline has: {', '.join(stage_names)}"
            )
        matched = [name for name in matched if name in wanted]
    if select.get("engine"):
        matched = [
            name for name in matched if config_cls.stage_config_cls(name).engine_stage
        ]
    excluded = select.get("exclude") or []
    if not isinstance(excluded, list):
        raise ValueError(f"{label}.select.exclude must be a list of names")
    matched = [name for name in matched if name not in excluded]
    if not matched:
        raise ValueError(f"{label}.select matches no stage of this pipeline")
    return matched


def patches_from_stages_mapping(
    stages_block: Any,
    config_cls: type[PipelineConfig],
    known_names: Iterable[str],
    *,
    origin: str = "",
) -> ConfigPatchSet:
    """Split the ``stages:`` mapping into per-leaf patches.

    Every name must be a stage the config class defines: stage topology --
    which stages exist, how they route, where requests enter -- lives in the
    model's ``config.py``, and a config file only overrides settings on it.
    Each leaf the body writes becomes one patch at the user-file layer.
    """
    if isinstance(stages_block, list):
        raise ValueError(_STAGES_LIST_GUIDANCE)
    if not isinstance(stages_block, Mapping):
        raise ValueError(
            "stages must be a mapping from stage name to stage settings, "
            f"got {type(stages_block).__name__}"
        )

    known = list(known_names)
    source = ConfigSource(SourceKind.YAML_FILE, origin)
    patches = ConfigPatchSet()

    for stage_name, body in stages_block.items():
        if not isinstance(stage_name, str) or not stage_name:
            raise ValueError(
                f"stages keys must be stage names written as strings, "
                f"got {stage_name!r}"
            )
        if not isinstance(body, Mapping):
            raise ValueError(f"stages.{stage_name} must be a mapping")
        if "name" in body:
            raise ValueError(
                f"stages.{stage_name} must not set name: the mapping key is "
                "the stage's name"
            )
        if stage_name not in known:
            raise ValueError(
                f"stages.{stage_name}: no stage named {stage_name!r} in "
                f"{config_cls.__name__}; stage topology lives in the model's "
                f"config class. This pipeline has: {', '.join(known)}"
            )
        prefix = f"stages.{stage_name}"
        for path, value in _flatten(prefix, dict(body), config_cls):
            patches.add(
                ConfigPatch.create(path, value, source, root=config_cls, coerce=False)
            )

    return patches


def sources_from_config_file(
    file_path: str,
) -> tuple[PipelineConfig, ConfigPatchSet]:
    """Split a config file into the config it declares and its overrides.

    ``ConfigManager.from_file`` folds the returned patches into the config,
    which is what launching needs and the opposite of what explaining needs:
    by the time the caller holds the config, the stage entry that set a value
    has already been absorbed into it. Keeping the two apart is what lets
    ``sgl-omni config explain`` name the file as a source.
    """
    # Local import: the registry pulls in model packages, several of which
    # import this module's callers.
    from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY

    with open(file_path, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config file {file_path!r} must contain a mapping")

    data = dict(data)
    for block, guidance in _REMOVED_TOP_LEVEL_BLOCKS.items():
        if block in data:
            raise ValueError(f"Config file {file_path!r}: {guidance}")

    config_cls = PIPELINE_CONFIG_REGISTRY.get_config_cls_by_name(data["config_cls"])
    stages_block = data.pop("stages", None)
    shared_block = data.pop("shared", None)
    if stages_block is None and shared_block is None:
        return config_cls(**data), ConfigPatchSet()

    config = config_cls(**data)
    patches = ConfigPatchSet()
    if stages_block is not None:
        patches = patches_from_stages_mapping(
            stages_block,
            config_cls,
            (stage.name for stage in config.stages),
            origin=str(file_path),
        )
    if shared_block is not None:
        # Expanded against the settled stage list, so a selector can reach a
        # stage this same file added.
        patches = patches.merge(
            patches_from_shared_block(
                shared_block,
                config_cls,
                (stage.name for stage in config.stages),
                origin=str(file_path),
            )
        )
    return config, patches


def dump_user_config(config: PipelineConfig) -> dict[str, Any]:
    """Dump a config in the shape a config file is written in.

    The internal stage list becomes the user-facing ``stages:`` mapping: the
    key carries the name (so ``name`` leaves the body), and a non-engine
    stage's ``engine: null`` placeholder is dropped because writing below it
    is a path error. ``entry_stage`` is dropped too -- it belongs to the
    model's config class, not to a config file. The result round-trips
    through :func:`sources_from_config_file` back to an equal config.
    """
    data = config.model_dump(mode="json")
    data.pop("entry_stage", None)
    stages: dict[str, Any] = {}
    for stage in data.get("stages", []):
        body = dict(stage)
        body.pop("name", None)
        if body.get("engine") is None:
            body.pop("engine", None)
        stages[stage["name"]] = body
    data["stages"] = stages
    return data


def _flatten(
    prefix: str,
    value: dict[str, Any],
    root: type[PipelineConfig],
) -> list[tuple[str, Any]]:
    """Split a nested stage entry into one patch per leaf.

    Stopping at schema leaves keeps the by-name merge a merge (an omitted
    sibling keeps its default) while keeping provenance per value rather
    than per block.
    """
    out: list[tuple[str, Any]] = []
    for key, child in value.items():
        path = f"{prefix}.{key}"
        if isinstance(child, dict) and not ConfigPath.parse(path, root).is_leaf:
            out.extend(_flatten(path, child, root))
        else:
            out.append((path, child))
    return out
