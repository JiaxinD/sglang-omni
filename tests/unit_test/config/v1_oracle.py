# SPDX-License-Identifier: Apache-2.0
"""A frozen copy of the V1 configuration merge chain.

**Do not refactor, tidy or "fix" anything in this file.** It is a verbatim
snapshot of ``sglang_omni/config/manager.py`` as of commit 89d4235e, taken
immediately before the resolver became authoritative. Its whole value is that
it does not change: it is the oracle ``test_v1_parity.py`` compares the
resolver against, so an edit here silently moves the compatibility baseline
rather than testing against it.

The production copies of these functions are deleted. If a behaviour below
looks like a bug, it probably is one -- V1 shipped with it, users may depend on
it, and the parity test's ``INTENTIONAL_DIFFERENCES`` table is where a
deliberate departure gets recorded and justified.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from sglang_omni.config.schema import PipelineConfig

__all__ = ["v1_merge_dotted_cli", "v1_apply_stage_overrides", "v1_convert_scalar"]


def v1_merge_dotted_cli(
    config: PipelineConfig,
    extra_args: dict[str, Any],
) -> PipelineConfig:
    """``ConfigManager.merge_config``: dotted CLI arguments onto a config."""
    extra_args = {key: v1_convert_scalar(value) for key, value in extra_args.items()}
    config_data = config.model_dump()
    config_cls = type(config)

    cfg_copy = deepcopy(config_data)
    for key, value in extra_args.items():
        current = cfg_copy
        keys = key.split(".")
        for k in keys[:-1]:
            current = _resolve_child(current, k)

        # update the value
        last_key = keys[-1]
        if isinstance(current, list):
            current[_resolve_list_index(current, last_key)] = value
        else:
            current[last_key] = value

    _sync_stage_parallelism_aliases(cfg_copy, set(extra_args))

    # validate the configuration
    return config_cls(**cfg_copy)


def v1_apply_stage_overrides(
    config: PipelineConfig,
    # Widened from ``dict[str, Any]`` so the ``isinstance`` guard below stays
    # reachable for a type checker -- V1 accepted whatever YAML produced, and
    # the parity probes deliberately feed it non-mappings. Runtime is unchanged.
    stage_overrides: Any,
) -> PipelineConfig:
    """``config.manager._apply_stage_overrides``: the YAML ``stage_overrides`` block."""
    if not isinstance(stage_overrides, dict):
        raise ValueError(
            "stage_overrides must be a mapping from stage name to overrides"
        )

    config_data = config.model_dump()
    stages = config_data["stages"]
    stage_by_name = {stage["name"]: stage for stage in stages}

    for stage_name, override in stage_overrides.items():
        if stage_name not in stage_by_name:
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
        stage = stage_by_name[stage_name]
        stage["runtime"] = _deep_merge_dict(
            stage.get("runtime", {}),
            runtime_override,
        )

    return type(config)(**config_data)


def _deep_merge_dict(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _sync_stage_parallelism_aliases(
    config_data: dict[str, Any],
    override_keys: set[str],
) -> None:
    """Keep StageConfig.tp_size and parallelism.tp coherent for dotted CLI args."""
    stages = config_data.get("stages")
    if not isinstance(stages, list):
        return

    for index, stage in enumerate(stages):
        if not isinstance(stage, dict):
            continue
        stage_keys = (str(index), stage["name"])
        has_tp_size_override = any(
            f"stages.{key}.tp_size" in override_keys for key in stage_keys
        )
        has_parallelism_override = any(
            f"stages.{key}.parallelism.tp" in override_keys for key in stage_keys
        )
        if has_tp_size_override == has_parallelism_override:
            continue

        if has_tp_size_override:
            parallelism = dict(stage.get("parallelism") or {})
            parallelism["tp"] = stage["tp_size"]
            stage["parallelism"] = parallelism
        else:
            stage["tp_size"] = stage["parallelism"]["tp"]


def _resolve_list_index(items: list[Any], key: str) -> int:
    """Resolve a dotted-path segment to a list index."""
    if key.isdigit():
        return int(key)

    for index, item in enumerate(items):
        if isinstance(item, dict) and item.get("name") == key:
            return index

    available = [
        item["name"] for item in items if isinstance(item, dict) and "name" in item
    ]
    raise KeyError(
        f"Could not resolve list element {key!r}; expected an integer index "
        f"or one of the named entries {available}"
    )


def _resolve_child(current: Any, key: str) -> Any:
    """Traverse one segment of a dotted config path."""
    if isinstance(current, list):
        return current[_resolve_list_index(current, key)]
    return current[key]


def v1_convert_scalar(value: Any) -> Any:
    """``ConfigManager._convert_scalar``: type inference from CLI text."""
    if not isinstance(value, str):
        return value

    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "none":
        return None

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        return value
