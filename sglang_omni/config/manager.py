from typing import Any

import yaml
from transformers import AutoConfig

from sglang_omni.config.compat import (
    patches_from_dotted_cli,
    patches_from_stage_overrides,
)
from sglang_omni.config.deprecation import warn_deprecations
from sglang_omni.config.resolver import ConfigResolver
from sglang_omni.config.schema import PipelineConfig
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from sglang_omni.utils import (
    architecture_from_hf_config,
    try_resolve_arch_from_mistral_config,
    try_resolve_arch_from_raw_config,
)


def resolve_config_cls_for_model_path(model_path: str):
    """Resolve a PipelineConfig class from HF config metadata."""
    hf_config = None
    try:
        hf_config = AutoConfig.from_pretrained(model_path)
    except (OSError, ValueError, KeyError):
        hf_config = None

    arch = architecture_from_hf_config(hf_config) if hf_config is not None else None
    if arch is None:
        arch = try_resolve_arch_from_raw_config(model_path)
    if arch is None:
        arch = try_resolve_arch_from_mistral_config(model_path)
    if arch is None:
        raise ValueError(f"Could not resolve model architecture for {model_path!r}")
    return PIPELINE_CONFIG_REGISTRY.get_config(arch)


class ConfigManager:
    """
    The ConfigManager is responsible for managing the configuration based on the user CLI arguments, configuration file
    given by the user, and the default configuration for the model. As the omni models have various architectures, setting a uniform
    list of arguments is not feasible. Thus, we take reference from the TorchTitan's configuration management system to allow users to
    dynamically configure their runtime settings.
    """

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config

    def parse_extra_args(self, args: list[str]) -> dict[str, Any]:
        """
        Parse the CLI arguments and return the configuration.
        """
        # we expect the arguments to be key-values pairs
        extra_args = {}
        cur_key, cur_value = None, None
        for arg in args:
            if "=" in arg and cur_key is None and cur_value is None:
                cur_key, cur_value = arg.split("=", 1)
            elif cur_key is None and cur_value is None:
                cur_key = arg
            elif cur_key is not None and cur_value is None:
                # record the key value pair
                cur_value = arg
            else:
                raise ValueError(f"Invalid argument: {arg}")

            if cur_key is not None and cur_value is not None:
                # remove the -- in front of the key
                formatted_key = cur_key.lstrip("-").replace("-", "_")
                extra_args[formatted_key] = cur_value
                cur_key, cur_value = None, None
        if cur_key is not None and cur_value is None:
            raise ValueError(f"Missing value for argument: {cur_key}")
        return extra_args

    def merge_config(self, extra_args: dict[str, Any]) -> PipelineConfig:
        """
        Merge the configuration and the extra arguments.

        The dotted keys are translated into canonical patches and applied by
        :class:`~sglang_omni.config.resolver.ConfigResolver`, which is the only
        code that writes into a configuration. Legacy spellings -- positional
        stage indices, paths the schema would rather people stopped using --
        are accepted and reported, never silently dropped.
        """
        patches = patches_from_dotted_cli(extra_args, self.config)
        resolved = ConfigResolver(self.config).resolve(patches)
        warn_deprecations(patches, context="command line")
        return resolved.config

    @staticmethod
    def from_model_path(model_path: str, variant: str | None = None) -> "ConfigManager":
        """Load config from model path, optionally selecting a variant."""
        import importlib

        config_cls = resolve_config_cls_for_model_path(model_path)

        if variant:
            module = importlib.import_module(config_cls.__module__)
            variants = getattr(module, "Variants", None)
            if variants and variant in variants:
                config_cls = variants[variant]
            else:
                raise ValueError(
                    f"Unknown variant '{variant}' for {config_cls.__name__}"
                )

        config = config_cls(model_path=model_path)
        return ConfigManager(config)

    @staticmethod
    def from_file(file_path: str) -> "ConfigManager":
        """
        Load the configuration from the file path.
        """
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Config file {file_path!r} must contain a mapping")

        data = dict(data)
        has_stage_overrides = "stage_overrides" in data
        stage_overrides = data.pop("stage_overrides", {})
        config_cls_str = data["config_cls"]
        config_cls = PIPELINE_CONFIG_REGISTRY.get_config_cls_by_name(config_cls_str)
        config = config_cls(**data)
        if has_stage_overrides:
            config = _apply_stage_overrides(config, stage_overrides)
        return ConfigManager(config)


def _apply_stage_overrides(
    config: PipelineConfig,
    stage_overrides: dict[str, Any],
) -> PipelineConfig:
    """Apply compact file-level runtime overrides by stage name.

    ``from_file`` folds the block into the configuration it returns, so callers
    that already hold a ``ConfigManager`` see one settled config rather than a
    config plus a pile of pending overrides. The validation messages are part
    of that block's documented contract and live in ``compat.py``.
    """
    patches = patches_from_stage_overrides(stage_overrides, config)
    resolved = ConfigResolver(config).resolve(patches)
    warn_deprecations(patches, context="the stage_overrides block")
    return resolved.config
