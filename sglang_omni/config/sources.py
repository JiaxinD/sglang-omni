# SPDX-License-Identifier: Apache-2.0
"""The canonical user-facing surfaces for setting a configuration path.

Three spellings, one meaning. In YAML, the ``set:`` block -- one flat dotted
path per entry::

    config_cls: MossTTSPipelineConfig
    model_path: OpenMOSS-Team/MOSS-TTS

    set:
      stages.tts_engine.runtime.max_seq_len: 8192
      stages.tts_engine.tp_size: 2
      entry_stage: preprocessing

Flat, not nested, because the key is then literally the path that
``sgl-omni config explain`` prints back and that an error message names. A
nested block would have to be flattened before it could be talked about, and
the flattening is exactly where ``stage_overrides`` lost track of which line
set which value.

On the command line the same paths appear twice over: as dotted flags and as
``--set`` assignments::

    sgl-omni serve --config omni.yaml \\
        --stages.tts_engine.runtime.max_seq_len 8192 \\
        --set stages.tts_engine.tp_size=2

``--set`` is the CLI counterpart of the YAML ``set:`` block; the dotted flag is
the pipeline's native override syntax. Both are first-class, land at the same
layer *and* the same specificity, and so writing one path with both is refused
rather than resolved by argument order -- see
:meth:`ConfigPatchSet.require_no_conflicts`.

What is *not* here is any legacy spelling: positional stage indices and the
``stage_overrides`` block live in :mod:`sglang_omni.config.compat`, whose job
is to translate them into the canonical paths this module deals in.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from sglang_omni.config.patch import (
    ConfigPatch,
    ConfigPatchSet,
    ConfigSource,
    SourceKind,
)
from sglang_omni.config.path import ConfigPathError
from sglang_omni.config.schema import PipelineConfig

__all__ = [
    "SET_BLOCK_KEY",
    "patches_from_dotted_cli",
    "patches_from_set_block",
    "patches_from_set_cli",
    "split_assignment",
]

SET_BLOCK_KEY = "set"
"""The YAML top-level key. Reserved: it is not a field of any config class."""

_EXAMPLE = "stages.thinker.runtime.max_seq_len=8192"


def split_assignment(text: str) -> tuple[str, str]:
    """Split one ``--set PATH=VALUE`` argument.

    The value keeps every ``=`` after the first, so JSON and query-string
    shaped values survive: ``--set stages.thinker.env={"A":"1=2"}``.
    """
    path, separator, value = text.partition("=")
    path = path.strip()
    if not separator or not path:
        # A ConfigPathError rather than a bare ValueError, so the CLI entry
        # points that already turn path errors into readable BadParameter
        # messages catch a malformed assignment the same way.
        raise ConfigPathError(
            f"--set expects PATH=VALUE, got {text!r}; e.g. {_EXAMPLE}", raw=text
        )
    return path, value


def patches_from_dotted_cli(
    extra_args: dict[str, Any],
    config: PipelineConfig,
    *,
    origin: str = "extra CLI args",
) -> ConfigPatchSet:
    """Normalize ``--stages.thinker.tp_size 4`` style arguments.

    The dotted flag is canonical syntax; the one legacy spelling it still
    accepts is a positional stage index (``--stages.1.tp_size``), which the
    compat layer translates into a name and flags as deprecated.
    """
    # Local import: the index translation is compat's job, and compat imports
    # this module for the ``set:`` helpers, so importing it at module level
    # would be a cycle.
    from sglang_omni.config.compat import canonicalize_dotted_key

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


def patches_from_set_cli(
    values: Iterable[str],
    config: PipelineConfig,
    *,
    origin: str = "command line",
) -> ConfigPatchSet:
    """Normalize repeated ``--set PATH=VALUE`` arguments.

    Positional stage indices are *not* accepted here. They are legacy syntax
    and this is new syntax; ``ConfigPath`` already answers
    ``--set stages.1.tp_size=2`` with "stages is addressed by name, not by
    index", which is the message to give someone writing the flag for the
    first time.
    """
    source = ConfigSource(SourceKind.CLI_SET, origin)
    patchset = ConfigPatchSet()
    for text in values:
        path, value = split_assignment(text)
        patchset.add(ConfigPatch.create(path, value, source, root=type(config)))
    return patchset


def patches_from_set_block(
    block: Any,
    config: PipelineConfig,
    *,
    origin: str = "",
) -> ConfigPatchSet:
    """Normalize the YAML ``set:`` block.

    Values arrive already typed by the YAML parser, and pass through
    :meth:`ConfigPath.coerce` untouched unless they are strings -- so a quoted
    ``"8192"`` still reaches an ``int`` field as an integer, and ``none``
    still clears an optional one, exactly as on the command line.
    """
    if not isinstance(block, Mapping):
        raise ValueError(
            f"{SET_BLOCK_KEY} must be a mapping from config path to value, "
            f"got {type(block).__name__}"
        )

    source = ConfigSource(SourceKind.YAML_FILE, origin)
    patchset = ConfigPatchSet()
    for key, value in block.items():
        if not isinstance(key, str):
            raise ValueError(
                f"{SET_BLOCK_KEY} keys must be config paths written as strings, "
                f"got {key!r}"
            )
        patchset.add(ConfigPatch.create(key, value, source, root=type(config)))
    return patchset
