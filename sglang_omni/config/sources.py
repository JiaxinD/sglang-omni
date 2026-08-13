# SPDX-License-Identifier: Apache-2.0
"""The canonical way to set a configuration path: ``set:`` and ``--set``.

Everything in :mod:`sglang_omni.config.compat` exists to translate a *legacy*
spelling into a canonical path. This module is its counterpart: the two
surfaces that take a canonical path and nothing else.

In YAML, one flat dotted path per entry::

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

On the command line the same paths, one per ``--set``::

    sgl-omni serve --config omni.yaml \\
        --set stages.tts_engine.runtime.max_seq_len=8192 \\
        --set stages.tts_engine.tp_size=2

``--set`` and the older ``--stages.tts_engine.tp_size 2`` form land at the same
layer *and* the same specificity, so writing one path with both is refused
rather than resolved by argument order -- see
:meth:`ConfigPatchSet.require_no_conflicts`. That is the intended relationship:
they are two spellings of one thing, and only one of them is the one to teach.
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
from sglang_omni.config.schema import PipelineConfig

__all__ = [
    "SET_BLOCK_KEY",
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
        raise ValueError(f"--set expects PATH=VALUE, got {text!r}; e.g. {_EXAMPLE}")
    return path, value


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
