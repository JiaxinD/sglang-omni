# SPDX-License-Identifier: Apache-2.0
"""Configuration patches: one normalized representation for every source.

Every configuration source — the model's Python defaults, a YAML file, a
Router worker entry, a CLI flag — is reduced to the same shape here::

    ConfigPatch(path=<ConfigPath>, value=..., source=..., layer=..., specificity=...)

Sources therefore only have to parse *their own syntax*. Deciding which value
wins is no longer spread across the sources; it is a property of the layer and
specificity attached to each patch, and is executed in exactly one place
(:mod:`sglang_omni.config.resolver`).

Precedence is ``(layer, specificity, path depth, insertion order)``:

* **layer** separates sources. A CLI value beats a YAML value because
  ``Layer.CLI > Layer.USER_FILE``, not because of the order helpers happen to
  run in ``cli/serve.py``.
* **specificity** separates aliases *within* one source. ``--mem-fraction-static``
  broadcasts to several stages, ``--talker-mem-fraction-static`` names one role;
  both are legal at once and the per-role value must win.
* **path depth** makes a container patch apply before a patch nested inside it,
  so ``stages.thinker.runtime`` never clobbers a sibling
  ``stages.thinker.runtime.max_seq_len``.
* **insertion order** is the final tie-break, and only ever applies to patches
  that a duplicate check has already accepted.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any

from pydantic import BaseModel

from sglang_omni.config.path import ConfigPath, ConfigPathError
from sglang_omni.config.schema import PipelineConfig

__all__ = [
    "ConfigPatch",
    "ConfigPatchSet",
    "ConfigSource",
    "DuplicatePatchError",
    "Layer",
    "SourceKind",
    "Specificity",
]


class Layer(IntEnum):
    """Cross-source precedence. Higher wins."""

    FRAMEWORK = 0
    """Schema defaults declared in ``config/schema.py``."""

    MODEL_DEFAULT = 10
    """The model's Python pipeline definition in ``models/*/config.py``."""

    PROFILE = 20
    """A named variant or shipped profile selected by the user."""

    USER_FILE = 30
    """The user's YAML document."""

    ROUTER = 40
    """Per-worker patches injected by ``sgl-omni-router``."""

    CLI = 50
    """Command line flags and ``--set``."""


class Specificity(IntEnum):
    """Precedence *within* one layer. Higher wins."""

    BROADCAST = 0
    """An alias that fans out to several stages, e.g. ``--mem-fraction-static``."""

    ROLE = 10
    """An alias naming one role, e.g. ``--talker-mem-fraction-static``."""

    EXPLICIT = 20
    """A path the user wrote out in full, e.g. ``--set stages.talker.…``."""


class SourceKind(str, Enum):
    """Where a patch came from."""

    FRAMEWORK_DEFAULT = "framework_default"
    MODEL_DEFAULT = "model_default"
    PROFILE = "profile"
    YAML_FILE = "yaml_file"
    YAML_STAGE_OVERRIDES = "yaml_stage_overrides"
    ROUTER = "router"
    CLI_FLAG = "cli_flag"
    CLI_DOTTED = "cli_dotted"
    CLI_SET = "cli_set"


_DEFAULT_LAYER: dict[SourceKind, Layer] = {
    SourceKind.FRAMEWORK_DEFAULT: Layer.FRAMEWORK,
    SourceKind.MODEL_DEFAULT: Layer.MODEL_DEFAULT,
    SourceKind.PROFILE: Layer.PROFILE,
    SourceKind.YAML_FILE: Layer.USER_FILE,
    SourceKind.YAML_STAGE_OVERRIDES: Layer.USER_FILE,
    SourceKind.ROUTER: Layer.ROUTER,
    SourceKind.CLI_FLAG: Layer.CLI,
    SourceKind.CLI_DOTTED: Layer.CLI,
    SourceKind.CLI_SET: Layer.CLI,
}


@dataclass(frozen=True)
class ConfigSource:
    """Human-facing identity of whatever produced a patch."""

    kind: SourceKind
    origin: str = ""
    """File path, flag name, worker id — whatever names this source instance."""

    detail: str = ""
    """Optional extra shown in ``config explain``, e.g. a deprecation note."""

    @property
    def layer(self) -> Layer:
        return _DEFAULT_LAYER[self.kind]

    def describe(self) -> str:
        label = self.kind.value.replace("_", " ")
        parts = [label]
        if self.origin:
            parts.append(f"({self.origin})")
        if self.detail:
            parts.append(f"- {self.detail}")
        return " ".join(parts)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.describe()


@dataclass(frozen=True)
class ConfigPatch:
    """A single ``path = value`` assignment with its provenance."""

    path: ConfigPath
    value: Any
    source: ConfigSource
    layer: Layer
    specificity: Specificity = Specificity.EXPLICIT
    deprecated: str = ""
    """Non-empty when the source syntax is on its way out; surfaced as a warning."""

    @classmethod
    def create(
        cls,
        path: str | ConfigPath,
        value: Any,
        source: ConfigSource,
        *,
        root: type[BaseModel] = PipelineConfig,
        layer: Layer | None = None,
        specificity: Specificity = Specificity.EXPLICIT,
        deprecated: str = "",
        coerce: bool = True,
    ) -> "ConfigPatch":
        """Build a patch, compiling ``path`` and coercing ``value`` to its type.

        Raises :class:`~sglang_omni.config.path.ConfigPathError` when the path
        is unknown or is not writable by a user-facing source. A path that is
        merely deprecated is accepted, and carries its reason forward as a
        deprecation notice alongside any the caller supplied.
        """
        compiled = (
            path if isinstance(path, ConfigPath) else ConfigPath.parse(path, root)
        )
        compiled.require_writable()
        notices = [note for note in (deprecated, compiled.visibility_reason) if note]
        return cls(
            path=compiled,
            value=compiled.coerce(value) if coerce else value,
            source=source,
            layer=source.layer if layer is None else layer,
            specificity=specificity,
            deprecated="; ".join(notices),
        )

    @property
    def key(self) -> str:
        return self.path.raw

    @property
    def rank(self) -> tuple[int, int, int]:
        """Precedence key, ignoring insertion order."""
        return (int(self.layer), int(self.specificity), len(self.path.segments))


class DuplicatePatchError(ConfigPathError):
    """Two equally-ranked patches disagree about the same path."""


@dataclass
class ConfigPatchSet:
    """An append-only, order-preserving collection of patches."""

    patches: list[ConfigPatch] = field(default_factory=list)

    # ------------------------------------------------------------------
    # building
    # ------------------------------------------------------------------

    def add(self, patch: ConfigPatch) -> "ConfigPatchSet":
        self.patches.append(patch)
        return self

    def extend(self, patches: Iterable[ConfigPatch]) -> "ConfigPatchSet":
        self.patches.extend(patches)
        return self

    def merge(self, other: "ConfigPatchSet") -> "ConfigPatchSet":
        return ConfigPatchSet(list(self.patches) + list(other.patches))

    def __iter__(self) -> Iterator[ConfigPatch]:
        return iter(self.patches)

    def __len__(self) -> int:
        return len(self.patches)

    def __bool__(self) -> bool:
        return bool(self.patches)

    # ------------------------------------------------------------------
    # precedence
    # ------------------------------------------------------------------

    def ordered(self) -> list[ConfigPatch]:
        """Patches in application order: weakest first, strongest last."""
        return [
            patch
            for _, patch in sorted(
                enumerate(self.patches),
                key=lambda item: (*item[1].rank, item[0]),
            )
        ]

    def winners(self) -> dict[str, ConfigPatch]:
        """The patch that ends up owning each path."""
        out: dict[str, ConfigPatch] = {}
        for patch in self.ordered():
            out[patch.key] = patch
        return out

    # ------------------------------------------------------------------
    # checks
    # ------------------------------------------------------------------

    def conflicts(self) -> list[tuple[ConfigPatch, ConfigPatch]]:
        """Pairs that write the same path at the same rank with different values.

        Equal values are not a conflict: two sources agreeing is harmless, and
        several V1 entry points legitimately restate the same default.
        """
        seen: dict[tuple[str, tuple[int, int, int]], ConfigPatch] = {}
        out: list[tuple[ConfigPatch, ConfigPatch]] = []
        for patch in self.patches:
            key = (patch.key, patch.rank)
            previous = seen.get(key)
            if previous is None:
                seen[key] = patch
            elif previous.value != patch.value:
                out.append((previous, patch))
        return out

    def require_no_conflicts(self) -> None:
        """Refuse a path two equally-ranked sources disagree about.

        Deliberately an error rather than a rule. Nothing about
        ``--stages.thinker.tp_size 4 --set stages.thinker.tp_size=8`` says which
        one the user meant, and any tie-break this code invented -- argument
        order, flag length, alphabetical -- would be a rule people had to learn
        in order to predict what their own command line does.
        """
        conflicts = self.conflicts()
        if not conflicts:
            return
        blocks = [
            f"{first.path.raw} is set twice at the same precedence "
            f"({first.layer.name.lower()} layer, "
            f"{first.specificity.name.lower()}):\n"
            f"  {first.value!r}  <- {first.source.describe()}\n"
            f"  {second.value!r}  <- {second.source.describe()}\n"
            f"Pick one."
            for first, second in conflicts
        ]
        raise DuplicatePatchError("\n\n".join(blocks), raw=conflicts[0][0].path.raw)

    def deprecations(self) -> list[ConfigPatch]:
        return [patch for patch in self.patches if patch.deprecated]
