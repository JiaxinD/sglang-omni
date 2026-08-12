# SPDX-License-Identifier: Apache-2.0
"""Canonical configuration paths compiled from the typed pipeline schema.

Today four independent implementations know how to address a configuration
value: the dotted-path walker in :mod:`sglang_omni.config.manager`, the
stage-name deep merge for ``stage_overrides``, the factory-kwarg merge in
:mod:`sglang_omni.config.runtime`, and a dozen hand-written CLI helpers in
``cli/serve.py``. This module is the single addressing primitive meant to
replace all four.

A :class:`ConfigPath` is *compiled against the typed schema*, not evaluated
against a dict. That buys three things the current walker cannot offer:

* an illegal path fails at parse time with nearby legal paths, instead of
  raising ``KeyError`` deep inside a ``model_dump()`` copy;
* every path knows its declared type, so a CLI string can be coerced by
  pydantic rather than by ``int()``/``float()`` guessing;
* every path knows whether it is part of the public surface, so internal
  wiring such as ``factory_args`` can be refused up front.

Stages are addressed **by name** (``stages.thinker.runtime.max_seq_len``).
Positional indices are deliberately not accepted here; the V1 compatibility
layer is the only place allowed to translate ``stages.4.…`` into a name.
"""

from __future__ import annotations

import difflib
import types
import typing
from dataclasses import dataclass
from enum import Enum
from typing import Any, get_args, get_origin

from pydantic import BaseModel, TypeAdapter

from sglang_omni.config.schema import PipelineConfig

__all__ = [
    "ConfigPath",
    "ConfigPathError",
    "PathVisibility",
    "Segment",
    "SegmentKind",
    "coerce_scalar_text",
    "iter_schema_paths",
    "suggest_nearby",
]

# Guard against a pathological schema; the real tree is only a few levels deep.
_MAX_SCHEMA_DEPTH = 12


class SegmentKind(str, Enum):
    """What one dotted segment addresses."""

    FIELD = "field"
    """A declared field of a pydantic model."""

    NAMED_ITEM = "named_item"
    """An element of a name-keyed collection, e.g. ``stages.thinker``."""

    MAPPING_KEY = "mapping_key"
    """A key of a typed mapping, e.g. ``env.CUDA_VISIBLE_DEVICES``."""

    FREEFORM = "freeform"
    """A key below an untyped ``Any``; no schema information remains."""


class PathVisibility(str, Enum):
    """Whether a path may be written by a user-facing configuration source."""

    PUBLIC = "public"
    IDENTITY = "identity"
    DERIVED = "derived"
    INTERNAL = "internal"
    DEPRECATED = "deprecated"


# Ordered specific -> general; the first matching pattern wins. ``*`` matches
# exactly one segment, ``**`` matches zero or more trailing segments.
_VISIBILITY_RULES: tuple[tuple[str, PathVisibility, str], ...] = (
    (
        "config_cls",
        PathVisibility.DERIVED,
        "derived from the config class name in PipelineConfig.model_post_init",
    ),
    (
        "stages.*.name",
        PathVisibility.IDENTITY,
        "a stage name is its address; rename the stage in the document instead",
    ),
    (
        "stages.*.factory_args.**",
        PathVisibility.DEPRECATED,
        "factory kwargs are owned by the model config; prefer the typed runtime "
        "fields. Still writable because shipped models expose knobs here",
    ),
    (
        "stages.*.runtime_arg_map.**",
        PathVisibility.INTERNAL,
        "runtime_arg_map is wiring owned by the model config author",
    ),
    (
        "runtime_overrides.**",
        PathVisibility.DEPRECATED,
        "untyped SGLang escape hatch; prefer the typed runtime fields",
    ),
)


class ConfigPathError(ValueError):
    """A dotted path could not be compiled against the schema."""

    def __init__(
        self,
        message: str,
        *,
        raw: str,
        resolved_prefix: str = "",
        suggestions: tuple[str, ...] = (),
    ) -> None:
        self.raw = raw
        self.resolved_prefix = resolved_prefix
        self.suggestions = tuple(suggestions)
        if self.suggestions:
            message = f"{message}\n  did you mean: " + ", ".join(self.suggestions)
        super().__init__(message)


@dataclass(frozen=True)
class Segment:
    """One compiled segment of a canonical path."""

    raw: str
    kind: SegmentKind
    annotation: Any
    """Declared type of the value *at* this segment."""

    container: Any
    """Declared type the segment was resolved against."""


@dataclass(frozen=True)
class ConfigPath:
    """A dotted path compiled against a :class:`PipelineConfig` schema."""

    raw: str
    root: type[BaseModel]
    segments: tuple[Segment, ...]

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    @classmethod
    def parse(
        cls,
        raw: str,
        root: type[BaseModel] = PipelineConfig,
    ) -> "ConfigPath":
        """Compile ``raw`` against ``root``, or raise :class:`ConfigPathError`."""
        if not isinstance(raw, str) or not raw.strip():
            raise ConfigPathError(
                "Config path must be a non-empty string", raw=str(raw)
            )

        parts = raw.split(".")
        if any(not part for part in parts):
            raise ConfigPathError(f"Config path {raw!r} has an empty segment", raw=raw)

        segments: list[Segment] = []
        current: Any = root
        for index, part in enumerate(parts):
            prefix = ".".join(parts[:index])
            segments.append(_descend(current, part, raw=raw, prefix=prefix))
            current = segments[-1].annotation

        return cls(raw=raw, root=root, segments=tuple(segments))

    # ------------------------------------------------------------------
    # shape
    # ------------------------------------------------------------------

    @property
    def parts(self) -> tuple[str, ...]:
        return tuple(segment.raw for segment in self.segments)

    @property
    def value_type(self) -> Any:
        return self.segments[-1].annotation

    @property
    def is_leaf(self) -> bool:
        """True when nothing can be addressed below this path.

        Lists of scalars are leaves: they are replaced as a whole, never
        indexed into.
        """
        return not _is_traversable(self.value_type)

    @property
    def stage_name(self) -> str | None:
        """Name of the stage this path lives in, if any."""
        for index, segment in enumerate(self.segments):
            if segment.kind is SegmentKind.NAMED_ITEM and index > 0:
                if self.segments[index - 1].raw == "stages":
                    return segment.raw
        return None

    @property
    def parent(self) -> "ConfigPath | None":
        if len(self.segments) == 1:
            return None
        return ConfigPath(
            raw=".".join(self.parts[:-1]),
            root=self.root,
            segments=self.segments[:-1],
        )

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.raw

    # ------------------------------------------------------------------
    # visibility
    # ------------------------------------------------------------------

    @property
    def visibility(self) -> PathVisibility:
        return self._visibility_rule[0]

    @property
    def visibility_reason(self) -> str:
        return self._visibility_rule[1]

    @property
    def _visibility_rule(self) -> tuple[PathVisibility, str]:
        generic = _generic_form(self.segments)
        for pattern, visibility, reason in _VISIBILITY_RULES:
            if _pattern_matches(pattern, generic):
                return visibility, reason
        return PathVisibility.PUBLIC, ""

    def is_public(self) -> bool:
        """True when this path is part of the recommended surface.

        Stricter than :meth:`is_writable`: a deprecated path is still writable
        but should not be advertised, e.g. by completion or ``config explain``.
        """
        return self.visibility is PathVisibility.PUBLIC

    def is_writable(self) -> bool:
        """True when a user-facing source is allowed to write this path.

        Deprecated paths are writable. Refusing them would break configurations
        that work today, which is what a deprecation period exists to avoid;
        they carry :attr:`visibility_reason` as a warning instead.
        """
        return self.visibility in (PathVisibility.PUBLIC, PathVisibility.DEPRECATED)

    def require_writable(self) -> None:
        """Raise unless this path may be written by a user-facing source."""
        if self.is_writable():
            return
        raise ConfigPathError(
            f"Config path {self.raw!r} is {self.visibility.value} and cannot be "
            f"set directly: {self.visibility_reason}",
            raw=self.raw,
        )

    # ------------------------------------------------------------------
    # values
    # ------------------------------------------------------------------

    def coerce(self, value: Any) -> Any:
        """Convert a raw (usually textual) value into this path's declared type."""
        annotation = self.value_type
        if annotation is Any or annotation is None:
            return coerce_scalar_text(value)
        if not isinstance(value, str):
            return value
        try:
            return TypeAdapter(annotation).validate_python(value)
        except Exception:
            pass
        try:
            return TypeAdapter(annotation).validate_json(value)
        except Exception:
            return coerce_scalar_text(value)

    def read(self, source: BaseModel | dict[str, Any]) -> Any:
        """Read the value at this path from a config instance or a dumped dict."""
        current: Any = source
        if isinstance(current, BaseModel):
            current = current.model_dump()
        for segment in self.segments:
            current = _read_segment(current, segment, path=self.raw)
        return current

    def write(self, data: dict[str, Any], value: Any) -> None:
        """Assign ``value`` at this path inside a dumped config dict, in place.

        ``data`` is expected to be the output of ``PipelineConfig.model_dump()``.
        Missing containers are created only where the schema allows a free
        mapping key; a missing stage is an error, never a silent insert.
        """
        current: Any = data
        for segment in self.segments[:-1]:
            current = _read_segment(
                current, segment, path=self.raw, create_missing=True
            )
        last = self.segments[-1]
        if last.kind is SegmentKind.NAMED_ITEM:
            index = _named_index(current, last.raw, path=self.raw)
            current[index] = value
            return
        if not isinstance(current, dict):
            raise ConfigPathError(
                f"Cannot set {self.raw!r}: {_join(self.parts[:-1])} is "
                f"{type(current).__name__}, not a mapping",
                raw=self.raw,
            )
        current[last.raw] = value


# ----------------------------------------------------------------------
# schema walking
# ----------------------------------------------------------------------


def _descend(container: Any, part: str, *, raw: str, prefix: str) -> Segment:
    """Resolve one segment against ``container``'s declared type."""
    core = _unwrap_optional(container)

    if _is_model(core):
        fields = core.model_fields
        if part in fields:
            return Segment(
                raw=part,
                kind=SegmentKind.FIELD,
                annotation=fields[part].annotation,
                container=core,
            )
        raise ConfigPathError(
            f"{core.__name__} has no field {part!r}"
            + (f" at {prefix!r}" if prefix else ""),
            raw=raw,
            resolved_prefix=prefix,
            suggestions=_suggest(part, sorted(fields), prefix),
        )

    item_type = _named_collection_item(core)
    if item_type is not None:
        if part.isdigit():
            raise ConfigPathError(
                f"{_join_prefix(prefix)} is addressed by name, not by index; "
                f"use e.g. {_join_prefix(prefix)}.thinker instead of {part!r}",
                raw=raw,
                resolved_prefix=prefix,
            )
        # Stage names come from the document, not the schema, so any identifier
        # is structurally valid here; existence is checked when the path is
        # bound to a concrete config.
        return Segment(
            raw=part,
            kind=SegmentKind.NAMED_ITEM,
            annotation=item_type,
            container=core,
        )

    value_type = _mapping_value_type(core)
    if value_type is not None:
        return Segment(
            raw=part,
            kind=SegmentKind.MAPPING_KEY,
            annotation=value_type,
            container=core,
        )

    if core is Any:
        return Segment(
            raw=part, kind=SegmentKind.FREEFORM, annotation=Any, container=Any
        )

    if _is_plain_sequence(core):
        raise ConfigPathError(
            f"{_join_prefix(prefix)} is a list value and is replaced as a whole; "
            f"it cannot be indexed by {part!r}",
            raw=raw,
            resolved_prefix=prefix,
        )

    raise ConfigPathError(
        f"{_join_prefix(prefix)} is a leaf of type {_type_name(core)}; "
        f"nothing can be addressed below it (got {part!r})",
        raw=raw,
        resolved_prefix=prefix,
    )


def _unwrap_optional(annotation: Any) -> Any:
    """Strip ``| None`` from a union, leaving unions of real types untouched."""
    origin = get_origin(annotation)
    if origin is typing.Union or origin is types.UnionType:
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(args) == 1:
            return args[0]
        return annotation
    return annotation


def _is_model(annotation: Any) -> bool:
    return isinstance(annotation, type) and issubclass(annotation, BaseModel)


def _named_collection_item(annotation: Any) -> type[BaseModel] | None:
    """Return the item type when ``annotation`` is a name-keyed model list."""
    if get_origin(annotation) not in (list, tuple):
        return None
    args = get_args(annotation)
    if not args:
        return None
    item = _unwrap_optional(args[0])
    if _is_model(item) and "name" in item.model_fields:
        return item
    return None


def _mapping_value_type(annotation: Any) -> Any | None:
    if get_origin(annotation) is not dict:
        return None
    args = get_args(annotation)
    return args[1] if len(args) == 2 else Any


def _is_plain_sequence(annotation: Any) -> bool:
    return get_origin(annotation) in (list, tuple, set, frozenset)


def _is_traversable(annotation: Any) -> bool:
    core = _unwrap_optional(annotation)
    if _is_model(core):
        return True
    if _named_collection_item(core) is not None:
        return True
    if _mapping_value_type(core) is not None:
        return True
    return core is Any


def _type_name(annotation: Any) -> str:
    if isinstance(annotation, type):
        return annotation.__name__
    return str(annotation).replace("typing.", "")


# ----------------------------------------------------------------------
# value access on dumped dicts
# ----------------------------------------------------------------------


def _read_segment(
    current: Any,
    segment: Segment,
    *,
    path: str,
    create_missing: bool = False,
) -> Any:
    if segment.kind is SegmentKind.NAMED_ITEM:
        if not isinstance(current, list):
            raise ConfigPathError(
                f"Cannot resolve {path!r}: expected a list of named entries, "
                f"got {type(current).__name__}",
                raw=path,
            )
        return current[_named_index(current, segment.raw, path=path)]

    if not isinstance(current, dict):
        raise ConfigPathError(
            f"Cannot resolve {path!r}: expected a mapping at {segment.raw!r}, "
            f"got {type(current).__name__}",
            raw=path,
        )

    if segment.raw not in current or current[segment.raw] is None:
        if not create_missing:
            if segment.raw in current:
                return current[segment.raw]
            raise ConfigPathError(
                f"Cannot resolve {path!r}: {segment.raw!r} is not present",
                raw=path,
                suggestions=_suggest(segment.raw, sorted(map(str, current)), ""),
            )
        current[segment.raw] = {}
    return current[segment.raw]


def _named_index(items: list[Any], name: str, *, path: str) -> int:
    for index, item in enumerate(items):
        if isinstance(item, dict) and item.get("name") == name:
            return index
    available = [
        str(item["name"]) for item in items if isinstance(item, dict) and "name" in item
    ]
    raise ConfigPathError(
        f"Cannot resolve {path!r}: no entry named {name!r}",
        raw=path,
        suggestions=_suggest(name, available, ""),
    )


# ----------------------------------------------------------------------
# patterns, suggestions, schema enumeration
# ----------------------------------------------------------------------


def _generic_form(segments: tuple[Segment, ...]) -> tuple[str, ...]:
    """Replace document-defined names with ``*`` so rules stay model-agnostic."""
    return tuple(
        "*" if segment.kind is SegmentKind.NAMED_ITEM else segment.raw
        for segment in segments
    )


def _pattern_matches(pattern: str, parts: tuple[str, ...]) -> bool:
    pattern_parts = pattern.split(".")
    if pattern_parts and pattern_parts[-1] == "**":
        head = pattern_parts[:-1]
        if len(parts) < len(head):
            return False
        return all(p in ("*", q) for p, q in zip(head, parts))
    if len(pattern_parts) != len(parts):
        return False
    return all(p in ("*", q) for p, q in zip(pattern_parts, parts))


def _suggest(word: str, candidates: list[str], prefix: str) -> tuple[str, ...]:
    if not candidates:
        return ()
    close = difflib.get_close_matches(word, candidates, n=3, cutoff=0.5)
    chosen = close or candidates[:5]
    return tuple(f"{prefix}.{name}" if prefix else name for name in chosen)


def _join(parts: tuple[str, ...]) -> str:
    return ".".join(parts)


def _join_prefix(prefix: str) -> str:
    return prefix or "<root>"


def coerce_scalar_text(value: Any) -> Any:
    """Best-effort scalar parsing for untyped (``Any``) positions.

    Mirrors the historical behaviour of ``ConfigManager._convert_scalar`` so
    free-form escape hatches keep parsing the same way.
    """
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


def iter_schema_paths(
    root: type[BaseModel] = PipelineConfig,
    *,
    include_non_public: bool = False,
) -> list[str]:
    """Enumerate every addressable path in the schema.

    Name-keyed collections and free mappings are rendered with a ``*``
    placeholder because their keys come from the document, not the schema.
    """
    out: list[str] = []

    def walk(annotation: Any, prefix: tuple[str, ...], depth: int) -> None:
        if depth > _MAX_SCHEMA_DEPTH:
            return
        core = _unwrap_optional(annotation)

        if _is_model(core):
            for name, field in core.model_fields.items():
                child = prefix + (name,)
                _emit(child)
                walk(field.annotation, child, depth + 1)
            return

        item = _named_collection_item(core)
        if item is not None:
            child = prefix + ("*",)
            walk(item, child, depth + 1)
            return

        value_type = _mapping_value_type(core)
        if value_type is not None and value_type is not Any:
            child = prefix + ("*",)
            _emit(child)
            walk(value_type, child, depth + 1)
            return

    def _emit(parts: tuple[str, ...]) -> None:
        if not include_non_public:
            for pattern, visibility, _ in _VISIBILITY_RULES:
                if visibility is not PathVisibility.PUBLIC and _pattern_matches(
                    pattern, parts
                ):
                    return
        out.append(".".join(parts))

    walk(root, (), 0)
    return out


def suggest_nearby(
    raw: str,
    root: type[BaseModel] = PipelineConfig,
    *,
    limit: int = 5,
) -> tuple[str, ...]:
    """Return legal paths close to ``raw``, for error messages and CLI hints.

    The comparison happens in generic form (document-defined keys replaced by
    ``*``) so that ``stages.thinker.runtime.max_seq`` is matched against
    ``stages.*.runtime.max_seq_len`` and reported back with ``thinker`` kept.
    """
    parts = raw.split(".")
    names: dict[int, str] = {}
    generic_parts = list(parts)
    for index, part in enumerate(parts[:-1]):
        if part == "stages":
            names[index + 1] = parts[index + 1]
            generic_parts[index + 1] = "*"

    candidates = iter_schema_paths(root, include_non_public=False)
    close = (
        difflib.get_close_matches(
            ".".join(generic_parts), candidates, n=limit, cutoff=0.4
        )
        or candidates[:limit]
    )

    out: list[str] = []
    for candidate in close:
        candidate_parts = candidate.split(".")
        for index, name in names.items():
            if index < len(candidate_parts) and candidate_parts[index] == "*":
                candidate_parts[index] = name
        out.append(".".join(candidate_parts))
    return tuple(out)
