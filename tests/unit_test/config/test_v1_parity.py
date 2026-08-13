# SPDX-License-Identifier: Apache-2.0
"""Differential proof that the resolver accepts V1 configuration unchanged.

The resolver replaces a hand-written merge chain that shipped for a long time,
so "does the new one behave like the old one" has to be answered over the
*whole path space*, not over the handful of paths existing tests happen to
touch. This module therefore generates its own corpus:

    for every path the schema can address
      for every probe value that path's declared type admits
        the frozen V1 oracle and the resolver must agree

"Agree" means one of:

* both produce a config, and :func:`diff_configs` finds no field that differs;
* both reject the input -- a probe value being illegal for a field is normal,
  and what matters is that the two chains draw the same accept/reject line.

Anything else is a divergence and fails, unless it is listed in
:data:`INTENTIONAL_DIFFERENCES` with a reason. That table is the deliverable:
it is the complete, reviewable set of ways V2 departs from V1, which is a
stronger statement than "shadow mode observed no divergence at runtime".
"""

from __future__ import annotations

import types
import typing
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Union, get_args, get_origin

import pytest
from pydantic import BaseModel

from sglang_omni.config.compat import patches_from_dotted_cli
from sglang_omni.config.path import ConfigPath, iter_schema_paths
from sglang_omni.config.resolver import ConfigResolver, diff_configs
from sglang_omni.config.schema import PipelineConfig
from tests.unit_test.config.conftest import build_pipeline_config
from tests.unit_test.config.v1_oracle import (
    v1_apply_stage_overrides,
    v1_convert_scalar,
    v1_merge_dotted_cli,
)

# ----------------------------------------------------------------------
# the intentional-difference table
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class IntentionalDifference:
    """One sanctioned departure from V1 behaviour.

    ``pattern`` is matched against the *generic* form of the path (document
    keys replaced by ``*``), using the same wildcard rules as the visibility
    table: ``*`` matches one segment, ``**`` matches any trailing segments.
    ``when`` narrows an entry to the probe values it actually describes, so a
    broad pattern does not quietly sanction unrelated divergences on the same
    path.
    """

    pattern: str
    kind: str
    reason: str
    when: Callable[[str], bool] = lambda text: True

    def covers(self, divergence: "Divergence", generic: tuple[str, ...]) -> bool:
        return (
            self.kind == divergence.kind
            and _pattern_matches(self.pattern, generic)
            and self.when(divergence.text)
        )


V2_ACCEPTS = "v2_accepts"
"""V1 rejected the input, the resolver takes it."""

V2_REJECTS = "v2_rejects"
"""V1 took the input, the resolver rejects it."""

VALUE = "value"
"""Both produce a config, but some field ends up different."""


def _is_json_text(text: str) -> bool:
    return text.startswith(("{", "["))


def _v1_retyped_the_text(text: str) -> bool:
    """True when ``_convert_scalar`` turned this text into a non-string.

    Deliberately excludes the ``none`` sentinel: that one is now honoured
    identically by both chains, and letting it fall under this predicate would
    disarm the probe that caught the divergence in the first place.
    """
    return text.lower() != _NONE_TEXT and v1_convert_scalar(text) != text


INTENTIONAL_DIFFERENCES: tuple[IntentionalDifference, ...] = (
    IntentionalDifference(
        pattern="stages.*.comm.**",
        kind=V2_ACCEPTS,
        reason=(
            "V1 walked the dumped config with plain subscripting, so writing "
            "into an optional container that is currently None -- "
            "``--stages.thinker.comm.credits 4`` with no ``comm`` block -- died "
            "with a bare TypeError. ConfigPath.write builds the container when "
            "the schema allows it, and the model's defaults fill the rest."
        ),
    ),
    IntentionalDifference(
        pattern="**",
        kind=V2_ACCEPTS,
        when=_is_json_text,
        reason=(
            "V1 assigned CLI text verbatim and let the final model rebuild "
            "reject it, so a container field could not be set from the command "
            "line at all. ConfigPath.coerce parses JSON, which is what makes "
            '``--stages.thinker.env \'{"OMP_NUM_THREADS": "8"}\'`` work.'
        ),
    ),
    IntentionalDifference(
        pattern="**",
        kind=V2_ACCEPTS,
        when=_v1_retyped_the_text,
        reason=(
            "V1 retyped ``true``/``false``/digits before knowing the target "
            "field, so a string field could never hold those words: "
            "``--stages.thinker.process 7`` handed pydantic an int. V2 coerces "
            "against the declared type, so the text survives where the field "
            "wants a string and is still parsed where it wants a number."
        ),
    ),
)


def _pattern_matches(pattern: str, parts: tuple[str, ...]) -> bool:
    """``*`` matches one segment, ``**`` matches zero or more trailing ones."""
    expected = pattern.split(".")
    for index, token in enumerate(expected):
        if token == "**":
            return True
        if index >= len(parts):
            return False
        if token != "*" and token != parts[index]:
            return False
    return len(parts) == len(expected)


def _generic_form(path: str, stage_names: frozenset[str]) -> tuple[str, ...]:
    parts = path.split(".")
    out: list[str] = []
    for index, part in enumerate(parts):
        previous = parts[index - 1] if index else ""
        out.append("*" if previous == "stages" and part in stage_names else part)
    return tuple(out)


# ----------------------------------------------------------------------
# probe corpus
# ----------------------------------------------------------------------

_NONE_TYPE = type(None)
_PROBE_KEY = "PROBE_KEY"


def _union_members(annotation: Any) -> tuple[list[Any], bool]:
    """Split a possibly-optional annotation into members and an optional flag."""
    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        args = get_args(annotation)
        members = [arg for arg in args if arg is not _NONE_TYPE]
        return members, len(members) != len(args)
    return [annotation], False


def sample_texts(annotation: Any) -> list[str]:
    """CLI-shaped probe values for a declared type.

    These are strings because that is what both chains actually receive: argv
    carries text, and turning text into a typed value is precisely the step
    where V1's ``_convert_scalar`` and V2's ``ConfigPath.coerce`` could drift.

    Every path also gets the words V1 rewrote *regardless of the declared
    type* -- ``none``, ``true``, and digits. Sampling only type-appropriate
    values would never reach the case the sentinel bug lived in.
    """
    members, optional = _union_members(annotation)
    out: list[str] = []
    for member in members:
        out.extend(_sample_member(member))
    out.extend(_V1_REWRITTEN_TEXTS)
    if optional:
        out.append(_NONE_TEXT)
    # Preserve order, drop duplicates from overlapping union members.
    return list(dict.fromkeys(out))


_NONE_TEXT = "none"
_V1_REWRITTEN_TEXTS = (_NONE_TEXT, "true", "7")
"""Texts ``ConfigManager._convert_scalar`` turned into non-strings."""


def _sample_member(annotation: Any) -> list[str]:
    if annotation is Any:
        return ["probe", "7", "true"]
    if annotation is bool:
        return ["true", "false"]
    if annotation is int:
        return ["7"]
    if annotation is float:
        return ["0.5"]
    if annotation is str:
        return ["probe"]

    origin = get_origin(annotation)
    if origin in (list, typing.List):
        (item,) = get_args(annotation) or (Any,)
        element = _sample_member(item)[:1] or ["probe"]
        return [f'["{element[0]}"]' if item is str else f"[{element[0]}]"]
    if origin in (dict, typing.Dict):
        return ['{"' + _PROBE_KEY + '": "probe"}']
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return ["{}"]
    return ["probe"]


def concrete_paths(config: PipelineConfig) -> list[str]:
    """Every schema path, with ``*`` bound to keys this config actually has."""
    stage_names = [stage.name for stage in config.stages]
    out: list[str] = []
    for generic in iter_schema_paths(type(config), include_non_public=True):
        for concrete in _bind(generic.split("."), stage_names):
            out.append(concrete)
    return out


def _bind(parts: list[str], stage_names: list[str]) -> list[str]:
    """Expand ``*`` placeholders: stage slots fan out, free keys get a probe."""
    results: list[list[str]] = [[]]
    for index, part in enumerate(parts):
        previous = parts[index - 1] if index else ""
        if part != "*":
            options = [part]
        elif previous == "stages":
            options = list(stage_names)
        else:
            options = [_PROBE_KEY]
        results = [prefix + [option] for prefix in results for option in options]
    return [".".join(parts) for parts in results]


# ----------------------------------------------------------------------
# the two chains
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Divergence:
    path: str
    text: str
    kind: str
    detail: str

    def render(self) -> str:
        return f"[{self.kind}] --{self.path} {self.text!r}: {self.detail}"


def _capture(fn) -> tuple[bool, Any]:
    try:
        return True, fn()
    except Exception as exc:  # noqa: BLE001 - the accept/reject line is the point
        return False, exc


def _v2_merge(config: PipelineConfig, extra_args: dict[str, Any]) -> PipelineConfig:
    patches = patches_from_dotted_cli(extra_args, config)
    return ConfigResolver(config).resolve(patches).config


def compare_dotted(
    config: PipelineConfig,
    path: str,
    text: str,
) -> Divergence | None:
    """Run one probe through both chains and classify any disagreement."""
    args = {path: text}
    v1_ok, v1_out = _capture(lambda: v1_merge_dotted_cli(config, dict(args)))
    v2_ok, v2_out = _capture(lambda: _v2_merge(config, dict(args)))

    if not v1_ok and not v2_ok:
        return None
    if v1_ok and not v2_ok:
        return Divergence(path, text, V2_REJECTS, f"{type(v2_out).__name__}: {v2_out}")
    if v2_ok and not v1_ok:
        return Divergence(path, text, V2_ACCEPTS, f"V1 raised {type(v1_out).__name__}")

    differences = diff_configs(v1_out, v2_out)
    if not differences:
        return None
    return Divergence(
        path,
        text,
        VALUE,
        "; ".join(difference.render() for difference in differences[:4]),
    )


def sweep(config: PipelineConfig) -> list[Divergence]:
    """Every path × every probe value for one config."""
    out: list[Divergence] = []
    for path in concrete_paths(config):
        try:
            annotation = ConfigPath.parse(path, type(config)).value_type
        except Exception:
            # A path the compiler itself rejects is covered by the probe below:
            # feed it anyway and let the accept/reject comparison speak.
            annotation = Any
        for text in sample_texts(annotation):
            divergence = compare_dotted(config, path, text)
            if divergence is not None:
                out.append(divergence)
    return out


_SWEEP_CACHE: dict[str, tuple[PipelineConfig, list[Divergence]]] = {}


def cached_sweep(label: str, config: PipelineConfig) -> list[Divergence]:
    """Sweep once per config; several tests read the same result."""
    if label not in _SWEEP_CACHE:
        _SWEEP_CACHE[label] = (config, sweep(config))
    return _SWEEP_CACHE[label][1]


def unsanctioned(
    divergences: list[Divergence],
    stage_names: frozenset[str],
) -> list[Divergence]:
    """Drop the divergences the table accounts for; everything else is a bug."""
    out: list[Divergence] = []
    for divergence in divergences:
        generic = _generic_form(divergence.path, stage_names)
        if any(entry.covers(divergence, generic) for entry in INTENTIONAL_DIFFERENCES):
            continue
        out.append(divergence)
    return out


def exercised_entries(
    divergences: list[Divergence],
    stage_names: frozenset[str],
) -> set[int]:
    """Indices of table entries that at least one divergence actually needed."""
    out: set[int] = set()
    for divergence in divergences:
        generic = _generic_form(divergence.path, stage_names)
        for index, entry in enumerate(INTENTIONAL_DIFFERENCES):
            if entry.covers(divergence, generic):
                out.add(index)
    return out


def _report(divergences: list[Divergence]) -> str:
    lines = [divergence.render() for divergence in divergences[:40]]
    if len(divergences) > 40:
        lines.append(f"... and {len(divergences) - 40} more")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------


class TestDottedCliParity:
    """The dotted-CLI surface, swept exhaustively."""

    def _sweep(self) -> tuple[PipelineConfig, list[Divergence], frozenset[str]]:
        config = build_pipeline_config()
        divergences = cached_sweep("fixture", config)
        return config, divergences, frozenset(s.name for s in config.stages)

    def test_no_unsanctioned_divergence(self) -> None:
        _, divergences, stage_names = self._sweep()
        remaining = unsanctioned(divergences, stage_names)
        assert (
            not remaining
        ), f"{len(remaining)} unsanctioned V1/V2 divergences:\n" + _report(remaining)

    def test_nothing_v1_accepted_is_rejected(self) -> None:
        """The property that actually defines compatibility.

        Every sanctioned difference is V2 being *more* permissive. If a probe
        ever lands in the other direction, some configuration that works today
        would stop launching -- so this stays a separate assertion rather than
        something the table can be widened to cover.
        """
        _, divergences, _ = self._sweep()
        regressions = [d for d in divergences if d.kind in (V2_REJECTS, VALUE)]
        assert (
            not regressions
        ), "the resolver refuses or alters input the V1 chain accepted:\n" + _report(
            regressions
        )

    def test_every_table_entry_is_exercised(self) -> None:
        """A sanction nothing hits is a sanction that has outlived its cause."""
        _, divergences, stage_names = self._sweep()
        exercised = exercised_entries(divergences, stage_names)
        stale = [
            entry.pattern
            for index, entry in enumerate(INTENTIONAL_DIFFERENCES)
            if index not in exercised
        ]
        assert not stale, (
            "INTENTIONAL_DIFFERENCES entries no longer match any divergence "
            f"and should be deleted: {stale}"
        )

    def test_corpus_is_not_trivially_small(self) -> None:
        """Guard against the sweep silently degenerating to a handful of probes."""
        config = build_pipeline_config()
        paths = concrete_paths(config)
        probes = sum(
            len(sample_texts(ConfigPath.parse(path, type(config)).value_type))
            for path in paths
        )
        assert len(paths) > 90
        assert probes > 300

    def test_optional_sentinel_is_probed(self) -> None:
        """``none`` must reach every field, or difference #1 goes unseen."""
        assert _NONE_TEXT in sample_texts(str | None)
        assert _NONE_TEXT in sample_texts(int | None)
        assert _NONE_TEXT in sample_texts(str | list[str] | None)
        assert _NONE_TEXT in sample_texts(str)


class TestKnownV1Behaviours:
    """Named cases the generated sweep cannot express."""

    def test_optional_sentinel_clears_the_field(self) -> None:
        """``--…merge_fn none`` clears the field; it does not store "none"."""
        config = build_pipeline_config()
        args = {"stages.thinker.merge_fn": "none"}
        assert v1_merge_dotted_cli(config, dict(args)).stages[1].merge_fn is None
        assert _v2_merge(config, dict(args)).stages[1].merge_fn is None

    def test_optional_sentinel_is_honoured_even_when_it_invalidates(self) -> None:
        """Clearing a required-by-cross-check field must fail, not quietly pass.

        ``process`` is optional on the field but required by the pipeline's own
        validation. Storing the string "none" here would sail through both and
        reach the process planner as a stage named "none".
        """
        config = build_pipeline_config()
        args = {"stages.thinker.process": "none"}
        with pytest.raises(ValueError, match="must declare process"):
            v1_merge_dotted_cli(config, dict(args))
        with pytest.raises(ValueError, match="must declare process"):
            _v2_merge(config, dict(args))

    @pytest.mark.parametrize(
        "key",
        ["stages.thinker.tp_size", "stages.thinker.parallelism.tp"],
    )
    def test_parallelism_alias_is_mirrored(self, key: str) -> None:
        config = build_pipeline_config()
        v1 = v1_merge_dotted_cli(config, {key: "4"})
        v2 = _v2_merge(config, {key: "4"})
        assert (v1.stages[1].tp_size, v1.stages[1].parallelism.tp) == (4, 4)
        assert not diff_configs(v1, v2)

    def test_positional_stage_index(self) -> None:
        config = build_pipeline_config()
        args = {"stages.1.tp_size": "2"}
        v1 = v1_merge_dotted_cli(config, dict(args))
        v2 = _v2_merge(config, dict(args))
        assert not diff_configs(v1, v2)

    def test_positional_index_is_deprecated(self) -> None:
        config = build_pipeline_config()
        patches = patches_from_dotted_cli({"stages.1.tp_size": "2"}, config)
        assert [patch.deprecated for patch in patches.deprecations()] != []

    def test_several_keys_at_once(self) -> None:
        config = build_pipeline_config()
        args = {
            "stages.preprocessing.tp_size": "2",
            "stages.thinker.runtime.max_seq_len": "4096",
            # A non-numeric env value on purpose: V1 retyped "8" to an int and
            # then failed the dict[str, str] check, which is the sanctioned
            # difference exercised by the sweep rather than something to assert
            # equality on here.
            "stages.thinker.env.CUDA_VISIBLE_DEVICES": "0,1",
            "name": "probe-pipeline",
        }
        v1 = v1_merge_dotted_cli(config, dict(args))
        v2 = _v2_merge(config, dict(args))
        assert not diff_configs(v1, v2)


class TestStageOverridesParity:
    """The YAML ``stage_overrides`` block."""

    def _v2(self, config: PipelineConfig, overrides: dict[str, Any]) -> PipelineConfig:
        from sglang_omni.config.compat import patches_from_stage_overrides

        patches = patches_from_stage_overrides(overrides, config)
        return ConfigResolver(config).resolve(patches).config

    @pytest.mark.parametrize(
        "overrides",
        [
            {},
            {"thinker": {}},
            {"thinker": {"runtime": {"max_seq_len": 8192}}},
            {"thinker": {"runtime": {"max_seq_len": 8192, "video_fps": 2.0}}},
            {
                "preprocessing": {"runtime": {"max_seq_len": 1024}},
                "thinker": {"runtime": {"max_seq_len": 2048}},
            },
        ],
    )
    def test_accepted_overrides_match(self, overrides: dict[str, Any]) -> None:
        config = build_pipeline_config()
        v1 = v1_apply_stage_overrides(config, overrides)
        v2 = self._v2(config, overrides)
        assert not diff_configs(v1, v2)

    @pytest.mark.parametrize(
        "overrides",
        [
            "not-a-mapping",
            {"nope": {"runtime": {}}},
            {"thinker": "not-a-mapping"},
            {"thinker": {"tp_size": 4}},
            {"thinker": {"runtime": "not-a-mapping"}},
        ],
    )
    def test_rejected_overrides_report_the_same_message(self, overrides: Any) -> None:
        config = build_pipeline_config()
        with pytest.raises(ValueError) as v1_error:
            v1_apply_stage_overrides(config, overrides)
        with pytest.raises(ValueError) as v2_error:
            self._v2(config, overrides)
        assert str(v1_error.value) == str(v2_error.value)


class TestParityOnShippedConfigs:
    """The same sweep against configs that ship with the repo.

    Requires the model packages (and therefore ``sglang``); skipped where they
    are not installed, which is why the fixture sweep above has to stand on its
    own.
    """

    def _registry_configs(self) -> list[tuple[str, PipelineConfig]]:
        registry = pytest.importorskip("sglang_omni.models.registry")
        configs = registry.PIPELINE_CONFIG_REGISTRY.configs
        out: list[tuple[str, PipelineConfig]] = []
        for arch in sorted(configs):
            try:
                out.append((arch, configs[arch](model_path=f"/models/{arch}")))
            except Exception:
                # A config that cannot be built without weights is out of scope
                # here; the fixture sweep still covers the shared schema.
                continue
        return out

    def test_shipped_configs(self) -> None:
        configs = self._registry_configs()
        if not configs:
            pytest.skip("no shipped pipeline config could be instantiated")
        failures: list[str] = []
        for name, config in configs:
            stage_names = frozenset(stage.name for stage in config.stages)
            divergences = unsanctioned(cached_sweep(name, config), stage_names)
            if divergences:
                failures.append(f"{name}:\n" + _report(divergences))
        assert not failures, "\n\n".join(failures)
