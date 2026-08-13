from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from enum import Enum
from typing import Annotated, Any, Iterator, NamedTuple

import typer
import yaml

from sglang_omni.config.compat import (
    canonicalize_dotted_key,
    patches_from_dotted_cli,
    sources_from_config_file,
)
from sglang_omni.config.manager import ConfigManager, resolve_config_cls_for_model_path
from sglang_omni.config.patch import ConfigPatchSet
from sglang_omni.config.path import ConfigPath, ConfigPathError
from sglang_omni.config.resolver import ConfigResolver, ResolvedConfig, diff_configs
from sglang_omni.config.schema import PipelineConfig
from sglang_omni.config.shadow import SHADOW_ENV

logger = logging.getLogger(__name__)

config_app = typer.Typer(help="Inspect, resolve and export the pipeline configuration")

_MODEL_PATH_HELP = "The Hugging Face model ID or the path to the model directory."
_CONFIG_HELP = "Path to a pipeline config file, as accepted by `sgl-omni serve`."
_TEXT_ONLY_HELP = "Use the thinker-only pipeline, as `sgl-omni serve --text-only` does."


def _dump_yaml(data: Any) -> str:
    return yaml.dump(
        data,
        sort_keys=False,  # preserve order
        default_flow_style=False,  # use block style (not inline)
        indent=2,  # control indentation
        allow_unicode=True,
    )


@config_app.command()
def view(
    model_path: Annotated[
        str,
        typer.Option(
            help="The Hugging Face model ID or the path to the model directory."
        ),
    ],
) -> None:
    """View the model's pipeline configuration."""
    config_cls = resolve_config_cls_for_model_path(model_path)
    config = config_cls(model_path=model_path)
    print(_dump_yaml(config.model_dump(mode="json")))


@config_app.command()
def export(
    model_path: Annotated[
        str,
        typer.Option(
            help="The Hugging Face model ID or the path to the model directory."
        ),
    ],
    output_path: Annotated[
        str, typer.Option(help="Path to the output JSON file.")
    ] = None,
) -> None:
    """Export the default pipeline configuration to a YAML file."""
    # get the default pipeline config for the model

    config_cls = resolve_config_cls_for_model_path(model_path)
    config = config_cls(model_path=model_path)

    # export config in a yaml file
    if output_path is None:
        output_path = f"./config_{config.name}.yaml"

    with open(output_path, "w") as f:
        f.write(_dump_yaml(config.model_dump(mode="json")))
    print(f"Pipeline config exported to {output_path}")


# ----------------------------------------------------------------------
# resolve / explain
# ----------------------------------------------------------------------


class ResolveOutput(str, Enum):
    """What ``config resolve`` writes to stdout."""

    config = "config"
    diff = "diff"
    provenance = "provenance"


@contextmanager
def _shadow_disabled() -> Iterator[None]:
    """Silence the in-process shadow comparison for the duration of a call.

    These commands run the V1 chain themselves and report the comparison as
    output. Letting ``ConfigManager`` log it as well would say the same thing
    twice, in a less useful form.
    """
    previous = os.environ.get(SHADOW_ENV)
    os.environ[SHADOW_ENV] = "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(SHADOW_ENV, None)
        else:
            os.environ[SHADOW_ENV] = previous


class Resolution(NamedTuple):
    """Everything these commands need to both answer and check themselves."""

    baseline: PipelineConfig
    """The config before any source was applied, for `--show diff`."""

    resolved: ResolvedConfig
    v1_config: PipelineConfig
    """What the still-authoritative V1 merge chain produced from the same input."""


def _resolve_sources(
    *,
    model_path: str | None,
    config_file: str | None,
    text_only: bool,
    argv: list[str],
) -> Resolution:
    """Build the configuration ``sgl-omni serve`` would build, twice.

    Once through the resolver, which carries provenance, and once through the
    V1 merge chain, which is still what a launch actually uses. Returning both
    is what lets these commands state whether the answer they print is the
    answer the server would compute, rather than asking the reader to trust it.
    """
    if config_file is None and model_path is None:
        raise typer.BadParameter("--model-path is required unless --config is set")

    # Every V1 call is made with the in-process shadow comparison off: this
    # command performs that comparison itself and prints the result.
    with _shadow_disabled():
        if config_file:
            baseline, patches = sources_from_config_file(config_file)
            manager = ConfigManager.from_file(config_file)
        else:
            manager = ConfigManager.from_model_path(
                str(model_path), variant="text" if text_only else None
            )
            baseline, patches = manager.config, ConfigPatchSet()
        extra_args = manager.parse_extra_args(list(argv))
        v1_config = manager.merge_config(extra_args)

    patches = patches.merge(
        patches_from_dotted_cli(extra_args, baseline, origin="command line")
    )
    return Resolution(baseline, ConfigResolver(baseline).resolve(patches), v1_config)


def _report_sources(resolution: Resolution) -> None:
    """Write deprecation notices and the V1 comparison to stderr.

    stderr, so that ``config resolve > pipeline.yaml`` yields a usable file.
    """
    for patch in resolution.resolved.patches.deprecations():
        typer.secho(
            f"deprecated: {patch.key} <- {patch.source.describe()}: {patch.deprecated}",
            err=True,
            fg=typer.colors.YELLOW,
        )

    differences = diff_configs(resolution.v1_config, resolution.resolved.config)
    if not differences:
        return
    typer.secho(
        f"warning: the resolver and the V1 merge chain disagree on "
        f"{len(differences)} field(s). The V1 result is what a launch would "
        f"use; this is a bug in the resolver, please report it.",
        err=True,
        fg=typer.colors.RED,
    )
    for difference in differences:
        typer.secho(f"  {difference.render()}", err=True, fg=typer.colors.RED)


@config_app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def resolve(
    ctx: typer.Context,
    model_path: Annotated[str | None, typer.Option(help=_MODEL_PATH_HELP)] = None,
    config: Annotated[str | None, typer.Option(help=_CONFIG_HELP)] = None,
    text_only: Annotated[
        bool, typer.Option("--text-only", help=_TEXT_ONLY_HELP)
    ] = False,
    show: Annotated[
        ResolveOutput,
        typer.Option(
            help=(
                "config: the whole resolved pipeline. "
                "diff: only the fields the sources changed. "
                "provenance: every source that contributed, winner last."
            )
        ),
    ] = ResolveOutput.config,
) -> None:
    """Show the configuration a `serve` command with these arguments would use.

    Takes the same arguments as `sgl-omni serve`, including dotted overrides:

        sgl-omni config resolve --model-path Qwen/Qwen3-Omni \\
            --stages.thinker.tp_size 4 --show diff
    """
    resolution = _resolve_sources(
        model_path=model_path,
        config_file=config,
        text_only=text_only,
        argv=ctx.args,
    )
    _report_sources(resolution)
    provenance = resolution.resolved.provenance

    if show is ResolveOutput.config:
        print(_dump_yaml(resolution.resolved.config.model_dump(mode="json")))
        return

    if show is ResolveOutput.diff:
        # Against the baseline config rather than against the recorded patches,
        # so that a field an alias carried along is shown as changed too.
        changes = diff_configs(resolution.baseline, resolution.resolved.config)
        if not changes:
            print("No configuration source changed the pipeline's defaults.")
            return
        for change in changes:
            print(f"{change.path}: {change.expected!r} -> {change.actual!r}")
        return

    if not provenance.paths():
        print("No configuration source touched the pipeline's defaults.")
        return
    print("\n\n".join(provenance.explain(path) for path in provenance.paths()))


@config_app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def explain(
    ctx: typer.Context,
    path: Annotated[
        str | None,
        typer.Argument(
            help=(
                "Canonical config path, e.g. stages.thinker.runtime.max_seq_len. "
                "Omit to list every path a source touched. Pass it before any "
                "dotted override so it is not mistaken for one."
            )
        ),
    ] = None,
    model_path: Annotated[str | None, typer.Option(help=_MODEL_PATH_HELP)] = None,
    config: Annotated[str | None, typer.Option(help=_CONFIG_HELP)] = None,
    text_only: Annotated[
        bool, typer.Option("--text-only", help=_TEXT_ONLY_HELP)
    ] = False,
) -> None:
    """Say where one configuration value came from, and what it overrode.

        sgl-omni config explain stages.thinker.runtime.max_seq_len \\
            --config omni.yaml --stages.thinker.runtime.max_seq_len 8192
    """
    resolution = _resolve_sources(
        model_path=model_path,
        config_file=config,
        text_only=text_only,
        argv=ctx.args,
    )
    _report_sources(resolution)
    provenance = resolution.resolved.provenance

    if path is None:
        if not provenance.paths():
            print("No configuration source touched the pipeline's defaults.")
            return
        for touched in provenance.paths():
            winner = provenance.winner(touched)
            assert winner is not None  # a touched path always has a winner
            print(f"{touched} = {winner.value!r}  <- {winner.source.describe()}")
        return

    canonical, deprecation = canonicalize_dotted_key(path, resolution.baseline)
    if deprecation:
        typer.secho(f"deprecated: {deprecation}", err=True, fg=typer.colors.YELLOW)
    try:
        compiled = ConfigPath.parse(canonical, type(resolution.baseline))
    except ConfigPathError as exc:
        # Carries `did you mean:` suggestions, which a traceback would bury.
        raise typer.BadParameter(str(exc)) from exc

    if provenance.touched(compiled.raw):
        print(provenance.explain(compiled.raw))
        return
    value = compiled.read(resolution.resolved.config)
    print(f"{compiled.raw} = {value!r}")
    print("  no source touched this path; the value is the model's own default")
