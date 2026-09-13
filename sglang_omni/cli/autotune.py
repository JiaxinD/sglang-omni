"""Restage search planning commands."""

from pathlib import Path
from typing import Annotated

import typer

from sglang_omni.cli.config import _resolve_sources
from sglang_omni.restage.plan import SearchSpace, write_plan

autotune_app = typer.Typer(help="Plan and inspect Restage configuration searches.")


@autotune_app.command()
def plan(
    config: Annotated[
        Path, typer.Option(exists=True, dir_okay=False, help="Serving pipeline YAML.")
    ],
    search_space: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="JSON device budget, replica counts and configuration dimensions.",
        ),
    ],
    output: Annotated[
        Path, typer.Option(help="New directory for candidate YAML and search records.")
    ],
    max_candidates: Annotated[
        int,
        typer.Option(
            min=1,
            help="Maximum examined records; includes rejections. Summary reports truncation.",
        ),
    ] = 256,
):
    """Export unmeasured candidates using the serving configuration rules."""
    try:
        space = SearchSpace.model_validate_json(
            search_space.read_text(encoding="utf-8")
        )
        resolution = _resolve_sources(
            model_path=None,
            config_file=str(config),
            text_only=False,
            mem_fraction_static=None,
            argv=[],
        )
        summary = write_plan(
            resolution.resolved.config, space, output, max_candidates=max_candidates
        )
    except (ValueError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    scope = "exhausted" if summary["complete"] else "truncated at the record limit"
    typer.echo(
        f"Exported {summary['accepted']} candidates; rejected {summary['rejected']}. "
        f"Search {scope}. Performance not measured. Results: {output}"
    )
