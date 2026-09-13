"""Restage planning and measured search commands."""

import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer

from sglang_omni.cli.config import _resolve_sources
from sglang_omni.restage.capacity import CapacityCatalog, CapacityContext
from sglang_omni.restage.plan import SearchSpace, write_plan

autotune_app = typer.Typer(
    help="Plan candidates and run measured Restage configuration searches."
)


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
    capacity_catalog: Annotated[
        Path | None,
        typer.Option(
            exists=True, dir_okay=False, help="Measured GPU-group capacity JSON."
        ),
    ] = None,
    capacity_context: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Verified model, hardware, stack and workload identity JSON.",
        ),
    ] = None,
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
            resolution.resolved.config,
            space,
            output,
            max_candidates=max_candidates,
            capacity_catalog=(
                CapacityCatalog.model_validate_json(
                    capacity_catalog.read_text(encoding="utf-8")
                )
                if capacity_catalog is not None
                else None
            ),
            capacity_context=(
                CapacityContext.model_validate_json(
                    capacity_context.read_text(encoding="utf-8")
                )
                if capacity_context is not None
                else None
            ),
        )
    except (ValueError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    scope = "exhausted" if summary["complete"] else "truncated at the record limit"
    typer.echo(
        f"Exported {summary['accepted']} candidates; rejected {summary['rejected']}. "
        f"Search {scope}. Performance not measured. Results: {output}"
    )


@autotune_app.command()
def run(
    spec: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Campaign JSON with configurations, workload and SLO.",
        ),
    ],
    output: Annotated[
        Path, typer.Option(help="Campaign results directory; new unless resuming.")
    ],
    resume: Annotated[
        bool,
        typer.Option(
            help="Resume using caller-verified run_identity and matching recorded inputs."
        ),
    ] = False,
):
    """Measure a campaign on caller-allocated GPUs and export its recommendation."""
    from benchmarks.benchmarker.restage_campaign import (
        execute_campaign,
        load_campaign_spec,
    )

    try:
        options = load_campaign_spec(spec)
    except (ValueError, OSError) as exc:
        raise typer.BadParameter(str(exc), param_hint="--spec") from exc
    try:
        campaign = execute_campaign(destination=output, resume=resume, **options)
    except TypeError as exc:
        raise typer.BadParameter(str(exc), param_hint="--spec") from exc
    selection = asyncio.run(campaign)
    typer.echo(json.dumps(asdict(selection), indent=2))
