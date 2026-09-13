"""Repeated TTS candidate measurements and an evidence-scoped recommendation."""

import argparse
import asyncio
import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path

from benchmarks.benchmarker.restage_tts import execute_tts_trial
from benchmarks.dataset.seedtts import SampleInput
from sglang_omni.restage.evaluation import SLO
from sglang_omni.restage.search import search_rates
from sglang_omni.restage.selection import Selection, select_candidate


async def execute_tts_campaign(
    *,
    configs: dict[str, Path],
    baseline: str,
    rates: list[float],
    repeats: int,
    arrival_seed: int,
    destination: Path,
    trial_options: dict,
) -> Selection:
    """Measure supplied candidates with identical workload/SLO and paired arrivals.

    This explicit finite measurement campaign does not perform prediction-based
    pruning. The caller supplies admitted hardware and model-specific options.
    Any trial exception stops the campaign, retaining completed trial evidence.
    """
    if baseline not in configs:
        raise ValueError("Include the baseline configuration")
    destination.mkdir(parents=True, exist_ok=False)
    snapshots = {}
    for index, (key, path) in enumerate(configs.items()):
        target = destination / f"candidate-{index:05d}.yaml"
        shutil.copyfile(path, target)
        snapshots[key] = target
    metadata = {
        "baseline": baseline,
        "rates": rates,
        "repeats": repeats,
        "arrival_seed": arrival_seed,
        "configs": {
            key: {
                "file": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for key, path in snapshots.items()
        },
    }
    (destination / "campaign.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    results = {}
    for index, (key, path) in enumerate(snapshots.items()):

        async def trial(rate, repeat):
            trial_dir = (
                destination / f"candidate-{index:05d}-rate-{rate.hex()}-repeat-{repeat}"
            )
            try:
                evaluation = await execute_tts_trial(
                    config_path=path,
                    destination=trial_dir,
                    rate=rate,
                    arrival_seed=arrival_seed + repeat,
                    **trial_options,
                )
            except BaseException as exc:
                (destination / "failure.json").write_text(
                    json.dumps(
                        {
                            "candidate": key,
                            "rate": rate,
                            "repeat": repeat,
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                raise
            with (destination / "trials.jsonl").open("a", encoding="utf-8") as log:
                log.write(
                    json.dumps(
                        {
                            "candidate": key,
                            "rate": rate,
                            "repeat": repeat,
                            "arrival_seed": arrival_seed + repeat,
                            "directory": trial_dir.name,
                            "evaluation": asdict(evaluation),
                        }
                    )
                    + "\n"
                )
            return evaluation

        results[key] = await search_rates(
            trial, [float(rate) for rate in rates], repeats=repeats
        )
        (destination / f"candidate-{index:05d}-search.json").write_text(
            json.dumps(asdict(results[key]), indent=2), encoding="utf-8"
        )
    selection = select_candidate(results, baseline=baseline)
    (destination / "selection.json").write_text(
        json.dumps(asdict(selection), indent=2), encoding="utf-8"
    )
    if selection.recommended is not None:
        shutil.copyfile(
            snapshots[selection.recommended], destination / "recommended.yaml"
        )
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    base = args.spec.resolve().parent
    spec["configs"] = {key: base / path for key, path in spec["configs"].items()}
    options = spec["trial_options"]
    options["samples"] = [SampleInput(**sample) for sample in options["samples"]]
    options["slo"] = SLO(**options["slo"])
    options["asr_config_path"] = base / options["asr_config_path"]
    selection = asyncio.run(execute_tts_campaign(destination=args.output, **spec))
    print(json.dumps(asdict(selection), indent=2))


if __name__ == "__main__":
    main()
