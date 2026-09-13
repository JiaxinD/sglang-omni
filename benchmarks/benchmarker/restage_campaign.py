"""Repeated candidate measurements and an evidence-scoped recommendation."""

import argparse
import asyncio
import hashlib
import json
import math
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path
from urllib.parse import urlparse

from benchmarks.benchmarker.restage_asr import execute_asr_trial
from benchmarks.benchmarker.restage_tts import execute_tts_trial
from benchmarks.dataset.seedtts import SampleInput
from sglang_omni.restage.evaluation import SLO, Evaluation
from sglang_omni.restage.search import search_rates
from sglang_omni.restage.selection import Selection, select_candidate


async def execute_campaign(
    *,
    configs: dict[str, Path],
    baseline: str,
    rates: list[float],
    repeats: int,
    arrival_seed: int,
    destination: Path,
    trial_options: dict,
    task: str = "tts",
    run_identity: str | None = None,
    resume: bool = False,
    adaptive_search: dict | None = None,
) -> Selection:
    """Measure supplied candidates with identical workload/SLO and paired arrivals.

    The caller supplies admitted hardware and model-specific options. Any trial
    exception stops the campaign, retaining completed trial evidence. Resume
    reuses completed trials and requires the recorded campaign identity.
    """
    if resume and not run_identity:
        raise ValueError("Resume requires an explicit frozen run_identity")
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if adaptive_search is not None:
        from benchmarks.benchmarker.restage_adaptive import execute_adaptive_campaign

        return await execute_adaptive_campaign(
            configs=configs,
            baseline=baseline,
            rates=rates,
            repeats=repeats,
            arrival_seed=arrival_seed,
            destination=destination,
            trial_options=trial_options,
            task=task,
            run_identity=run_identity,
            resume=resume,
            adaptive_search=adaptive_search,
        )
    return await _execute_campaign(
        configs=configs,
        baseline=baseline,
        rates=rates,
        repeats=repeats,
        arrival_seed=arrival_seed,
        destination=destination,
        trial_options=trial_options,
        task=task,
        run_identity=run_identity,
        resume=resume,
    )


def _json_default(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value.resolve())
    raise TypeError(f"Unsupported campaign option: {type(value).__name__}")


def _atomic_write(path, text):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _is_media_reference(value):
    return urlparse(str(value)).scheme in {"http", "https", "data", "file"}


def _input_identity(options):
    return json.loads(json.dumps(options, default=_json_default, allow_nan=False))


async def _execute_campaign(
    *,
    configs,
    baseline,
    rates,
    repeats,
    arrival_seed,
    destination,
    trial_options,
    task,
    run_identity,
    resume,
):
    runners = {"tts": execute_tts_trial, "asr": execute_asr_trial}
    if task not in runners:
        raise ValueError(f"Unsupported campaign task: {task}")
    if not rates or any(not math.isfinite(rate) or rate <= 0 for rate in rates):
        raise ValueError("Rates must be nonempty, finite and positive")
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    run_trial = runners[task]
    if baseline not in configs:
        raise ValueError("Include the baseline configuration")
    config_bytes = {key: path.read_bytes() for key, path in configs.items()}
    snapshots = {
        key: destination / f"candidate-{index:05d}.yaml"
        for index, key in enumerate(configs)
    }
    metadata = {
        "schema_version": 1,
        "run_identity": run_identity,
        "trial_inputs": _input_identity(trial_options),
        "task": task,
        "baseline": baseline,
        "rates": rates,
        "repeats": repeats,
        "arrival_seed": arrival_seed,
        "configs": {
            key: {
                "file": path.name,
                "sha256": hashlib.sha256(config_bytes[key]).hexdigest(),
            }
            for key, path in snapshots.items()
        },
    }
    manifest = destination / "campaign.json"
    completed = {}
    checkpoint = destination / "completed-trials.json"

    def write_trial_log():
        _atomic_write(
            destination / "trials.jsonl",
            "".join(json.dumps(row) + "\n" for row in completed.values()),
        )

    if resume:
        if json.loads(manifest.read_text(encoding="utf-8")) != metadata:
            raise ValueError("Campaign identity differs from the recorded run")
        if checkpoint.exists():
            for row in json.loads(checkpoint.read_text(encoding="utf-8")):
                completed[row["candidate"], row["rate"], row["repeat"]] = row
        write_trial_log()
    else:
        destination.mkdir(parents=True, exist_ok=False)
        for key, path in snapshots.items():
            path.write_bytes(config_bytes[key])
        manifest.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    results = {}
    for index, (key, path) in enumerate(snapshots.items()):

        async def trial(rate, repeat):
            if (key, rate, repeat) in completed:
                return Evaluation(**completed[key, rate, repeat]["evaluation"])
            base_dir = (
                destination / f"candidate-{index:05d}-rate-{rate.hex()}-repeat-{repeat}"
            )
            trial_dir = base_dir
            attempt = 0
            while trial_dir.exists():
                attempt += 1
                trial_dir = base_dir.with_name(f"{base_dir.name}-attempt-{attempt:05d}")
            try:
                evaluation = await run_trial(
                    config_path=path,
                    destination=trial_dir,
                    rate=rate,
                    arrival_seed=arrival_seed + repeat,
                    **trial_options,
                )
            except BaseException as exc:
                _atomic_write(
                    destination / "failure.json",
                    json.dumps(
                        {
                            "candidate": key,
                            "rate": rate,
                            "repeat": repeat,
                            "directory": trial_dir.name,
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                        indent=2,
                    ),
                )
                raise
            row = {
                "candidate": key,
                "rate": rate,
                "repeat": repeat,
                "arrival_seed": arrival_seed + repeat,
                "directory": trial_dir.name,
                "evaluation": asdict(evaluation),
            }
            _atomic_write(checkpoint, json.dumps([*completed.values(), row], indent=2))
            completed[key, rate, repeat] = row
            write_trial_log()
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


def load_campaign_spec(path: Path) -> dict:
    """Load a campaign with local asset/config paths relative to its spec."""
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
        base = path.resolve().parent
        if "plan_directory" in spec:
            if "configs" in spec:
                raise ValueError("Specify either configs or plan_directory, not both")
            from sglang_omni.restage.plan import load_plan

            spec["configs"] = load_plan(
                base / spec.pop("plan_directory"), spec["baseline"]
            )
        else:
            spec["configs"] = {
                key: base / value for key, value in spec["configs"].items()
            }
        options = spec["trial_options"]
        # Note (Jiaxin Deng): existing campaigns explicitly disabled profiling.
        if options.pop("profile", False) is not False:
            raise ValueError("profile support was removed from autotune campaigns")
        task = spec.get("task", "tts")
        sender = options.get("sender_options") or {}
        needs_audio = task == "asr" or (
            sender.get("voice_clone", False)
            if options.get("api") == "chat"
            else not sender.get("no_ref_audio", False)
        )
        options["samples"] = [SampleInput(**sample) for sample in options["samples"]]
        inputs = list(options["samples"])
        if options.get("warmup_sample") is not None:
            options["warmup_sample"] = SampleInput(**options["warmup_sample"])
            inputs.append(options["warmup_sample"])
        for sample in inputs:
            media_reference = _is_media_reference(sample.ref_audio)
            if sample.ref_audio and not media_reference:
                sample.ref_audio = str(base / Path(sample.ref_audio).expanduser())
            if needs_audio:
                if task == "asr" and media_reference:
                    raise ValueError("ASR samples require local audio files")
                if not media_reference and (
                    not sample.ref_audio or not Path(sample.ref_audio).is_file()
                ):
                    raise ValueError(
                        f"Reference audio file does not exist: {sample.ref_audio}"
                    )
        options["slo"] = SLO(**options["slo"])
        if task == "tts":
            options["asr_config_path"] = base / options["asr_config_path"]
            if not options["asr_config_path"].is_file():
                raise ValueError(
                    f"ASR configuration file does not exist: {options['asr_config_path']}"
                )
        return spec
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"Invalid campaign spec {path}: {exc}") from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    spec = load_campaign_spec(args.spec)
    selection = asyncio.run(
        execute_campaign(destination=args.output, resume=args.resume, **spec)
    )
    print(json.dumps(asdict(selection), indent=2))


if __name__ == "__main__":
    main()
