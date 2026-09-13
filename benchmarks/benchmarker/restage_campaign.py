"""Repeated candidate measurements and an evidence-scoped recommendation."""

import argparse
import asyncio
import hashlib
import json
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path

from filelock import FileLock

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
) -> Selection:
    """Measure supplied candidates with identical workload/SLO and paired arrivals.

    This explicit finite measurement campaign does not perform prediction-based
    pruning. The caller supplies admitted hardware and model-specific options.
    Any trial exception stops the campaign, retaining completed trial evidence.
    Resume requires the caller's unchanged source/image/hardware/checkpoint
    identity as well as matching recorded options and local input contents.
    """
    if resume and not run_identity:
        raise ValueError("Resume requires an explicit frozen run_identity")
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(destination.parent / f".{destination.name}.lock"), timeout=0):
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


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path, text):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _input_identity(options):
    serialized = json.loads(json.dumps(options, default=_json_default, allow_nan=False))
    references = [sample.get("ref_audio") for sample in serialized.get("samples", [])]
    references.append(serialized.get("asr_config_path"))
    files = {
        str(Path(path).resolve()): _file_hash(path)
        for path in references
        if path and Path(path).is_file()
    }
    return {"options": serialized, "local_input_sha256": files}


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
        for key, path in snapshots.items():
            if _file_hash(path) != metadata["configs"][key]["sha256"]:
                raise ValueError(f"Candidate snapshot changed: {key}")
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
                failure = json.dumps(
                    {
                        "candidate": key,
                        "rate": rate,
                        "repeat": repeat,
                        "directory": trial_dir.name,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                    indent=2,
                )
                trial_dir.mkdir(parents=True, exist_ok=True)
                _atomic_write(trial_dir / "execution-failure.json", failure)
                _atomic_write(destination / "failure.json", failure)
                raise
            completed[key, rate, repeat] = {
                "candidate": key,
                "rate": rate,
                "repeat": repeat,
                "arrival_seed": arrival_seed + repeat,
                "directory": trial_dir.name,
                "evaluation": asdict(evaluation),
            }
            _atomic_write(checkpoint, json.dumps(list(completed.values()), indent=2))
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    base = args.spec.resolve().parent
    spec["configs"] = {key: base / path for key, path in spec["configs"].items()}
    options = spec["trial_options"]
    options["samples"] = [SampleInput(**sample) for sample in options["samples"]]
    options["slo"] = SLO(**options["slo"])
    if spec.get("task", "tts") == "tts":
        options["asr_config_path"] = base / options["asr_config_path"]
    selection = asyncio.run(
        execute_campaign(destination=args.output, resume=args.resume, **spec)
    )
    print(json.dumps(asdict(selection), indent=2))


if __name__ == "__main__":
    main()
