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

from filelock import FileLock

from benchmarks.benchmarker.restage_asr import execute_asr_trial
from benchmarks.benchmarker.restage_trial import restore_measurement
from benchmarks.benchmarker.restage_tts import evaluate_tts_batch, execute_tts_trial
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
    batch_quality: bool = False,
    plan_evidence: dict | None = None,
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
            batch_quality=batch_quality,
            plan_evidence=plan_evidence,
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


def _is_media_reference(value):
    return urlparse(str(value)).scheme in {"http", "https", "data", "file"}


def _input_identity(options):
    serialized = json.loads(json.dumps(options, default=_json_default, allow_nan=False))
    references = [sample.get("ref_audio") for sample in serialized.get("samples", [])]
    if serialized.get("warmup_sample") is not None:
        references.append(serialized["warmup_sample"].get("ref_audio"))
    references.append(serialized.get("asr_config_path"))
    files = {
        str(Path(path).resolve()): _file_hash(path)
        for path in references
        if path and not _is_media_reference(path) and Path(path).is_file()
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
    batch_quality,
    plan_evidence,
):
    runners = {"tts": execute_tts_trial, "asr": execute_asr_trial}
    if task not in runners:
        raise ValueError(f"Unsupported campaign task: {task}")
    if batch_quality and task != "tts":
        raise ValueError("Batch quality is available for TTS campaigns only")
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
    if batch_quality:
        metadata["batch_quality"] = True
    if plan_evidence is not None:
        metadata["plan_evidence"] = plan_evidence
    manifest = destination / "campaign.json"
    completed = {}
    measured = {}
    measurement_checkpoint = destination / "measured-trials.json"
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
        if measurement_checkpoint.exists():
            for row in json.loads(measurement_checkpoint.read_text(encoding="utf-8")):
                measured[row["candidate"], row["rate"], row["repeat"]] = row
        write_trial_log()
    else:
        destination.mkdir(parents=True, exist_ok=False)
        for key, path in snapshots.items():
            path.write_bytes(config_bytes[key])
        manifest.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def record_failure(key, rate, repeat, trial_dir, exc, phase):
        failure = json.dumps(
            {
                "candidate": key,
                "rate": rate,
                "repeat": repeat,
                "directory": trial_dir.name if trial_dir is not None else None,
                "phase": phase,
                "error": f"{type(exc).__name__}: {exc}",
            },
            indent=2,
        )
        if trial_dir is not None:
            trial_dir.mkdir(parents=True, exist_ok=True)
            _atomic_write(trial_dir / "execution-failure.json", failure)
        _atomic_write(destination / "failure.json", failure)

    def record_completed(key, rate, repeat, trial_dir, evaluation):
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

    results = {}
    for index, (key, path) in enumerate(snapshots.items()):

        async def trial(rate, repeat, *, defer_quality=False):
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
            saved = measured.get((key, rate, repeat)) if defer_quality else None
            try:
                if saved is not None:
                    if (
                        _file_hash(
                            destination / saved["directory"] / "measurement.json"
                        )
                        != saved["receipt_sha256"]
                    ):
                        raise ValueError(
                            f"Measurement receipt changed since generation: {destination / saved['directory'] / 'measurement.json'}"
                        )
                    evaluation = restore_measurement(
                        destination / saved["directory"], destination=trial_dir
                    )
                else:
                    evaluation = await run_trial(
                        config_path=path,
                        destination=trial_dir,
                        rate=rate,
                        arrival_seed=arrival_seed + repeat,
                        **({"defer_quality": True} if defer_quality else {}),
                        **trial_options,
                    )
            except BaseException as exc:
                record_failure(
                    key,
                    rate,
                    repeat,
                    trial_dir,
                    exc,
                    (
                        "restore"
                        if saved is not None
                        else ("generation" if defer_quality else "trial")
                    ),
                )
                raise
            if defer_quality:
                row = {
                    "candidate": key,
                    "rate": rate,
                    "repeat": repeat,
                    "directory": trial_dir.name,
                    "receipt_sha256": _file_hash(trial_dir / "measurement.json"),
                }
                updated = {**measured, (key, rate, repeat): row}
                _atomic_write(
                    measurement_checkpoint, json.dumps(list(updated.values()), indent=2)
                )
                measured[key, rate, repeat] = row
                return evaluation
            record_completed(key, rate, repeat, trial_dir, evaluation)
            return evaluation

        if batch_quality:
            pending = []
            cells = {}
            for repeat in range(repeats):
                for rate in sorted(set(float(rate) for rate in rates)):
                    if (key, rate, repeat) in completed:
                        continue
                    measurement = await trial(rate, repeat, defer_quality=True)
                    pending.append(measurement)
                    cells[measurement.destination] = (rate, repeat)

            checkpoint_error = None

            def checkpoint_quality(measurement, evaluation):
                nonlocal checkpoint_error
                rate, repeat = cells[measurement.destination]
                try:
                    record_completed(
                        key, rate, repeat, measurement.destination, evaluation
                    )
                except BaseException:
                    checkpoint_error = measurement
                    raise

            if pending:
                try:
                    await evaluate_tts_batch(
                        pending,
                        on_evaluated=checkpoint_quality,
                        **{
                            name: trial_options[name]
                            for name in (
                                "samples",
                                "asr_config_path",
                                "asr_model_path",
                                "port",
                                "lang",
                                "max_wer",
                                "startup_timeout_s",
                                "request_timeout_s",
                                "asr_concurrency",
                            )
                            if name in trial_options
                        },
                    )
                except BaseException as exc:
                    unfinished = checkpoint_error or next(
                        (
                            m
                            for m in pending
                            if (key, *cells[m.destination]) not in completed
                        ),
                        None,
                    )
                    if unfinished is None:
                        record_failure(key, None, None, None, exc, "batch_shutdown")
                    else:
                        rate, repeat = cells[unfinished.destination]
                        record_failure(
                            key,
                            rate,
                            repeat,
                            unfinished.destination,
                            exc,
                            (
                                "checkpoint"
                                if checkpoint_error is not None
                                else "batch_quality"
                            ),
                        )
                    raise

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
            if "plan_evidence" in spec:
                raise ValueError("plan_evidence is generated from plan_directory")
            from sglang_omni.restage.plan import load_plan

            spec["configs"], spec["plan_evidence"] = load_plan(
                base / spec.pop("plan_directory"), spec["baseline"]
            )
        else:
            spec["configs"] = {
                key: base / value for key, value in spec["configs"].items()
            }
        options = spec["trial_options"]
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
