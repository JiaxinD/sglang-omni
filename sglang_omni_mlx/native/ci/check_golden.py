# SPDX-License-Identifier: Apache-2.0
"""Checks the native runtime against a model's golden outputs on the frozen corpus.

    python check_golden.py --runtime-bin DIR --data-root DIR --golden FILE [--write]

Runs qwen3_asr_transcribe once over every corpus clip with the golden file's
request (Voxt's Final request), then requires each clip's text, language,
token count and finish reason to equal the golden file. Word, character and
mixed error rates are reported next to the original Voxt backend's (Swift on
MLX Audio) as a summary; they do not gate. --write regenerates the golden
clips and metrics from the current runtime.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

CI_DIRECTORY = Path(__file__).resolve().parent
COMPARED_FIELDS = ("text", "language", "generated_token_count", "finish_reason")
QUALITY_GROUPS = {
    "wer_en": lambda clip: clip["lang"] == "en"
    and clip["stratum"] not in ("silence", "noise"),
    "cer_zh": lambda clip: clip["lang"] == "zh",
    "mer_mixed": lambda clip: clip["stratum"] == "mixed",
}


def tokens(text: str, lang: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower()
    if lang == "en":
        text = re.sub(r"[^a-z0-9' ]+", " ", text)
        return [token.strip("'") for token in text.split() if token.strip("'")]
    else:
        return re.findall(r"[㐀-鿿豈-﫿]|[a-z0-9']+", text)


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, expected in enumerate(reference, 1):
        current = [i]
        for j, actual in enumerate(hypothesis, 1):
            current.append(
                min(
                    current[j - 1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (expected != actual),
                )
            )
        previous = current
    return previous[-1]


def quality(manifest: dict[str, dict], texts: dict[str, str]) -> dict[str, float]:
    metrics = {}
    for name, keep in QUALITY_GROUPS.items():
        errors = total = 0
        for clip_id, text in texts.items():
            clip = manifest[clip_id]
            if keep(clip):
                lang = "en" if clip["lang"] == "en" else "zh"
                reference = tokens(clip["reference"], lang)
                errors += edit_distance(reference, tokens(text, lang))
                total += len(reference)
            else:
                pass
        metrics[name] = round(errors / total, 5) if total else 0.0
    return metrics


def transcribe(
    runtime_bin: Path, model_directory: Path, clips: list[Path], request: dict
) -> dict[str, dict]:
    command = [
        str(runtime_bin / "qwen3_asr_transcribe"),
        "--model-path", str(model_directory),
        "--layout", request["layout"],
        "--language", request["language"],
        "--max-new-tokens", str(request["max_new_tokens"]),
    ]  # fmt: skip
    if request["stop_at_end_of_text"]:
        command.append("--stop-at-end-of-text")
    else:
        pass
    if request["stop_on_token_loop"]:
        command.append("--stop-on-token-loop")
    else:
        pass
    completed = subprocess.run(
        command + [str(clip) for clip in clips],
        check=True,
        capture_output=True,
        text=True,
    )
    results = {}
    for line in completed.stdout.splitlines():
        row = json.loads(line)
        results[Path(row["file"]).stem] = {
            field: row[field] for field in COMPARED_FIELDS
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-bin", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    arguments = parser.parse_args()

    golden = json.loads(arguments.golden.read_text())
    manifest = {
        row["id"]: row
        for row in map(
            json.loads,
            (CI_DIRECTORY / "corpus" / "manifest.jsonl").read_text().splitlines(),
        )
    }
    model_directory = arguments.data_root / "models" / golden["model"].replace("/", "_")
    clips = [
        arguments.data_root / "corpus" / "v1" / "clips" / f"{clip_id}.wav"
        for clip_id in manifest
    ]
    results = transcribe(
        arguments.runtime_bin, model_directory, clips, golden["request"]
    )
    metrics = quality(
        manifest, {clip_id: row["text"] for clip_id, row in results.items()}
    )

    if arguments.write:
        golden["metrics"] = metrics
        golden["clips"] = results
        arguments.golden.write_text(
            json.dumps(golden, ensure_ascii=False, indent=1) + "\n"
        )
        print(f"wrote {len(results)} clips to {arguments.golden}")
        return
    else:
        pass

    mismatches = [
        clip_id
        for clip_id in manifest
        if results.get(clip_id) != golden["clips"].get(clip_id)
    ]
    lines = [
        f"### {golden['model']}",
        "",
        f"Golden clips identical: {len(manifest) - len(mismatches)}/{len(manifest)}",
        "",
        "| | Original Voxt (Swift) | Native runtime | Δ |",
        "|---|---|---|---|",
    ]
    for name, value in metrics.items():
        baseline = golden["baseline"][name]
        lines.append(
            f"| {name} | {baseline:.2%} | {value:.2%} | {(value - baseline) * 100:+.2f} pp |"
        )
    for clip_id in mismatches[:10]:
        lines.append(f"\n- `{clip_id}` differs from its golden output")
    report = "\n".join(lines) + "\n"
    print(report)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as handle:
            handle.write(report)
    else:
        pass
    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
