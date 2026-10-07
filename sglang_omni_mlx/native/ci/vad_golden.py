# SPDX-License-Identifier: Apache-2.0
"""Silero VAD golden check, called by check_golden.py for golden files of kind
silero_vad.

The golden file holds the original Voxt's outputs (Swift MLXAudioVAD), not the
native runtime's: Voxt runs MLX 0.31.1 and the runtime 0.32.3, whose kernels
differ in the last bits. The same C++ built on MLX 0.31.1 reproduces Swift bit
for bit, so the checks allow for that kernel difference and nothing more:

- stream probabilities (one feed per 512-sample chunk, a 40-clip subset): within
  the tolerance, and on the same side of 0.5 for every chunk;
- speech timestamps for every clip and Voxt sensitivity profile: the same
  number of ranges, each boundary within one chunk.
"""

from __future__ import annotations

import base64
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def run_probe(
    runtime_bin: Path, model_directory: Path, clips: list[Path], profiles: dict
) -> Path:
    output = Path(tempfile.mkdtemp(prefix="silero-golden-"))
    (output / "profiles.json").write_text(json.dumps(profiles))
    subprocess.run(
        [
            str(runtime_bin / "silero_vad_probe"),
            "--model-path", str(model_directory),
            "--out", str(output),
            "--profiles", str(output / "profiles.json"),
        ]  # fmt: skip
        + [str(clip) for clip in clips],
        check=True,
    )
    return output


def ranges_match(expected: list, actual: list, boundary_samples: int) -> bool:
    return len(expected) == len(actual) and all(
        abs(e[0] - a[0]) <= boundary_samples and abs(e[1] - a[1]) <= boundary_samples
        for e, a in zip(expected, actual)
    )


def check(
    golden: dict, runtime_bin: Path, data_root: Path, clip_ids: list[str]
) -> tuple[list[str], list[str]]:
    """Returns (report lines, failures)."""
    model_directory = data_root / "models" / golden["model"].replace("/", "_")
    clips = [data_root / "corpus" / "v1" / "clips" / f"{clip}.wav" for clip in clip_ids]
    output = run_probe(runtime_bin, model_directory, clips, golden["profiles"])
    tolerance = golden["tolerance"]
    failures = []

    largest = 0.0
    flips = chunks = 0
    for clip, encoded in golden["stream_probabilities"].items():
        expected = np.frombuffer(base64.b64decode(encoded), dtype="<f4")
        actual = np.fromfile(output / f"{clip}.stream.f32", dtype="<f4")
        if expected.shape != actual.shape:
            failures.append(
                f"`{clip}`: {actual.size} stream chunks, expected {expected.size}"
            )
            continue
        else:
            pass
        difference = float(np.max(np.abs(expected - actual))) if expected.size else 0.0
        largest = max(largest, difference)
        flips += int(np.sum((expected >= 0.5) != (actual >= 0.5)))
        chunks += expected.size
        if difference > tolerance["max_abs_probability"]:
            failures.append(f"`{clip}`: stream probability off by {difference:.4f}")
        else:
            pass
    if flips:
        failures.append(f"{flips} stream chunks fall on the other side of 0.5")
    else:
        pass

    exact = total = 0
    for clip in clip_ids:
        actual = json.loads((output / f"{clip}.timestamps.json").read_text())
        for profile, expected in golden["timestamps"][clip].items():
            total += 1
            exact += int(actual[profile] == expected)
            if not ranges_match(
                expected, actual[profile], tolerance["boundary_samples"]
            ):
                failures.append(f"`{clip}` ({profile}): speech ranges differ")
            else:
                pass

    lines = [
        f"### {golden['model']}",
        "",
        f"Reference: {golden['reference']}",
        "",
        f"- Stream probabilities: {chunks} chunks on {len(golden['stream_probabilities'])} clips, "
        f"max |Δ| {largest:.2g}, {flips} decisions flipped at 0.5",
        f"- Speech timestamps: {exact}/{total} identical, the rest within "
        f"{tolerance['boundary_samples']} samples per boundary",
    ]
    return lines, failures
