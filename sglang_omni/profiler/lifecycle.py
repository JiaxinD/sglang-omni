# SPDX-License-Identifier: Apache-2.0
"""Recorder evidence, separate from request events and GPU service measurements."""

import json
import uuid
from pathlib import Path


def write_inventory(directory: Path, run_id: str, workers: list[dict]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"inventory_{uuid.uuid4().hex}.json"
    path.write_text(
        json.dumps({"run_id": run_id, "workers": workers}), encoding="utf-8"
    )


def _records(data: bytes) -> tuple[list[dict], int]:
    records, invalid = [], 0
    for line in data.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("record is not an object")
            records.append(record)
        except ValueError:
            invalid += 1
    return records, invalid


def recorder_coverage(source: Path, run_id: str) -> dict:
    """Report observed joins and file outcomes; absence never means no work."""
    directory = source if source.is_dir() else source.parent
    inventories, sessions, errors = [], [], []
    for path in sorted(directory.glob("inventory_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data["run_id"] == run_id:
                if data["workers"] not in inventories:
                    inventories.append(data["workers"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append({"file": path.name, "error": str(exc)})
    for path in sorted(directory.glob("lifecycle_*.jsonl")):
        try:
            records, invalid = _records(path.read_bytes())
            records = [r for r in records if r.get("run_id") == run_id]
            if not records:
                continue
            opened = next((r for r in records if r.get("kind") == "open"), None)
            closed = next((r for r in records if r.get("kind") == "close"), None)
            observed = [r for r in records if r.get("kind") == "join"]
            row = {
                "file": path.name,
                "open": opened,
                "joins": observed,
                "close": closed,
                "lifecycle_unparsed_lines": invalid,
                "close_status": (
                    "unobserved"
                    if closed is None
                    else ("clean" if closed["close_ok"] else "failed")
                ),
                "events_written": None if closed is None else closed["events_written"],
                "write_failures": None if closed is None else closed["write_failures"],
                "events_parsed": None,
                "unparsed_lines": None,
            }
            sessions.append(row)
            # Note (Jiaxin Deng): event files append across runs. Only count
            # the byte interval claimed by this recorder session's endpoints.
            if opened and closed and closed.get("events_end_bytes") is not None:
                event_name = opened["events_file"]
                if Path(event_name).name != event_name:
                    raise ValueError("events_file must be a filename")
                start, end = opened["events_start_bytes"], closed["events_end_bytes"]
                event_path = directory / event_name
                if not 0 <= start <= end <= event_path.stat().st_size:
                    raise ValueError("event byte interval is unavailable")
                with event_path.open("rb") as fp:
                    fp.seek(start)
                    events, damaged = _records(fp.read(end - start))
                row.update(events_parsed=len(events), unparsed_lines=damaged)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append({"file": path.name, "error": str(exc)})
    workers = []
    if len(inventories) == 1:
        for worker in inventories[0]:
            matched = [
                session["file"]
                for session in sessions
                if any(
                    j.get("stage") == worker["stage"]
                    and j.get("pid") == worker["pid"]
                    and all(
                        j.get("worker", {}).get(k) == worker.get(k)
                        for k in ("role", "tp_rank")
                    )
                    for j in session["joins"]
                )
            ]
            workers.append(
                {
                    **worker,
                    "status": "recorder_observed" if matched else "recorder_missing",
                    "sessions": matched,
                }
            )
    return {
        "inventory_status": (
            "available"
            if len(inventories) == 1
            else ("unavailable" if not inventories else "ambiguous")
        ),
        "workers": workers,
        "sessions": sessions,
        "read_errors": errors,
        "scope": "Recorder joins and file outcomes only; no proof of instrumented work coverage or GPU capacity.",
    }


def work_unit_report(source: Path, run_id: str) -> dict:
    directory = source if source.is_dir() else source.parent
    sessions, errors = [], []
    for path in sorted(directory.glob("work_units_*.jsonl")):
        try:
            records, invalid = _records(path.read_bytes())
            records = [r for r in records if r.get("run_id") == run_id]
            if not records:
                continue
            begins = {r["unit_id"]: r for r in records if r.get("kind") == "begin"}
            ends = {r["unit_id"]: r for r in records if r.get("kind") == "end"}
            units = []
            for unit_id, begin in begins.items():
                end = ends.get(unit_id)
                duration = (
                    None if end is None else (end["end_ns"] - end["start_ns"]) / 1e9
                )
                units.append(
                    {
                        "unit_id": unit_id,
                        "batch_id": begin["batch_id"],
                        "attempt": begin["attempt"],
                        "member_count": len(begin["members"]),
                        "component_id": begin.get("component_id"),
                        "constructed_in": begin.get("constructed_in"),
                        "end_observed": end is not None,
                        "error_type": None if end is None else end["error_type"],
                        "host_execution_s": duration,
                        "executions": None if end is None else end.get("executions"),
                    }
                )
            sessions.append(
                {
                    "file": path.name,
                    "units": units,
                    "unparsed_lines": invalid,
                    "stop_observed": any(r.get("kind") == "stop" for r in records),
                    "stop_records": [r for r in records if r.get("kind") == "stop"],
                    "unmatched_ends": [e for k, e in ends.items() if k not in begins],
                    "metadata_errors": [
                        r for r in records if r.get("kind") == "metadata_error"
                    ],
                }
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append({"file": path.name, "error": str(exc)})
    return {
        "sessions": sessions,
        "read_errors": errors,
        "scope": "Per-execution host envelopes, not GPU-only or isolated stage capacity. Component/PID identity is not a request or stage-work coverage proof.",
    }
