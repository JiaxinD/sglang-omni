import json

import pytest

from benchmarks.benchmarker import restage_profile
from benchmarks.benchmarker.data import RequestResult
from benchmarks.benchmarker.restage_profile import write_profile_report
from sglang_omni.profiler.event_recorder import RequestEventRecorder


def test_profile_reports_missing_rank_and_unobserved_close(tmp_path):
    import os

    from sglang_omni.profiler.lifecycle import write_inventory

    worker = dict(stage="engine", pid=os.getpid(), role="leader", tp_rank=0)
    write_inventory(
        tmp_path,
        "run",
        [worker, {**worker, "pid": 99999, "role": "follower", "tp_rank": 1}],
    )
    rec = RequestEventRecorder()
    rec.start("run", str(tmp_path), "engine", worker={"role": "leader", "tp_rank": 0})
    output = tmp_path / "report.json"
    try:
        write_profile_report([], source=tmp_path, run_id="run", output=output)
        report = json.loads(output.read_text())
        coverage = report["recorder_coverage"]
        assert [w["status"] for w in coverage["workers"]] == [
            "recorder_observed",
            "recorder_missing",
        ]
        assert coverage["sessions"][0]["close_status"] == "unobserved"
        assert report["calibration_ready"] is False
    finally:
        rec.stop()
    write_profile_report([], source=tmp_path, run_id="run", output=output)
    session = json.loads(output.read_text())["recorder_coverage"]["sessions"][0]
    assert session["close_status"] == "clean"
    assert session["events_written"] == session["events_parsed"] == 0


def test_profile_lifecycle_reports_parse_damage_and_absent_inventory(tmp_path):
    rec = RequestEventRecorder()
    path = rec.start("run", str(tmp_path), "engine")
    rec.emit(request_id="a", stage="engine", event_name="encoder_start")
    # Simulate a damaged line on disk, distinct from a writer exception.
    rec._fp.write("broken json\n")
    rec.stop()
    output = tmp_path / "report.json"
    write_profile_report([], source=tmp_path, run_id="run", output=output)
    coverage = json.loads(output.read_text())["recorder_coverage"]
    assert coverage["inventory_status"] == "unavailable"
    assert coverage["sessions"][0]["events_written"] == 1
    assert coverage["sessions"][0]["events_parsed"] == 1
    assert coverage["sessions"][0]["unparsed_lines"] == 1
    assert coverage["sessions"][0]["write_failures"] == 0


def test_profile_preserves_restarts_and_sessions_with_unavailable_event_bytes(tmp_path):
    import os
    from pathlib import Path

    from sglang_omni.profiler.lifecycle import recorder_coverage, write_inventory

    worker = dict(stage="engine", pid=os.getpid(), role="single", tp_rank=0)
    write_inventory(tmp_path, "run", [worker])
    write_inventory(tmp_path, "run", [worker])
    rec = RequestEventRecorder()
    for _ in range(2):
        path = rec.start(
            "run", str(tmp_path), "engine", worker={"role": "single", "tp_rank": 0}
        )
        rec.emit(request_id="a", stage="engine", event_name="encoder_start")
        rec.stop()
    Path(path).write_text("")
    report = recorder_coverage(tmp_path, "run")
    assert report["inventory_status"] == "available"
    assert len(report["workers"][0]["sessions"]) == 2
    assert len(report["sessions"]) == len(report["read_errors"]) == 2
    assert all(s["events_parsed"] is None for s in report["sessions"])
    write_inventory(tmp_path, "run", [{**worker, "pid": 99999}])
    assert recorder_coverage(tmp_path, "run")["inventory_status"] == "ambiguous"


def test_profile_report_excludes_warmup_and_pairs_within_worker(tmp_path):
    events = []
    for run, request, pid, name, timestamp in [
        ("run", "server-a", 1, "encoder_start", 100),
        ("run", "server-a", 2, "encoder_start", 120),
        ("run", "server-a", 2, "encoder_end", 150),
        ("run", "server-a", 1, "encoder_end", 200),
        ("run", "warmup", 1, "encoder_start", 10),
        ("other-run", "server-a", 1, "encoder_end", 300),
    ]:
        events.append(
            dict(
                run_id=run,
                request_id=request,
                pid=pid,
                stage="encoder",
                event_name=name,
                timestamp_ns=timestamp,
            )
        )
    source = tmp_path / "events.jsonl"
    source.write_text("\n".join(json.dumps(e) for e in events))
    output = tmp_path / "report.json"
    write_profile_report(
        [
            RequestResult(request_id="a", server_request_id="server-a"),
            RequestResult(request_id="b"),
        ],
        source=source,
        run_id="run",
        output=output,
    )
    report = json.loads(output.read_text())
    assert report["requests"]["a"]["event_count"] == 4
    assert report["requests"]["b"]["event_count"] == 0
    assert report["requests"]["b"]["server_request_id"] is None
    intervals = report["requests"]["a"]["intervals"]
    assert {i["pid"]: i["close_ns"] - i["open_ns"] for i in intervals} == {
        1: 100,
        2: 30,
    }
    assert report["calibration_ready"] is False


@pytest.mark.asyncio
async def test_profile_stops_exact_run_on_dispatch_failure(tmp_path, monkeypatch):
    calls = []

    class Response:
        def raise_for_status(self):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    class Session(Response):
        def __init__(self, **kwargs):
            pass

        def post(self, url, json):
            calls.append((url, json))
            return Response()

    monkeypatch.setattr(restage_profile.aiohttp, "ClientSession", Session)
    with pytest.raises(RuntimeError, match="dispatch"):
        async with restage_profile.request_profile(
            "http://localhost:18000", event_dir=tmp_path, run_id="run"
        ):
            raise RuntimeError("dispatch failed")
    assert calls == [
        (
            "http://localhost:18000/start_request_profile",
            {"run_id": "run", "event_dir": str(tmp_path.resolve())},
        ),
        ("http://localhost:18000/stop_request_profile", {"run_id": "run"}),
    ]


def test_profile_keeps_asr_children_and_retries_separate(tmp_path):
    events = []
    for rid, start, end in [
        ("transcription-a-chunk-0", 100, 300),
        ("transcription-a-chunk-1", 110, 200),
        ("transcription-a-chunk-0-retry", 400, 450),
        ("transcription-warmup-chunk-0", 1, 2),
        ("transcription-a-unrelated", 1, 2),
    ]:
        for name, timestamp in [("encoder_start", start), ("encoder_end", end)]:
            events.append(
                dict(
                    run_id="run",
                    request_id=rid,
                    pid=1,
                    stage="encoder",
                    event_name=name,
                    timestamp_ns=timestamp,
                )
            )
    source = tmp_path / "events.jsonl"
    source.write_text("\n".join(json.dumps(e) for e in events))
    output = tmp_path / "report.json"
    write_profile_report(
        [RequestResult(request_id="a", server_request_id="transcription-a")],
        source=source,
        run_id="run",
        output=output,
    )
    report = json.loads(output.read_text())
    row = report["requests"]["a"]
    assert row["event_count"] == 6
    assert {
        i["request_id"]: i["close_ns"] - i["open_ns"] for i in row["intervals"]
    } == {
        "transcription-a-chunk-0": 200,
        "transcription-a-chunk-1": 90,
        "transcription-a-chunk-0-retry": 50,
    }
    assert report["calibration_ready"] is False
