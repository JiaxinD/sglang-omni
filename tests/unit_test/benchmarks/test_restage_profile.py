import json

import pytest

from benchmarks.benchmarker import restage_profile
from benchmarks.benchmarker.data import RequestResult
from benchmarks.benchmarker.restage_profile import write_profile_report


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
