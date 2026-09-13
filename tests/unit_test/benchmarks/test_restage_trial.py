import asyncio
import json
from contextlib import asynccontextmanager, contextmanager

import pytest

from benchmarks.benchmarker import restage_trial
from benchmarks.benchmarker.data import RequestResult
from sglang_omni.restage.evaluation import SLO


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", [False, True])
async def test_trial_stops_server_before_quality_and_saves_requests(
    tmp_path, monkeypatch, profile
):
    running = False
    profiling = False

    @asynccontextmanager
    async def profiler(url, *, event_dir, run_id):
        nonlocal profiling
        assert running
        profiling = True
        try:
            yield
        finally:
            assert running
            profiling = False
            event_dir.mkdir()
            (event_dir / "events_encoder_1.jsonl").write_text(
                "\n".join(
                    json.dumps(
                        dict(
                            run_id=run_id,
                            request_id=rid,
                            pid=1,
                            stage="encoder",
                            event_name="encoder_start",
                            timestamp_ns=1,
                        )
                    )
                    for rid in ("a", "b")
                )
            )

    monkeypatch.setattr(restage_trial, "request_profile", profiler)

    @contextmanager
    def server(**kwargs):
        nonlocal running
        assert kwargs["wait_for_gpu_release"] is False
        running = True
        try:
            yield
        finally:
            running = False

    monkeypatch.setattr(restage_trial, "managed_omni_server", server)

    def sender(url, audio_dir):
        assert url == "http://127.0.0.1:18000"

        async def send(session, sample):
            assert running
            assert profiling == profile
            return RequestResult(
                request_id=sample, server_request_id=sample, is_success=True
            )

        return send

    async def quality(results):
        assert not running
        assert (tmp_path / "trial/requests.jsonl").exists()
        return {item.request_id: True for item in results}

    result = await restage_trial.execute_trial(
        config_path=tmp_path / "candidate.yaml",
        model_path="dummy",
        samples=["a", "b"],
        send_factory=sender,
        quality=quality,
        slo=SLO(max_latency_s=1),
        rate=100,
        destination=tmp_path / "trial",
        port=18000,
        warmup=0,
        profile=profile,
    )
    assert result.feasible
    assert result.good_requests == 2
    saved = json.loads((tmp_path / "trial/result.json").read_text())
    assert saved["status"] == "complete"
    assert saved["evaluation"]["feasible"] is True
    if profile:
        report = json.loads((tmp_path / "trial/profile-report.json").read_text())
        assert report["requests"]["a"]["event_count"] == 1
    else:
        assert not (tmp_path / "trial/profile-report.json").exists()


@pytest.mark.asyncio
async def test_trial_preserves_failure_and_stops_its_server(tmp_path, monkeypatch):
    events = []

    @contextmanager
    def server(**kwargs):
        events.append("start")
        try:
            yield
        finally:
            events.append("stop")

    monkeypatch.setattr(restage_trial, "managed_omni_server", server)

    def sender(url, audio_dir):
        async def send(session, sample):
            raise RuntimeError("sender failed")

        return send

    async def quality(results):
        raise AssertionError("quality must not run after failed dispatch")

    with pytest.raises(RuntimeError, match="sender failed"):
        await restage_trial.execute_trial(
            config_path=tmp_path / "candidate.yaml",
            model_path="dummy",
            samples=["a"],
            send_factory=sender,
            quality=quality,
            slo=SLO(max_latency_s=1),
            rate=100,
            destination=tmp_path / "trial",
            port=18000,
            warmup=0,
        )
    assert events == ["start", "stop"]
    assert (
        json.loads((tmp_path / "trial/result.json").read_text())["status"] == "failed"
    )


@pytest.mark.asyncio
async def test_trial_keeps_completed_request_evidence_after_later_failure(
    tmp_path, monkeypatch
):
    @contextmanager
    def server(**kwargs):
        yield

    monkeypatch.setattr(restage_trial, "managed_omni_server", server)

    def sender(url, audio_dir):
        async def send(session, sample):
            if sample == "fail":
                await asyncio.sleep(0.01)
                raise RuntimeError("later request failed")
            return RequestResult(request_id=sample, is_success=True)

        return send

    with pytest.raises(RuntimeError, match="later request failed"):
        await restage_trial.execute_trial(
            config_path=tmp_path / "candidate.yaml",
            model_path="dummy",
            samples=["completed", "fail"],
            send_factory=sender,
            quality=lambda results: None,
            slo=SLO(max_latency_s=1),
            rate=1000,
            destination=tmp_path / "trial",
            port=18000,
            warmup=0,
            arrival_seed=42,
        )
    rows = [
        json.loads(line)
        for line in (tmp_path / "trial/requests.jsonl").read_text().splitlines()
    ]
    assert [row["request_id"] for row in rows] == ["completed"]
    assert rows[0]["scheduled_s"] <= rows[0]["dispatched_s"] <= rows[0]["completed_s"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_phase", ["shutdown", "quality"])
async def test_trial_preserves_completed_measurement_before_postprocessing_failure(
    tmp_path, monkeypatch, failure_phase
):
    output = tmp_path / "trial/result.json"
    measured_elapsed = None

    @contextmanager
    def server(**kwargs):
        nonlocal measured_elapsed
        yield
        saved = json.loads(output.read_text())
        assert saved["measurement_complete"] is True
        measured_elapsed = saved["elapsed_s"]
        assert measured_elapsed > 0
        if failure_phase == "shutdown":
            raise RuntimeError("shutdown failed")

    monkeypatch.setattr(restage_trial, "managed_omni_server", server)

    def sender(url, audio_dir):
        async def send(session, sample):
            return RequestResult(request_id=sample, is_success=True)

        return send

    async def quality(results):
        assert failure_phase == "quality"
        raise RuntimeError("quality failed")

    with pytest.raises(RuntimeError, match=f"{failure_phase} failed"):
        await restage_trial.execute_trial(
            config_path=tmp_path / "candidate.yaml",
            model_path="dummy",
            samples=["a"],
            send_factory=sender,
            quality=quality,
            slo=SLO(max_latency_s=1),
            rate=100,
            destination=tmp_path / "trial",
            port=18000,
            warmup=0,
            arrival_seed=42,
        )
    saved = json.loads(output.read_text())
    assert saved["status"] == "failed"
    assert saved["measurement_complete"] is True
    assert saved["elapsed_s"] == measured_elapsed
    assert "evaluation" not in saved
    assert (tmp_path / "trial/measurement.json").exists() is (
        failure_phase == "quality"
    )
    rows = (tmp_path / "trial/requests.jsonl").read_text().splitlines()
    assert len(rows) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("quality_outcome", ["pass", "reject", "error"])
async def test_measurement_waits_for_quality_without_restarting_generation(
    tmp_path, monkeypatch, quality_outcome
):
    events = []

    @contextmanager
    def server(**kwargs):
        events.append("start")
        try:
            yield
        finally:
            events.append("stop")

    monkeypatch.setattr(restage_trial, "managed_omni_server", server)

    def sender(url, audio_dir):
        async def send(session, sample):
            events.append("generate")
            return RequestResult(request_id=sample, is_success=True)

        return send

    measurement = await restage_trial.measure_trial(
        config_path=tmp_path / "candidate.yaml",
        model_path="dummy",
        samples=["a"],
        send_factory=sender,
        slo=SLO(max_latency_s=1),
        rate=100,
        destination=tmp_path / "trial",
        port=18000,
        warmup=0,
        arrival_seed=42,
    )
    output = tmp_path / "trial/result.json"
    before = json.loads(output.read_text())
    assert before["status"] == "awaiting_quality"
    assert before["measurement_complete"] is True
    assert "evaluation" not in before
    assert events == ["start", "generate", "stop"]
    raw_requests = (tmp_path / "trial/requests.jsonl").read_bytes()

    async def quality(results):
        assert events == ["start", "generate", "stop"]
        events.append("quality")
        assert [result.request_id for result in results] == ["a"]
        if quality_outcome == "error":
            raise RuntimeError("evaluator unavailable")
        return {"a": quality_outcome == "pass"}

    if quality_outcome == "error":
        with pytest.raises(RuntimeError, match="evaluator unavailable"):
            await restage_trial.evaluate_trial(measurement, quality=quality)
        assert json.loads(output.read_text())["status"] == "failed"
    else:
        evaluation = await restage_trial.evaluate_trial(measurement, quality=quality)
        assert evaluation.feasible is (quality_outcome == "pass")
        assert json.loads(output.read_text())["status"] == "complete"
    assert events == ["start", "generate", "stop", "quality"]
    assert json.loads(output.read_text())["elapsed_s"] == before["elapsed_s"]
    assert (tmp_path / "trial/requests.jsonl").read_bytes() == raw_requests
    final_record = output.read_bytes()
    with pytest.raises(ValueError, match="awaiting quality"):
        await restage_trial.evaluate_trial(measurement, quality=quality)
    assert output.read_bytes() == final_record
    assert events.count("quality") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("quality_failed", [False, True])
@pytest.mark.parametrize("tamper", [None, "audio", "requests"])
async def test_restore_measurement_preserves_original_and_checks_inputs(
    tmp_path, monkeypatch, quality_failed, tamper
):
    @contextmanager
    def server(**kwargs):
        yield

    monkeypatch.setattr(restage_trial, "managed_omni_server", server)

    def sender(url, audio_dir):
        async def send(session, sample):
            wav = audio_dir / "a.wav"
            wav.write_bytes(b"saved audio")
            return RequestResult(
                request_id=sample,
                is_success=True,
                wav_path=str(wav),
                text="first\u2028second\u2029third\u0085fourth",
            )

        return send

    original = await restage_trial.measure_trial(
        config_path=tmp_path / "config.yaml",
        model_path="tts",
        samples=["a"],
        send_factory=sender,
        slo=SLO(),
        rate=100,
        destination=tmp_path / "original",
        port=18000,
        warmup=0,
        arrival_seed=42,
    )
    if quality_failed:

        async def fail(results):
            raise RuntimeError("ASR unavailable")

        with pytest.raises(RuntimeError):
            await restage_trial.evaluate_trial(original, quality=fail)
    before = (original.destination / "result.json").read_bytes()
    if tamper:
        path = original.destination / (
            "audio/a.wav" if tamper == "audio" else "requests.jsonl"
        )
        path.write_bytes(path.read_bytes() + b"changed")
        with pytest.raises(ValueError, match="changed"):
            restage_trial.restore_measurement(
                original.destination, destination=tmp_path / "retry"
            )
        assert not (tmp_path / "retry").exists()
    else:
        restored = restage_trial.restore_measurement(
            original.destination, destination=tmp_path / "retry"
        )
        assert restored.metadata["status"] == "awaiting_quality"
        assert restored.metadata["elapsed_s"] == original.metadata["elapsed_s"]
        assert restored.results == original.results
        assert restored.metadata["measurement_source"] == str(original.destination)

        async def passed(results):
            return {"a": True}

        assert (await restage_trial.evaluate_trial(restored, quality=passed)).feasible
    assert (original.destination / "result.json").read_bytes() == before


@pytest.mark.asyncio
@pytest.mark.parametrize("warmup_sample", [None, "warm"])
async def test_restored_measurement_preserves_input_order_and_excludes_warmup(
    tmp_path, monkeypatch, warmup_sample
):
    import asyncio

    @contextmanager
    def server(**kwargs):
        yield

    monkeypatch.setattr(restage_trial, "managed_omni_server", server)
    calls = []

    def sender(url, audio_dir):
        async def send(session, sample):
            calls.append(sample)
            if sample == "a":
                await asyncio.sleep(0.02)
            return RequestResult(request_id=sample, is_success=True)

        return send

    original = await restage_trial.measure_trial(
        config_path=tmp_path / "config.yaml",
        model_path="tts",
        samples=["a", "b"],
        send_factory=sender,
        slo=SLO(),
        rate=1e9,
        destination=tmp_path / "original",
        port=18000,
        warmup=2,
        warmup_sample=warmup_sample,
        arrival_seed=42,
    )
    rows = [
        json.loads(line)
        for line in (original.destination / "requests.jsonl").read_text().splitlines()
    ]
    assert [row["request_id"] for row in rows] == ["b", "a"]
    assert calls.count("a") == (3 if warmup_sample is None else 1)
    assert calls.count("b") == 1
    assert calls.count("warm") == (0 if warmup_sample is None else 2)
    restored = restage_trial.restore_measurement(
        original.destination, destination=tmp_path / "retry"
    )
    assert restored.results == original.results
    assert [result.request_id for result in restored.results] == ["a", "b"]

    if warmup_sample is None:
        assert "warmup_sample" not in restored.metadata
    else:
        assert restored.metadata["warmup_sample"] == warmup_sample
