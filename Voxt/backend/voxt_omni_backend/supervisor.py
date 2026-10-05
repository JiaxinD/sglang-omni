# SPDX-License-Identifier: Apache-2.0
"""Owns one local SGLang-Omni server for Voxt: launch, verify, report, reap.

Events go to stdout as JSON lines. Voxt holds this process's stdin open; a
shutdown command or end-of-file (Voxt exited or crashed) stops the server and
every process it started. Logs carry state and timing only.
"""

from __future__ import annotations

import argparse
import http.client
import json
import logging
import os
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Literal

import psutil
from voxt_omni_backend.model_views import build_whisper_hf_view

LOOPBACK_HOST = "127.0.0.1"
ModelKind = Literal["qwen3_asr", "moss_transcribe_diarize", "whisper"]
MODEL_KINDS: tuple[ModelKind, ...] = ("qwen3_asr", "moss_transcribe_diarize", "whisper")
HEALTH_POLL_INTERVAL_S = 0.1
HTTP_PROBE_TIMEOUT_S = 1.0
TERMINATE_GRACE_S = 5.0
CONTROL_POLL_INTERVAL_S = 0.1

logger = logging.getLogger("voxt_omni_backend.supervisor")
loopback_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


@dataclass(kw_only=True, frozen=True)
class ServerLaunch:
    command: list[str]
    environment: dict[str, str]
    model_name: str
    port: int


class SupervisorStopped(Exception):
    """A termination signal asked the supervisor to stop."""


def emit(event: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(event) + "\n")
    sys.stdout.flush()


def free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((LOOPBACK_HOST, 0))
        return int(probe.getsockname()[1])


def get_json(url: str) -> dict[str, object] | None:
    try:
        with loopback_opener.open(url, timeout=HTTP_PROBE_TIMEOUT_S) as response:
            decoded = json.loads(response.read())
    except (
        urllib.error.URLError,
        http.client.HTTPException,
        ConnectionError,
        TimeoutError,
        ValueError,
    ):
        return None
    if isinstance(decoded, dict):
        return decoded
    else:
        return None


class OwnedProcessTree:
    """The server and every descendant seen so far, keyed by pid and start time.

    A descendant reparented after the server dies is still recognised, and a
    recycled pid with a different start time is never signalled.
    """

    def __init__(self, process: subprocess.Popen[bytes]) -> None:
        self.process = process
        self.start_times_by_pid: dict[int, float] = {}
        self.refresh()

    def refresh(self) -> None:
        try:
            server = psutil.Process(self.process.pid)
            members = [server, *server.children(recursive=True)]
        except psutil.NoSuchProcess:
            members = []
        for member in members:
            try:
                self.start_times_by_pid[member.pid] = member.create_time()
            except psutil.NoSuchProcess:
                pass

    def live_members(self) -> list[psutil.Process]:
        live: list[psutil.Process] = []
        for pid, start_time in self.start_times_by_pid.items():
            try:
                member = psutil.Process(pid)
                if member.create_time() == start_time:
                    live.append(member)
                else:
                    pass
            except psutil.NoSuchProcess:
                pass
        return live

    def reap(self) -> None:
        """Stop the server and its descendants; no other process is signalled."""
        self.refresh()
        members = self.live_members()
        for member in members:
            try:
                member.terminate()
            except psutil.NoSuchProcess:
                pass
        _gone, alive = psutil.wait_procs(members, timeout=TERMINATE_GRACE_S)
        for survivor in alive:
            try:
                survivor.kill()
            except psutil.NoSuchProcess:
                pass
        self.process.wait()


def ready_failure(
    owned_tree: OwnedProcessTree, launch: ServerLaunch, deadline: float
) -> str | None:
    """None once our own instance answers; otherwise why it never will."""
    base_url = f"http://{LOOPBACK_HOST}:{launch.port}"
    while time.monotonic() < deadline:
        owned_tree.refresh()
        exit_code = owned_tree.process.poll()
        if exit_code is not None:
            return f"server exited with code {exit_code} before it was ready"
        else:
            pass
        health = get_json(f"{base_url}/health")
        models = get_json(f"{base_url}/v1/models") if health is not None else None
        if health is not None and health.get("status") == "healthy" and models:
            served = models.get("data")
            served_names = (
                [entry.get("id") for entry in served if isinstance(entry, dict)]
                if isinstance(served, list)
                else []
            )
            if launch.model_name in served_names:
                return None
            else:
                return "a different server instance answered on the launch port"
        else:
            pass
        time.sleep(HEALTH_POLL_INTERVAL_S)
    return "startup timeout"


def control_commands(command_queue: queue.Queue[str | None]) -> None:
    for line in sys.stdin:
        command_queue.put(line)
    command_queue.put(None)


def server_launch(arguments: argparse.Namespace) -> ServerLaunch:
    model_kind: ModelKind = arguments.model_kind
    model_directory = Path(arguments.model_directory)
    model_name = f"voxt-{model_kind}-{uuid.uuid4().hex[:12]}"
    port = free_loopback_port()
    if model_kind == "whisper":
        model_path = build_whisper_hf_view(
            model_directory,
            Path(arguments.derived_root) / "omni-whisper-view" / model_directory.name,
        )
    else:
        model_path = model_directory
    if arguments.server_command:
        base_command = list(json.loads(arguments.server_command))
    else:
        base_command = [
            str(Path(sys.executable).with_name("sgl-omni")),
            "serve",
            "--asr.engine.max_running_requests",
            "1",
            *(["--enable-realtime"] if model_kind == "qwen3_asr" else []),
        ]
    environment = dict(os.environ)
    environment.update(
        {
            "SGLANG_USE_MLX": "1",
            "HF_HUB_OFFLINE": "1",
            "NO_PROXY": f"{LOOPBACK_HOST},localhost",
            "no_proxy": f"{LOOPBACK_HOST},localhost",
        }
    )
    if arguments.ffmpeg_library_directory:
        environment["DYLD_LIBRARY_PATH"] = arguments.ffmpeg_library_directory
    else:
        pass
    return ServerLaunch(
        command=[
            *base_command,
            "--model-path",
            str(model_path),
            "--model-name",
            model_name,
            "--host",
            LOOPBACK_HOST,
            "--port",
            str(port),
        ],
        environment=environment,
        model_name=model_name,
        port=port,
    )


def run(arguments: argparse.Namespace) -> int:
    launch = server_launch(arguments)
    log_directory = Path(arguments.derived_root) / "omni-logs"
    log_directory.mkdir(parents=True, exist_ok=True)
    started_s = time.monotonic()
    with open(log_directory / f"{launch.model_name}.log", "wb") as server_log:
        process = subprocess.Popen(
            launch.command,
            env=launch.environment,
            stdin=subprocess.DEVNULL,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    logger.info(f"server started pid={process.pid} port={launch.port}")
    owned_tree = OwnedProcessTree(process)
    try:
        failure = ready_failure(
            owned_tree, launch, started_s + float(arguments.startup_timeout_s)
        )
        if failure is not None:
            emit({"event": "failed", "reason": failure})
            return 1
        else:
            pass
        emit(
            {
                "event": "ready",
                "host": LOOPBACK_HOST,
                "port": launch.port,
                "model_name": launch.model_name,
                "server_pid": process.pid,
                "startup_s": round(time.monotonic() - started_s, 3),
            }
        )
        command_queue: queue.Queue[str | None] = queue.Queue()
        threading.Thread(
            target=control_commands, args=(command_queue,), daemon=True
        ).start()
        while True:
            owned_tree.refresh()
            exit_code = process.poll()
            if exit_code is not None:
                emit({"event": "exited", "exit_code": exit_code})
                return 1
            else:
                pass
            try:
                line = command_queue.get(timeout=CONTROL_POLL_INTERVAL_S)
            except queue.Empty:
                continue
            if line is None:
                logger.info("control pipe closed")
                return 0
            elif json.loads(line).get("command") == "shutdown":
                owned_tree.reap()
                emit({"event": "stopped"})
                return 0
            else:
                logger.warning("ignored unknown control command")
    finally:
        owned_tree.reap()


def stop_on_signal(signal_number: int, frame: FrameType | None) -> None:
    del frame
    raise SupervisorStopped(signal.Signals(signal_number).name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-kind", choices=MODEL_KINDS, required=True)
    parser.add_argument("--model-directory", required=True)
    parser.add_argument("--derived-root", required=True)
    parser.add_argument("--ffmpeg-library-directory", default=None)
    parser.add_argument("--startup-timeout-s", type=float, default=180.0)
    parser.add_argument(
        "--server-command",
        default=None,
        help="JSON argv replacing sgl-omni serve, for tests",
    )
    arguments = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(message)s"
    )
    signal.signal(signal.SIGTERM, stop_on_signal)
    signal.signal(signal.SIGINT, stop_on_signal)
    try:
        return run(arguments)
    except SupervisorStopped as stopped:
        logger.info(f"stopped by {stopped}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
