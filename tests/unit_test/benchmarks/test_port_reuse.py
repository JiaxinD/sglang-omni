import os
import socket

import pytest

from benchmarks.benchmarker.utils import _ensure_port_available
from sglang_omni.serve.launcher import _find_available_port

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="POSIX socket reuse semantics"
)


@pytest.mark.parametrize("probe", [_ensure_port_available, _find_available_port])
def test_sequential_service_can_reuse_closed_listener(probe, monkeypatch):
    monkeypatch.setenv("SGLANG_OMNI_STRICT_PORT", "1")
    host = "127.0.0.1"
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((host, 0))
        port = listener.getsockname()[1]
        listener.listen()
        with socket.create_connection((host, port)) as client:
            conn, _ = listener.accept()
            with conn:
                conn.shutdown(socket.SHUT_WR)
                assert client.recv(1) == b""
    probe(host, port)


@pytest.mark.parametrize("probe", [_ensure_port_available, _find_available_port])
def test_reusable_live_listener_remains_busy(probe, monkeypatch):
    monkeypatch.setenv("SGLANG_OMNI_STRICT_PORT", "1")
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        with pytest.raises(RuntimeError, match="already in use"):
            probe("127.0.0.1", listener.getsockname()[1])
