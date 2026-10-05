# SPDX-License-Identifier: Apache-2.0
"""Stands in for sgl-omni serve: one HTTP process with a long-lived child."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--model-name", required=True)
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--host", required=True)
parser.add_argument("--pid-file", required=True)
parser.add_argument("--startup-delay-s", type=float, default=0.0)
parser.add_argument("--exit-after-s", type=float, default=0.0)
parser.add_argument("--report-model-name", default=None)
arguments, _unknown = parser.parse_known_args()

# A stage child with its own child, like a launcher's worker and its tracker.
stage_child = subprocess.Popen(
    [
        sys.executable,
        "-c",
        "import subprocess, sys, time; "
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)']); "
        "time.sleep(600)",
    ]
)
time.sleep(0.3)
Path(arguments.pid_file).write_text(json.dumps([os.getpid(), stage_child.pid]))
Path(arguments.pid_file + ".env").write_text(
    json.dumps({"SGLANG_OMNI_STRICT_PORT": os.environ.get("SGLANG_OMNI_STRICT_PORT")})
)
time.sleep(arguments.startup_delay_s)
reported_name = arguments.report_model_name or arguments.model_name


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/health":
            body = {"status": "healthy", "running": True}
        elif self.path == "/v1/models":
            body = {"data": [{"id": reported_name}]}
        else:
            self.send_response(404)
            self.end_headers()
            return
        payload = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        pass


server = ThreadingHTTPServer((arguments.host, arguments.port), Handler)
if arguments.exit_after_s > 0:
    threading.Timer(arguments.exit_after_s, lambda: os._exit(3)).start()
else:
    pass
server.serve_forever()
