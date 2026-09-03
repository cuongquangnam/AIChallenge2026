#!/usr/bin/env python3
"""Print persistent Colab worker status. Run on the Colab VM."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

PID_FILE = Path("/tmp/video-retrieval-worker.pid")
LOG_FILE = Path("/content/video-retrieval/worker.log")
PORT = int(os.environ.get("COLAB_WORKER_PORT", "8765"))


def main() -> None:
    pid = None
    if PID_FILE.is_file():
        try:
            pid = int(PID_FILE.read_text(encoding="utf-8").strip())
        except ValueError:
            pid = None

    alive = False
    if pid is not None:
        try:
            os.kill(pid, 0)
            alive = True
        except OSError:
            alive = False

    health = None
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as resp:
            health = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        health = None

    print(
        json.dumps(
            {
                "pid": pid,
                "pid_alive": alive,
                "port": PORT,
                "healthy": health is not None,
                "health": health,
                "log": str(LOG_FILE) if LOG_FILE.is_file() else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
