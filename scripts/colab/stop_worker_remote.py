#!/usr/bin/env python3
"""Stop the persistent Colab worker. Run on the Colab VM."""
from __future__ import annotations

import os
import signal
import time
from pathlib import Path

PID_FILE = Path("/tmp/video-retrieval-worker.pid")


def main() -> None:
    if not PID_FILE.is_file():
        print("No PID file; worker not tracked.")
        return
    try:
        pid = int(PID_FILE.read_text(encoding="utf-8").strip())
    except ValueError:
        PID_FILE.unlink(missing_ok=True)
        print("Invalid PID file removed.")
        return

    try:
        os.kill(pid, 0)
    except OSError:
        PID_FILE.unlink(missing_ok=True)
        print(f"Worker pid={pid} not running.")
        return

    print(f"Stopping worker pid={pid}...")
    os.kill(pid, signal.SIGTERM)
    for _ in range(40):
        try:
            os.kill(pid, 0)
        except OSError:
            break
        time.sleep(0.25)
    else:
        os.kill(pid, signal.SIGKILL)
    PID_FILE.unlink(missing_ok=True)
    print("Worker stopped.")


if __name__ == "__main__":
    main()
