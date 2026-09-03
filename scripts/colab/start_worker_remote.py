#!/usr/bin/env python3
"""Start the persistent Colab worker (models loaded once). Run on the Colab VM."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path("/content/video-retrieval")
PID_FILE = Path("/tmp/video-retrieval-worker.pid")
LOG_FILE = REPO_ROOT / "worker.log"
DEFAULT_PORT = int(os.environ.get("COLAB_WORKER_PORT", "8765"))
DEFAULT_DATA_DIR = os.environ.get("COLAB_REMOTE_DATA_DIR", "/content/data")
READY_TIMEOUT_SEC = float(os.environ.get("COLAB_WORKER_READY_TIMEOUT_SEC", "900"))


def _health_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/health"


def _is_healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(_health_url(port), timeout=2) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _read_pid() -> int | None:
    if not PID_FILE.is_file():
        return None
    try:
        return int(PID_FILE.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _stop_existing(port: int) -> None:
    pid = _read_pid()
    if pid and _pid_alive(pid):
        print(f"Stopping existing worker pid={pid}...")
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
        for _ in range(20):
            if not _pid_alive(pid):
                break
            time.sleep(0.25)
        if _pid_alive(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
    PID_FILE.unlink(missing_ok=True)
    # Also reclaim the port if a stale process is listening.
    if _is_healthy(port):
        print(f"Warning: something is still healthy on port {port}")


def main() -> None:
    port = DEFAULT_PORT
    data_dir = DEFAULT_DATA_DIR

    if not REPO_ROOT.is_dir():
        raise SystemExit(f"Repo not found at {REPO_ROOT}. Run laptop_clone.sh first.")

    if _is_healthy(port):
        print(f"Worker already healthy at {_health_url(port)}")
        return

    _stop_existing(port)
    REPO_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["COLAB_REMOTE_DATA_DIR"] = data_dir
    env["COLAB_WORKER_PORT"] = str(port)
    # Prefer package import path from editable install.
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "video_retrieval.remote.worker_server:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "info",
    ]
    print(f"Starting worker: {' '.join(cmd)}")
    print(f"Logs: {LOG_FILE}")
    log_handle = LOG_FILE.open("a", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    PID_FILE.write_text(str(proc.pid), encoding="utf-8")
    print(f"Worker pid={proc.pid}; waiting for health (timeout={READY_TIMEOUT_SEC:.0f}s)...")

    deadline = time.time() + READY_TIMEOUT_SEC
    while time.time() < deadline:
        if proc.poll() is not None:
            tail = LOG_FILE.read_text(encoding="utf-8")[-2000:] if LOG_FILE.is_file() else ""
            raise SystemExit(
                f"Worker exited early with code {proc.returncode}. Log tail:\n{tail}"
            )
        if _is_healthy(port):
            print(f"Worker ready at {_health_url(port)}")
            return
        time.sleep(2)

    raise SystemExit(
        f"Timed out waiting for worker health at {_health_url(port)}. "
        f"Check {LOG_FILE}"
    )


if __name__ == "__main__":
    main()
