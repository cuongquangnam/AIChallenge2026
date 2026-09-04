#!/usr/bin/env python3
"""Start a Cloudflare quick tunnel to the Colab worker (run on the Colab VM)."""
from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path("/content/video-retrieval")
BIN = Path("/content/bin/cloudflared")
PID_FILE = Path("/tmp/video-retrieval-cloudflared.pid")
LOG_FILE = REPO_ROOT / "cloudflared.log"
URL_FILE = Path("/tmp/video-retrieval-worker-public-url.txt")
DRIVE_URL_FILE = Path("/content/drive/MyDrive/video-retrieval/worker_public_url.txt")
PORT = int(os.environ.get("COLAB_WORKER_PORT", "8765"))
CLOUDFLARED_URL = (
    "https://github.com/cloudflare/cloudflared/releases/latest/download/"
    "cloudflared-linux-amd64"
)
URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


def _worker_healthy() -> bool:
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/health", method="GET")
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _ensure_cloudflared() -> Path:
    BIN.parent.mkdir(parents=True, exist_ok=True)
    if BIN.is_file() and os.access(BIN, os.X_OK):
        return BIN
    print(f"Downloading cloudflared to {BIN} ...", flush=True)
    urllib.request.urlretrieve(CLOUDFLARED_URL, BIN)
    BIN.chmod(BIN.stat().st_mode | 0o111)
    return BIN


def _stop_existing() -> None:
    if PID_FILE.is_file():
        try:
            pid = int(PID_FILE.read_text(encoding="utf-8").strip())
            os.kill(pid, signal.SIGTERM)
        except (ValueError, OSError):
            pass
        PID_FILE.unlink(missing_ok=True)
    subprocess.run(["pkill", "-f", "cloudflared tunnel --url"], check=False)


def _parse_url(text: str) -> str | None:
    match = URL_RE.search(text)
    return match.group(0) if match else None


def main() -> None:
    if not _worker_healthy():
        raise SystemExit(
            f"Worker not healthy on :{PORT}. Run start_worker_remote.py first."
        )

    _stop_existing()
    binary = _ensure_cloudflared()
    REPO_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text("", encoding="utf-8")

    cmd = [
        str(binary),
        "tunnel",
        "--url",
        f"http://127.0.0.1:{PORT}",
        "--no-autoupdate",
    ]
    print(f"Starting tunnel: {' '.join(cmd)}", flush=True)
    log_handle = LOG_FILE.open("a", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    PID_FILE.write_text(str(proc.pid), encoding="utf-8")

    public_url = None
    deadline = time.time() + 90
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit(
                f"cloudflared exited early ({proc.returncode}). "
                f"See {LOG_FILE}:\n{LOG_FILE.read_text(encoding='utf-8')[-2000:]}"
            )
        text = LOG_FILE.read_text(encoding="utf-8")
        public_url = _parse_url(text)
        if public_url:
            break
        time.sleep(1)

    if not public_url:
        raise SystemExit(
            f"Timed out waiting for trycloudflare.com URL. See {LOG_FILE}"
        )

    URL_FILE.write_text(public_url + "\n", encoding="utf-8")
    try:
        DRIVE_URL_FILE.parent.mkdir(parents=True, exist_ok=True)
        DRIVE_URL_FILE.write_text(public_url + "\n", encoding="utf-8")
        drive_note = str(DRIVE_URL_FILE)
    except OSError:
        drive_note = "(Drive not writable)"

    print(f"Tunnel ready: {public_url}", flush=True)
    print(f"Saved: {URL_FILE}", flush=True)
    print(f"Drive: {drive_note}", flush=True)
    print("", flush=True)
    print("Add to laptop .env:", flush=True)
    print(f"COLAB_WORKER_PUBLIC_URL={public_url}", flush=True)
    print("COLAB_WORKER_MODE=tunnel", flush=True)
    print("", flush=True)
    print("Leave this process running (keepalive cell / notebook session).", flush=True)


if __name__ == "__main__":
    main()
