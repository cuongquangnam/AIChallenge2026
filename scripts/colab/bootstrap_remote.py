#!/usr/bin/env python3
"""Install the package on Colab and pull index data from Drive (run on the VM)."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path("/content/video-retrieval")
REMOTE_DATA_DIR = Path("/content/data")


def main() -> None:
    if not REPO_ROOT.is_dir():
        raise SystemExit(
            f"Repo not found at {REPO_ROOT}. "
            "On Colab run: python scripts/colab/clone_remote.py "
            "(see scripts/colab/MANUAL_SETUP.md)"
        )

    env_file = REPO_ROOT / ".env.colab"
    if not env_file.is_file():
        raise SystemExit(
            f"Missing {env_file}. Run: ./scripts/colab/laptop_upload_env.sh"
        )

    t0 = time.monotonic()
    print("[bootstrap] Installing video-retrieval[ml] (may take several minutes)...", flush=True)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-e", f"{REPO_ROOT}[ml]"],
    )
    print(f"[bootstrap] pip install done in {time.monotonic() - t0:.1f}s", flush=True)

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from video_retrieval.config import get_settings
    from video_retrieval.remote.worker import run_request

    settings = get_settings(data_dir=str(REMOTE_DATA_DIR), colab=True)
    drive_root = Path(settings.drive_mount) / "MyDrive"
    if not drive_root.is_dir():
        raise SystemExit(
            f"Google Drive not mounted ({drive_root} missing).\n"
            "From your laptop (not inside colab console python) run:\n"
            f"  colab drivemount -s video-retrieval {settings.drive_mount}\n"
            "or: ./scripts/colab/laptop_drivemount.sh\n"
            "Then re-run: ./scripts/colab/laptop_bootstrap.sh"
        )

    request = {
        "job": "session_pull",
        "drive_mount": settings.drive_mount,
        "drive_data_path": settings.drive_data_path,
        "drive_local_path": "",
        "remote_data_dir": str(REMOTE_DATA_DIR),
        "settings_overrides": settings.remote_settings_overrides(),
    }
    print(
        "[bootstrap] Pulling Drive data + starting Qdrant/Elasticsearch "
        "(progress + log tails will print while waiting)...",
        flush=True,
    )
    t1 = time.monotonic()
    response = run_request(request)
    print(f"[bootstrap] session_pull finished in {time.monotonic() - t1:.1f}s", flush=True)
    if not response.ok:
        raise SystemExit(response.error or "session_pull failed")
    print(response.result)
    print(f"[bootstrap] total elapsed {time.monotonic() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
