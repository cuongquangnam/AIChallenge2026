#!/usr/bin/env python3
"""Install the package on Colab and pull index data from Drive (run on the VM)."""
from __future__ import annotations

import subprocess
import sys
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

    print("Installing video-retrieval[ml]...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "-e", f"{REPO_ROOT}[ml]"],
    )

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
    print("Pulling index data from Drive and loading Elasticsearch...")
    response = run_request(request)
    if not response.ok:
        raise SystemExit(response.error or "session_pull failed")
    print(response.result)


if __name__ == "__main__":
    main()
