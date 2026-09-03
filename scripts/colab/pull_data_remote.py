#!/usr/bin/env python3
"""Re-pull Drive index data + hydrate Elasticsearch (run ON the Colab VM).

Shows live progress. Skips pip install.

Usage (on the VM, Drive already mounted):
  python3 /content/video-retrieval/scripts/colab/pull_data_remote.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path("/content/video-retrieval")
REMOTE_DATA_DIR = Path("/content/data")


def main() -> None:
    if not REPO_ROOT.is_dir():
        raise SystemExit(f"Repo not found at {REPO_ROOT}")
    env_file = REPO_ROOT / ".env.colab"
    if not env_file.is_file():
        raise SystemExit(f"Missing {env_file}")

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from video_retrieval.config import get_settings
    from video_retrieval.remote.worker import run_request
    from video_retrieval.storage.sync_paths import SESSION_PULL_PATHS

    settings = get_settings(data_dir=str(REMOTE_DATA_DIR), colab=True)
    drive_root = Path(settings.drive_mount) / "MyDrive"
    if not drive_root.is_dir():
        raise SystemExit(
            f"Google Drive not mounted ({drive_root} missing).\n"
            "Mount in the Colab notebook UI, or from laptop:\n"
            "  ./scripts/colab/laptop_drivemount.sh"
        )

    print("==> Pulling Drive data + loading Elasticsearch (with progress)", flush=True)
    print(f"    remote_data_dir={REMOTE_DATA_DIR}", flush=True)
    print(f"    drive={settings.drive_mount}/{settings.drive_data_path}", flush=True)
    print(f"    paths={list(SESSION_PULL_PATHS)}", flush=True)

    response = run_request(
        {
            "job": "session_pull",
            "drive_mount": settings.drive_mount,
            "drive_data_path": settings.drive_data_path,
            "drive_local_path": "",
            "remote_data_dir": str(REMOTE_DATA_DIR),
            "pull_paths": list(SESSION_PULL_PATHS),
            "settings_overrides": settings.remote_settings_overrides(),
        }
    )
    if not response.ok:
        raise SystemExit(response.error or "session_pull failed")
    print("==> Done", flush=True)
    print(response.result, flush=True)


if __name__ == "__main__":
    main()
