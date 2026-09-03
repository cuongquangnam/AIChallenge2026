#!/usr/bin/env python3
"""Re-pull Drive index data + hydrate Elasticsearch (run ON the Colab VM).

Shows live progress. Skips pip install. Resumes by skipping files already copied.

Usage (on the VM, Drive already mounted):
  python3 /content/video-retrieval/scripts/colab/pull_data_remote.py
  python3 .../pull_data_remote.py --skip-keyframes
  python3 .../pull_data_remote.py --only manifests,elasticsearch
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path("/content/video-retrieval")
REMOTE_DATA_DIR = Path("/content/data")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-keyframes",
        action="store_true",
        help="Do not pull keyframes/ (Drive FUSE is slow for many small files).",
    )
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated subset of paths (elasticsearch,qdrant,keyframes,manifests).",
    )
    args, _unknown = parser.parse_known_args()

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

    paths = list(SESSION_PULL_PATHS)
    if args.only.strip():
        paths = [p.strip() for p in args.only.split(",") if p.strip()]
    if args.skip_keyframes:
        paths = [p for p in paths if p != "keyframes"]

    print("==> Pulling Drive data + loading Elasticsearch (with progress)", flush=True)
    print(f"    remote_data_dir={REMOTE_DATA_DIR}", flush=True)
    print(f"    drive={settings.drive_mount}/{settings.drive_data_path}", flush=True)
    print(f"    paths={paths}", flush=True)
    print("    tip: already-copied files are skipped; Ctrl+C and re-run to resume", flush=True)

    response = run_request(
        {
            "job": "session_pull",
            "drive_mount": settings.drive_mount,
            "drive_data_path": settings.drive_data_path,
            "drive_local_path": "",
            "remote_data_dir": str(REMOTE_DATA_DIR),
            "pull_paths": paths,
            "settings_overrides": settings.remote_settings_overrides(),
        }
    )
    if not response.ok:
        raise SystemExit(response.error or "session_pull failed")
    print("==> Done", flush=True)
    print(response.result, flush=True)
    if "keyframes" not in paths:
        print(
            "Note: keyframes were skipped. Search still works; "
            "rerank/UI thumbs may need a later keyframe pull.",
            flush=True,
        )


if __name__ == "__main__":
    main()
