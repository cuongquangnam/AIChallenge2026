#!/usr/bin/env python3
"""Install the package on Colab and pull index data from Drive (run on the VM)."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path("/content/video-retrieval")
REMOTE_DATA_DIR = Path("/content/data")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-keyframes",
        action="store_true",
        help="Also extract Drive keyframes/*.zip into DATA_DIR/keyframes for rerank/QA.",
    )
    args, _unknown = parser.parse_known_args()

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
    from video_retrieval.storage.sync_paths import SESSION_PULL_PATHS

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

    pull_paths = list(SESSION_PULL_PATHS)
    if args.with_keyframes and "keyframes" not in pull_paths:
        pull_paths.append("keyframes")

    if args.with_keyframes:
        from video_retrieval.storage.drive_sync import (
            DriveDataSync,
            _discover_keyframe_zips,
        )

        sync = DriveDataSync(
            mount_point=settings.drive_mount,
            data_path=settings.drive_data_path,
            local_dir=REMOTE_DATA_DIR,
            local_mirror="",
            mount_on_access=True,
        )
        source_root = sync.remote_root()
        zips = _discover_keyframe_zips(source_root)
        print(f"[bootstrap] Drive root for keyframes: {source_root}", flush=True)
        if not zips:
            raise SystemExit(
                f"PULL_KEYFRAMES/--with-keyframes set, but no keyframes zip found under "
                f"{source_root}. Expected one of:\n"
                f"  {source_root / 'keyframes.zip'}\n"
                f"  {source_root / 'keyframes_*.zip'}\n"
                f"  {source_root / 'keyframes' / '*.zip'}\n"
                "Check DRIVE_DATA_PATH in .env.colab and that Drive is mounted."
            )
        for zip_path in zips:
            try:
                size_gb = zip_path.stat().st_size / (1024**3)
            except OSError:
                size_gb = 0.0
            print(f"[bootstrap] will extract: {zip_path} ({size_gb:.1f} GiB)", flush=True)

    request = {
        "job": "session_pull",
        "drive_mount": settings.drive_mount,
        "drive_data_path": settings.drive_data_path,
        "drive_local_path": "",
        "remote_data_dir": str(REMOTE_DATA_DIR),
        "pull_paths": pull_paths,
        "settings_overrides": settings.remote_settings_overrides(),
    }
    print(
        "[bootstrap] Pulling Drive data + starting Qdrant/Elasticsearch "
        f"(paths={pull_paths})...",
        flush=True,
    )
    if args.with_keyframes:
        print(
            "[bootstrap] Keyframes unzip enabled: Drive zip(s) extract into "
            f"{REMOTE_DATA_DIR / 'keyframes'} (large archives can take a long time).",
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
