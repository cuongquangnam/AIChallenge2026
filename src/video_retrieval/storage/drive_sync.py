from __future__ import annotations

import logging
import shutil
from pathlib import Path

from video_retrieval.storage.sync_paths import QA_PULL_PATHS, SEARCH_PULL_PATHS

logger = logging.getLogger(__name__)


def mount_google_drive(mount_point: str | Path = "/content/drive", *, force_remount: bool = False) -> Path:
    """Mount Google Drive on a Colab VM (no-op if already mounted)."""
    mount = Path(mount_point)
    my_drive = mount / "MyDrive"
    if my_drive.is_dir() and not force_remount:
        logger.info("Google Drive already mounted at %s", mount)
        return mount

    try:
        from google.colab import drive
    except ImportError as exc:
        raise RuntimeError(
            "google.colab.drive is only available on a Colab runtime. "
            "On your laptop, set DRIVE_LOCAL_PATH to the Google Drive desktop sync folder."
        ) from exc

    logger.info("Mounting Google Drive at %s", mount)
    drive.mount(str(mount), force_remount=force_remount)
    return mount


class DriveDataSync:
    """Sync DATA_DIR layout to/from a folder on Google Drive."""

    def __init__(
        self,
        *,
        mount_point: str | Path = "/content/drive",
        data_path: str = "MyDrive/video-retrieval",
        local_dir: Path,
        local_mirror: str | Path = "",
        mount_on_access: bool = True,
    ):
        if not str(data_path).strip() and not str(local_mirror).strip():
            raise ValueError("drive_data_path or drive_local_path must be configured for Drive sync")
        self.mount_point = Path(mount_point)
        self.data_path = data_path.strip("/")
        self.local_dir = Path(local_dir)
        self.local_mirror = Path(local_mirror).expanduser() if str(local_mirror).strip() else None
        self.mount_on_access = mount_on_access

    def remote_root(self) -> Path:
        if self.local_mirror is not None:
            mirror = self.local_mirror.expanduser().resolve()
            if mirror.is_dir():
                return mirror
            raise FileNotFoundError(
                f"DRIVE_LOCAL_PATH does not exist: {mirror}. "
                "Point it at the synced Google Drive folder on your laptop."
            )

        if self.mount_on_access:
            mount_google_drive(self.mount_point)
        root = self.mount_point / self.data_path if self.data_path else self.mount_point
        if not root.is_dir():
            raise FileNotFoundError(
                f"Drive data folder not found: {root}. "
                f"Create {self.data_path} on Google Drive or adjust DRIVE_DATA_PATH."
            )
        return root.resolve()

    def pull(self, *, paths: list[str] | tuple[str, ...] | None = None) -> int:
        """Copy selected subfolders from Drive into local_dir."""
        selected = list(paths or SEARCH_PULL_PATHS)
        source_root = self.remote_root()
        copied = 0
        for path_name in selected:
            copied += _copy_tree(source_root / path_name, self.local_dir / path_name)
        return copied

    def push(self, *, paths: list[str] | tuple[str, ...] | None = None) -> int:
        """Copy selected local_dir subfolders up to Drive."""
        selected = list(paths or QA_PULL_PATHS)
        dest_root = self.remote_root()
        copied = 0
        for path_name in selected:
            copied += _copy_tree(self.local_dir / path_name, dest_root / path_name)
        return copied

    def upload_file(self, local_path: Path, *, dest_relative: str) -> None:
        dest = self.remote_root() / dest_relative.strip("/")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, dest)

    def download_file(self, *, src_relative: str, local_path: Path) -> None:
        src = self.remote_root() / src_relative.strip("/")
        if not src.is_file():
            raise FileNotFoundError(src)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, local_path)

    def download_paths(self, relative_paths: list[str]) -> int:
        copied = 0
        for relative in relative_paths:
            relative = relative.strip("/")
            if not relative:
                continue
            target = self.local_dir / relative
            if target.is_file():
                continue
            src = self.remote_root() / relative
            if not src.is_file():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
            copied += 1
        return copied


def _copy_tree(source: Path, destination: Path) -> int:
    if not source.exists():
        return 0
    copied = 0
    if source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return 1

    for src in source.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(source)
        dest = destination / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        copied += 1
    return copied
