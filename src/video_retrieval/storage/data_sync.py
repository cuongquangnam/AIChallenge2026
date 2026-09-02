from __future__ import annotations

from pathlib import Path
from typing import Protocol

from video_retrieval.config import Settings
from video_retrieval.storage.drive_sync import DriveDataSync


class DataSync(Protocol):
    def pull(self, *, paths: list[str] | tuple[str, ...] | None = None) -> int: ...

    def push(self, *, paths: list[str] | tuple[str, ...] | None = None) -> int: ...

    def upload_file(self, local_path: Path, *, dest_relative: str) -> None: ...

    def download_file(self, *, src_relative: str, local_path: Path) -> None: ...

    def download_paths(self, relative_paths: list[str]) -> int: ...


def create_data_sync(
    settings: Settings,
    *,
    local_dir: Path | None = None,
    mount_drive: bool = True,
) -> DataSync:
    """Return a Google Drive sync client."""
    target = Path(local_dir) if local_dir is not None else settings.data_dir
    return DriveDataSync(
        mount_point=settings.drive_mount,
        data_path=settings.drive_data_path,
        local_dir=target,
        local_mirror=settings.drive_local_path,
        mount_on_access=mount_drive,
    )


def validate_remote_storage(settings: Settings) -> None:
    if not settings.drive_data_path.strip() and not settings.drive_local_path.strip():
        raise ValueError("DRIVE_DATA_PATH or DRIVE_LOCAL_PATH must be set for remote storage")
