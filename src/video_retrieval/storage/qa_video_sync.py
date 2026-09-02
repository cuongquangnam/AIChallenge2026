from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from video_retrieval.config import Settings
from video_retrieval.storage.data_sync import create_data_sync

logger = logging.getLogger(__name__)

_VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v")


def should_lazy_pull_qa_videos(settings: Settings) -> bool:
    """Pull QA source videos from Drive on demand (Colab remote worker only)."""
    return settings.colab_runtime


def local_video_path(settings: Settings, video_id: str) -> Path | None:
    manifest_path = settings.manifests_dir / f"{video_id}.json"
    if manifest_path.is_file():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            video_path = Path(payload["video_path"])
            if video_path.is_file():
                return video_path
            name = video_path.name
            if name:
                candidate = settings.videos_dir / name
                if candidate.is_file():
                    return candidate
        except (KeyError, TypeError, json.JSONDecodeError, OSError):
            pass

    if settings.videos_dir.is_dir():
        for path in sorted(settings.videos_dir.iterdir()):
            if path.is_file() and path.stem == video_id:
                return path
        for ext in _VIDEO_EXTENSIONS:
            candidate = settings.videos_dir / f"{video_id}{ext}"
            if candidate.is_file():
                return candidate
    return None


def ensure_qa_videos_from_drive(settings: Settings, video_ids: set[str]) -> dict[str, str]:
    """Download missing QA videos from Google Drive into ``settings.videos_dir``."""
    if not should_lazy_pull_qa_videos(settings):
        return {}

    settings.videos_dir.mkdir(parents=True, exist_ok=True)
    sync = create_data_sync(settings, mount_drive=True)
    pulled: dict[str, str] = {}

    for video_id in sorted(video_ids):
        existing = local_video_path(settings, video_id)
        if existing is not None:
            continue
        dest = _download_video_from_drive(sync, settings, video_id)
        if dest is not None:
            pulled[video_id] = dest.name
            logger.info("Pulled QA video %s from Drive to %s", video_id, dest)

    return pulled


def _download_video_from_drive(sync, settings: Settings, video_id: str) -> Path | None:
    from video_retrieval.storage.drive_sync import DriveDataSync

    if not isinstance(sync, DriveDataSync):
        return None

    source_root = sync.remote_root()
    videos_remote = source_root / "videos"
    if not videos_remote.is_dir():
        logger.warning("Drive videos folder not found: %s", videos_remote)
        return None

    for candidate in sorted(videos_remote.iterdir()):
        if candidate.is_file() and candidate.stem == video_id:
            dest = settings.videos_dir / candidate.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, dest)
            return dest

    for ext in _VIDEO_EXTENSIONS:
        candidate = videos_remote / f"{video_id}{ext}"
        if candidate.is_file():
            dest = settings.videos_dir / candidate.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, dest)
            return dest

    logger.warning("QA video %s was not found under Drive:%s", video_id, videos_remote)
    return None
