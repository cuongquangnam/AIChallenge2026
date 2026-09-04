from __future__ import annotations

import json
import logging
import shutil
import time
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

    ids = sorted(video_ids)
    logger.info("QA Drive pull: checking %s video(s): %s", len(ids), ", ".join(ids))
    settings.videos_dir.mkdir(parents=True, exist_ok=True)

    logger.info("QA Drive pull: opening Drive sync (mount=%s path=%s)", settings.drive_mount, settings.drive_data_path)
    sync = create_data_sync(settings, mount_drive=True)
    pulled: dict[str, str] = {}
    skipped = 0

    for video_id in ids:
        existing = local_video_path(settings, video_id)
        if existing is not None:
            skipped += 1
            logger.info("QA Drive pull: %s already local (%s)", video_id, existing)
            continue
        logger.info("QA Drive pull: %s missing locally; fetching from Drive ...", video_id)
        dest = _download_video_from_drive(sync, settings, video_id)
        if dest is not None:
            pulled[video_id] = dest.name
            logger.info("QA Drive pull: %s ready at %s", video_id, dest)

    logger.info(
        "QA Drive pull: done pulled=%s skipped_local=%s missing=%s",
        len(pulled),
        skipped,
        len(ids) - len(pulled) - skipped,
    )
    return pulled


def _download_video_from_drive(sync, settings: Settings, video_id: str) -> Path | None:
    from video_retrieval.storage.drive_sync import DriveDataSync

    if not isinstance(sync, DriveDataSync):
        logger.warning("QA Drive pull: sync backend is not DriveDataSync; skip %s", video_id)
        return None

    logger.info("QA Drive pull: resolving Drive root for %s ...", video_id)
    t0 = time.monotonic()
    source_root = sync.remote_root()
    logger.info(
        "QA Drive pull: Drive root ready in %.1fs (%s)",
        time.monotonic() - t0,
        source_root,
    )

    videos_remote = source_root / "videos"
    logger.info("QA Drive pull: looking up %s under %s ...", video_id, videos_remote)
    if not videos_remote.is_dir():
        logger.warning("Drive videos folder not found: %s", videos_remote)
        return None

    t_list = time.monotonic()
    try:
        candidates = sorted(videos_remote.iterdir())
    except OSError as exc:
        logger.error("QA Drive pull: failed listing %s: %s", videos_remote, exc)
        return None
    logger.info(
        "QA Drive pull: listed %s entr(y/ies) in %.1fs",
        len(candidates),
        time.monotonic() - t_list,
    )

    for candidate in candidates:
        if candidate.is_file() and candidate.stem == video_id:
            return _copy_drive_video(candidate, settings.videos_dir / candidate.name, video_id)

    for ext in _VIDEO_EXTENSIONS:
        candidate = videos_remote / f"{video_id}{ext}"
        logger.info("QA Drive pull: probing %s ...", candidate)
        if candidate.is_file():
            return _copy_drive_video(candidate, settings.videos_dir / candidate.name, video_id)

    logger.warning("QA video %s was not found under Drive:%s", video_id, videos_remote)
    return None


def _copy_drive_video(src: Path, dest: Path, video_id: str) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        size_mb = src.stat().st_size / (1024 * 1024)
    except OSError:
        size_mb = -1.0
    logger.info(
        "QA Drive pull: copying %s -> %s (%.1f MiB) ...",
        src,
        dest,
        size_mb,
    )
    t0 = time.monotonic()
    shutil.copy2(src, dest)
    logger.info(
        "QA Drive pull: copied %s in %.1fs",
        video_id,
        time.monotonic() - t0,
    )
    return dest
