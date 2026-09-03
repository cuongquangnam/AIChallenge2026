from __future__ import annotations

from pathlib import Path

from video_retrieval.config import Settings

_MEMORY_URLS = {":memory:", "memory", "local", "file"}


def is_memory_elasticsearch(url: str) -> bool:
    return url.strip().lower() in _MEMORY_URLS


def is_local_qdrant(url: str) -> bool:
    normalised = url.strip().lower()
    return normalised in {"local", "file"} or (
        normalised not in {":memory:", "memory"}
        and not normalised.startswith("http://")
        and not normalised.startswith("https://")
    )


def qdrant_storage_path(settings: Settings) -> str:
    normalised = settings.qdrant_url.strip().lower()
    if normalised in {"local", "file"}:
        base = settings.data_dir / "qdrant"
        db = base / "db"
        if _looks_like_qdrant_storage(base) and not _looks_like_qdrant_storage(db):
            return str(base)
        return str(db)
    return settings.qdrant_url


def _looks_like_qdrant_storage(path: Path) -> bool:
    if not path.is_dir():
        return False
    markers = ("meta.json", "collections", "collection")
    if any((path / marker).exists() for marker in markers):
        return True
    return any(child.is_dir() and child.name.startswith("collection") for child in path.iterdir())
