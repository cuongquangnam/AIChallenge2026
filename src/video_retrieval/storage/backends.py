from __future__ import annotations

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
        return str(settings.data_dir / "qdrant")
    return settings.qdrant_url
