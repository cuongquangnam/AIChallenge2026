from __future__ import annotations

from video_retrieval.config import Settings
from video_retrieval.storage.elasticsearch_store import ElasticsearchStore
from video_retrieval.storage.memory_text_store import MemoryTextStore
from video_retrieval.storage.qdrant_store import QdrantStore

__all__ = [
    "MemoryTextStore",
    "create_qdrant_store",
    "create_text_store",
]


def create_qdrant_store(settings: Settings, client=None) -> QdrantStore:
    return QdrantStore(settings, client=client)


def create_text_store(settings: Settings, client=None):
    """Return Elasticsearch or an in-memory text store based on settings."""
    from video_retrieval.storage.backends import is_memory_elasticsearch

    if is_memory_elasticsearch(settings.elasticsearch_url):
        if client is not None:
            return client
        return MemoryTextStore(manifests_dir=settings.manifests_dir)
    if client is not None:
        return ElasticsearchStore(settings, client=client)
    return ElasticsearchStore(settings)
