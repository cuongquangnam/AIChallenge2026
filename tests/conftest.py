from __future__ import annotations

from pathlib import Path

import pytest

from video_retrieval.config import Settings
from video_retrieval.models import FrameRole, KeyFrame
from video_retrieval.storage.elasticsearch_store import ElasticsearchStore
from video_retrieval.storage.qdrant_store import QdrantStore

from tests.helpers import write_dummy_image, write_dummy_video


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        qdrant_url=":memory:",
        elasticsearch_url="http://localhost:9200",
        qdrant_collection="test_keyframes",
        es_index="test_video_text",
        visual_backend="mock",
        ocr_backend="mock",
        asr_backend="mock",
        shot_backend="opencv",
        siglip_dim=32,
        beit3_dim=32,
    )


@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    return write_dummy_video(tmp_path / "clip.mp4")


@pytest.fixture
def sample_keyframe(tmp_path: Path) -> KeyFrame:
    path = write_dummy_image(tmp_path / "shot_0000_middle.jpg")
    return KeyFrame(
        video_id="clip",
        shot_index=0,
        role=FrameRole.MIDDLE,
        frame_index=10,
        timestamp_sec=1.0,
        path=path,
    )


@pytest.fixture
def qdrant_store(settings: Settings) -> QdrantStore:
    return QdrantStore(settings)


def _es_ping(url: str) -> bool:
    try:
        from elasticsearch import Elasticsearch

        client = Elasticsearch(url)
        return bool(client.ping())
    except Exception:
        return False


@pytest.fixture
def es_store(settings: Settings):
    if not _es_ping(settings.elasticsearch_url):
        pytest.skip("Elasticsearch is not available on localhost:9200")
    store = ElasticsearchStore(settings)
    if store.client.indices.exists(index=settings.es_index):
        store.client.indices.delete(index=settings.es_index)
    store.ensure_index()
    yield store
    if store.client.indices.exists(index=settings.es_index):
        store.client.indices.delete(index=settings.es_index)
