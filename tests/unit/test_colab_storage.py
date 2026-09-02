import json
from pathlib import Path

import pytest

from video_retrieval.config import get_settings
from video_retrieval.models import FrameRole, TextDocument
from video_retrieval.storage.factory import create_qdrant_store, create_text_store
from video_retrieval.storage.memory_text_store import MemoryTextStore
from video_retrieval.storage.qdrant_store import QdrantStore


@pytest.mark.unit
def test_get_settings_colab_uses_local_backends(tmp_path: Path) -> None:
    settings = get_settings(data_dir=tmp_path / "data", colab=True)
    assert settings.qdrant_url == "local"
    assert settings.elasticsearch_url == "http://localhost:9200"
    assert settings.colab_runtime is True


@pytest.mark.unit
def test_memory_text_store_hydrates_from_manifest(tmp_path: Path) -> None:
    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir()
    doc = TextDocument(
        doc_id="clip:ocr:0",
        video_id="clip",
        source="ocr",
        text="music awards ceremony daesang",
        shot_index=0,
        frame_index=12,
        role=FrameRole.MIDDLE,
        start_sec=4.0,
        metadata={"keyframe_path": "/tmp/frame.jpg"},
    )
    manifest = {"video_id": "clip", "text_docs": [doc.model_dump(mode="json")]}
    (manifests_dir / "clip.json").write_text(json.dumps(manifest), encoding="utf-8")

    store = MemoryTextStore(manifests_dir=manifests_dir)
    hits = store.search("daesang", source="ocr", strict=True)
    assert len(hits) == 1
    assert hits[0].video_id == "clip"
    assert hits[0].keyframe_path == "/tmp/frame.jpg"


@pytest.mark.unit
def test_memory_text_store_strict_requires_all_terms() -> None:
    store = MemoryTextStore()
    store.index_documents(
        [
            TextDocument(
                doc_id="a",
                video_id="clip",
                source="asr",
                text="the winner is announced",
            )
        ]
    )
    assert store.search("winner announced", strict=True)
    assert store.search("winner missing", strict=True) == []


@pytest.mark.unit
def test_create_text_store_uses_memory_backend(settings) -> None:
    settings = settings.model_copy(update={"elasticsearch_url": "memory"})
    store = create_text_store(settings)
    assert isinstance(store, MemoryTextStore)


@pytest.mark.unit
def test_create_qdrant_store_uses_local_path(settings, tmp_path: Path) -> None:
    settings = settings.model_copy(
        update={
            "data_dir": tmp_path / "data",
            "qdrant_url": "local",
            "qdrant_collection": "colab_keyframes",
        }
    )
    store = create_qdrant_store(settings)
    assert isinstance(store, QdrantStore)
    assert store.client is not None
