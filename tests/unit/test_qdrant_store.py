from pathlib import Path

import pytest

from qdrant_client.http import models as qm
from video_retrieval.encoders.visual import VisualEncoder
from video_retrieval.models import FrameRole, KeyFrame, VisualEmbedding
from video_retrieval.storage.qdrant_store import QdrantStore, _stable_uuid
from tests.helpers import write_dummy_image


@pytest.mark.unit
def test_stable_uuid_is_deterministic() -> None:
    assert _stable_uuid("a") == _stable_uuid("a")
    assert _stable_uuid("a") != _stable_uuid("b")


@pytest.mark.unit
def test_qdrant_upsert_and_search_roundtrip(settings, tmp_path: Path) -> None:
    store = QdrantStore(settings)
    encoder = VisualEncoder(settings)
    path = write_dummy_image(tmp_path / "shot_0000_middle.jpg")
    kf = KeyFrame(
        video_id="clip",
        shot_index=0,
        role=FrameRole.MIDDLE,
        frame_index=10,
        timestamp_sec=1.0,
        path=path,
    )
    siglip = encoder.encode_image(path)
    n = store.upsert_embeddings([VisualEmbedding(keyframe=kf, siglip=siglip)])
    assert n == 1

    hits = store.search(siglip, vector_name="siglip", limit=5)
    assert len(hits) == 1
    assert hits[0].video_id == "clip"
    assert hits[0].source == "visual:siglip"
    assert hits[0].shot_index == 0

    filtered = store.search(siglip, vector_name="siglip", video_id="other")
    assert filtered == []


@pytest.mark.unit
def test_qdrant_upsert_empty(settings) -> None:
    store = QdrantStore(settings)
    assert store.upsert_embeddings([]) == 0


@pytest.mark.unit
def test_qdrant_recreates_legacy_dual_vector_collection(settings) -> None:
    store = QdrantStore(settings)
    store.client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config={
            "siglip": qm.VectorParams(size=32, distance=qm.Distance.COSINE),
            "beit3": qm.VectorParams(size=32, distance=qm.Distance.COSINE),
        },
    )
    store.ensure_collection()
    assert store._vector_names() == {"siglip"}
