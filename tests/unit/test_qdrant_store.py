from pathlib import Path

import pytest

from video_retrieval.encoders.visual import VisualEncoder
from video_retrieval.models import (
    FrameObjectDetections,
    FrameRole,
    KeyFrame,
    ObjectDetection,
    VisualEmbedding,
)
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
    siglip, beit3 = encoder.encode_image(path)
    n = store.upsert_embeddings([VisualEmbedding(keyframe=kf, siglip=siglip, beit3=beit3)])
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
def test_qdrant_roundtrip_includes_compact_object_payload(settings, tmp_path: Path) -> None:
    store = QdrantStore(settings)
    encoder = VisualEncoder(settings)
    path = write_dummy_image(tmp_path / "object-frame.jpg")
    keyframe = KeyFrame(
        video_id="objects",
        shot_index=1,
        role=FrameRole.START,
        frame_index=20,
        timestamp_sec=2.0,
        path=path,
    )
    siglip, beit3 = encoder.encode_image(path)
    frames = [
        FrameObjectDetections(
            keyframe=keyframe,
            detections=[
                ObjectDetection(label="person", confidence=0.9, bbox_xyxy=(0, 0, 10, 10)),
                ObjectDetection(label="person", confidence=0.8, bbox_xyxy=(10, 0, 20, 10)),
            ],
        )
    ]
    store.upsert_embeddings(
        [VisualEmbedding(keyframe=keyframe, siglip=siglip, beit3=beit3)],
        object_detections=frames,
    )

    hit = store.search(siglip, limit=1)[0]
    assert hit.payload["objects_indexed"] is True
    assert hit.payload["objects"] == ["person"]
    assert hit.payload["object_counts"] == {"person": 2}


@pytest.mark.unit
def test_qdrant_can_add_objects_after_visual_indexing(settings, tmp_path: Path) -> None:
    store = QdrantStore(settings)
    encoder = VisualEncoder(settings)
    path = write_dummy_image(tmp_path / "late-objects.jpg")
    keyframe = KeyFrame(
        video_id="late",
        shot_index=0,
        role=FrameRole.MIDDLE,
        frame_index=5,
        timestamp_sec=0.5,
        path=path,
    )
    siglip, beit3 = encoder.encode_image(path)
    store.upsert_embeddings([VisualEmbedding(keyframe=keyframe, siglip=siglip, beit3=beit3)])
    store.set_object_payload(
        [
            FrameObjectDetections(
                keyframe=keyframe,
                detections=[
                    ObjectDetection(label="car", confidence=0.75, bbox_xyxy=(0, 0, 8, 8))
                ],
            )
        ]
    )

    hit = store.search(siglip, limit=1)[0]
    assert hit.payload["object_counts"] == {"car": 1}
