from pathlib import Path
import json

import pytest

from video_retrieval.pipeline.indexer import VideoIndexer
from video_retrieval.models import FrameObjectDetections, ObjectDetection
from video_retrieval.storage.qdrant_store import QdrantStore
from tests.fakes import FakeElasticsearchStore
from tests.helpers import write_dummy_video


@pytest.mark.unit
def test_index_video_missing_raises(settings) -> None:
    indexer = VideoIndexer(
        settings,
        qdrant=QdrantStore(settings),
        es=FakeElasticsearchStore(),  # type: ignore[arg-type]
    )
    with pytest.raises(FileNotFoundError):
        indexer.index_video(Path("/no/such/video.mp4"))


@pytest.mark.unit
def test_index_video_writes_manifest_and_counts(settings, tmp_path: Path) -> None:
    video = write_dummy_video(tmp_path / "sample.mp4")
    fake_es = FakeElasticsearchStore()
    indexer = VideoIndexer(
        settings,
        qdrant=QdrantStore(settings),
        es=fake_es,  # type: ignore[arg-type]
    )

    result = indexer.index_video(video, video_id="sample")
    assert result.video_id == "sample"
    assert result.num_shots >= 1
    assert result.num_keyframes >= 3
    assert result.num_visual_points == result.num_keyframes
    assert result.num_text_docs >= 1
    assert result.audio_path is not None and result.audio_path.exists()

    manifest = settings.data_dir / "manifests" / "sample.json"
    assert manifest.exists()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    asr_docs = [doc for doc in payload["text_docs"] if doc["source"] == "asr"]
    assert asr_docs
    assert all(doc["frame_index"] is not None for doc in asr_docs)
    assert all(doc["shot_index"] is not None for doc in asr_docs)


@pytest.mark.unit
def test_index_directory_filters_extensions(settings, tmp_path: Path) -> None:
    write_dummy_video(tmp_path / "a.mp4")
    (tmp_path / "notes.txt").write_text("skip", encoding="utf-8")
    indexer = VideoIndexer(
        settings,
        qdrant=QdrantStore(settings),
        es=FakeElasticsearchStore(),  # type: ignore[arg-type]
    )
    results = indexer.index_directory(tmp_path)
    assert len(results) == 1
    assert results[0].video_id == "a"


@pytest.mark.unit
def test_index_video_visual_only_skips_text(settings, tmp_path: Path) -> None:
    video = write_dummy_video(tmp_path / "clip.mp4")
    fake_es = FakeElasticsearchStore()
    indexer = VideoIndexer(
        settings,
        qdrant=QdrantStore(settings),
        es=fake_es,  # type: ignore[arg-type]
    )
    result = indexer.index_video(video, video_id="clip", stages=["visual"])
    assert result.stages == ["visual"]
    assert result.num_visual_points >= 3
    assert result.num_text_docs == 0
    assert fake_es.docs == {}


@pytest.mark.unit
def test_index_video_ocr_only_then_asr(settings, tmp_path: Path) -> None:
    video = write_dummy_video(tmp_path / "clip.mp4")
    fake_es = FakeElasticsearchStore()
    indexer = VideoIndexer(
        settings,
        qdrant=QdrantStore(settings),
        es=fake_es,  # type: ignore[arg-type]
    )
    ocr_result = indexer.index_video(video, video_id="clip", stages=["ocr"])
    assert ocr_result.num_visual_points == 0
    assert ocr_result.num_text_docs >= 1
    assert all(doc.source == "ocr" for doc in fake_es.docs.values())

    asr_result = indexer.index_video(video, video_id="clip", stages=["asr"], reuse_extract=True)
    assert asr_result.num_text_docs >= 1
    sources = {doc.source for doc in fake_es.docs.values()}
    assert sources == {"ocr", "asr"}
    manifest = json.loads((settings.manifests_dir / "clip.json").read_text(encoding="utf-8"))
    assert {doc["source"] for doc in manifest["text_docs"]} == {"ocr", "asr"}


@pytest.mark.unit
def test_index_video_objects_writes_manifest_and_qdrant_payload(settings, tmp_path: Path) -> None:
    class _FakeDetector:
        def detect_keyframes(self, keyframes):
            return [
                FrameObjectDetections(
                    keyframe=keyframe,
                    detections=[
                        ObjectDetection(
                            label="person",
                            confidence=0.9,
                            bbox_xyxy=(1, 2, 10, 20),
                        )
                    ],
                )
                for keyframe in keyframes
            ]

    video = write_dummy_video(tmp_path / "objects.mp4")
    qdrant = QdrantStore(settings)
    indexer = VideoIndexer(
        settings,
        objects=_FakeDetector(),  # type: ignore[arg-type]
        qdrant=qdrant,
        es=FakeElasticsearchStore(),  # type: ignore[arg-type]
    )

    result = indexer.index_video(video, video_id="objects", stages=["visual", "objects"])

    assert result.num_object_detections == result.num_keyframes
    manifest = json.loads((settings.manifests_dir / "objects.json").read_text(encoding="utf-8"))
    assert len(manifest["object_detections"]) == result.num_keyframes
    assert manifest["object_detections"][0]["detections"][0]["label"] == "person"
    query = indexer.visual.encode_text("person")
    hit = qdrant.search(query, limit=1)[0]
    assert hit.payload["object_counts"] == {"person": 1}
