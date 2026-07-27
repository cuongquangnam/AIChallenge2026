from pathlib import Path

import pytest

from video_retrieval.pipeline.indexer import VideoIndexer
from video_retrieval.storage.qdrant_store import QdrantStore
from tests.fakes import FakeElasticsearchStore
from tests.helpers import write_dummy_image, write_dummy_video


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
    assert "sample" in manifest.read_text(encoding="utf-8")


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
def test_index_keyframe_directory_visual_only(settings, tmp_path: Path) -> None:
    root = tmp_path / "keyframes"
    write_dummy_image(root / "L27_V001" / "001.jpg")
    write_dummy_image(root / "L27_V001" / "002.jpg")
    write_dummy_image(root / "L27_V002" / "001.jpg")
    fake_es = FakeElasticsearchStore()
    qdrant = QdrantStore(settings)
    indexer = VideoIndexer(
        settings,
        qdrant=qdrant,
        es=fake_es,  # type: ignore[arg-type]
    )

    results = indexer.index_keyframe_directory(root, limit=2)

    assert len(results) == 1
    assert results[0].video_id == "L27_V001"
    assert results[0].num_keyframes == 2
    assert results[0].num_visual_points == 2
    assert results[0].num_text_docs == 0
    assert fake_es.docs == {}
    manifest = settings.data_dir / "manifests" / "L27_V001.json"
    assert manifest.exists()
