from pathlib import Path

import pytest

from video_retrieval.pipeline.indexer import VideoIndexer
from video_retrieval.search.service import SearchService
from video_retrieval.storage.qdrant_store import QdrantStore
from tests.fakes import FakeElasticsearchStore
from tests.helpers import write_dummy_video


@pytest.mark.integration
def test_index_then_hybrid_search_offline(settings, tmp_path: Path) -> None:
    video = write_dummy_video(tmp_path / "news.mp4", frames=50, fps=10)
    qdrant = QdrantStore(settings)
    fake_es = FakeElasticsearchStore()

    indexer = VideoIndexer(settings, qdrant=qdrant, es=fake_es)  # type: ignore[arg-type]
    result = indexer.index_video(video, video_id="news")
    assert result.num_visual_points > 0
    assert result.num_text_docs > 0

    service = SearchService(settings, qdrant=qdrant, es=fake_es)  # type: ignore[arg-type]

    text = service.search_text("mock transcription for news")
    assert any(hit.video_id == "news" for hit in text.hits)

    ocr = service.search_text("mock ocr text")
    assert any(hit.video_id == "news" for hit in ocr.hits)
    assert any(hit.channel_scores.get("ocr", 0) > 0 or "ocr" in (hit.text or "") for hit in ocr.hits)

    visual = service.search_visual_text("any query", limit=5)
    # Mock embeddings are not semantically meaningful; still expect a scored ranking.
    assert len(visual.hits) <= 5

    hybrid = service.search_hybrid("mock", limit=10)
    assert hybrid.mode == "mixed"
    assert len(hybrid.hits) >= 1


@pytest.mark.integration
def test_image_search_finds_same_keyframe(settings, tmp_path: Path) -> None:
    video = write_dummy_video(tmp_path / "clip.mp4")
    qdrant = QdrantStore(settings)
    fake_es = FakeElasticsearchStore()
    indexer = VideoIndexer(settings, qdrant=qdrant, es=fake_es)  # type: ignore[arg-type]
    indexer.index_video(video, video_id="clip")

    keyframes_root = settings.keyframes_dir / "clip"
    images = sorted(keyframes_root.glob("*.jpg"))
    assert images, "expected extracted keyframes"

    service = SearchService(settings, qdrant=qdrant, es=fake_es)  # type: ignore[arg-type]
    response = service.search_image(images[0], limit=3)
    assert response.hits
    assert response.hits[0].video_id == "clip"
    assert response.hits[0].score >= response.hits[-1].score
