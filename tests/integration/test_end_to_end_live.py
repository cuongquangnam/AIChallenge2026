from pathlib import Path

import pytest

from video_retrieval.pipeline.indexer import VideoIndexer
from video_retrieval.search.service import SearchService
from video_retrieval.storage.elasticsearch_store import ElasticsearchStore
from video_retrieval.storage.qdrant_store import QdrantStore
from tests.helpers import write_dummy_video


@pytest.mark.integration
def test_full_pipeline_against_live_elasticsearch(settings, es_store, tmp_path: Path) -> None:
    """Requires Elasticsearch on localhost:9200; uses in-memory Qdrant."""
    video = write_dummy_video(tmp_path / "broadcast.mp4", frames=45, fps=10)
    qdrant = QdrantStore(settings)
    # Reuse the cleaned ES index from es_store fixture.
    es = ElasticsearchStore(settings, client=es_store.client)

    indexer = VideoIndexer(settings, qdrant=qdrant, es=es)
    result = indexer.index_video(video, video_id="broadcast")
    assert result.num_text_docs >= 1
    assert result.num_visual_points >= 3

    service = SearchService(settings, qdrant=qdrant, es=es)
    text = service.search_text("mock transcription for broadcast")
    assert any(h.video_id == "broadcast" for h in text.hits)

    hybrid = service.search_hybrid("mock", limit=5)
    assert hybrid.hits
